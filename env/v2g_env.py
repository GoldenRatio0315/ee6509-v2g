"""Half-hourly building BESS and V2G fleet scheduling environment."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from numbers import Integral, Real
from pathlib import Path
import math

import gymnasium as gym
import numpy as np

from .data import DATA_DIR, ROOT, load_day, load_params, split_dates
from .physics import (BatterySpec, E_TOL, P_TOL, POWER_INPUT_TOL, advance_energy,
                      feasible_interval, settle_cost, _nonnegative)
from .scenario import Scenario, parameter_id, valid_seed


class V2GEnv(gym.Env):
    """One local day per episode; power order is BESS then physical EV slots."""
    metadata = {'render_modes': []}

    def __init__(self, n_slots: int = 5, params: dict | None = None,
                 split: str = 'train', timeseries_dir: Path = DATA_DIR / 'timeseries'):
        if isinstance(n_slots, (bool, np.bool_)) or not isinstance(n_slots, Integral) or n_slots < 1:
            raise ValueError('n_slots must be a positive integer')
        self.n_slots = int(n_slots)
        self._params = deepcopy(load_params() if params is None else params)
        self.params_id = parameter_id(self._params)
        self.split = split
        self.dates = split_dates(self._params, split)
        self.timeseries_dir = Path(timeseries_dir)
        self.bess = BatterySpec.from_params(self._params['bess'])
        self.ev = BatterySpec.from_params(self._params['ev'])
        physical_values = [value for spec in (self.bess,self.ev) for value in
                           (spec.capacity_kwh,spec.charge_max_kw,spec.discharge_max_kw)]
        if max(physical_values) > np.finfo(np.float32).max:
            raise ValueError('battery scale cannot be represented in float32 observations')
        sim = self._params['simulation']
        if (sim['delta_t_h'], sim['steps_per_day'], sim['history_steps'], sim['forecast_steps']) != (.5,48,8,12):
            raise ValueError('environment requires 30-minute, 48-step days and 8/12 observation windows')
        self.dt = float(sim['delta_t_h'])
        tariff = self._params['tariff']
        divisor = _nonnegative(tariff['month_to_day_divisor'],'month_to_day_divisor')
        if divisor == 0:
            raise ValueError('month_to_day_divisor must be positive')
        ucc_month = _nonnegative(tariff['ucc_month_sgd_per_kw'],'ucc_month_sgd_per_kw')
        contract_month = _nonnegative(tariff['contract_month_sgd_per_kw'],'contract_month_sgd_per_kw')
        self._ucc = ucc_month / divisor
        self._adder = _nonnegative(tariff['buy_adder_sgd_per_kwh'],'buy_adder_sgd_per_kwh')
        self._contract = _nonnegative(self._params['building']['contract_capacity_kw'],'contract_capacity_kw')
        self._pv_limit = _nonnegative(self._params['pv']['installed_kwp'],'installed_kwp')
        self._fixed_pv = _nonnegative(self._params['pv']['daily_fixed_cost_sgd'],'daily_fixed_cost_sgd')
        self._fixed_contract = contract_month * self._contract / divisor
        if not np.isfinite(self._ucc) or not np.isfinite(self._fixed_contract):
            raise ValueError('derived daily tariffs must be finite')
        self.action_space = gym.spaces.Box(-1.,1.,shape=(self.n_slots+1,),dtype=np.float32)
        low_ts = np.vstack([np.full(20,-np.inf),np.zeros((2,20))]).astype(np.float32)
        self.observation_space = gym.spaces.Dict({
            'timeseries': gym.spaces.Box(low_ts,np.full((3,20),np.inf,dtype=np.float32),dtype=np.float32),
            'global': gym.spaces.Box(np.array([0,0,-1,-1],np.float32),
                                     np.array([1,np.inf,1,1],np.float32),dtype=np.float32),
            'ev': gym.spaces.Box(np.zeros((self.n_slots,4),np.float32),
                                 np.tile(np.array([1,1,48,self.ev.capacity_kwh],np.float32),(self.n_slots,1)),dtype=np.float32),
            'ev_present': gym.spaces.MultiBinary(self.n_slots)})
        self._scenario = None
        self._done = True

    @property
    def params(self):
        """A detached copy: editing caller parameters cannot change an active episode."""
        return deepcopy(self._params)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self._done = True
        self._scenario = None
        if seed is not None:
            seed = valid_seed(seed)
        super().reset(seed=seed)
        options = {} if options is None else dict(options)
        if set(options) - {'date','scenario'}:
            raise ValueError('unsupported reset option')
        if 'date' in options and 'scenario' in options:
            raise ValueError('scenario and date are mutually exclusive')
        self._done = True
        if 'scenario' in options:
            scenario = options['scenario']
            if isinstance(scenario,dict):
                scenario = Scenario.from_dict(scenario)
            if not isinstance(scenario,Scenario):
                raise ValueError('scenario must be a Scenario or exported dict')
            # Read-only NumPy data still permits shape/dtype metadata changes.
            # Own an independent, revalidated snapshot of caller-supplied arrays.
            scenario = replace(scenario)
        else:
            scenario = self._sample_scenario(seed,options)
        if len(scenario.vehicles) != self.n_slots or scenario.params_id != self.params_id:
            raise ValueError('scenario slot count or params_id mismatch')
        feasible_interval(scenario.bess_initial_kwh,scenario.bess_initial_kwh,scenario.T,self.bess,self.dt)
        for vehicle in scenario.vehicles:
            if vehicle is not None:
                feasible_interval(vehicle.initial_kwh,vehicle.target_kwh,
                                  vehicle.departure-vehicle.arrival,self.ev,self.dt)
        timeseries = self._make_timeseries(scenario)
        if max(scenario.initial_peak_kw, float(scenario.truth[1].max()) + self.bess.charge_max_kw
               + self.n_slots*self.ev.charge_max_kw) > np.finfo(np.float32).max:
            raise ValueError('scenario peak cannot be represented in float32 observations')
        self._scenario = scenario
        self._timeseries = timeseries
        self.t = 0
        self._bess_energy = scenario.bess_initial_kwh
        self._ev_energy = np.zeros(self.n_slots,dtype=np.float64)
        self._peak = scenario.initial_peak_kw
        self._step_costs = []
        self._departures = []
        self._arrive()
        self._done = False
        return self._observation(), self._metadata()

    def _sample_scenario(self,seed,options):
        from .scenario import generate_scenario
        explicit_date = 'date' in options
        date = options['date'] if explicit_date else str(self.np_random.choice(self.dates))
        if seed is not None and explicit_date:
            scenario_seed = seed
        else:
            pool = self._params['data']['test_ev_noise_seeds' if self.split == 'test' else 'train_ev_noise_seeds']
            scenario_seed = int(self.np_random.integers(pool[0],pool[1]+1))
        return generate_scenario(date,scenario_seed,self.n_slots,self._params,self.timeseries_dir)

    def _metadata(self):
        return {'date':self._scenario.date,'scenario_seed':self._scenario.seed,
                'params_id':self.params_id,'t':self.t}

    def export_scenario(self) -> dict:
        if self._scenario is None:
            raise RuntimeError('reset required before exporting a scenario')
        return self._scenario.to_dict()

    def _present(self):
        return np.array([v is not None and v.arrival <= self.t < v.departure
                         for v in self._scenario.vehicles],dtype=np.int8)

    def _arrive(self):
        for i, vehicle in enumerate(self._scenario.vehicles):
            if vehicle is not None and vehicle.arrival == self.t:
                self._ev_energy[i] = vehicle.initial_kwh

    def _check_live(self):
        if self._scenario is None or self._done:
            raise RuntimeError('reset required before stepping an uninitialised or terminated episode')

    @staticmethod
    def _correction(device,kind,delta,unit):
        return {'device':int(device),'kind':kind,'delta':float(delta),'unit':unit}

    def _intervals(self):
        self._check_live()
        bounds = np.zeros((self.n_slots+1,2),dtype=np.float64)
        corrections = []
        lo,hi,delta = feasible_interval(self._bess_energy,self._scenario.bess_initial_kwh,
                                       self._scenario.T-self.t,self.bess,self.dt)
        bounds[0] = (lo,hi)
        if delta:
            corrections.append(self._correction(0,'interval_lo',delta,'kW'))
        for i,vehicle in enumerate(self._scenario.vehicles):
            if vehicle is not None and vehicle.arrival <= self.t < vehicle.departure:
                lo,hi,delta = feasible_interval(self._ev_energy[i],vehicle.target_kwh,
                                               vehicle.departure-self.t,self.ev,self.dt)
                bounds[i+1] = (lo,hi)
                if delta:
                    corrections.append(self._correction(i+1,'interval_lo',delta,'kW'))
        return bounds,corrections

    def feasible_intervals(self) -> np.ndarray:
        return self._intervals()[0]

    def _vector(self,value,name):
        array = np.asarray(value)
        if array.shape != (self.n_slots+1,) or array.dtype.kind not in 'iuf':
            raise ValueError(f'{name} must be a numeric vector of shape ({self.n_slots+1},)')
        array = array.astype(np.float64,copy=True)
        if not np.isfinite(array).all():
            raise ValueError(f'{name} must be finite')
        return array

    def step(self,action):
        bounds,corrections = self._intervals()
        a = np.clip(self._vector(action,'action'),-1.,1.)
        mapped = bounds[:,0] + (a+1.)*.5*(bounds[:,1]-bounds[:,0])
        power = np.clip(mapped,bounds[:,0],bounds[:,1])
        for device in np.flatnonzero(power != mapped):
            delta = power[device]-mapped[device]
            if abs(delta) > P_TOL:
                raise ValueError('action mapping roundoff exceeds power tolerance')
            corrections.append(self._correction(device,'power',delta,'kW'))
        return self._settle(power,bounds,corrections)

    def step_power(self,power,tol: float = POWER_INPUT_TOL):
        bounds,corrections = self._intervals()
        if isinstance(tol,(bool,np.bool_)) or not isinstance(tol,Real) or not np.isfinite(tol) or tol < 0:
            raise ValueError('tol must be finite and nonnegative')
        p = self._vector(power,'power')
        if np.any(p < bounds[:,0]-tol) or np.any(p > bounds[:,1]+tol):
            raise ValueError('infeasible_input: power outside feasible interval')
        clipped = np.clip(p,bounds[:,0],bounds[:,1])
        for device in np.flatnonzero(clipped != p):
            corrections.append(self._correction(device,'power',clipped[device]-p[device],'kW'))
        return self._settle(clipped,bounds,corrections)

    def _settle(self,power,bounds,corrections):
        present = self._present()
        next_bess,delta = advance_energy(self._bess_energy,power[0],self.bess,self.dt)
        if delta:
            corrections.append(self._correction(0,'energy',delta,'kWh'))
        next_ev = self._ev_energy.copy()
        departures = []
        for i in np.flatnonzero(present):
            next_ev[i],delta = advance_energy(self._ev_energy[i],power[i+1],self.ev,self.dt)
            if delta:
                corrections.append(self._correction(i+1,'energy',delta,'kWh'))
            vehicle = self._scenario.vehicles[i]
            if vehicle.departure == self.t+1:
                success = bool(next_ev[i] >= vehicle.target_kwh-E_TOL)
                if not success:
                    raise ValueError('infeasible_input: departure target not met')
                departures.append({'slot':int(i),'departure_t':self.t+1,'energy_kwh':float(next_ev[i]),
                                   'target_kwh':vehicle.target_kwh,'success':success})
        terminal = self.t+1 == self._scenario.T
        if terminal and next_bess < self._scenario.bess_initial_kwh-E_TOL:
            raise ValueError('infeasible_input: BESS terminal target not met')
        usep,load,pv = self._scenario.truth[:,self.t]
        info = self._metadata()
        info.update(settle_cost(load_kw=load,pv_kw=pv,usep=usep,power=power,bess=self.bess,ev=self.ev,
                                peak_before_kw=self._peak,contract_capacity_kw=self._contract,
                                ucc_daily=self._ucc,adder=self._adder,dt=self.dt))
        info.update({'intervals':bounds.copy(),'ev_present':present.copy(),
                     'e_bess_before_kwh':self._bess_energy,'e_ev_before_kwh':self._ev_energy.copy(),
                     'e_bess_kwh':next_bess,'e_ev_kwh':next_ev.copy(),
                     'departures':deepcopy(departures),'corrections':deepcopy(corrections)})
        self._bess_energy = next_bess
        self._ev_energy = next_ev
        self._peak = info['peak_after_kw']
        self._step_costs.append(info['step_cost_sgd'])
        self._departures.extend(deepcopy(departures))
        for record in departures:
            self._ev_energy[record['slot']] = 0.
        self.t += 1
        self._arrive()
        self._done = terminal
        if terminal:
            info['episode_summary'] = self._summary()
        return self._observation(),-info['step_cost_sgd'],terminal,False,info

    def _summary(self):
        full_day = self._scenario.T == 48
        variable = math.fsum(self._step_costs)
        fixed_pv = self._fixed_pv if full_day else 0.
        fixed_contract = self._fixed_contract if full_day else 0.
        return {'date':self._scenario.date,'scenario_seed':self._scenario.seed,'params_id':self.params_id,
                'variable_cost_sgd':variable,'fixed_pv_sgd':fixed_pv,'fixed_contract_sgd':fixed_contract,
                'total_cost_sgd':variable+fixed_pv+fixed_contract,'peak_grid_kw':self._peak,
                'bess_final_kwh':self._bess_energy,'departures':deepcopy(self._departures),
                'departure_success_rate':(sum(v['success'] for v in self._departures)/len(self._departures)
                                          if self._departures else None)}

    def _make_timeseries(self,scenario):
        issues = np.arange(scenario.T+1)[:,None]
        history_indices = np.clip(issues+np.arange(-7,1),0,scenario.T-1)
        forecast_indices = np.clip(issues+np.arange(1,13),0,scenario.T-1)
        past = scenario.truth[:,history_indices]
        with np.errstate(over='ignore',invalid='ignore'):
            future = scenario.truth[:,forecast_indices]*(1+scenario.noise)
        # Overflow is invalid even when subsequent physical clipping could hide it.
        if not np.isfinite(future).all():
            raise ValueError('nonfinite forecast')
        future[1] = np.maximum(future[1],0.)
        future[2] = np.clip(future[2],0.,self._pv_limit)
        result = np.concatenate([past,future],axis=2).transpose(1,0,2)
        if np.any(np.abs(result) > np.finfo(np.float32).max):
            raise ValueError('scenario cannot be represented in float32 observations')
        return result.astype(np.float32)

    def _observation(self):
        scenario = self._scenario
        ev_state = np.zeros((self.n_slots,4),dtype=np.float32)
        present = self._present()
        for i in np.flatnonzero(present):
            vehicle = scenario.vehicles[i]
            ev_state[i] = (1.,self._ev_energy[i]/self.ev.capacity_kwh,vehicle.departure-self.t,
                           max(0.,vehicle.target_kwh-self._ev_energy[i]))
        angle = 2*np.pi*self.t/48
        return {'timeseries':self._timeseries[self.t].copy(),
                'global':np.array([self._bess_energy/self.bess.capacity_kwh,self._peak,
                                   np.sin(angle),np.cos(angle)],dtype=np.float32),
                'ev':ev_state,'ev_present':present}
