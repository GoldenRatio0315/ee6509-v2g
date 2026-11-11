"""Immutable, serialisable scenario shared by simulation and optimisation."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date as Date
import hashlib
import json
from numbers import Integral

import numpy as np

from .data import DATA_DIR, load_day
from .physics import BatterySpec, feasible_interval, _finite_real


def parameter_id(params: dict) -> str:
    try:
        payload = json.dumps(params, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError('params must contain finite JSON-compatible values') from error
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def valid_seed(seed: int) -> int:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    return int(seed)


def _readonly(values) -> np.ndarray:
    try:
        raw = np.asarray(values)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError('scenario arrays must contain real numeric values') from error
    if raw.dtype.kind not in 'iuf':
        raise ValueError('scenario arrays must contain real numeric values')
    with np.errstate(over='ignore',invalid='ignore'):
        array = raw.astype(np.float64,copy=False)
    if not np.isfinite(array).all():
        raise ValueError('scenario arrays must be finite')
    # A bytes-backed array cannot be made writable again by setflags().
    return np.frombuffer(array.tobytes(), dtype=np.float64).reshape(array.shape)


@dataclass(frozen=True)
class Vehicle:
    arrival: int
    departure: int
    initial_kwh: float
    target_kwh: float

    def __post_init__(self):
        for key in ('arrival', 'departure'):
            value = getattr(self, key)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
                raise ValueError('vehicle times must be integer step indices')
            object.__setattr__(self, key, int(value))
        if not 0 <= self.arrival < self.departure:
            raise ValueError('vehicle requires 0 <= arrival < departure')
        for key in ('initial_kwh', 'target_kwh'):
            value = _finite_real(getattr(self, key),key)
            if not np.isfinite(value) or value < 0:
                raise ValueError('vehicle energy must be finite and nonnegative')
            object.__setattr__(self, key, value)


@dataclass(frozen=True, eq=False)
class Scenario:
    date: str
    seed: int
    params_id: str
    truth: np.ndarray
    vehicles: tuple
    bess_initial_kwh: float
    initial_peak_kw: float
    noise: np.ndarray

    def __post_init__(self):
        if not isinstance(self.date, str):
            raise ValueError('date must be an ISO date string')
        Date.fromisoformat(self.date)
        object.__setattr__(self, 'seed', valid_seed(self.seed))
        truth = _readonly(self.truth)
        if truth.ndim != 2 or truth.shape[0] != 3 or not 1 <= truth.shape[1] <= 48:
            raise ValueError('truth must have shape (3,T), 1 <= T <= 48')
        if np.any(truth[1:] < 0):
            raise ValueError('load/PV truth must be nonnegative')
        noise = _readonly(self.noise)
        if noise.shape != (3, truth.shape[1]+1, 12):
            raise ValueError('noise must have shape (3,T+1,12)')
        vehicles = tuple(self.vehicles)
        if not vehicles or any(v is not None and not isinstance(v, Vehicle) for v in vehicles):
            raise ValueError('vehicles must contain Vehicle or None for each slot')
        if any(v is not None and v.departure > truth.shape[1] for v in vehicles):
            raise ValueError('departure exceeds episode horizon')
        if not isinstance(self.params_id, str) or not self.params_id:
            raise ValueError('params_id is required')
        for key in ('bess_initial_kwh', 'initial_peak_kw'):
            value = _finite_real(getattr(self, key),key)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f'{key} must be finite and nonnegative')
            object.__setattr__(self, key, value)
        object.__setattr__(self, 'truth', truth)
        object.__setattr__(self, 'noise', noise)
        object.__setattr__(self, 'vehicles', vehicles)

    @property
    def T(self) -> int:
        return self.truth.shape[1]

    def to_dict(self) -> dict:
        return {'date': self.date, 'seed': self.seed, 'params_id': self.params_id,
                'truth': self.truth.tolist(),
                'vehicles': [asdict(v) if v is not None else None for v in self.vehicles],
                'bess_initial_kwh': self.bess_initial_kwh, 'initial_peak_kw': self.initial_peak_kw,
                'noise': self.noise.tolist()}

    @classmethod
    def from_dict(cls, values: dict) -> Scenario:
        try:
            values = dict(values)
            values['vehicles'] = tuple(Vehicle(**v) if v is not None else None for v in values['vehicles'])
            return cls(**values)
        except (TypeError, KeyError) as error:
            raise ValueError('invalid scenario fields') from error


def _arrival_departure_distribution(group, prefix):
    mean, sd, lower, upper = (
        _finite_real(group[f'{prefix}_{suffix}_h'], f'{prefix}_{suffix}_h')
        for suffix in ('mean', 'sd', 'min', 'max'))
    if sd <= 0 or not 0 <= lower < upper <= 24:
        raise ValueError(f'{prefix} requires positive sd and 0 <= min < max <= 24 h')
    return mean, sd, lower, upper


def _truncated_hour(rng, distribution):
    mean, sd, lower, upper = distribution
    for _ in range(100000):
        sample = float(rng.normal(mean, sd))
        if lower <= sample <= upper:
            return sample
    raise ValueError('truncated normal sampling failed after 100000 attempts')


def generate_scenario(date, seed, n_slots, params,
                      timeseries_dir=DATA_DIR / 'timeseries') -> Scenario:
    """Create one frozen day using fixed truth and independent random streams.

    Each physical slot draws arrival, departure, then initial SoC from its own
    stream. Forecast errors use separate variable streams and already include
    their horizon-dependent sigma. Changing fleet size does not alter old slots
    or forecast noise, and no global RNG state is read or modified.
    """
    if isinstance(n_slots, (bool, np.bool_)) or not isinstance(n_slots, Integral) or n_slots < 1:
        raise ValueError('n_slots must be a positive integer')
    n_slots = int(n_slots)
    seed = valid_seed(seed)
    if not isinstance(date, str):
        raise ValueError('date must be an ISO date string')
    day = Date.fromisoformat(date)
    data = params['data']
    first, last = Date.fromisoformat(data['first_date']), Date.fromisoformat(data['last_date'])
    if not first <= day <= last:
        raise ValueError('date must lie within the supplied data range')
    sim = params['simulation']
    if (sim['delta_t_h'], sim['steps_per_day'], sim['forecast_steps']) != (.5, 48, 12):
        raise ValueError('scenario generation requires 30-minute, 48-step days and 12 forecasts')
    dt = .5
    ev_group, bess_group = params['ev'], params['bess']
    ev, bess = BatterySpec.from_params(ev_group), BatterySpec.from_params(bess_group)
    arrival = _arrival_departure_distribution(ev_group, 'arrival')
    departure = _arrival_departure_distribution(ev_group, 'departure')
    initial_min = _finite_real(ev_group['initial_soc_min'], 'initial_soc_min')
    initial_max = _finite_real(ev_group['initial_soc_max'], 'initial_soc_max')
    if not ev.soc_min <= initial_min < initial_max <= ev.soc_max:
        raise ValueError('initial SoC range must be ordered and within EV physical bounds')
    target = ev.capacity_kwh * _finite_real(ev_group['departure_target_soc'],
                                               'departure_target_soc')
    if not ev.energy_min <= target <= ev.energy_max:
        raise ValueError('infeasible_input: EV target outside physical energy bounds')
    for key, expected in [('weekday_cars_per_slot', 1), ('weekend_cars_per_slot', 0)]:
        value = ev_group[key]
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value != expected:
            raise ValueError('only one car per weekday slot and no weekend cars are supported')
    bess_initial = bess.capacity_kwh * _finite_real(bess_group['initial_soc'], 'initial_soc')
    feasible_interval(bess_initial, bess_initial, 48, bess, dt)
    initial_peak = _finite_real(sim['initial_running_peak_kw'], 'initial_running_peak_kw')
    if initial_peak < 0:
        raise ValueError('initial_running_peak_kw must be nonnegative')

    forecast = params['forecast']
    sigmas = []
    for name in ('usep', 'load', 'pv'):
        start = _finite_real(forecast[f'{name}_sigma_at_h1'], f'{name}_sigma_at_h1')
        increment = _finite_real(forecast[f'{name}_sigma_increment'],
                                     f'{name}_sigma_increment')
        if start < 0 or increment < 0:
            raise ValueError('forecast sigma and increments must be nonnegative')
        sigma = start + increment * np.arange(12, dtype=np.float64)
        if not np.isfinite(sigma).all():
            raise ValueError('forecast sigma must be finite at every horizon')
        sigmas.append(sigma)

    source = load_day(day.isoformat(), timeseries_dir)
    truth = np.stack([source['usep'], source['load'], source['pv']])
    vehicles = []
    for slot in range(n_slots):
        if day.weekday() >= 5:
            vehicles.append(None)
            continue
        rng = np.random.default_rng(np.random.SeedSequence([seed, day.toordinal(), 10, slot]))
        arrival_step = int(np.floor(_truncated_hour(rng, arrival) / dt + .5))
        departure_step = int(np.floor(_truncated_hour(rng, departure) / dt + .5))
        initial = float(rng.uniform(initial_min, initial_max)) * ev.capacity_kwh
        vehicle = Vehicle(arrival_step, departure_step, initial, target)
        if vehicle.departure > 48:
            raise ValueError('vehicle departure exceeds the day horizon')
        feasible_interval(initial, target, departure_step - arrival_step, ev, dt)
        vehicles.append(vehicle)
    noise = np.empty((3, 49, 12), dtype=np.float64)
    for variable, sigma in enumerate(sigmas):
        rng = np.random.default_rng(np.random.SeedSequence([seed, day.toordinal(), 20, variable]))
        noise[variable] = rng.normal(0., sigma, size=(49, 12))
    return Scenario(day.isoformat(), seed, parameter_id(params), truth, tuple(vehicles),
                    bess_initial, initial_peak, noise)
