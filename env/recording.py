"""Shared result columns and occupancy masks for runners and plots."""
from __future__ import annotations

import csv
from itertools import chain
from numbers import Integral
from pathlib import Path

import numpy as np

from .scenario import Scenario
from .physics import _finite_real


COST_COLUMNS = ('buy_price', 'sell_price', 'purchase_sgd', 'revenue_sgd',
                'peak_increment_sgd', 'bess_wear_sgd', 'ev_wear_sgd', 'step_cost_sgd')


def csv_columns(n_slots: int) -> list:
    if isinstance(n_slots,(bool,np.bool_)) or not isinstance(n_slots,Integral) or n_slots < 1:
        raise ValueError('n_slots must be a positive integer')
    return (['strategy', 'date', 'seed', 't', 'p_grid_kw', 'p_bess_kw']
            + [f'p_ev_{i}_kw' for i in range(1,n_slots+1)] + ['e_bess_kwh']
            + [f'e_ev_{i}_kwh' for i in range(1,n_slots+1)] + list(COST_COLUMNS))


def info_to_row(info: dict, *, strategy: str) -> dict:
    """Use post-step, pre-departure-clear energies and the original root schema."""
    if not isinstance(strategy,str) or not strategy.strip():
        raise ValueError('strategy must be a nonempty string')
    power = np.asarray(info['p_ev_kw'])
    energy = np.asarray(info['e_ev_kwh'])
    if (power.ndim != 1 or power.size < 1 or energy.shape != power.shape
            or power.dtype.kind not in 'iuf' or energy.dtype.kind not in 'iuf'):
        raise ValueError('EV power and energy must be matching real numeric vectors')
    for key in ('scenario_seed','t'):
        value = info[key]
        if isinstance(value,(bool,np.bool_)) or not isinstance(value,Integral) or value < 0:
            raise ValueError(f'{key} must be a nonnegative integer')
    row = {'strategy':strategy, 'date':info['date'], 'seed':int(info['scenario_seed']),
           't':int(info['t']), 'p_grid_kw':_finite_real(info['p_grid_kw'],'p_grid_kw'),
           'p_bess_kw':_finite_real(info['p_bess_kw'],'p_bess_kw')}
    row.update({f'p_ev_{i+1}_kw':_finite_real(value,'p_ev_kw') for i,value in enumerate(power)})
    row['e_bess_kwh'] = _finite_real(info['e_bess_kwh'],'e_bess_kwh')
    row.update({f'e_ev_{i+1}_kwh':_finite_real(value,'e_ev_kwh') for i,value in enumerate(energy)})
    row.update({key:_finite_real(info[key],key) for key in COST_COLUMNS})
    if not all(np.isfinite(value) for key,value in row.items() if key not in ('strategy','date','seed','t')):
        raise ValueError('CSV numeric values must be finite')
    return row


def write_csv(path, infos, *, strategy: str) -> int:
    """Write one trajectory without rounding physical quantities or currency."""
    iterator = iter(infos)
    first = next(iterator,None)
    if first is None:
        raise ValueError('cannot write an empty trajectory')
    n_slots = len(first['p_ev_kw'])
    rows = []
    for info in chain((first,),iterator):
        if len(info['p_ev_kw']) != n_slots:
            raise ValueError('slot count changed within trajectory')
        rows.append(info_to_row(info,strategy=strategy))
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8',newline='') as handle:
        writer = csv.DictWriter(handle,fieldnames=csv_columns(n_slots))
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)



def occupancy_mask(scenario) -> np.ndarray:
    """Boolean (T,N) mask for CSV energies, including each departure's last step."""
    if isinstance(scenario,dict):
        scenario = Scenario.from_dict(scenario)
    if not isinstance(scenario,Scenario):
        raise ValueError('scenario must be a Scenario or exported dict')
    result = np.zeros((scenario.T,len(scenario.vehicles)),dtype=bool)
    for slot,vehicle in enumerate(scenario.vehicles):
        if vehicle is not None:
            result[vehicle.arrival:vehicle.departure,slot] = True
    return result
