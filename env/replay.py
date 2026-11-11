"""Replay a physical schedule through the same settlement used for control."""
from __future__ import annotations

import numpy as np

from .scenario import Scenario
from .v2g_env import V2GEnv


def replay_power_schedule(scenario, powers, *, params=None, tol=1e-6):
    """Return (step_infos, episode_summary) for a complete (T,N+1) kW schedule.

    An optimisation objective must use the same variable-cost or total-cost
    convention before comparison. No solver or objective is supplied here.
    """
    if isinstance(scenario,dict):
        scenario = Scenario.from_dict(scenario)
    if not isinstance(scenario,Scenario):
        raise ValueError('scenario must be a Scenario or exported dict')
    schedule = np.asarray(powers)
    expected = (scenario.T,len(scenario.vehicles)+1)
    if schedule.shape != expected or schedule.dtype.kind not in 'iuf' or not np.isfinite(schedule).all():
        raise ValueError(f'powers must be a finite numeric schedule of shape {expected}')
    environment = V2GEnv(n_slots=len(scenario.vehicles),params=params)
    try:
        environment.reset(options={'scenario':scenario})
        infos = [environment.step_power(power,tol=tol)[4] for power in schedule]
        return infos, infos[-1]['episode_summary']
    finally:
        environment.close()
