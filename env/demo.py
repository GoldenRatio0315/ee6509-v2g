"""Run one fixed day with an independent random-action stream and save its data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform

import gymnasium
import numpy as np

from .recording import write_csv
from .v2g_env import V2GEnv


def run_demo(*, date='2026-10-01', seed=0, n_slots=5, action_seed=6509,
             output_dir='results/env_demo'):
    environment = V2GEnv(n_slots=n_slots)
    try:
        obs, _ = environment.reset(seed=seed,options={'date':date})
        scenario = environment.export_scenario()
        rng = np.random.default_rng(action_seed)
        infos = []
        while True:
            obs, _, terminated, truncated, info = environment.step(rng.uniform(-1,1,n_slots+1))
            infos.append(info)
            if terminated or truncated:
                break
        summary = dict(infos[-1]['episode_summary'])
        summary.update(strategy='random', action_seed=action_seed, n_slots=n_slots,
                       python=platform.python_version(),numpy=np.__version__,gymnasium=gymnasium.__version__)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True,exist_ok=True)
        stem = f'random_{date}_seed{seed}_n{n_slots}'
        csv_path = output_dir / f'{stem}.csv'
        write_csv(csv_path,infos,strategy='random')
        (output_dir / f'{stem}.scenario.json').write_text(
            json.dumps(scenario,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        (output_dir / f'{stem}.summary.json').write_text(
            json.dumps(summary,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        return summary,csv_path
    finally:
        environment.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--date',default='2026-10-01')
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--n-slots',type=int,default=5)
    parser.add_argument('--action-seed',type=int,default=6509)
    parser.add_argument('--output-dir',default='results/env_demo')
    args = parser.parse_args()
    summary,path = run_demo(**vars(args))
    print(json.dumps(summary,indent=2,allow_nan=False))
    print(f'CSV: {path}')


if __name__ == '__main__':
    main()
