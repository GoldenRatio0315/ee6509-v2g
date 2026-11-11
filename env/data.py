"""Read the pinned parameter values and half-hourly source data."""
from __future__ import annotations

import csv
import json
from datetime import date as Date, datetime, time, timedelta, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data' / 'D'


def load_params(path: Path = DATA_DIR / 'params.yaml') -> dict:
    raw = json.loads(Path(path).read_text(encoding='utf-8'))
    return {group: {key: leaf['value'] for key, leaf in items.items()
                    if isinstance(leaf, dict) and 'value' in leaf}
            for group, items in raw.items() if isinstance(items, dict)}


def load_day(date: str, timeseries_dir: Path = DATA_DIR / 'timeseries') -> dict:
    """Read 48 ordered half-hours for the requested Singapore local date."""
    try:
        day = Date.fromisoformat(date)
    except (TypeError, ValueError):
        raise ValueError('date must be an ISO calendar date') from None
    columns = [('usep', 'usep_sgd_per_kwh'), ('load', 'load_kw'), ('pv', 'pv_kw')]
    required = ['t', 'start_local'] + [column for _, column in columns]
    with open(Path(timeseries_dir) / f'{day.isoformat()}.csv', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if any(fieldnames.count(column) != 1 for column in required):
            raise ValueError(f'{date}: each required column must appear once: {required}')
        rows = list(reader)
    if len(rows) != 48:
        raise ValueError(f'{date}: expected 48 consecutive half-hour rows')
    local_offset = timedelta(hours=8)
    midnight = datetime.combine(day, time.min, timezone(local_offset))
    arrays = {key: np.empty(48, dtype=np.float64) for key, _ in columns}
    for t, row in enumerate(rows):
        try:
            index = int(row['t'])
            stamp = datetime.fromisoformat(row['start_local'])
            values = [float(row[column]) for _, column in columns]
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f'{date}: invalid index, timestamp or numeric value at row {t}') from None
        if index != t:
            raise ValueError(f'{date}: row indices must be ordered from 0 through 47')
        if stamp.utcoffset() != local_offset or stamp != midnight + timedelta(minutes=30 * t):
            raise ValueError(f'{date}: row {t} must match the UTC+08:00 half-hour boundary')
        for (key, _), value in zip(columns, values):
            arrays[key][t] = value
    if not all(np.isfinite(a).all() for a in arrays.values()):
        raise ValueError(f'{date}: timeseries values must be finite')
    if np.any(arrays['load'] < 0) or np.any(arrays['pv'] < 0):
        raise ValueError(f'{date}: load and PV must be nonnegative')
    return arrays


def split_dates(params: dict, split: str) -> tuple:
    data = params['data']
    bounds = {'train': (data['first_date'], data['train_last_date']),
              'validation': ((Date.fromisoformat(data['train_last_date']) + timedelta(days=1)).isoformat(),
                             data['validation_last_date']),
              'test': ((Date.fromisoformat(data['validation_last_date']) + timedelta(days=1)).isoformat(),
                       data['last_date'])}
    if split not in bounds:
        raise ValueError('split must be train, validation or test')
    first, last = map(Date.fromisoformat, bounds[split])
    if first > last:
        raise ValueError('invalid split dates')
    return tuple((first + timedelta(days=i)).isoformat() for i in range((last-first).days+1))
