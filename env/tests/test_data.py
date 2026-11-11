"""Daily input validation against the pinned local half-hour time grid."""
from copy import deepcopy
import csv
from datetime import date as Date, datetime, time, timedelta, timezone
from pathlib import Path
import tempfile
import unittest

import numpy as np

from env.data import DATA_DIR, load_day, load_params


class DataTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.date = '2026-10-01'
        self.fields = ['t', 'start_local', 'usep_sgd_per_kwh', 'load_kw', 'pv_kw']
        midnight = datetime.combine(Date.fromisoformat(self.date), time.min,
                                    timezone(timedelta(hours=8)))
        self.rows = [
            {'t': str(t),
             'start_local': (midnight + timedelta(minutes=30 * t)).isoformat(timespec='minutes'),
             'usep_sgd_per_kwh': '.20', 'load_kw': str(100 + t), 'pv_kw': '5'}
            for t in range(48)
        ]

    def write_day(self, rows=None, fields=None):
        with (self.directory / f'{self.date}.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=self.fields if fields is None else fields,
                                    extrasaction='ignore')
            writer.writeheader()
            writer.writerows(self.rows if rows is None else rows)

    def test_all_28_real_days(self):
        files = sorted((DATA_DIR / 'timeseries').glob('*.csv'))
        self.assertEqual(len(files), 28)
        total = 0
        for path in files:
            with self.subTest(date=path.stem):
                arrays = load_day(path.stem)
                self.assertEqual(set(arrays), {'usep', 'load', 'pv'})
                for values in arrays.values():
                    self.assertEqual(values.shape, (48,))
                    self.assertEqual(values.dtype, np.float64)
                    self.assertTrue(np.isfinite(values).all())
                self.assertTrue((arrays['load'] >= 0).all())
                self.assertTrue((arrays['pv'] >= 0).all())
                total += arrays['usep'].size
        self.assertEqual(total, 1344)

    def test_parameter_value_interface_preserved(self):
        params = load_params()
        self.assertEqual(params['simulation']['delta_t_h'], .5)
        self.assertEqual(params['ev']['capacity_kwh'], 40)
        self.assertEqual(params['bess']['capacity_kwh'], 120)
        self.assertEqual(params['tariff']['buy_adder_sgd_per_kwh'], .03)

    def test_extra_columns_and_negative_price_are_allowed(self):
        self.rows[0]['usep_sgd_per_kwh'] = '-.10'
        for row in self.rows:
            row['source_note'] = 'local fixture'
        self.write_day(fields=self.fields + ['source_note'])
        arrays = load_day(self.date, self.directory)
        self.assertEqual(arrays['usep'][0], -.10)
        np.testing.assert_array_equal(arrays['load'], np.arange(100, 148))
        np.testing.assert_array_equal(arrays['pv'], np.full(48, 5))

    def test_missing_required_columns(self):
        for missing in self.fields:
            with self.subTest(missing=missing):
                self.write_day(fields=[field for field in self.fields if field != missing])
                with self.assertRaises(ValueError):
                    load_day(self.date, self.directory)
        (self.directory / f'{self.date}.csv').write_text('')
        with self.assertRaises(ValueError):
            load_day(self.date, self.directory)

    def test_duplicate_required_column(self):
        self.write_day(fields=self.fields + ['pv_kw'])
        with self.assertRaises(ValueError):
            load_day(self.date, self.directory)

    def test_wrong_row_count(self):
        for rows in [self.rows[:-1], self.rows + [self.rows[-1]], []]:
            with self.subTest(count=len(rows)):
                self.write_day(rows=rows)
                with self.assertRaises(ValueError):
                    load_day(self.date, self.directory)

    def test_disordered_duplicate_and_invalid_indices(self):
        for indices in [(1, 0), (0, 0), ('invalid', 1), ('0.0', 1), (None, 1)]:
            with self.subTest(indices=indices):
                rows = deepcopy(self.rows)
                rows[0]['t'], rows[1]['t'] = indices
                self.write_day(rows)
                with self.assertRaises(ValueError):
                    load_day(self.date, self.directory)

    def test_wrong_date_offset_and_time_boundary(self):
        for stamp in ['2026-10-02T00:00+08:00', '2026-10-01T00:00+09:00',
                      '2026-09-30T16:00+00:00', '2026-10-01T00:00',
                      '2026-10-01T00:15+08:00', '2026-10-01T00:30+08:00',
                      '2026-10-01T00:00:01+08:00', 'invalid', None]:
            with self.subTest(stamp=stamp):
                rows = deepcopy(self.rows)
                rows[0]['start_local'] = stamp
                self.write_day(rows)
                with self.assertRaises(ValueError):
                    load_day(self.date, self.directory)
        rows = deepcopy(self.rows)
        rows[47]['start_local'] = '2026-10-02T00:00+08:00'
        self.write_day(rows)
        with self.assertRaises(ValueError):
            load_day(self.date, self.directory)

    def test_nonfinite_invalid_and_missing_numeric_values(self):
        for column in ['usep_sgd_per_kwh', 'load_kw', 'pv_kw']:
            for value in ['nan', 'inf', '-inf', '1e999', 'not-a-number', None]:
                with self.subTest(column=column, value=value):
                    rows = deepcopy(self.rows)
                    rows[8][column] = value
                    self.write_day(rows)
                    with self.assertRaises(ValueError):
                        load_day(self.date, self.directory)
        rows = deepcopy(self.rows)
        del rows[8]['pv_kw']
        self.write_day(rows)
        with self.assertRaises(ValueError):
            load_day(self.date, self.directory)

    def test_negative_load_and_pv(self):
        for column in ['load_kw', 'pv_kw']:
            rows = deepcopy(self.rows)
            rows[8][column] = '-.001'
            self.write_day(rows)
            with self.subTest(column=column), self.assertRaises(ValueError):
                load_day(self.date, self.directory)

    def test_invalid_requested_date(self):
        for date in [None, 'not-a-date', '2026-13-01']:
            with self.subTest(date=date), self.assertRaises(ValueError):
                load_day(date, self.directory)


if __name__ == '__main__':
    unittest.main()
