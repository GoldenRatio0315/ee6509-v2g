"""Frozen daily scenarios, random-stream isolation and reset sampling rules."""
from copy import deepcopy
from datetime import date as Date, timedelta
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import Mock, patch, call

import numpy as np

from env.data import load_day, load_params, split_dates
from env.physics import BatterySpec, feasible_interval
from env.scenario import generate_scenario, parameter_id
from env.v2g_env import V2GEnv


def scenario_hash(scenario):
    encoded = json.dumps(scenario.to_dict(), sort_keys=True, separators=(',', ':'),
                         allow_nan=False).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


class GenerationTests(unittest.TestCase):
    def setUp(self):
        self.params = load_params()

    def test_all_dates_and_fleet_sizes(self):
        ev = BatterySpec.from_params(self.params['ev'])
        first = Date(2026, 10, 1)
        for offset in range(28):
            day = first + timedelta(days=offset)
            with self.subTest(date=day):
                five = generate_scenario(day.isoformat(), 100, 5, self.params)
                ten = generate_scenario(day.isoformat(), 100, 10, self.params)
                self.assertEqual((five.T, ten.T), (48, 48))
                self.assertEqual(len(five.vehicles), 5)
                self.assertEqual(len(ten.vehicles), 10)
                self.assertEqual(ten.vehicles[:5], five.vehicles)
                np.testing.assert_array_equal(ten.noise, five.noise)
                np.testing.assert_array_equal(ten.truth, five.truth)
                self.assertEqual(five.params_id, parameter_id(self.params))
                self.assertEqual(five.bess_initial_kwh, 60)
                self.assertEqual(five.initial_peak_kw, 0)
                self.assertEqual(five.noise.shape, (3, 49, 12))
                source = load_day(day.isoformat())
                np.testing.assert_array_equal(
                    five.truth, np.stack([source['usep'], source['load'], source['pv']]))
                if day.weekday() >= 5:
                    self.assertTrue(all(vehicle is None for vehicle in ten.vehicles))
                else:
                    for vehicle in ten.vehicles:
                        self.assertIsNotNone(vehicle)
                        self.assertTrue(14 <= vehicle.arrival <= 20)
                        self.assertTrue(32 <= vehicle.departure <= 40)
                        self.assertTrue(12 <= vehicle.initial_kwh <= 28)
                        self.assertEqual(vehicle.target_kwh, 26)
                        feasible_interval(vehicle.initial_kwh, vehicle.target_kwh,
                                          vehicle.departure - vehicle.arrival, ev)

    def test_reproducibility_streams_and_fixed_truth(self):
        first = generate_scenario('2026-10-22', 100, 5, self.params)
        changed = generate_scenario('2026-10-22', 101, 5, self.params)
        repeated = generate_scenario('2026-10-22', 100, 5, self.params)
        self.assertEqual(scenario_hash(first), scenario_hash(repeated))
        self.assertNotEqual(first.vehicles, changed.vehicles)
        self.assertFalse(np.array_equal(first.noise, changed.noise))
        np.testing.assert_array_equal(first.truth, changed.truth)
        ordinal = Date(2026, 10, 22).toordinal()
        for variable, name in enumerate(('usep', 'load', 'pv')):
            rng = np.random.default_rng(np.random.SeedSequence([100, ordinal, 20, variable]))
            sigma = (self.params['forecast'][f'{name}_sigma_at_h1']
                     + np.arange(12) * self.params['forecast'][f'{name}_sigma_increment'])
            np.testing.assert_array_equal(first.noise[variable],
                                          rng.normal(0., sigma, size=(49, 12)))
        self.assertFalse(np.array_equal(first.noise[0], first.noise[1]))

    def test_cross_process_hash_does_not_use_python_hash(self):
        code = (
            'import hashlib,json; from env.data import load_params; '
            'from env.scenario import generate_scenario; '
            's=generate_scenario("2026-10-22",100,5,load_params()); '
            'print(hashlib.sha256(json.dumps(s.to_dict(),sort_keys=True,'
            'separators=(",",":"),allow_nan=False).encode()).hexdigest())'
        )
        expected = scenario_hash(generate_scenario('2026-10-22', 100, 5, self.params))
        for hash_seed in ['1', '777']:
            process_env = dict(os.environ, PYTHONHASHSEED=hash_seed)
            output = subprocess.check_output(
                [sys.executable, '-B', '-c', code], env=process_env,
                cwd=Path(__file__).resolve().parents[2], text=True).strip()
            self.assertEqual(output, expected)

    def test_generated_arrays_are_immutable(self):
        scenario = generate_scenario('2026-10-22', 100, 5, self.params)
        for array in (scenario.truth, scenario.noise):
            with self.assertRaises(ValueError):
                array.setflags(write=True)
        exported = scenario.to_dict()
        exported['truth'][0][0] = -999
        self.assertNotEqual(scenario.truth[0, 0], -999)

    def test_reject_before_quantising_and_half_ties_round_later(self):
        vehicle_rng = Mock()
        vehicle_rng.normal.side_effect = [6.99, 7.25, 20.01, 17.25]
        vehicle_rng.uniform.return_value = .5
        noise_rngs = [np.random.default_rng(seed) for seed in range(3)]
        with patch('env.scenario.np.random.default_rng',
                   side_effect=[vehicle_rng] + noise_rngs):
            scenario = generate_scenario('2026-10-01', 0, 1, self.params)
        vehicle = scenario.vehicles[0]
        self.assertEqual((vehicle.arrival, vehicle.departure), (15, 35))
        self.assertEqual(vehicle.initial_kwh, 20)
        self.assertEqual(vehicle_rng.normal.call_args_list,
                         [call(8.5,.5),call(8.5,.5),call(18.,.75),call(18.,.75)])
        vehicle_rng.uniform.assert_called_once_with(.3, .7)

    def test_invalid_generation_inputs(self):
        for date in ['2026-09-30', '2026-10-29', 'not-a-date', Date(2026,10,1)]:
            with self.subTest(date=date), self.assertRaises(ValueError):
                generate_scenario(date, 0, 5, self.params)
        for seed in [-1, True, 1.5]:
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                generate_scenario('2026-10-01', seed, 5, self.params)
        for slots in [0, -1, True, 1.5]:
            with self.subTest(slots=slots), self.assertRaises(ValueError):
                generate_scenario('2026-10-01', 0, slots, self.params)

    def test_invalid_distributions_and_rejection_limit(self):
        invalid = [('ev', 'arrival_sd_h', 0), ('ev', 'departure_sd_h', np.inf),
                   ('ev', 'arrival_mean_h', np.nan), ('ev', 'arrival_min_h', 11),
                   ('ev', 'departure_max_h', 25), ('ev', 'initial_soc_min', .1),
                   ('ev', 'departure_target_soc', .95), ('ev', 'weekday_cars_per_slot', 2),
                   ('ev', 'weekend_cars_per_slot', 1), ('bess', 'initial_soc', .95),
                   ('forecast', 'usep_sigma_at_h1', -.01),
                   ('forecast', 'pv_sigma_increment', np.inf)]
        for group, key, value in invalid:
            params = deepcopy(self.params)
            params[group][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                generate_scenario('2026-10-01', 0, 5, params)
        params = deepcopy(self.params)
        params['ev']['arrival_mean_h'] = 1e6
        with self.assertRaisesRegex(ValueError, '100000 attempts'):
            generate_scenario('2026-10-01', 0, 1, params)

    def test_date_split_boundaries(self):
        train = split_dates(self.params, 'train')
        validation = split_dates(self.params, 'validation')
        test = split_dates(self.params, 'test')
        self.assertEqual((len(train), len(validation), len(test)), (14, 7, 7))
        self.assertEqual((train[0], train[-1]), ('2026-10-01', '2026-10-14'))
        self.assertEqual((validation[0], validation[-1]), ('2026-10-15', '2026-10-21'))
        self.assertEqual((test[0], test[-1]), ('2026-10-22', '2026-10-28'))
        self.assertEqual(len(set(train + validation + test)), 28)


class ResetSamplingTests(unittest.TestCase):
    def test_default_date_and_seed_ranges(self):
        for split, lower, upper in [('train', 0, 99), ('validation', 0, 99), ('test', 100, 119)]:
            env = V2GEnv(split=split)
            scenarios = []
            for iteration in range(20):
                _, info = env.reset(seed=7654 if iteration == 0 else None)
                self.assertIn(info['date'], env.dates)
                self.assertTrue(lower <= info['scenario_seed'] <= upper)
                scenarios.append((info['date'], info['scenario_seed']))
            self.assertGreater(len(set(scenarios)), 1)
            env.close()

    def test_explicit_date_seed_and_reset_history_independence(self):
        env = V2GEnv(split='train')
        _, info = env.reset(seed=123456, options={'date': '2026-10-22'})
        first = env.export_scenario()
        self.assertEqual(info['scenario_seed'], 123456)
        self.assertEqual(info['date'], '2026-10-22')
        env.reset(seed=99, options={'date': '2026-10-03'})
        env.reset(seed=123456, options={'date': '2026-10-22'})
        self.assertEqual(env.export_scenario(), first)
        other = V2GEnv(split='test')
        other.reset(seed=123456, options={'date': '2026-10-22'})
        self.assertEqual(other.export_scenario(), first)
        env.close()
        other.close()

    def test_generated_noisy_scenario_json_round_trip(self):
        first, replay = V2GEnv(n_slots=5), V2GEnv(n_slots=5)
        obs_a, meta_a = first.reset(seed=100, options={'date':'2026-10-22'})
        exported = json.loads(json.dumps(first.export_scenario(),allow_nan=False))
        self.assertTrue(np.any(np.asarray(exported['noise']) != 0))
        obs_b, meta_b = replay.reset(seed=999, options={'scenario':exported})
        self.assertEqual(meta_a,meta_b)
        for key in obs_a:
            np.testing.assert_array_equal(obs_a[key],obs_b[key])
        for _ in range(24):
            bounds = first.feasible_intervals()
            np.testing.assert_array_equal(bounds,replay.feasible_intervals())
            power = bounds.mean(axis=1)
            a,b = first.step_power(power),replay.step_power(power)
            for key in a[0]:
                np.testing.assert_array_equal(a[0][key],b[0][key])
            self.assertEqual(a[1:4],b[1:4])
            self.assertEqual(json.dumps(a[4],default=lambda x:x.tolist(),sort_keys=True),
                             json.dumps(b[4],default=lambda x:x.tolist(),sort_keys=True))
        first.close()
        replay.close()

    def test_date_without_seed_and_seed_without_date(self):
        for split, lower, upper in [('train', 0, 99), ('validation', 0, 99), ('test', 100, 119)]:
            first, second = V2GEnv(split=split), V2GEnv(split=split)
            first.reset(seed=24)
            second.reset(seed=24)
            self.assertEqual(first.export_scenario(), second.export_scenario())
            for _ in range(5):
                _, info = first.reset(options={'date': '2026-10-01'})
                second.reset(options={'date': '2026-10-01'})
                self.assertEqual(first.export_scenario(), second.export_scenario())
                self.assertEqual(info['date'], '2026-10-01')
                self.assertTrue(lower <= info['scenario_seed'] <= upper)
            first.close()
            second.close()


if __name__ == '__main__':
    unittest.main()
