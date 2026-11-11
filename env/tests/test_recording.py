"""Shared CSV schema, physical schedule replay and the one-day example."""
from copy import deepcopy
import csv
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import env
from env.data import load_params
from env.demo import run_demo
from env.recording import csv_columns, info_to_row, occupancy_mask, write_csv
from env.replay import replay_power_schedule
from env.scenario import Vehicle
from env.tests.test_env import hand_columns, hand_scenario, read_rows


def expected_columns(n_slots):
    return (['strategy', 'date', 'seed', 't', 'p_grid_kw', 'p_bess_kw']
            + [f'p_ev_{slot}_kw' for slot in range(1, n_slots + 1)]
            + ['e_bess_kwh']
            + [f'e_ev_{slot}_kwh' for slot in range(1, n_slots + 1)]
            + ['buy_price', 'sell_price', 'purchase_sgd', 'revenue_sgd',
               'peak_increment_sgd', 'bess_wear_sgd', 'ev_wear_sgd', 'step_cost_sgd'])


def hand_schedule(n_slots=1):
    powers = np.zeros((4, n_slots + 1))
    for t, row in enumerate(read_rows('hand_case.csv')):
        powers[t, :2] = row['p_bess_kw'], row['p_ev_kw']
    return powers


class RecordingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def test_public_imports(self):
        required = {'V2GEnv', 'Scenario', 'Vehicle', 'generate_scenario',
                    'load_params', 'load_day'}
        self.assertTrue(required.issubset(env.__all__))
        for name in required:
            self.assertTrue(callable(getattr(env, name)))
        self.assertEqual(env.load_params(), load_params())

    def test_root_csv_schema_order_and_no_diagnostic_columns(self):
        for slots in [1, 5, 10]:
            with self.subTest(slots=slots):
                self.assertEqual(csv_columns(slots), expected_columns(slots))
                infos, _ = replay_power_schedule(hand_scenario(slots), hand_schedule(slots))
                row = info_to_row(infos[0], strategy='hand_case')
                self.assertEqual(list(row), expected_columns(slots))
                self.assertEqual(row['strategy'], 'hand_case')
                self.assertEqual(row['date'], '2026-10-01')
                self.assertEqual(row['seed'], 0)
                self.assertEqual(row['t'], 0)
                self.assertTrue({'params_id', 'scenario_seed', 'peak_before_kw',
                                 'peak_after_kw', 'corrections', 'departures'}.isdisjoint(row))

    def test_all_real_day_csv_values_masks_and_power_replay(self):
        for slots in (5,10):
            original,replay = env.V2GEnv(n_slots=slots),env.V2GEnv(n_slots=slots)
            original.reset(seed=100,options={'date':'2026-10-22'})
            scenario = original.export_scenario()
            replay.reset(options={'scenario':scenario})
            masks = occupancy_mask(scenario)
            rng = np.random.default_rng(6509)
            for t in range(48):
                observation,reward,terminated,truncated,info = original.step(rng.uniform(-1,1,slots+1))
                row = info_to_row(info,strategy='sample')
                self.assertEqual(list(row),expected_columns(slots))
                self.assertEqual(row['strategy'],'sample')
                self.assertEqual(row['seed'],info['scenario_seed'])
                for key in ('date','t','p_grid_kw','p_bess_kw','e_bess_kwh','buy_price','sell_price',
                            'purchase_sgd','revenue_sgd','peak_increment_sgd','bess_wear_sgd',
                            'ev_wear_sgd','step_cost_sgd'):
                    self.assertEqual(row[key],info[key])
                for i in range(slots):
                    self.assertEqual(row[f'p_ev_{i+1}_kw'],info['p_ev_kw'][i])
                    self.assertEqual(row[f'e_ev_{i+1}_kwh'],info['e_ev_kwh'][i])
                np.testing.assert_array_equal(masks[t],info['ev_present'])
                powers = np.concatenate(([info['p_bess_kw']],info['p_ev_kw']))
                replayed = replay.step_power(powers)
                self.assertEqual(replayed[1:4],(reward,terminated,truncated))
                for key in observation:
                    np.testing.assert_array_equal(replayed[0][key],observation[key])
                for key in ('e_ev_kwh','e_bess_kwh','p_grid_kw','step_cost_sgd'):
                    np.testing.assert_array_equal(replayed[4][key],info[key])
            self.assertEqual(replayed[4]['episode_summary'],info['episode_summary'])
            original.close()
            replay.close()

    def test_replay_matches_all_twenty_hand_case_columns(self):
        scenario = hand_scenario()
        powers = hand_schedule()
        before = powers.copy()
        infos, summary = replay_power_schedule(scenario, powers)
        np.testing.assert_array_equal(powers, before)
        expected = read_rows('hand_case_expected.csv')
        self.assertEqual(len(infos), 4)
        for actual_info, expected_row in zip(infos, expected):
            actual = hand_columns(actual_info)
            self.assertEqual(len(expected_row), 20)
            self.assertEqual(set(actual), set(expected_row))
            for column, value in expected_row.items():
                with self.subTest(t=actual_info['t'], column=column):
                    self.assertAlmostEqual(actual[column], value, delta=1e-9)
        self.assertAlmostEqual(summary['variable_cost_sgd'], 96.10157925, delta=1e-9)
        self.assertAlmostEqual(summary['total_cost_sgd'], 96.10157925, delta=1e-9)
        self.assertEqual(summary['fixed_pv_sgd'], 0)
        self.assertEqual(summary['fixed_contract_sgd'], 0)
        self.assertEqual(summary['peak_grid_kw'], 252)
        self.assertAlmostEqual(summary['bess_final_kwh'], 60, delta=1e-9)
        self.assertEqual(summary['departure_success_rate'], 1)
        dict_infos, dict_summary = replay_power_schedule(scenario.to_dict(), powers)
        self.assertEqual(dict_summary, summary)
        self.assertEqual(len(dict_infos), 4)

    def test_csv_preserves_departure_energy_before_clear(self):
        infos, _ = replay_power_schedule(hand_scenario(2), hand_schedule(2))
        path = self.directory / 'hand.csv'
        self.assertEqual(write_csv(path, iter(infos), strategy='hand_case'), 4)
        with path.open(newline='') as stream:
            reader = csv.DictReader(stream)
            rows = list(reader)
        self.assertEqual(reader.fieldnames, expected_columns(2))
        self.assertEqual(len(rows), 4)
        self.assertAlmostEqual(float(rows[-1]['e_ev_1_kwh']), 26, delta=1e-9)
        self.assertEqual(float(rows[-1]['p_ev_1_kw']), 6)
        self.assertEqual(float(rows[-1]['e_ev_2_kwh']), 0)
        self.assertEqual(float(rows[-1]['p_ev_2_kw']), 0)
        self.assertAlmostEqual(infos[-1]['departures'][0]['energy_kwh'], 26, delta=1e-9)
        self.assertAlmostEqual(sum(float(row['step_cost_sgd']) for row in rows),
                               96.10157925, delta=1e-9)

    def test_large_integer_seed_is_preserved_exactly(self):
        seed = 2**80 + 7
        scenario = replace(hand_scenario(), seed=seed)
        infos, _ = replay_power_schedule(scenario, hand_schedule())
        row = info_to_row(infos[0], strategy='hand_case')
        self.assertEqual(row['seed'], seed)
        path = self.directory / 'large_seed.csv'
        write_csv(path, infos, strategy='hand_case')
        with path.open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        self.assertTrue(all(row['seed'] == str(seed) for row in rows))

    def test_occupancy_mask_includes_last_action_not_departure_boundary(self):
        scenario = replace(hand_scenario(2),
                           vehicles=(Vehicle(1, 3, 24.8, 26), None))
        expected = np.array([[False, False], [True, False],
                             [True, False], [False, False]], dtype=bool)
        mask = occupancy_mask(scenario)
        self.assertEqual(mask.dtype, np.bool_)
        self.assertEqual(mask.shape, (4, 2))
        np.testing.assert_array_equal(mask, expected)
        np.testing.assert_array_equal(occupancy_mask(scenario.to_dict()), expected)
        mask[:] = False
        np.testing.assert_array_equal(occupancy_mask(scenario), expected)
        self.assertTrue(occupancy_mask(hand_scenario())[-1, 0])

    def test_replay_rejects_parameter_mismatch(self):
        wrong_id = replace(hand_scenario(), params_id='different-parameters')
        with self.assertRaisesRegex(ValueError, 'params_id'):
            replay_power_schedule(wrong_id, hand_schedule())
        parameters = load_params()
        parameters['tariff']['buy_adder_sgd_per_kwh'] = .04
        with self.assertRaisesRegex(ValueError, 'params_id'):
            replay_power_schedule(hand_scenario(), hand_schedule(), params=parameters)

    def test_replay_rejects_bad_shape_length_and_nonfinite(self):
        valid = hand_schedule()
        invalid = [valid[:3], np.vstack([valid, valid[-1]]), valid[:, :1],
                   np.zeros((4, 3)), np.zeros((2, 4)), np.zeros(8),
                   np.full((4, 2), np.nan), np.full((4, 2), np.inf),
                   np.zeros((4, 2), dtype=bool), np.full((4, 2), '0')]
        for powers in invalid:
            with self.subTest(shape=powers.shape, dtype=powers.dtype), self.assertRaises(ValueError):
                replay_power_schedule(hand_scenario(), powers)

    def test_replay_rejects_infeasible_powers(self):
        for step, device, power in [(0, 1, -6.1), (3, 0, 39), (3, 1, 5.9)]:
            powers = hand_schedule()
            powers[step, device] = power
            with self.subTest(step=step, device=device), self.assertRaisesRegex(
                    ValueError, 'infeasible_input'):
                replay_power_schedule(hand_scenario(), powers)
        powers = hand_schedule(2)
        powers[0, 2] = .01
        with self.assertRaisesRegex(ValueError, 'infeasible_input'):
            replay_power_schedule(hand_scenario(2), powers)

    def test_row_validation_and_empty_trajectory(self):
        infos, _ = replay_power_schedule(hand_scenario(), hand_schedule())
        for strategy in ['', '  ', None]:
            with self.subTest(strategy=strategy), self.assertRaises(ValueError):
                info_to_row(infos[0], strategy=strategy)
        for field, value in [('p_ev_kw', [[0]]), ('e_ev_kwh', [0, 0]),
                             ('p_ev_kw', [np.nan]), ('step_cost_sgd', np.inf)]:
            info = deepcopy(infos[0])
            info[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                info_to_row(info, strategy='case')
        with self.assertRaises(ValueError):
            write_csv(self.directory / 'empty.csv', [], strategy='case')

    def test_rejects_nonreal_csv_values_without_discarding_information(self):
        infos, _ = replay_power_schedule(hand_scenario(),hand_schedule())
        for key in ('p_ev_kw','e_ev_kwh'):
            for bad in (np.array([2+3j]),np.array([True]),np.array(['2']),np.array([2],dtype=object)):
                changed = deepcopy(infos[0])
                changed[key] = bad
                with self.subTest(key=key,dtype=bad.dtype), self.assertRaises(ValueError):
                    info_to_row(changed,strategy='case')
        for key in ('p_grid_kw','p_bess_kw','e_bess_kwh','buy_price','sell_price',
                    'purchase_sgd','revenue_sgd','peak_increment_sgd',
                    'bess_wear_sgd','ev_wear_sgd','step_cost_sgd'):
            for bad in (np.complex128(2+3j),True,'2'):
                changed = deepcopy(infos[0])
                changed[key] = bad
                with self.subTest(key=key,value=bad), self.assertRaises(ValueError):
                    info_to_row(changed,strategy='case')

    def test_rejects_fractional_or_boolean_record_identifiers(self):
        infos, _ = replay_power_schedule(hand_scenario(),hand_schedule())
        for key in ('scenario_seed','t'):
            for bad in (100.9,-1,True,np.bool_(False),'100'):
                changed = deepcopy(infos[0])
                changed[key] = bad
                with self.subTest(key=key,value=bad), self.assertRaises(ValueError):
                    info_to_row(changed,strategy='case')
        for bad in (0,-1,True,1.5):
            with self.subTest(n_slots=bad), self.assertRaises(ValueError):
                csv_columns(bad)

    def test_bad_later_row_preserves_existing_csv(self):
        infos, _ = replay_power_schedule(hand_scenario(), hand_schedule())
        path = self.directory / 'existing.csv'
        original = 'existing results must remain intact\n'
        path.write_text(original)
        for defect in ['nonfinite_cost', 'different_slot_count']:
            broken = deepcopy(infos)
            if defect == 'nonfinite_cost':
                broken[2]['step_cost_sgd'] = np.nan
            else:
                broken[2]['p_ev_kw'] = np.array([0., 0.])
                broken[2]['e_ev_kwh'] = np.array([0., 0.])
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                write_csv(path, (info for info in broken), strategy='case')
            self.assertEqual(path.read_text(), original)

    def test_real_demo_csv_replays_same_scenario(self):
        summary, csv_path = run_demo(date='2026-10-01', seed=0, n_slots=5,
                                     action_seed=6509, output_dir=self.directory)
        with csv_path.open(newline='') as stream:
            reader = csv.DictReader(stream)
            rows = list(reader)
        self.assertEqual(reader.fieldnames, expected_columns(5))
        self.assertEqual(len(rows), 48)
        self.assertEqual([int(row['t']) for row in rows], list(range(48)))
        self.assertTrue(all(row['strategy'] == 'random' for row in rows))
        self.assertTrue(all(row['date'] == '2026-10-01' for row in rows))
        self.assertTrue(all(int(row['seed']) == 0 for row in rows))
        scenario_path = csv_path.with_suffix('.scenario.json')
        summary_path = csv_path.with_suffix('.summary.json')
        scenario = json.loads(scenario_path.read_text())
        saved_summary = json.loads(summary_path.read_text())
        self.assertEqual(saved_summary, summary)
        self.assertEqual(set(self.directory.iterdir()), {csv_path, scenario_path, summary_path})
        power_columns = ['p_bess_kw'] + [f'p_ev_{slot}_kw' for slot in range(1, 6)]
        powers = np.array([[float(row[column]) for column in power_columns] for row in rows])
        infos, replayed = replay_power_schedule(scenario, powers)
        self.assertEqual(len(infos), 48)
        for key in ['variable_cost_sgd', 'fixed_pv_sgd', 'fixed_contract_sgd',
                    'total_cost_sgd', 'bess_final_kwh', 'peak_grid_kw']:
            with self.subTest(summary_field=key):
                self.assertAlmostEqual(replayed[key], summary[key], delta=1e-8)
        self.assertAlmostEqual(sum(float(row['step_cost_sgd']) for row in rows),
                               replayed['variable_cost_sgd'], delta=1e-8)
        for row, info in zip(rows, infos):
            self.assertAlmostEqual(float(row['step_cost_sgd']), info['step_cost_sgd'], delta=1e-9)
        self.assertEqual(replayed['departure_success_rate'], 1)
        self.assertGreaterEqual(replayed['bess_final_kwh'], 60 - 1e-9)


if __name__ == '__main__':
    unittest.main()
