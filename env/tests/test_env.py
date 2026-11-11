"""Production-path regression tests against D's independent hand-case oracle."""

from __future__ import annotations

import csv
from dataclasses import replace
import json
from pathlib import Path
import unittest

import numpy as np

from env.scenario import Scenario, Vehicle, parameter_id
from env.physics import P_TOL
from env.v2g_env import V2GEnv, load_params


DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "D"


def read_rows(name):
    with (DATA_DIR / name).open(encoding="utf-8", newline="") as handle:
        return [{key: float(value) for key, value in row.items()}
                for row in csv.DictReader(handle)]


def hand_scenario(n_slots=1):
    """Load the four-step fixture; no production equations are duplicated here."""
    rows = read_rows("hand_case.csv")
    params = load_params()
    truth = np.array([[row[key] for row in rows]
                      for key in ("usep_sgd_per_kwh", "load_kw", "pv_kw")])
    case = params["hand_case"]
    return Scenario(
        date="2026-10-01", seed=0, params_id=parameter_id(params),
        truth=truth,
        vehicles=(Vehicle(arrival=0, departure=4,
                          initial_kwh=case["ev_initial_kwh"],
                          target_kwh=26.0),) + (None,) * (n_slots - 1),
        bess_initial_kwh=case["bess_initial_kwh"],
        initial_peak_kw=case["initial_running_peak_kw"],
        noise=np.zeros((3, 5, 12)),
    )


def hand_columns(info, slot=0):
    """Map public info names to all twenty columns of D's reference CSV."""
    return {
        "t": info["t"],
        "ev_lo": info["intervals"][slot+1, 0],
        "ev_hi": info["intervals"][slot+1, 1],
        "bess_lo": info["intervals"][0, 0],
        "bess_hi": info["intervals"][0, 1],
        "ev_start": info["e_ev_before_kwh"][slot],
        "ev_end": info["e_ev_kwh"][slot],
        "bess_start": info["e_bess_before_kwh"],
        "bess_end": info["e_bess_kwh"],
        "p_grid": info["p_grid_kw"],
        "buy_price": info["buy_price"],
        "sell_price": info["sell_price"],
        "purchase": info["purchase_sgd"],
        "revenue": info["revenue_sgd"],
        "peak_before": info["peak_before_kw"],
        "peak_after": info["peak_after_kw"],
        "peak_increment": info["peak_increment_sgd"],
        "bess_wear": info["bess_wear_sgd"],
        "ev_wear": info["ev_wear_sgd"],
        "step_cost": info["step_cost_sgd"],
    }


class HandCaseTests(unittest.TestCase):
    def run_case(self, n_slots, use_actions):
        environment = V2GEnv(n_slots=n_slots)
        obs, reset_info = environment.reset(options={"scenario": hand_scenario(n_slots)})
        self.assertEqual(reset_info["t"], 0)
        self.assertEqual(reset_info["scenario_seed"], 0)
        self.assertEqual(obs["ev_present"].tolist(), [1] + [0] * (n_slots - 1))
        expected_rows = read_rows("hand_case_expected.csv")
        trajectory, cost = [], 0.0
        for t, (source, expected) in enumerate(zip(read_rows("hand_case.csv"), expected_rows)):
            power = np.zeros(n_slots + 1)
            power[:2] = source["p_bess_kw"], source["p_ev_kw"]
            if use_actions:
                intervals = environment.feasible_intervals()
                width = intervals[:, 1] - intervals[:, 0]
                action = np.zeros(n_slots + 1)
                nonzero = width > 1e-12
                action[nonzero] = 2 * (power[nonzero] - intervals[nonzero, 0]) / width[nonzero] - 1
                if t == 3:
                    self.assertAlmostEqual(width[1], 0.0, delta=1e-9)
                    self.assertEqual(action[1], 0.0)
                obs, reward, terminated, truncated, info = environment.step(action)
            else:
                obs, reward, terminated, truncated, info = environment.step_power(power)
            actual = hand_columns(info)
            self.assertEqual(set(actual), set(expected))
            for column, value in expected.items():
                with self.subTest(n_slots=n_slots, action_path=use_actions, t=t, column=column):
                    self.assertAlmostEqual(actual[column], value, delta=1e-9)
            self.assertEqual(terminated, t == 3)
            self.assertFalse(truncated)
            self.assertAlmostEqual(reward, -expected["step_cost"], delta=1e-9)
            np.testing.assert_allclose(info["p_ev_kw"][1:], 0.0, atol=0, rtol=0)
            np.testing.assert_allclose(info["e_ev_kwh"][1:], 0.0, atol=0, rtol=0)
            self.assertEqual(len(info["departures"]), int(t == 3))
            cost += info["step_cost_sgd"]
            trajectory.append(actual)
        self.assertAlmostEqual(cost, 96.10157925, delta=1e-9)
        self.assertAlmostEqual(info["peak_after_kw"], 252.0, delta=1e-9)
        self.assertAlmostEqual(info["e_bess_kwh"], 60.0, delta=1e-9)
        self.assertAlmostEqual(info["e_ev_kwh"][0], 26.0, delta=1e-9)
        self.assertEqual(info["ev_present"].tolist(), [1] + [0] * (n_slots - 1))
        np.testing.assert_array_equal(obs["ev_present"], np.zeros(n_slots, dtype=np.int8))
        np.testing.assert_array_equal(obs["ev"], np.zeros((n_slots, 4)))
        departure = info["departures"][0]
        self.assertEqual(departure["slot"], 0)
        self.assertEqual(departure["departure_t"], 4)
        self.assertAlmostEqual(departure["energy_kwh"], 26.0, delta=1e-9)
        self.assertAlmostEqual(departure["target_kwh"], 26.0, delta=1e-9)
        self.assertTrue(departure["success"])
        summary = info["episode_summary"]
        self.assertAlmostEqual(summary["variable_cost_sgd"], cost, delta=1e-9)
        self.assertEqual(summary["fixed_pv_sgd"], 0.0)
        self.assertEqual(summary["fixed_contract_sgd"], 0.0)
        self.assertAlmostEqual(summary["total_cost_sgd"], cost, delta=1e-9)
        self.assertEqual(summary["departure_success_rate"], 1.0)
        self.assertEqual(summary["departures"], info["departures"])
        return trajectory

    def test_four_steps_all_reference_columns_step_power_one_slot(self):
        self.run_case(1, False)

    def test_four_steps_all_reference_columns_step_power_five_slots(self):
        self.run_case(5, False)

    def test_hand_case_in_nonzero_slot(self):
        original = hand_scenario(5)
        scenario = replace(original, vehicles=(None, None, None, original.vehicles[0], None))
        environment = V2GEnv(n_slots=5)
        environment.reset(options={'scenario': scenario})
        for source, expected in zip(read_rows('hand_case.csv'), read_rows('hand_case_expected.csv')):
            power = np.zeros(6)
            power[0], power[4] = source['p_bess_kw'], source['p_ev_kw']
            info = environment.step_power(power)[4]
            for column, value in hand_columns(info, slot=3).items():
                self.assertAlmostEqual(value, expected[column], delta=1e-9)
        self.assertEqual(info['departures'][0]['slot'], 3)

    def test_action_mapping_matches_power_replay_and_padding(self):
        reference = self.run_case(1, False)
        for n_slots in (1, 5):
            actual = self.run_case(n_slots, True)
            for t, (expected_row, actual_row) in enumerate(zip(reference, actual)):
                for key in expected_row:
                    with self.subTest(n_slots=n_slots, t=t, column=key):
                        self.assertAlmostEqual(actual_row[key], expected_row[key], delta=1e-9)


class EnvironmentContractTests(unittest.TestCase):
    def make_env(self):
        environment = V2GEnv(n_slots=1)
        environment.reset(options={"scenario": hand_scenario()})
        return environment

    def test_arrival_and_departure_boundaries(self):
        scenario = replace(hand_scenario(), vehicles=(
            Vehicle(arrival=1, departure=3, initial_kwh=26.0, target_kwh=26.0),))
        environment = V2GEnv(n_slots=1)
        obs, _ = environment.reset(options={"scenario": scenario})
        self.assertEqual(obs["ev_present"].tolist(), [0])
        np.testing.assert_array_equal(environment.feasible_intervals()[1], [0.0, 0.0])
        obs, _, terminated, _, info = environment.step_power([0.0, 0.0])
        self.assertFalse(terminated)
        self.assertEqual(info["ev_present"].tolist(), [0])
        self.assertEqual(info["e_ev_kwh"].tolist(), [0.0])
        self.assertEqual(obs["ev_present"].tolist(), [1])
        np.testing.assert_allclose(obs["ev"][0], [1.0, 0.65, 2.0, 0.0], atol=1e-7)
        obs, _, _, _, info = environment.step_power([0.0, 0.0])
        self.assertEqual(info["departures"], [])
        self.assertEqual(obs["ev"][0, 2], 1.0)
        obs, _, _, _, info = environment.step_power([0.0, 0.0])
        self.assertEqual(info["e_ev_kwh"].tolist(), [26.0])
        self.assertTrue(info["departures"][0]["success"])
        self.assertEqual(info["departures"][0]["departure_t"], 3)
        self.assertEqual(obs["ev_present"].tolist(), [0])
        np.testing.assert_array_equal(environment.feasible_intervals()[1], [0.0, 0.0])
        _, _, terminated, _, info = environment.step_power([0.0, 0.0])
        self.assertTrue(terminated)
        self.assertEqual(info["e_ev_kwh"].tolist(), [0.0])
        self.assertEqual(info["departures"], [])
        self.assertEqual(len(info["episode_summary"]["departures"]), 1)

    def test_invalid_action_shape_and_nonfinite_values_do_not_advance(self):
        invalid = (0.0, [0.0], [0.0, 0.0, 0.0], [[0.0, 0.0]],
                   [np.nan, 0.0], [0.0, np.inf], [-np.inf, 0.0])
        for method_name in ("step", "step_power"):
            for action in invalid:
                with self.subTest(method=method_name, action=action):
                    environment = self.make_env()
                    with self.assertRaises(ValueError):
                        getattr(environment, method_name)(action)
                    _, _, _, _, info = environment.step_power([20.0, -2.7])
                    self.assertEqual(info["t"], 0)
                    self.assertAlmostEqual(info["step_cost_sgd"], 13.306093, delta=1e-9)

    def test_step_clips_normalised_actions(self):
        first, second = self.make_env(), self.make_env()
        actual = first.step([-2.0, 2.0])
        expected = second.step([-1.0, 1.0])
        for key in actual[0]:
            np.testing.assert_array_equal(actual[0][key], expected[0][key])
        self.assertEqual(actual[1:4], expected[1:4])
        self.assertEqual(actual[4]["p_bess_kw"], expected[4]["p_bess_kw"])
        np.testing.assert_array_equal(actual[4]["p_ev_kw"], expected[4]["p_ev_kw"])

    def test_power_outside_bounds_and_invalid_tolerance_are_rejected(self):
        environment = self.make_env()
        for power in ([61.0, 0.0], [0.0, -7.0]):
            with self.subTest(power=power), self.assertRaises(ValueError):
                environment.step_power(power)
        for tolerance in (-1.0, np.nan, np.inf):
            with self.subTest(tolerance=tolerance), self.assertRaises(ValueError):
                environment.step_power([20.0, -2.7], tol=tolerance)
        self.assertEqual(environment.step_power([20.0, -2.7])[4]["t"], 0)

    def test_power_tolerance_correction_is_recorded(self):
        environment = self.make_env()
        _, _, _, _, info = environment.step_power([60.0 + 0.5e-6, 0.0])
        self.assertEqual(info["p_bess_kw"], 60.0)
        power_corrections = [entry for entry in info["corrections"] if entry["kind"] == "power"]
        self.assertEqual(len(power_corrections), 1)
        self.assertEqual(power_corrections[0]["device"], 0)
        self.assertEqual(power_corrections[0]["unit"], "kW")
        self.assertAlmostEqual(power_corrections[0]["delta"], -0.5e-6, delta=1e-12)

    def test_interval_tolerance_correction_reaches_public_step_info(self):
        # A one-step deficit below the reachability tolerance collapses [lo,6].
        # It must not silently disappear between interval calculation and info.
        fixture = hand_scenario()
        scenario = replace(
            fixture, truth=fixture.truth[:, :1], noise=np.zeros((3, 2, 12)),
            vehicles=(Vehicle(arrival=0, departure=1,
                              initial_kwh=23.3 - 4.5e-11, target_kwh=26.0),),
        )
        environment = V2GEnv(n_slots=1)
        environment.reset(options={"scenario": scenario})
        _, _, terminated, truncated, info = environment.step([-1.0, 0.0])
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        np.testing.assert_array_equal(info["intervals"][1], [6.0, 6.0])
        correction = [item for item in info["corrections"] if item["kind"] == "interval_lo"]
        self.assertEqual(len(correction), 1)
        self.assertEqual(set(correction[0]), {"device", "kind", "delta", "unit"})
        self.assertEqual(correction[0]["device"], 1)
        self.assertEqual(correction[0]["unit"], "kW")
        self.assertAlmostEqual(correction[0]["delta"], -1e-10, delta=2e-14)
        self.assertEqual(info["p_ev_kw"].tolist(), [6.0])
        self.assertLess(info["e_ev_kwh"][0], 26.0)
        self.assertGreaterEqual(info["e_ev_kwh"][0], 26.0 - 1e-9)
        self.assertTrue(info["departures"][0]["success"])

    def test_energy_tolerance_correction_reaches_public_step_info(self):
        # Float64 inverse/update at this valid initial energy undershoots 12 kWh
        # by one representable step. Start through Scenario, not private state.
        scenario = replace(hand_scenario(), bess_initial_kwh=23.1, vehicles=(None,))
        environment = V2GEnv(n_slots=1)
        environment.reset(options={"scenario": scenario})
        obs, _, terminated, _, info = environment.step([-1.0, 0.0])
        self.assertFalse(terminated)
        self.assertEqual(info["e_bess_before_kwh"], 23.1)
        self.assertEqual(info["e_bess_kwh"], 12.0)
        self.assertAlmostEqual(float(obs["global"][0]), 0.1, delta=1e-7)
        correction = [item for item in info["corrections"] if item["kind"] == "energy"]
        self.assertEqual(len(correction), 1)
        self.assertEqual(set(correction[0]), {"device", "kind", "delta", "unit"})
        self.assertEqual(correction[0]["device"], 0)
        self.assertEqual(correction[0]["unit"], "kWh")
        self.assertGreater(correction[0]["delta"], 0.0)
        self.assertLessEqual(correction[0]["delta"], 1e-9)
        self.assertEqual(info["e_ev_kwh"].tolist(), [0.0])

    def test_step_requires_reset_and_cannot_follow_termination(self):
        environment = V2GEnv(n_slots=1)
        for method in (environment.step, environment.step_power):
            with self.assertRaises((RuntimeError, ValueError)):
                method([0.0, 0.0])
        environment.reset(options={"scenario": hand_scenario()})
        for row in read_rows("hand_case.csv"):
            environment.step_power([row["p_bess_kw"], row["p_ev_kw"]])
        for method in (environment.step, environment.step_power):
            with self.assertRaises((RuntimeError, ValueError)):
                method([0.0, 0.0])
        environment.reset(options={"scenario": hand_scenario()})
        self.assertEqual(environment.step_power([20.0, -2.7])[4]["t"], 0)

    def test_injected_seed_does_not_replace_scenario(self):
        scenario = hand_scenario()
        first, second = V2GEnv(n_slots=1), V2GEnv(n_slots=1)
        obs_a, info_a = first.reset(seed=12, options={"scenario": scenario})
        obs_b, info_b = second.reset(seed=99, options={"scenario": scenario})
        self.assertEqual(info_a, info_b)
        self.assertEqual(info_a["scenario_seed"], scenario.seed)
        for key in obs_a:
            np.testing.assert_array_equal(obs_a[key], obs_b[key])
        self.assertEqual(first.export_scenario(), second.export_scenario())

    def test_injection_rejects_ambiguous_date_and_parameter_mismatch(self):
        environment = V2GEnv(n_slots=1)
        with self.assertRaises(ValueError):
            environment.reset(options={"scenario": hand_scenario(), "date": "2026-10-01"})
        with self.assertRaises(ValueError):
            mismatched = replace(hand_scenario(), params_id="0" * 64)
            environment.reset(options={"scenario": mismatched})

    def test_failed_reset_clears_export_and_bad_export_fields_raise_value_error(self):
        environment = self.make_env()
        with self.assertRaises(ValueError):
            environment.reset(options={'scenario': replace(hand_scenario(), params_id='mismatch')})
        with self.assertRaises(RuntimeError):
            environment.export_scenario()
        invalid = hand_scenario().to_dict()
        invalid['unknown'] = 1
        with self.assertRaises(ValueError):
            Scenario.from_dict(invalid)

    def test_infeasible_vehicle_is_rejected_without_relaxing_target(self):
        with self.assertRaisesRegex(ValueError, "infeasible_input"):
            scenario = replace(hand_scenario(), vehicles=(
                Vehicle(arrival=3, departure=4, initial_kwh=12.0, target_kwh=26.0),))
            V2GEnv(n_slots=1).reset(options={"scenario": scenario})

    def test_returned_observations_info_and_export_do_not_mutate_state(self):
        environment = V2GEnv(n_slots=1)
        reference = self.make_env()
        obs, _ = environment.reset(options={"scenario": hand_scenario()})
        for value in obs.values():
            value[...] = 0
        exported = environment.export_scenario()
        exported["truth"][0][0] = 99999.0
        exported["vehicles"][0]["initial_kwh"] = 0.0
        intervals = environment.feasible_intervals()
        intervals[...] = 0
        actual = environment.step_power([20.0, -2.7])
        expected = reference.step_power([20.0, -2.7])
        for key in actual[0]:
            np.testing.assert_array_equal(actual[0][key], expected[0][key])
        self.assertAlmostEqual(actual[1], expected[1], delta=1e-12)
        actual[4]["e_ev_kwh"][...] = 0
        actual[4]["p_ev_kw"][...] = 99999
        actual[4]["intervals"][...] = 0
        actual[4]["ev_present"][...] = 0
        actual_next = environment.step_power([-38.0, 0.0])
        expected_next = reference.step_power([-38.0, 0.0])
        self.assertAlmostEqual(actual_next[1], expected_next[1], delta=1e-12)
        self.assertAlmostEqual(actual_next[4]["e_ev_before_kwh"][0], 23.3, delta=1e-9)
        for key in actual_next[0]:
            np.testing.assert_array_equal(actual_next[0][key], expected_next[0][key])
        json.dumps(environment.export_scenario(), allow_nan=False)

    def test_scenario_arrays_are_immutable_copies_and_export_round_trips(self):
        original = hand_scenario()
        truth, noise = original.truth.copy(), original.noise.copy()
        scenario = replace(original, truth=truth, noise=noise)
        truth[...] = 0
        noise[...] = 1
        np.testing.assert_array_equal(scenario.truth, original.truth)
        np.testing.assert_array_equal(scenario.noise, original.noise)
        for array in (scenario.truth, scenario.noise):
            with self.assertRaises(ValueError):
                array.flat[0] = 1
            with self.assertRaises(ValueError):
                array.setflags(write=True)
        environment = V2GEnv(n_slots=1)
        environment.reset(options={"scenario": scenario})
        exported = environment.export_scenario()
        restored = Scenario.from_dict(json.loads(json.dumps(exported, allow_nan=False)))
        replay = V2GEnv(n_slots=1)
        replay.reset(options={"scenario": restored})
        self.assertEqual(replay.export_scenario(), exported)
        result_a = environment.step_power([20.0, -2.7])
        result_b = replay.step_power([20.0, -2.7])
        self.assertEqual(result_a[1:4], result_b[1:4])
        for key in result_a[0]:
            np.testing.assert_array_equal(result_a[0][key], result_b[0][key])

    def check_external_array_metadata_isolated(self, field, metadata):
        source = hand_scenario()
        environment = V2GEnv(n_slots=1)
        control = V2GEnv(n_slots=1)
        self.addCleanup(environment.close)
        self.addCleanup(control.close)
        environment.reset(options={'scenario': source})
        exported = environment.export_scenario()
        control.reset(options={'scenario': exported})
        array = getattr(source, field)
        if metadata == 'shape':
            array.shape = (array.size,)
        else:
            array.dtype = np.int64
        self.assertEqual(environment.export_scenario(), exported)
        for row in read_rows('hand_case.csv'):
            power = [row['p_bess_kw'], row['p_ev_kw']]
            actual = environment.step_power(power)
            expected = control.step_power(power)
            self.assertEqual(actual[1:4], expected[1:4])
            for key in actual[0]:
                np.testing.assert_array_equal(actual[0][key], expected[0][key])
            self.assertEqual(hand_columns(actual[4]), hand_columns(expected[4]))
        self.assertEqual(environment.export_scenario(), exported)

    def test_external_truth_shape_does_not_change_active_scenario(self):
        self.check_external_array_metadata_isolated('truth', 'shape')

    def test_external_truth_dtype_does_not_change_settlement(self):
        self.check_external_array_metadata_isolated('truth', 'dtype')

    def test_external_noise_metadata_does_not_change_export_or_observations(self):
        for metadata in ('shape', 'dtype'):
            with self.subTest(metadata=metadata):
                self.check_external_array_metadata_isolated('noise', metadata)

    def test_saturated_action_roundoff_is_clipped_and_replays_without_tolerance(self):
        environment, control = V2GEnv(n_slots=5), V2GEnv(n_slots=5)
        self.addCleanup(environment.close)
        self.addCleanup(control.close)
        environment.reset(seed=0, options={'date': '2026-10-01'})
        control.reset(options={'scenario': environment.export_scenario()})
        first_power = [-37.05, 0, 0, 0, 0, 0]
        first = environment.step_power(first_power)[4]
        control.step_power(first_power, tol=0)
        self.assertEqual(first['e_bess_kwh'], 40.5)
        bounds = environment.feasible_intervals()
        raw_power = bounds[0, 0] + (bounds[0, 1] - bounds[0, 0])
        self.assertGreater(raw_power, bounds[0, 1])
        actual = environment.step([1, 0, 0, 0, 0, 0])
        info = actual[4]
        self.assertEqual(info['p_bess_kw'], 60)
        self.assertEqual(info['p_bess_kw'], bounds[0, 1])
        correction = next(item for item in info['corrections']
                          if item['device'] == 0 and item['kind'] == 'power')
        self.assertEqual(correction['unit'], 'kW')
        self.assertEqual(correction['delta'], bounds[0, 1] - raw_power)
        self.assertLess(correction['delta'], 0)
        self.assertLessEqual(abs(correction['delta']), P_TOL)
        power = np.concatenate(([info['p_bess_kw']], info['p_ev_kw']))
        replayed = control.step_power(power, tol=0)
        self.assertEqual(actual[1:4], replayed[1:4])
        self.assertEqual(info['e_bess_kwh'], replayed[4]['e_bess_kwh'])
        self.assertEqual(info['step_cost_sgd'], replayed[4]['step_cost_sgd'])
        np.testing.assert_array_equal(info['e_ev_kwh'], replayed[4]['e_ev_kwh'])

    def test_real_bangbang_days_replay_with_zero_power_tolerance(self):
        for slots in (5, 10):
            environment, control = V2GEnv(n_slots=slots), V2GEnv(n_slots=slots)
            self.addCleanup(environment.close)
            self.addCleanup(control.close)
            environment.reset(seed=0, options={'date': '2026-10-01'})
            control.reset(options={'scenario': environment.export_scenario()})
            for t in range(48):
                with self.subTest(slots=slots, t=t):
                    bounds = environment.feasible_intervals()
                    np.testing.assert_array_equal(bounds, control.feasible_intervals())
                    action = np.where((np.arange(slots + 1) + t) % 2, 1., -1.)
                    actual = environment.step(action)
                    info = actual[4]
                    power = np.concatenate(([info['p_bess_kw']], info['p_ev_kw']))
                    self.assertTrue(np.all(power >= bounds[:, 0]))
                    self.assertTrue(np.all(power <= bounds[:, 1]))
                    replayed = control.step_power(power, tol=0)
                    self.assertEqual(actual[1:4], replayed[1:4])
                    self.assertEqual(info['e_bess_kwh'], replayed[4]['e_bess_kwh'])
                    self.assertEqual(info['step_cost_sgd'], replayed[4]['step_cost_sgd'])
                    np.testing.assert_array_equal(info['e_ev_kwh'], replayed[4]['e_ev_kwh'])
                    for key in actual[0]:
                        np.testing.assert_array_equal(actual[0][key], replayed[0][key])
            self.assertTrue(actual[2])
            self.assertEqual(actual[4]['episode_summary'], replayed[4]['episode_summary'])

    def test_constructor_rejects_invalid_runtime_tariff_and_capacity_scalars(self):
        fields = [('tariff', 'month_to_day_divisor'), ('tariff', 'ucc_month_sgd_per_kw'),
                  ('tariff', 'buy_adder_sgd_per_kwh'), ('tariff', 'contract_month_sgd_per_kw'),
                  ('building', 'contract_capacity_kw'), ('pv', 'installed_kwp'),
                  ('pv', 'daily_fixed_cost_sgd')]
        for group, field in fields:
            for invalid in (True, '1', 1 + 2j, np.complex128(1 + 2j), -1):
                params = load_params()
                params[group][field] = invalid
                with self.subTest(field=field, value=invalid), self.assertRaises(ValueError):
                    V2GEnv(params=params)
        params = load_params()
        params['tariff']['month_to_day_divisor'] = 0
        with self.assertRaises(ValueError):
            V2GEnv(params=params)

    def test_constructor_accepts_json_compatible_numpy_real_scalars(self):
        params = load_params()
        fields = [('tariff', 'month_to_day_divisor'), ('tariff', 'ucc_month_sgd_per_kw'),
                  ('tariff', 'buy_adder_sgd_per_kwh'), ('tariff', 'contract_month_sgd_per_kw'),
                  ('building', 'contract_capacity_kw'), ('pv', 'installed_kwp'),
                  ('pv', 'daily_fixed_cost_sgd')]
        for group, field in fields:
            params[group][field] = np.float64(params[group][field])
        environment = V2GEnv(params=params)
        self.addCleanup(environment.close)
        observation, _ = environment.reset(seed=0, options={'date': '2026-10-01'})
        self.assertTrue(environment.observation_space.contains(observation))
        _, reward, _, _, info = environment.step(np.zeros(6))
        self.assertTrue(np.isfinite(reward))
        self.assertAlmostEqual(info['buy_price'], .24594 + .03, delta=1e-12)

    def test_scenario_rejects_nonreal_or_nonnumeric_array_types(self):
        original = hand_scenario()
        for field in ('truth', 'noise'):
            values = getattr(original, field)
            cases = {
                'complex': values.astype(np.complex128) + 123j,
                'bool': values.astype(bool),
                'string': values.astype(str),
                'object': values.astype(object),
            }
            for kind, invalid in cases.items():
                with self.subTest(field=field, kind=kind), self.assertRaises(ValueError):
                    replace(original, **{field: invalid})

    def test_vehicle_and_scenario_reject_complex_scalar_values(self):
        original = hand_scenario()
        for field in ('initial_kwh', 'target_kwh'):
            for value in (24 + 123j, np.complex64(24 + 123j), np.complex128(24 + 123j)):
                with self.subTest(field=field, dtype=type(value)), self.assertRaises(ValueError):
                    replace(original.vehicles[0], **{field: value})
        for field in ('bess_initial_kwh', 'initial_peak_kw'):
            for value in (60 + 123j, np.complex64(60 + 123j), np.complex128(60 + 123j)):
                with self.subTest(field=field, dtype=type(value)), self.assertRaises(ValueError):
                    replace(original, **{field: value})

    def test_real_integer_and_float_scenario_inputs_remain_supported(self):
        original = hand_scenario()
        for dtype in (np.int16, np.float32, np.float64):
            with self.subTest(dtype=dtype):
                scenario = replace(
                    original, seed=np.int64(1), truth=original.truth.astype(dtype),
                    noise=np.zeros(original.noise.shape, dtype=dtype),
                    vehicles=(Vehicle(np.int64(0), np.int32(4), np.float64(24), np.float32(26)),),
                    bess_initial_kwh=np.int64(60), initial_peak_kw=np.float64(0),
                )
                json.dumps(scenario.to_dict(), allow_nan=False)
                environment = V2GEnv(n_slots=1)
                self.addCleanup(environment.close)
                observation, _ = environment.reset(options={'scenario': scenario})
                self.assertTrue(environment.observation_space.contains(observation))
                for _ in range(4):
                    observation, _, terminated, _, info = environment.step(np.zeros(2))
                    self.assertTrue(environment.observation_space.contains(observation))
                self.assertTrue(terminated)
                self.assertGreaterEqual(info['departures'][0]['energy_kwh'], 26 - 1e-9)

    def test_multiple_slots_simultaneous_departure_and_later_arrival(self):
        scenario = replace(hand_scenario(4), vehicles=(None,
            Vehicle(0, 2, 26., 26.), Vehicle(0, 2, 27., 26.), Vehicle(1, 4, 24., 26.)))
        environment = V2GEnv(n_slots=4)
        obs, _ = environment.reset(options={'scenario': scenario})
        self.assertEqual(obs['ev_present'].tolist(), [0, 1, 1, 0])
        np.testing.assert_allclose(environment.feasible_intervals()[1:],
                                   [[0,0],[-4.86,6],[-6,6],[0,0]], atol=1e-12, rtol=0)
        obs, _, _, _, first = environment.step_power([0,0,2,-1.8,0])
        np.testing.assert_allclose(first['p_ev_kw'], [0,2,-1.8,0], atol=1e-12)
        np.testing.assert_allclose(first['e_ev_kwh'], [0,26.9,26,0], atol=1e-12)
        self.assertEqual(obs['ev_present'].tolist(), [0,1,1,1])
        np.testing.assert_allclose(environment.feasible_intervals()[1:],
                                   [[0,0],[-1.62,6],[0,6],[-6,6]], atol=1e-12, rtol=0)
        obs, _, _, _, second = environment.step_power([0,0,-1.62,0,2])
        np.testing.assert_allclose(second['e_ev_before_kwh'], [0,26.9,26,24], atol=1e-12)
        np.testing.assert_allclose(second['e_ev_kwh'], [0,26,26,24.9], atol=1e-12)
        self.assertEqual([d['slot'] for d in second['departures']], [1,2])
        self.assertTrue(all(d['success'] for d in second['departures']))
        self.assertEqual(obs['ev_present'].tolist(), [0,0,0,1])
        np.testing.assert_allclose(environment.feasible_intervals()[1:],
                                   [[0,0],[0,0],[0,0],[-2.88,6]], atol=1e-12, rtol=0)
        _, _, _, _, third = environment.step_power([0,0,0,0,2])
        np.testing.assert_allclose(third['e_ev_before_kwh'], [0,0,0,24.9], atol=1e-12)
        np.testing.assert_allclose(third['e_ev_kwh'], [0,0,0,25.8], atol=1e-12)
        obs, _, terminated, _, last = environment.step_power([0,0,0,0,4/9])
        self.assertTrue(terminated)
        self.assertEqual(last['departures'][0]['slot'], 3)
        self.assertAlmostEqual(last['departures'][0]['energy_kwh'], 26, delta=1e-12)
        self.assertEqual(len(last['episode_summary']['departures']), 3)
        np.testing.assert_array_equal(obs['ev_present'], np.zeros(4, dtype=np.int8))
        environment.reset(options={'scenario': scenario})
        info = environment.step_power([0,0,6+5e-7,0,0])[4]
        correction = next(c for c in info['corrections'] if c['kind'] == 'power')
        self.assertEqual(correction['device'], 2)
        self.assertAlmostEqual(correction['delta'], -5e-7, delta=1e-12)

    def test_empty_episode_reports_no_departure_success_rate(self):
        scenario = replace(hand_scenario(), vehicles=(None,))
        environment = V2GEnv(n_slots=1)
        environment.reset(options={"scenario": scenario})
        for _ in range(4):
            _, _, _, _, info = environment.step_power([0.0, 0.0])
        self.assertIsNone(info["episode_summary"]["departure_success_rate"])
        self.assertEqual(info["episode_summary"]["departures"], [])


if __name__ == "__main__":
    unittest.main()
