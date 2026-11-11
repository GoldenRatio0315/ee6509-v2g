"""Observation timing, units, masking and Gymnasium interface checks."""
from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np
from gymnasium.utils.env_checker import check_env

from env.scenario import Scenario, Vehicle, parameter_id
from env.v2g_env import V2GEnv, load_params


def observation_scenario(n_slots=5, steps=48):
    """Distinct deterministic rows make history/forecast indexing observable."""
    time = np.arange(steps, dtype=np.float64)
    truth = np.stack((-0.4 + 0.01 * time, 100.0 + time, time))
    return Scenario(
        date="2026-10-01", seed=0, params_id=parameter_id(load_params()),
        truth=truth, vehicles=(None,) * n_slots, bess_initial_kwh=60.0,
        initial_peak_kw=0.0, noise=np.zeros((3, steps + 1, 12)),
    )


def reset_scenario(scenario):
    environment = V2GEnv(n_slots=len(scenario.vehicles))
    observation, _ = environment.reset(options={"scenario": scenario})
    return environment, observation


def advance_zero_power(environment, steps):
    observation = None
    for _ in range(steps):
        observation, _, _, _, _ = environment.step_power(np.zeros(environment.n_slots + 1))
    return observation


class ObservationTests(unittest.TestCase):
    def test_unrepresentable_battery_scale_is_rejected(self):
        from copy import deepcopy
        from env.data import load_params
        for group in ('ev','bess'):
            params = deepcopy(load_params())
            params[group]['capacity_kwh'] = 1e100
            with self.subTest(group=group), self.assertRaises(ValueError):
                V2GEnv(params=params)

    def test_initial_peak_float32_overflow_is_rejected(self):
        with self.assertRaises(ValueError):
            reset_scenario(replace(observation_scenario(),initial_peak_kw=1e39))

    def test_production_gymnasium_checker_five_and_ten_slots(self):
        for n_slots in (5, 10):
            with self.subTest(n_slots=n_slots):
                environment = V2GEnv(n_slots=n_slots)
                check_env(environment, skip_render_check=True)
                environment.close()

    def test_history_includes_current_and_pads_first_day_value(self):
        scenario = observation_scenario()
        environment, observation = reset_scenario(scenario)
        expected_history = np.repeat(scenario.truth[:, :1], 8, axis=1).astype(np.float32)
        np.testing.assert_array_equal(observation["timeseries"][:, :8], expected_history)
        np.testing.assert_array_equal(observation["timeseries"][:, 8:],
                                      scenario.truth[:, 1:13].astype(np.float32))
        observation = advance_zero_power(environment, 3)
        expected_history = scenario.truth[:, [0, 0, 0, 0, 0, 1, 2, 3]].astype(np.float32)
        np.testing.assert_array_equal(observation["timeseries"][:, :8], expected_history)
        observation = advance_zero_power(environment, 5)
        np.testing.assert_array_equal(observation["timeseries"][:, :8],
                                      scenario.truth[:, 1:9].astype(np.float32))
        np.testing.assert_array_equal(observation["timeseries"][:, 8:],
                                      scenario.truth[:, 9:21].astype(np.float32))

    def test_given_forecast_errors_use_leads_one_to_twelve_and_clipping(self):
        scenario = observation_scenario()
        noise = scenario.noise.copy()
        noise[0, 10] = np.arange(1, 13) * 0.01
        noise[1, 10] = [-2.0, -1.0, -0.5, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        noise[2, 10] = 2.0
        scenario = replace(scenario, noise=noise)
        environment, _ = reset_scenario(scenario)
        observation = advance_zero_power(environment, 10)
        expected = scenario.truth[:, 11:23] * (1 + noise[:, 10])
        expected[1] = np.maximum(expected[1], 0.0)
        expected[2] = np.clip(expected[2], 0.0, 60.0)
        np.testing.assert_array_equal(observation["timeseries"][:, 8:], expected.astype(np.float32))
        self.assertTrue(np.all(observation["timeseries"][0, 8:] < 0.0))
        self.assertEqual(observation["timeseries"][1, 8], 0.0)
        self.assertEqual(observation["timeseries"][1, 9], 0.0)
        self.assertEqual(observation["timeseries"][2, -1], 60.0)
        np.testing.assert_array_equal(observation["timeseries"][:, :8],
                                      scenario.truth[:, 3:11].astype(np.float32))

    def test_last_day_padding_and_distinct_terminal_noise(self):
        scenario = observation_scenario()
        noise = scenario.noise.copy()
        noise[0, 47] = 0.1
        noise[0, 48] = 0.2
        scenario = replace(scenario, noise=noise)
        environment, _ = reset_scenario(scenario)
        observation = advance_zero_power(environment, 47)
        np.testing.assert_array_equal(observation["timeseries"][:, :8],
                                      scenario.truth[:, 40:48].astype(np.float32))
        expected = np.repeat(scenario.truth[:, -1:], 12, axis=1)
        expected[0] *= 1.1
        np.testing.assert_array_equal(observation["timeseries"][:, 8:], expected.astype(np.float32))
        observation, _, terminated, truncated, info = environment.step_power(np.zeros(6))
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertTrue(environment.observation_space.contains(observation))
        history = scenario.truth[:, [41, 42, 43, 44, 45, 46, 47, 47]]
        np.testing.assert_array_equal(observation["timeseries"][:, :8], history.astype(np.float32))
        expected[0] = scenario.truth[0, -1] * 1.2
        np.testing.assert_array_equal(observation["timeseries"][:, 8:], expected.astype(np.float32))
        self.assertEqual(info["t"], 47)
        self.assertAlmostEqual(observation["global"][0], 0.5, delta=1e-7)
        self.assertEqual(observation["global"][1], 100.0)
        np.testing.assert_array_equal(observation["ev_present"], np.zeros(5, dtype=np.int8))
        np.testing.assert_array_equal(observation["ev"], np.zeros((5, 4), dtype=np.float32))

    def test_truth_beyond_t_plus_twelve_is_not_visible(self):
        scenario = observation_scenario()
        changed = scenario.truth.copy()
        changed[:, 16:] += np.array([0.5, 50.0, 10.0])[:, None]
        first, _ = reset_scenario(scenario)
        second, _ = reset_scenario(replace(scenario, truth=changed))
        first_observation = advance_zero_power(first, 3)
        second_observation = advance_zero_power(second, 3)
        for key in first_observation:
            np.testing.assert_array_equal(first_observation[key], second_observation[key])
        visible_change = scenario.truth.copy()
        visible_change[0, 15] += 0.5
        third, _ = reset_scenario(replace(scenario, truth=visible_change))
        third_observation = advance_zero_power(third, 3)
        self.assertNotEqual(first_observation["timeseries"][0, -1],
                            third_observation["timeseries"][0, -1])

    def test_forecasts_are_independent_of_actions_and_read_queries(self):
        scenario = observation_scenario()
        noise = np.arange(3 * 49 * 12, dtype=np.float64).reshape(3, 49, 12) * 1e-5
        scenario = replace(scenario, noise=noise,
                           vehicles=(Vehicle(0, 48, 24.8, 26.0),) + (None,) * 4)
        first, observation_a = reset_scenario(scenario)
        second, observation_b = reset_scenario(scenario)
        states_differed = False
        for _ in range(48):
            np.testing.assert_array_equal(observation_a["timeseries"], observation_b["timeseries"])
            # Public read-only queries must not consume forecast randomness.
            first.feasible_intervals()
            first.export_scenario()
            first.feasible_intervals()
            observation_a, _, _, _, _ = first.step(-np.ones(6))
            observation_b, _, _, _, _ = second.step(np.ones(6))
            states_differed |= not np.array_equal(observation_a["global"], observation_b["global"])
        self.assertTrue(states_differed)
        np.testing.assert_array_equal(observation_a["timeseries"], observation_b["timeseries"])

    def test_forecast_noise_does_not_enter_physics_or_settlement(self):
        scenario = observation_scenario()
        scenario = replace(scenario, vehicles=(Vehicle(0, 48, 24.8, 26.0),) + (None,) * 4)
        noisy = replace(scenario, noise=np.full_like(scenario.noise, 0.5))
        first, observation_a = reset_scenario(scenario)
        second, observation_b = reset_scenario(noisy)
        self.assertFalse(np.array_equal(observation_a["timeseries"], observation_b["timeseries"]))
        cost_fields = ("p_grid_kw", "buy_price", "sell_price", "purchase_sgd", "revenue_sgd",
                       "peak_increment_sgd", "bess_wear_sgd", "ev_wear_sgd", "step_cost_sgd",
                       "e_bess_kwh", "peak_after_kw")
        for t in range(48):
            action = np.full(6, -1.0 if t % 2 == 0 else 1.0)
            result_a, result_b = first.step(action), second.step(action)
            self.assertEqual(result_a[1:4], result_b[1:4])
            for key in cost_fields:
                self.assertEqual(result_a[4][key], result_b[4][key])
            np.testing.assert_array_equal(result_a[4]["e_ev_kwh"], result_b[4]["e_ev_kwh"])
            np.testing.assert_array_equal(result_a[4]["intervals"], result_b[4]["intervals"])
        self.assertEqual(result_a[4]["episode_summary"], result_b[4]["episode_summary"])

    def test_ev_mask_dtype_shape_remaining_steps_gap_and_empty_rows(self):
        for n_slots in (5, 10):
            with self.subTest(n_slots=n_slots):
                scenario = observation_scenario(n_slots=n_slots, steps=4)
                scenario = replace(scenario, vehicles=(
                    Vehicle(2, 4, 24.8, 26.0),) + (None,) * (n_slots - 1))
                environment, observation = reset_scenario(scenario)
                self.assertEqual(observation["ev_present"].dtype, np.dtype(np.int8))
                self.assertEqual(observation["ev_present"].shape, (n_slots,))
                np.testing.assert_array_equal(observation["ev"], np.zeros((n_slots, 4)))
                observation = advance_zero_power(environment, 2)
                self.assertEqual(observation["ev_present"].tolist(), [1] + [0] * (n_slots - 1))
                np.testing.assert_allclose(observation["ev"][0], [1.0, 0.62, 2.0, 1.2], atol=1e-7)
                np.testing.assert_array_equal(observation["ev"][1:], np.zeros((n_slots - 1, 4)))
                self.assertEqual((observation["ev_present"] < 0.5).tolist(),
                                 [False] + [True] * (n_slots - 1))
                observation = advance_zero_power(environment, 1)
                self.assertEqual(observation["ev"][0, 2], 1.0)
                power = np.zeros(n_slots + 1)
                power[1] = 6.0
                observation, _, terminated, _, info = environment.step_power(power)
                self.assertTrue(terminated)
                self.assertTrue(info["departures"][0]["success"])
                np.testing.assert_array_equal(observation["ev_present"], np.zeros(n_slots, dtype=np.int8))
                np.testing.assert_array_equal(observation["ev"], np.zeros((n_slots, 4)))

    def test_observations_keep_raw_units_and_rewards_are_not_scaled(self):
        scenario = observation_scenario()
        environment, observation = reset_scenario(scenario)
        self.assertEqual(observation["timeseries"][1, 7], 100.0)
        self.assertAlmostEqual(float(observation["timeseries"][0, 7]), -0.4, delta=1e-7)
        self.assertEqual(observation["timeseries"][2, -1], 12.0)
        np.testing.assert_array_equal(observation["global"], [0.5, 0.0, 0.0, 1.0])
        observation, reward, _, _, info = environment.step_power(np.zeros(6))
        self.assertEqual(info["p_grid_kw"], 100.0)
        self.assertAlmostEqual(info["purchase_sgd"], -18.5, delta=1e-12)
        self.assertAlmostEqual(reward, 18.5, delta=1e-12)
        self.assertEqual(observation["global"][1], 100.0)
        np.testing.assert_allclose(observation["global"][2:],
                                   [np.sin(2 * np.pi / 48), np.cos(2 * np.pi / 48)], atol=1e-7)

    def test_all_production_observations_including_terminal_are_finite_and_in_space(self):
        for n_slots, day in ((5, "2026-10-01"), (10, "2026-10-03")):
            with self.subTest(n_slots=n_slots, day=day):
                environment = V2GEnv(n_slots=n_slots)
                observation, _ = environment.reset(seed=100, options={"date": day})
                for t in range(49):
                    self.assertTrue(environment.observation_space.contains(observation))
                    for key, value in observation.items():
                        self.assertTrue(np.isfinite(value).all(), (t, key))
                        expected_dtype = np.int8 if key == "ev_present" else np.float32
                        self.assertEqual(value.dtype, np.dtype(expected_dtype))
                    if t < 48:
                        observation, _, terminated, truncated, _ = environment.step(np.ones(n_slots + 1))
                        self.assertEqual(terminated, t == 47)
                        self.assertFalse(truncated)

    def test_extreme_finite_truth_is_rejected_before_returning_observation(self):
        scenario = observation_scenario()
        for variable in (0, 1):
            for t in (0, 47):
                with self.subTest(variable=variable, t=t):
                    truth = scenario.truth.copy()
                    truth[variable, t] = 1e100
                    invalid = replace(scenario, truth=truth)
                    with self.assertRaises(ValueError):
                        reset_scenario(invalid)

    def test_extreme_finite_noise_including_terminal_is_rejected_at_reset(self):
        scenario = observation_scenario()
        for variable in (0, 1):
            for t in (0, 48):
                with self.subTest(variable=variable, t=t):
                    noise = scenario.noise.copy()
                    noise[variable, t, -1] = 1e100
                    invalid = replace(scenario, noise=noise)
                    with self.assertRaises(ValueError):
                        reset_scenario(invalid)


if __name__ == "__main__":
    unittest.main()
