"""Numeric acceptance cases from the model and parameter handoff."""

from dataclasses import FrozenInstanceError, replace
import unittest

import numpy as np

from env.data import load_params
from env.physics import (
    BatterySpec, E_TOL, P_TOL, POWER_INPUT_TOL, advance_energy,
    feasible_interval, settle_cost,
)


EV = BatterySpec(40, 6, 6, 0.2, 0.9, 0.9, 0.9, 0.03118)
BESS = BatterySpec(120, 60, 60, 0.1, 0.9, 0.95, 0.95, 0.04395)


def cost(**changes):
    arguments = dict(load_kw=0, pv_kw=0, usep=0.1, power=[0, 0],
                     bess=BESS, ev=EV, peak_before_kw=0,
                     contract_capacity_kw=240, ucc_daily=21.32 / 30,
                     adder=0.03)
    arguments.update(changes)
    return settle_cost(**arguments)


class PhysicsTests(unittest.TestCase):
    def assertClose(self, actual, expected, tol=1e-9):
        self.assertLessEqual(abs(actual - expected), tol, (actual, expected))

    def assertInterval(self, energy, target, steps, spec, expected):
        lo, hi, correction = feasible_interval(energy, target, steps, spec)
        self.assertClose(lo, expected[0])
        self.assertClose(hi, expected[1])
        self.assertEqual(correction, 0)

    def test_spec_from_params_and_frozen(self):
        group = dict(capacity_kwh=40, charge_max_kw=6, discharge_max_kw=6,
                     soc_min=.2, soc_max=.9, eta_ch=.9, eta_dis=.9,
                     wear_sgd_per_kwh=.03118, irrelevant="allowed")
        spec = BatterySpec.from_params(group)
        self.assertEqual(spec, EV)
        self.assertEqual((spec.energy_min, spec.energy_max), (8, 36))
        group["capacity_kwh"] = 90
        self.assertEqual(spec.capacity_kwh, 40)
        with self.assertRaises(FrozenInstanceError):
            spec.capacity_kwh = 90
        with self.assertRaises(ValueError):
            BatterySpec.from_params({})

    def test_real_parameter_binding(self):
        params = load_params()
        bess = BatterySpec.from_params(params["bess"])
        ev = BatterySpec.from_params(params["ev"])
        self.assertEqual(bess, BESS)
        self.assertEqual(ev, EV)
        tariff = params["tariff"]
        contract = params["building"]["contract_capacity_kw"]
        self.assertEqual(params["simulation"]["delta_t_h"], .5)
        self.assertEqual(tariff["ucc_month_sgd_per_kw"], 21.32)
        self.assertEqual(tariff["month_to_day_divisor"], 30)
        ucc_daily = tariff["ucc_month_sgd_per_kw"] / tariff["month_to_day_divisor"]
        self.assertClose(ucc_daily, 533 / 750, 1e-15)
        self.assertEqual(tariff["buy_adder_sgd_per_kwh"], .03)
        self.assertEqual(contract, 240)
        self.assertEqual(tariff["contract_month_sgd_per_kw"], 14.21)
        self.assertClose(contract * tariff["contract_month_sgd_per_kw"]
                         / tariff["month_to_day_divisor"], 113.68)
        result = settle_cost(
            load_kw=252, pv_kw=0, usep=.4, power=[0, 0], bess=bess, ev=ev,
            peak_before_kw=0, contract_capacity_kw=contract, ucc_daily=ucc_daily,
            adder=tariff["buy_adder_sgd_per_kwh"],
            dt=params["simulation"]["delta_t_h"],
        )
        self.assertClose(result["buy_price"], .43)
        self.assertClose(result["peak_increment_sgd"], 8.528)

    def test_seeded_feasible_trajectories(self):
        rng = np.random.default_rng(6509)
        self.interval_correction_count = 0
        for kind, count, spec, target in [("EV", 2000, EV, 26),
                                           ("BESS", 200, BESS, 60)]:
            for trajectory in range(count):
                if kind == "EV":
                    energy = float(rng.uniform(12, 28))
                    steps = int(rng.integers(12, 27))
                else:
                    energy, steps = 60., 48
                with self.subTest(kind=kind, trajectory=trajectory):
                    for remaining in range(steps, 0, -1):
                        lo, hi, interval_correction = feasible_interval(
                            energy, target, remaining, spec)
                        self.assertLessEqual(lo, hi)
                        self.assertGreaterEqual(lo, -spec.discharge_max_kw - P_TOL)
                        self.assertLessEqual(hi, spec.charge_max_kw + P_TOL)
                        self.assertLessEqual(abs(interval_correction), P_TOL)
                        self.assertLessEqual(interval_correction, 0)
                        self.interval_correction_count += int(interval_correction != 0)
                        mode = int(rng.integers(0, 3))
                        action = (-1. if mode == 0 else 1. if mode == 1
                                  else float(rng.uniform(-1, 1)))
                        power = lo + (action + 1) * .5 * (hi - lo)
                        energy, energy_correction = advance_energy(energy, power, spec)
                        self.assertLessEqual(abs(energy_correction), E_TOL)
                        self.assertGreaterEqual(energy, spec.energy_min)
                        self.assertLessEqual(energy, spec.energy_max)
                        possible_final = (energy + spec.eta_ch * spec.charge_max_kw
                                          * (remaining - 1) * .5)
                        self.assertGreaterEqual(possible_final, target - E_TOL)
                    self.assertGreaterEqual(energy, target - E_TOL)
        self.assertGreater(self.interval_correction_count, 0)

    def test_mixed_fleet_power_and_absolute_throughput(self):
        for ev_power, expected_grid in [([6, -4, -2], 90), ([6, -4, 2], 94)]:
            with self.subTest(ev_power=ev_power):
                result = cost(load_kw=100, pv_kw=30, power=[20] + ev_power)
                self.assertEqual(result["p_grid_kw"], expected_grid)
                np.testing.assert_array_equal(result["p_ev_kw"], ev_power)
                self.assertClose(result["bess_wear_sgd"], .4395)
                self.assertClose(result["ev_wear_sgd"], .18708)
                self.assertClose(result["step_cost_sgd"],
                                 expected_grid * .13 * .5 + .4395 + .18708)

    def test_invalid_specs(self):
        for field, value in [("capacity_kwh", 0), ("capacity_kwh", True),
                             ("charge_max_kw", -1), ("discharge_max_kw", 0),
                             ("soc_min", -.1), ("soc_min", .9),
                             ("soc_max", 1.1), ("eta_ch", 0),
                             ("eta_dis", 1.01), ("eta_ch", np.nan),
                             ("wear_sgd_per_kwh", -1)]:
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                replace(EV, **{field: value})

    def test_upper_and_lower_energy_limits(self):
        self.assertInterval(35.5, 26, 8, EV, (-6, 10 / 9))
        energy, correction = advance_energy(35.5, 10 / 9, EV)
        self.assertClose(energy, 36)
        self.assertEqual(correction, 0)
        self.assertInterval(36, 26, 8, EV, (-6, 0))
        self.assertInterval(107.5, 60, 48, BESS, (-60, 20 / 19))
        self.assertClose(advance_energy(107.5, 20 / 19, BESS)[0], 108)
        self.assertInterval(12, 60, 48, BESS, (0, 60))

    def test_forced_ev_and_bess_charging(self):
        self.assertInterval(25, 26, 1, EV, (20 / 9, 6))
        self.assertClose(advance_energy(25, 20 / 9, EV)[0], 26)
        self.assertInterval(23.3, 26, 1, EV, (6, 6))
        self.assertInterval(41, 60, 1, BESS, (40, 60))
        self.assertClose(advance_energy(41, 40, BESS)[0], 60)

    def test_unreachable_target_and_invalid_energy(self):
        for energy, target, steps, spec in [(22, 26, 1, EV), (30, 60, 1, BESS),
                                             (7, 26, 8, EV), (37, 26, 8, EV),
                                             (24, 37, 8, EV), (24, 7, 8, EV)]:
            with self.subTest(energy=energy, target=target), self.assertRaisesRegex(
                    ValueError, "infeasible_input"):
                feasible_interval(energy, target, steps, spec)

    def test_tiny_interval_inversion_only(self):
        lo, hi, correction = feasible_interval(23.3 - 1e-10, 26, 1, EV)
        self.assertEqual((lo, hi), (6, 6))
        self.assertLess(correction, 0)
        self.assertGreaterEqual(correction, -P_TOL)
        raw_lo = (26 - (23.3 - 1e-10)) / (.9 * .5)
        self.assertEqual(correction, hi - raw_lo)
        lo, hi, correction = feasible_interval(23.3 + 1e-10, 26, 1, EV)
        self.assertLess(lo, hi)
        self.assertEqual(correction, 0)
        with self.assertRaisesRegex(ValueError, "infeasible_input"):
            feasible_interval(23.3 - 1e-8, 26, 1, EV)

    def test_energy_snap_corrections_and_no_inside_snap(self):
        charge = E_TOL / (2 * EV.eta_ch * .5)
        discharge = -E_TOL * EV.eta_dis / (2 * .5)
        energy, correction = advance_energy(36, charge, EV)
        self.assertEqual(energy, 36)
        self.assertLess(correction, 0)
        self.assertClose(correction, -E_TOL / 2, 1e-14)
        energy, correction = advance_energy(8, discharge, EV)
        self.assertEqual(energy, 8)
        self.assertGreater(correction, 0)
        self.assertClose(correction, E_TOL / 2, 1e-14)
        energy, correction = advance_energy(8, charge, EV)
        self.assertGreater(energy, 8)
        self.assertEqual(correction, 0)
        energy, correction = advance_energy(36, discharge, EV)
        self.assertLess(energy, 36)
        self.assertEqual(correction, 0)
        for initial, power in [(36, charge * 4), (8, discharge * 4)]:
            with self.assertRaisesRegex(ValueError, "infeasible_input"):
                advance_energy(initial, power, EV)

    def test_power_rating_tolerance(self):
        self.assertEqual(POWER_INPUT_TOL, 1e-6)
        for power in [6 + .5 * P_TOL, -6 - .5 * P_TOL]:
            energy, correction = advance_energy(24, power, EV)
            self.assertTrue(np.isfinite(energy))
            self.assertEqual(correction, 0)
        for power in [6 + 2 * P_TOL, -6 - 2 * P_TOL]:
            with self.assertRaisesRegex(ValueError, "infeasible_input"):
                advance_energy(24, power, EV)

    def test_invalid_dynamic_inputs(self):
        for value in [0, -.5, np.nan, np.inf, True]:
            with self.subTest(dt=value), self.assertRaises(ValueError):
                feasible_interval(24, 26, 8, EV, dt=value)
            with self.subTest(dt=value), self.assertRaises(ValueError):
                advance_energy(24, 0, EV, dt=value)
        for value in [0, -1, 1.5, True, np.nan]:
            with self.subTest(steps=value), self.assertRaises(ValueError):
                feasible_interval(24, 26, value, EV)
        with self.assertRaises(ValueError):
            feasible_interval(24, 26, 8, replace(EV, eta_ch=.1), dt=5e-324)
        for value in [np.nan, np.inf, True, "24"]:
            with self.subTest(energy=value), self.assertRaises(ValueError):
                feasible_interval(value, 26, 8, EV)
            with self.subTest(power=value), self.assertRaises(ValueError):
                advance_energy(24, value, EV)

    def test_four_step_hand_case(self):
        # t, load, PV, USEP, BESS power, EV power, expected interval bounds,
        # expected BESS/EV energy, grid power, purchase, revenue, peak, wear, cost.
        rows = [
            (220, 40, .10, 20, -2.7, (-60, 60), (-6, 6),
             69.5, 23.3, 197.3, 12.8245, 0, 0, .4395, .042093, 13.306093),
            (300, 10, .40, -38, 0, (-60, 60), (-4.86, 6),
             49.5, 23.3, 252, 54.18, 0, 8.528, .83505, 0, 63.54305),
            (60, 50, .20, -16.15, 0, (-34.2, 60), (0, 6),
             41, 23.3, -6.15, 0, .615, 0, .35489625, 0, -.26010375),
            (180, 20, .15, 40, 6, (40, 60), (6, 6),
             60, 26, 206, 18.54, 0, 0, .879, .09354, 19.51254),
        ]
        e_bess, e_ev, peak, total = 60., 24.8, 0., 0.
        for t, row in enumerate(rows):
            (load, pv, usep, p_bess, p_ev, b_bounds, ev_bounds,
             expected_b, expected_ev, grid, purchase, revenue, peak_fee,
             b_wear, ev_wear, step_cost) = row
            with self.subTest(t=t):
                self.assertInterval(e_bess, 60, 4 - t, BESS, b_bounds)
                self.assertInterval(e_ev, 26, 4 - t, EV, ev_bounds)
                e_bess = advance_energy(e_bess, p_bess, BESS)[0]
                e_ev = advance_energy(e_ev, p_ev, EV)[0]
                result = cost(load_kw=load, pv_kw=pv, usep=usep,
                              power=[p_bess, p_ev], peak_before_kw=peak)
                for name, expected in [
                    ("p_grid_kw", grid), ("purchase_sgd", purchase),
                    ("revenue_sgd", revenue), ("peak_increment_sgd", peak_fee),
                    ("bess_wear_sgd", b_wear), ("ev_wear_sgd", ev_wear),
                    ("step_cost_sgd", step_cost),
                ]:
                    self.assertClose(result[name], expected)
                self.assertClose(e_bess, expected_b)
                self.assertClose(e_ev, expected_ev)
                peak = result["peak_after_kw"]
                total += result["step_cost_sgd"]
        self.assertClose(total, 96.10157925)
        self.assertEqual(peak, 252)
        self.assertClose(e_bess, 60)
        self.assertClose(e_ev, 26)

    def test_peak_increment_and_export(self):
        first = cost(load_kw=270, peak_before_kw=250)
        self.assertClose(first["peak_increment_sgd"], 1066 / 75)
        self.assertEqual(first["peak_after_kw"], 270)
        for load, pv in [(265, 0), (0, 10)]:
            with self.subTest(load=load, pv=pv):
                result = cost(load_kw=load, pv_kw=pv, peak_before_kw=270)
                self.assertEqual(result["peak_increment_sgd"], 0)
                self.assertEqual(result["peak_after_kw"], 270)
        self.assertEqual(cost(pv_kw=10)["peak_after_kw"], 0)

    def test_negative_price_signs(self):
        exported = cost(pv_kw=10, usep=-.10)
        self.assertClose(exported["revenue_sgd"], -.5)
        self.assertClose(exported["step_cost_sgd"], .5)
        imported = cost(load_kw=10, usep=-.10)
        self.assertClose(imported["buy_price"], -.07)
        self.assertClose(imported["purchase_sgd"], -.35)
        self.assertClose(imported["step_cost_sgd"], -.35)

    def test_no_ev_and_no_input_mutation(self):
        power = np.array([20., -2.7])
        original = power.copy()
        result = cost(power=power)
        np.testing.assert_array_equal(power, original)
        result["p_ev_kw"][0] = 100
        np.testing.assert_array_equal(power, original)
        result = cost(power=[0], load_kw=1)
        self.assertEqual(result["p_ev_kw"].shape, (0,))
        self.assertEqual(result["ev_wear_sgd"], 0)

    def test_invalid_settlement_inputs(self):
        for power in [0, [], [[0, 0]], [np.nan, 0], [np.inf, 0],
                      [True, False], ["0", "1"], [0j, 0j]]:
            with self.subTest(power=power), self.assertRaises(ValueError):
                cost(power=power)
        for key in ["load_kw", "pv_kw", "peak_before_kw", "contract_capacity_kw",
                    "ucc_daily", "adder"]:
            for value in [-1, np.inf, True]:
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    cost(**{key: value})
        with self.assertRaises(ValueError):
            cost(usep=np.nan)
        with self.assertRaises(ValueError):
            cost(usep=1e308, load_kw=1e308)
        with self.assertRaises(ValueError):
            advance_energy(24, 1, EV, dt=1e308)


if __name__ == "__main__":
    unittest.main()
