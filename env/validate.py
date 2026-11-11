"""Run the fixed 504-episode physical and accounting acceptance matrix.

Usage from the repository root:
    python -m env.validate --output results/env_validation.json

This checks simulation invariants; it does not evaluate a learned policy or an
optimisation solver. Random actions use their own RNG, independent of scenarios.
"""
from __future__ import annotations

import argparse
from datetime import date as Date, datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time

import gymnasium
import numpy as np

from .scenario import parameter_id
from .v2g_env import V2GEnv, load_params


ENERGY_TOL = 1e-9
POWER_TOL = 1e-9
LEDGER_TOL = 1e-8
SLOT_COUNTS = (5, 10)
SCENARIO_SEEDS = (0, 1, 100)
ACTION_CASES = ("random", "minus_one", "plus_one")


class ValidationError(RuntimeError):
    """An acceptance invariant failed, with its episode and step context."""


def _require(condition, message, context):
    if not bool(condition):
        raise ValidationError(f"{context}: {message}")


class Residuals:
    """Track largest absolute discrepancies while enforcing unit-specific limits."""
    def __init__(self):
        self.maximum = {}

    def check(self, key, actual, expected, tolerance, context):
        actual = np.asarray(actual, dtype=np.float64)
        expected = np.asarray(expected, dtype=np.float64)
        _require(actual.shape == expected.shape,
                 f"{key}: shape {actual.shape} != {expected.shape}", context)
        _require(np.isfinite(actual).all() and np.isfinite(expected).all(),
                 f"{key}: nonfinite value", context)
        error = float(np.max(np.abs(actual - expected))) if actual.size else 0.0
        self.maximum[key] = max(self.maximum.get(key, 0.0), error)
        _require(error <= tolerance, f"{key}: residual {error:.12g} > {tolerance}", context)


def _check_observation(environment, observation, context):
    _require(environment.observation_space.contains(observation),
             "observation lies outside observation_space", context)
    for key, value in observation.items():
        _require(np.isfinite(value).all(), f"nonfinite observation[{key}]", context)


def _signature(scenario):
    payload = json.dumps(scenario, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _energy_corrections(info, n_slots, present, context, correction_counts, correction_maxima):
    changes = np.zeros(n_slots + 1, dtype=np.float64)
    expected_units = {"interval_lo": "kW", "power": "kW", "energy": "kWh"}
    for correction in info["corrections"]:
        _require(set(correction) == {"device", "kind", "delta", "unit"},
                 "unexpected correction fields", context)
        device, kind, delta = correction["device"], correction["kind"], correction["delta"]
        _require(isinstance(device, int) and not isinstance(device, bool) and 0 <= device <= n_slots,
                 "invalid correction device", context)
        _require(kind in expected_units, "unknown correction kind", context)
        _require(correction["unit"] == expected_units[kind], "incorrect correction unit", context)
        _require(math.isfinite(delta) and delta != 0, "invalid correction magnitude", context)
        correction_counts[kind] += 1
        correction_maxima[kind] = max(correction_maxima[kind],abs(delta))
        if kind == "energy":
            _require(abs(delta) <= ENERGY_TOL, "energy correction exceeds tolerance", context)
            _require(device == 0 or present[device - 1], "energy correction on an empty slot", context)
            changes[device] += delta
        elif kind == "interval_lo":
            _require(-POWER_TOL <= delta < 0, "invalid reversed-interval correction", context)
        else:
            # step() may clip only floating-point affine-map roundoff. Its
            # stricter limit differs from step_power's 1e-6 kW input tolerance.
            _require(abs(delta) <= POWER_TOL, "action mapping correction exceeds tolerance", context)
    _require(np.all(np.abs(changes) <= ENERGY_TOL), "total energy correction exceeds tolerance", context)
    return changes


def _run_episode(day, n_slots, seed, action_case, params, residuals, signatures, totals):
    label = f"date={day}, N={n_slots}, seed={seed}, action={action_case}"
    environment = V2GEnv(n_slots=n_slots, params=params)
    try:
        observation, reset_info = environment.reset(seed=seed, options={"date": day})
        scenario = environment.export_scenario()
        _check_observation(environment, observation, f"{label}, reset")
        _require(reset_info["date"] == day and reset_info["scenario_seed"] == seed,
                 "explicit date/seed changed during reset", label)
        _require(reset_info["params_id"] == parameter_id(params), "parameter identifier mismatch", label)
        case_key = (day, n_slots, seed)
        signature = _signature(scenario)
        if case_key in signatures:
            _require(signatures[case_key] == signature, "strategies received different scenarios", label)
        else:
            signatures[case_key] = signature
        truth = np.asarray(scenario["truth"], dtype=np.float64)
        vehicles = scenario["vehicles"]
        _require(truth.shape == (3, 48), "production scenario is not 48 steps", label)
        weekday = Date.fromisoformat(day).weekday() < 5
        expected_departures = n_slots if weekday else 0
        _require(sum(vehicle is not None for vehicle in vehicles) == expected_departures,
                 "unexpected weekday/weekend fleet size", label)
        action_rng = np.random.default_rng(
            np.random.SeedSequence([seed, Date.fromisoformat(day).toordinal(), n_slots, 6509]))
        bess, ev, tariff = params["bess"], params["ev"], params["tariff"]
        dt = params["simulation"]["delta_t_h"]
        lower = np.array([bess["capacity_kwh"] * bess["soc_min"]]
                         + [ev["capacity_kwh"] * ev["soc_min"]] * n_slots)
        upper = np.array([bess["capacity_kwh"] * bess["soc_max"]]
                         + [ev["capacity_kwh"] * ev["soc_max"]] * n_slots)
        charge_limit = np.array([bess["charge_max_kw"]] + [ev["charge_max_kw"]] * n_slots)
        discharge_limit = np.array([bess["discharge_max_kw"]] + [ev["discharge_max_kw"]] * n_slots)
        eta_ch = np.array([bess["eta_ch"]] + [ev["eta_ch"]] * n_slots)
        eta_dis = np.array([bess["eta_dis"]] + [ev["eta_dis"]] * n_slots)
        previous_energy = np.zeros(n_slots + 1, dtype=np.float64)
        previous_energy[0] = scenario["bess_initial_kwh"]
        peak = scenario["initial_peak_kw"]
        contract = params["building"]["contract_capacity_kw"]
        ucc_daily = tariff["ucc_month_sgd_per_kw"] / tariff["month_to_day_divisor"]
        costs, peak_fees, departures = [], [], []
        for t in range(48):
            context = f"{label}, t={t}"
            present = np.array([vehicle is not None and vehicle["arrival"] <= t < vehicle["departure"]
                                for vehicle in vehicles], dtype=bool)
            active = np.concatenate(([True], present))
            for slot, vehicle in enumerate(vehicles):
                if vehicle is not None and vehicle["arrival"] == t:
                    previous_energy[slot + 1] = vehicle["initial_kwh"]
            _require(np.array_equal(observation["ev_present"], present),
                     "observation occupancy disagrees with scenario", context)
            if action_case == "random":
                action = action_rng.uniform(-1.0, 1.0, size=n_slots + 1)
            else:
                action = np.full(n_slots + 1, -1.0 if action_case == "minus_one" else 1.0)
            try:
                observation, reward, terminated, truncated, info = environment.step(action)
            except Exception as error:
                raise ValidationError(f"{context}: step raised {type(error).__name__}: {error}") from error
            _require(terminated == (t == 47), "termination must occur only on step 48", context)
            _require(not truncated, "unexpected truncation", context)
            _require(info["t"] == t, "info step index mismatch", context)
            _require(np.array_equal(info["ev_present"], present), "info occupancy mismatch", context)
            _check_observation(environment, observation, f"{context}, next_obs")
            power = np.concatenate(([info["p_bess_kw"]], np.asarray(info["p_ev_kw"])))
            intervals = np.asarray(info["intervals"])
            _require(power.shape == (n_slots + 1,) and intervals.shape == (n_slots + 1, 2),
                     "incorrect power/interval shape", context)
            _require(np.isfinite(power).all() and np.isfinite(intervals).all(),
                     "nonfinite power or interval", context)
            _require(np.all(intervals[:, 0] <= intervals[:, 1]), "reversed public interval", context)
            violations = np.maximum(np.maximum(intervals[:, 0] - power, power - intervals[:, 1]), 0.0)
            residuals.check("power_interval_violation_kw", violations, np.zeros(n_slots + 1), POWER_TOL, context)
            violations = np.maximum(np.maximum(power - charge_limit, -power - discharge_limit), 0.0)
            residuals.check("power_rating_violation_kw", violations, np.zeros(n_slots + 1), POWER_TOL, context)
            residuals.check("empty_slot_power_kw", power[~active], np.zeros(np.sum(~active)), POWER_TOL, context)
            before = np.concatenate(([info["e_bess_before_kwh"]], info["e_ev_before_kwh"]))
            after = np.concatenate(([info["e_bess_kwh"]], info["e_ev_kwh"]))
            residuals.check("state_continuity_kwh", before, previous_energy, ENERGY_TOL, context)
            corrections = _energy_corrections(info, n_slots, present, context, totals["corrections"],
                                              totals["max_absolute_corrections"])
            # Independently apply the physical equation without calling physics.py.
            expected_energy = previous_energy + np.where(power >= 0, eta_ch * power * dt,
                                                         power * dt / eta_dis) + corrections
            residuals.check("energy_balance_kwh", after, expected_energy, ENERGY_TOL, context)
            _require(np.all(after[active] >= lower[active] - ENERGY_TOL)
                     and np.all(after[active] <= upper[active] + ENERGY_TOL),
                     "stored energy outside physical limits", context)
            residuals.check("empty_slot_energy_kwh", after[~active], np.zeros(np.sum(~active)), ENERGY_TOL, context)
            grid = float(truth[1, t] - truth[2, t]) + math.fsum(float(value) for value in power)
            residuals.check("grid_balance_kw", info["p_grid_kw"], grid, POWER_TOL, context)
            buy, sell = float(truth[0, t]) + tariff["buy_adder_sgd_per_kwh"], float(truth[0, t])
            residuals.check("buy_price_sgd_per_kwh", info["buy_price"], buy, LEDGER_TOL, context)
            residuals.check("sell_price_sgd_per_kwh", info["sell_price"], sell, LEDGER_TOL, context)
            peak_after = max(peak, grid)
            expected_cost = {
                "purchase_sgd": buy * max(grid, 0.0) * dt,
                "revenue_sgd": sell * max(-grid, 0.0) * dt,
                "peak_increment_sgd": ucc_daily * (max(0.0, peak_after - contract) - max(0.0, peak - contract)),
                "bess_wear_sgd": bess["wear_sgd_per_kwh"] * abs(float(power[0])) * dt,
                "ev_wear_sgd": ev["wear_sgd_per_kwh"] * math.fsum(abs(float(v)) for v in power[1:]) * dt,
            }
            for field, expected in expected_cost.items():
                residuals.check(f"step_{field}", info[field], expected, LEDGER_TOL, context)
            cost = (expected_cost["purchase_sgd"] - expected_cost["revenue_sgd"]
                    + expected_cost["peak_increment_sgd"] + expected_cost["bess_wear_sgd"]
                    + expected_cost["ev_wear_sgd"])
            residuals.check("step_cost_sgd", info["step_cost_sgd"], cost, LEDGER_TOL, context)
            residuals.check("reward_sgd", reward, -cost, LEDGER_TOL, context)
            residuals.check("peak_before_kw", info["peak_before_kw"], peak, POWER_TOL, context)
            residuals.check("peak_after_kw", info["peak_after_kw"], peak_after, POWER_TOL, context)
            due = [slot for slot, vehicle in enumerate(vehicles)
                   if vehicle is not None and vehicle["departure"] == t + 1]
            _require(sorted(record["slot"] for record in info["departures"]) == due,
                     "departure records missing, duplicated, or emitted at wrong step", context)
            for record in info["departures"]:
                slot = record["slot"]
                vehicle = vehicles[slot]
                _require(record["success"] is True, "unsuccessful departure", context)
                _require(record["departure_t"] == t + 1, "incorrect departure time", context)
                residuals.check("departure_energy_kwh", record["energy_kwh"], after[slot + 1], ENERGY_TOL, context)
                residuals.check("departure_target_kwh", record["target_kwh"], vehicle["target_kwh"], ENERGY_TOL, context)
                margin = record["energy_kwh"] - vehicle["target_kwh"]
                _require(margin >= -ENERGY_TOL, "departure energy target violated", context)
                totals["minimum_ev_departure_margin_kwh"] = min(totals["minimum_ev_departure_margin_kwh"], margin)
            departures.extend(info["departures"])
            previous_energy = expected_energy
            for slot in due:
                previous_energy[slot + 1] = 0.0
            peak = peak_after
            costs.append(float(info["step_cost_sgd"]))
            peak_fees.append(float(info["peak_increment_sgd"]))
            totals["steps"] += 1
            _require(("episode_summary" in info) == terminated,
                     "summary must appear only on terminal step", context)
        summary = info["episode_summary"]
        residuals.check("episode_variable_cost_sgd", summary["variable_cost_sgd"], math.fsum(costs), LEDGER_TOL, label)
        expected_peak_fee = ucc_daily * (max(0.0, peak - contract)
                                        - max(0.0, scenario["initial_peak_kw"] - contract))
        residuals.check("episode_peak_fee_sgd", math.fsum(peak_fees), expected_peak_fee, LEDGER_TOL, label)
        fixed_pv = params["pv"]["daily_fixed_cost_sgd"]
        fixed_contract = tariff["contract_month_sgd_per_kw"] * contract / tariff["month_to_day_divisor"]
        residuals.check("daily_fixed_pv_sgd", summary["fixed_pv_sgd"], fixed_pv, LEDGER_TOL, label)
        residuals.check("daily_fixed_contract_sgd", summary["fixed_contract_sgd"], fixed_contract, LEDGER_TOL, label)
        residuals.check("episode_total_cost_sgd", summary["total_cost_sgd"],
                        math.fsum(costs) + fixed_pv + fixed_contract, LEDGER_TOL, label)
        residuals.check("episode_peak_kw", summary["peak_grid_kw"], peak, POWER_TOL, label)
        residuals.check("episode_bess_final_kwh", summary["bess_final_kwh"], previous_energy[0], ENERGY_TOL, label)
        margin = summary["bess_final_kwh"] - scenario["bess_initial_kwh"]
        _require(margin >= -ENERGY_TOL, "BESS terminal energy below initial energy", label)
        _require(scenario["bess_initial_kwh"] == 60.0, "unexpected default BESS initial energy", label)
        _require(summary["departures"] == departures and len(departures) == expected_departures,
                 "episode departure summary mismatch", label)
        _require(summary["departure_success_rate"] == (1.0 if expected_departures else None),
                 "incorrect departure success rate or empty-day convention", label)
        _require(summary["date"] == day and summary["scenario_seed"] == seed
                 and summary["params_id"] == reset_info["params_id"], "summary metadata mismatch", label)
        _require(not observation["ev_present"].any(), "occupied slot after day-end", label)
        totals["minimum_bess_terminal_margin_kwh"] = min(totals["minimum_bess_terminal_margin_kwh"], margin)
        totals["departures"] += len(departures)
        totals["expected_departures"] += expected_departures
        totals["weekend_episodes"] += int(not weekday)
        totals["weekday_episodes"] += int(weekday)
        totals["episodes"] += 1
    except ValidationError:
        raise
    except Exception as error:
        raise ValidationError(f"{label}: {type(error).__name__}: {error}") from error
    finally:
        environment.close()


def validate_matrix():
    """Run every case and return a JSON-compatible acceptance report."""
    start = time.perf_counter()
    params = load_params()
    first = Date.fromisoformat(params["data"]["first_date"])
    last = Date.fromisoformat(params["data"]["last_date"])
    dates = [(first + timedelta(days=i)).isoformat() for i in range((last - first).days + 1)]
    _require(len(dates) == 28, "the acceptance matrix requires 28 supplied dates", "matrix")
    totals = {"episodes": 0, "steps": 0, "departures": 0, "expected_departures": 0,
              "weekend_episodes": 0, "weekday_episodes": 0,
              "minimum_bess_terminal_margin_kwh": math.inf,
              "minimum_ev_departure_margin_kwh": math.inf,
              "corrections": {"interval_lo": 0, "power": 0, "energy": 0},
              "max_absolute_corrections": {"interval_lo": 0., "power": 0., "energy": 0.}}
    residuals, signatures = Residuals(), {}
    for day in dates:
        for n_slots in SLOT_COUNTS:
            for seed in SCENARIO_SEEDS:
                for action_case in ACTION_CASES:
                    _run_episode(day, n_slots, seed, action_case, params, residuals, signatures, totals)
    _require(totals["episodes"] == 504 and totals["steps"] == 24192, "matrix size mismatch", "matrix")
    _require(totals["departures"] == totals["expected_departures"], "departure count mismatch", "matrix")
    return {
        "status": "passed", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": time.perf_counter() - start,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "gymnasium": gymnasium.__version__},
        "params_id": parameter_id(params),
        "matrix": {"dates": dates, "n_slots": list(SLOT_COUNTS), "scenario_seeds": list(SCENARIO_SEEDS),
                   "actions": list(ACTION_CASES),
                   "random_action_rng": "SeedSequence([scenario_seed,date_ordinal,n_slots,6509])"},
        "tolerances": {"energy_kwh": ENERGY_TOL, "power_kw": POWER_TOL, "ledger_sgd": LEDGER_TOL},
        **totals,
        "max_absolute_residuals": dict(sorted(residuals.maximum.items())),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional path for the JSON acceptance report")
    args = parser.parse_args(argv)
    try:
        report = validate_matrix()
    except ValidationError as error:
        print(f"VALIDATION FAILED: {error}", file=sys.stderr)
        return 1
    text = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
