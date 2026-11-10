"""Exact model/fixture assertions for EE6509 D v0.3; Python 3.9+, stdlib only.

All physical and settlement checks use Fraction, never binary floating point.
This checks D's model specification and data, not A's environment or C's solver.
"""
import csv
import json
from datetime import date, datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP, localcontext
from fractions import Fraction as F
from pathlib import Path

from build_timeseries import FIELDS, csv_text, make_days, make_manifest, values

HERE = Path(__file__).resolve().parent
ANNOTATED = json.loads((HERE / "params.yaml").read_text(encoding="utf-8"), parse_float=F)
CFG = values(ANNOTATED)
DT = CFG["simulation"]["delta_t_h"]


def num(value):
    return value if isinstance(value, F) else F(value)


def rounded(value, places):
    with localcontext() as context:
        context.prec = 50
        number = Decimal(value.numerator) / Decimal(value.denominator)
        return F(str(number.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)))


def energy_next(energy, power, eta_ch, eta_dis, dt=DT):
    return energy + eta_ch * max(power, 0) * dt - max(-power, 0) / eta_dis * dt


def power_for_delta(delta_energy, eta_ch, eta_dis, dt=DT):
    """Inverse of the strictly increasing, piecewise-linear energy update."""
    return delta_energy / (eta_ch * dt) if delta_energy >= 0 else delta_energy * eta_dis / dt


def feasible_interval(energy, device, k=None, target=None, present=True):
    if not present:
        return F(0), F(0)
    capacity = device["capacity_kwh"]
    minimum, maximum = capacity * device["soc_min"], capacity * device["soc_max"]
    if not minimum <= energy <= maximum:
        raise ValueError("infeasible_input: initial energy outside operating bounds")
    required = minimum
    if k is not None:
        if k < 1 or target is None:
            raise ValueError("An occupied EV requires k>=1 and a target")
        required = max(minimum, target - device["eta_ch"] * device["charge_max_kw"] * (k - 1) * DT)
    lo = max(-device["discharge_max_kw"],
             power_for_delta(required - energy, device["eta_ch"], device["eta_dis"]))
    hi = min(device["charge_max_kw"],
             power_for_delta(maximum - energy, device["eta_ch"], device["eta_dis"]))
    if lo > hi:
        raise ValueError("infeasible_input: departure/end-of-day target cannot be reached")
    return lo, hi


def project(action, interval):
    assert -1 <= action <= 1
    lo, hi = interval
    return lo + (action + 1) * (hi - lo) / 2


def settlement(load, pv, p_bess, p_evs, usep, peak_before):
    grid = load - pv + p_bess + sum(p_evs)
    peak_after = max(peak_before, grid)
    cc = CFG["building"]["contract_capacity_kw"]
    day_rate = CFG["tariff"]["ucc_month_sgd_per_kw"] / CFG["tariff"]["month_to_day_divisor"]
    buy_price = usep + CFG["tariff"]["buy_adder_sgd_per_kwh"]
    purchase = buy_price * max(grid, 0) * DT
    revenue = usep * max(-grid, 0) * DT
    peak_increment = day_rate * (max(0, peak_after - cc) - max(0, peak_before - cc))
    bess_wear = CFG["bess"]["wear_sgd_per_kwh"] * abs(p_bess) * DT
    ev_wear = CFG["ev"]["wear_sgd_per_kwh"] * sum(abs(power) for power in p_evs) * DT
    return dict(p_grid=grid, buy_price=buy_price, sell_price=usep, purchase=purchase,
                revenue=revenue, peak_before=peak_before, peak_after=peak_after,
                peak_increment=peak_increment, bess_wear=bess_wear, ev_wear=ev_wear,
                step_cost=purchase - revenue + peak_increment + bess_wear + ev_wear)


def check_metadata():
    registry = json.loads((HERE / "sources/source_manifest.json").read_text(encoding="utf-8"))
    ids = {source["id"] for source in registry["sources"]} | registry["assumptions"].keys()

    def walk(node, path=""):
        if not isinstance(node, dict):
            return
        if "value" in node:
            assert {"value", "unit", "source", "status", "note"} <= node.keys(), path
            assert node["unit"] and node["source"] and node["note"], path
            assert set(node["source"]) <= ids, path
        else:
            for key, value in node.items():
                walk(value, path + "." + key)

    walk(ANNOTATED)
    assert DT == F("0.5") and CFG["simulation"]["steps_per_day"] == 48
    assert CFG["simulation"]["forecast_steps"] == 12
    print("PASS: parameter units, registered sources/assumptions, 48 half-hours and 12-step horizon")


def check_economics():
    factor = CFG["economics"]["sgd_per_usd"]
    for key, raw_expected in (("ev", F(873, 28000)), ("bess", F(45, 1024))):
        device = CFG[key]
        price = device["pack_price_usd_per_kwh"] * factor * device["capacity_kwh"]
        usable = device["capacity_kwh"] * (device["soc_max"] - device["soc_min"])
        assert usable == device["capacity_kwh"] * device["usable_fraction_for_wear"]
        coefficient = price / (device["cycle_life"] * usable
                               * CFG["economics"]["throughput_directions_per_cycle"])
        assert coefficient == raw_expected
        assert rounded(coefficient, CFG["economics"]["wear_round_places"]) == device["wear_sgd_per_kwh"]
    pv = CFG["pv"]
    capital = pv["installed_kwp"] * pv["capital_cost_sgd_per_kwp"]
    daily = (F(capital) / pv["life_years"] + capital * pv["maintenance_fraction_per_year"]) / pv["calendar_days_per_year"]
    assert capital == 72000 and daily == F(864, 73)
    assert rounded(daily, 8) == pv["daily_fixed_cost_sgd"]
    fixed_contract = (CFG["building"]["contract_capacity_kw"]
                      * CFG["tariff"]["contract_month_sgd_per_kw"]
                      / CFG["tariff"]["month_to_day_divisor"])
    assert fixed_contract == F("113.68")
    assert CFG["tariff"]["ucc_month_sgd_per_kw"] / 30 == F(533, 750)
    print("PASS: separate wear derivations/rounding, PV 11.83561644 SGD/day, fixed contract 113.68 SGD/day")


def check_hand_case():
    def read(name):
        with (HERE / name).open(encoding="utf-8", newline="") as stream:
            return [{key: F(value) for key, value in row.items()} for row in csv.DictReader(stream)]

    rows, expected_rows = read("hand_case.csv"), read("hand_case_expected.csv")
    assert len(rows) == len(expected_rows) == 4
    ev, bess = CFG["ev"], CFG["bess"]
    energy_ev = CFG["hand_case"]["ev_initial_kwh"]
    energy_bess = CFG["hand_case"]["bess_initial_kwh"]
    peak = CFG["hand_case"]["initial_running_peak_kw"]
    totals = {key: F(0) for key in ("purchase", "revenue", "peak_increment", "bess_wear", "ev_wear", "step_cost")}
    target = ev["capacity_kwh"] * ev["departure_target_soc"]
    bess_start = energy_bess
    assert CFG["simulation"]["bess_terminal_rule"] == "end_ge_start"
    for row, expected in zip(rows, expected_rows):
        ev_bounds = feasible_interval(energy_ev, ev, int(row["k"]), target)
        # BESS end-of-day rule: the 4-step fixture is a complete "day", so k is also steps-to-end.
        bess_bounds = feasible_interval(energy_bess, bess, int(row["k"]), bess_start)
        for bounds, power in ((ev_bounds, row["p_ev_kw"]), (bess_bounds, row["p_bess_kw"])):
            assert bounds[0] <= power <= bounds[1]
            assert project(F(-1), bounds) == bounds[0]
            assert project(F(1), bounds) == bounds[1]
        ev_end = energy_next(energy_ev, row["p_ev_kw"], ev["eta_ch"], ev["eta_dis"])
        bess_end = energy_next(energy_bess, row["p_bess_kw"], bess["eta_ch"], bess["eta_dis"])
        got = settlement(row["load_kw"], row["pv_kw"], row["p_bess_kw"],
                         [row["p_ev_kw"]], row["usep_sgd_per_kwh"], peak)
        got.update(t=row["t"], ev_lo=ev_bounds[0], ev_hi=ev_bounds[1],
                   bess_lo=bess_bounds[0], bess_hi=bess_bounds[1],
                   ev_start=energy_ev, ev_end=ev_end,
                   bess_start=energy_bess, bess_end=bess_end)
        assert got == expected, f"Hand-case row {row['t']}: {got} != {expected}"
        assert ev["capacity_kwh"] * ev["soc_min"] <= ev_end <= ev["capacity_kwh"] * ev["soc_max"]
        assert bess["capacity_kwh"] * bess["soc_min"] <= bess_end <= bess["capacity_kwh"] * bess["soc_max"]
        assert ev_end + ev["eta_ch"] * ev["charge_max_kw"] * (row["k"] - 1) * DT >= target
        assert bess_end + bess["eta_ch"] * bess["charge_max_kw"] * (row["k"] - 1) * DT >= bess_start
        for key in totals:
            totals[key] += got[key]
        energy_ev, energy_bess, peak = ev_end, bess_end, got["peak_after"]
    assert energy_ev == target == 26 and energy_bess == bess_start == 60
    assert totals == dict(purchase=F("85.5445"), revenue=F("0.615"), peak_increment=F("8.528"),
                          bess_wear=F("2.50844625"), ev_wear=F("0.135633"), step_cost=F("96.10157925"))
    assert totals["peak_increment"] == F(533, 750) * max(0, peak - 240)
    print("PASS: all 4 hand-case rows, including feasible intervals, energy, import/export, peak and both wear terms")
    print("HAND CASE: total=96.10157925 SGD; EV=26 kWh; BESS=60 kWh (end-of-day rule met); peak=252 kW")


def check_boundaries():
    ev, bess = CFG["ev"], CFG["bess"]
    upper_ev = feasible_interval(F("35.5"), ev, 8, F(26))
    assert upper_ev == (F(-6), F(10, 9))
    assert energy_next(F("35.5"), project(F(1), upper_ev), F("0.9"), F("0.9")) == 36
    upper_bess = feasible_interval(F("107.5"), bess)
    assert upper_bess == (F(-60), F(20, 19))
    assert energy_next(F("107.5"), project(F(1), upper_bess), F("0.95"), F("0.95")) == 108
    assert feasible_interval(F(36), ev, 8, F(26))[1] == 0
    assert feasible_interval(F(12), bess)[0] == 0
    print("PASS: upper-SoC projection reaches exactly 36/108 kWh; charging at ceiling and discharge at floor blocked")

    last = feasible_interval(F(25), ev, 1, F(26))
    assert last == (F(20, 9), F(6))
    assert energy_next(F(25), project(F(-1), last), F("0.9"), F("0.9")) == 26
    assert feasible_interval(F("23.3"), ev, 1, F(26)) == (F(6), F(6))
    try:
        feasible_interval(F(22), ev, 1, F(26))
    except ValueError as error:
        assert "infeasible_input" in str(error)
    else:
        raise AssertionError("An impossible last-step target was silently accepted")
    assert feasible_interval(None, ev, present=False) == (F(0), F(0))
    print("PASS: imminent departure forces charge; tight interval [6,6]; unreachable input rejected; empty-slot power zero")

    end_day = feasible_interval(F(41), bess, 1, F(60))
    assert end_day == (F(40), F(60))
    assert energy_next(F(41), project(F(-1), end_day), F("0.95"), F("0.95")) == 60
    assert feasible_interval(F("49.5"), bess, 2, F(60)) == (F("-34.2"), F(60))
    try:
        feasible_interval(F(30), bess, 1, F(60))
    except ValueError as error:
        assert "infeasible_input" in str(error)
    else:
        raise AssertionError("An unreachable BESS end-of-day target was silently accepted")
    print("PASS: BESS end-of-day rule forces 40 kW at k=1 from 41 kWh; unreachable end-of-day target rejected")

    first = settlement(F(270), F(0), F(0), [], F("0.2"), F(250))
    assert first["peak_after"] == 270 and first["peak_increment"] == F(1066, 75)
    lower = settlement(F(265), F(0), F(0), [], F("0.2"), first["peak_after"])
    export = settlement(F(0), F(10), F(0), [], F("0.2"), lower["peak_after"])
    assert lower["peak_increment"] == export["peak_increment"] == 0
    assert export["peak_after"] == 270
    print("PASS: a new 270 kW peak adds 1066/75 SGD; repeated/lower peaks and export add zero")

    # USEP can be negative: a negative 'sale revenue' is a real export charge.
    negative_export = settlement(F(0), F(10), F(0), [], F("-0.10"), F(0))
    negative_import = settlement(F(10), F(0), F(0), [], F("-0.10"), F(0))
    assert negative_export["revenue"] == F("-0.5") and negative_export["step_cost"] == F("0.5")
    assert negative_import["purchase"] == negative_import["step_cost"] == F("-0.35")
    print("PASS: negative USEP signs retained for both import and export")


def check_scale_and_data():
    days = make_days()
    from build_timeseries import load_params
    cfg_float = load_params()
    expected_manifest = make_manifest(days, cfg_float)
    recorded = json.loads((HERE / "data_manifest.json").read_text(encoding="utf-8"))
    assert recorded == expected_manifest
    files = sorted((HERE / "timeseries").glob("*.csv"))
    assert len(files) == len(days) == 28
    assert {file.stem for file in files} == set(days)
    raw = {}
    with (HERE / recorded["source_file"]).open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            d = datetime.strptime(row["DATE"], "%d-%b-%Y").date().isoformat()
            raw[(d, int(row["PERIOD"]) - 1)] = F(row["USEP ($/MWh)"])
    all_rows, noon_rows = [], []
    for file in files:
        assert file.read_text(encoding="utf-8") == csv_text(days[file.stem]), file.name
        with file.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            assert reader.fieldnames == FIELDS
            rows = list(reader)
        assert len(rows) == 48
        first_stamp = datetime.fromisoformat(rows[0]["start_local"])
        assert first_stamp.date() == date.fromisoformat(file.stem) and first_stamp.hour == 0
        for t, row in enumerate(rows):
            stamp = datetime.fromisoformat(row["start_local"])
            assert int(row["t"]) == t and stamp == first_stamp + timedelta(minutes=30*t)
            assert stamp.utcoffset() == timedelta(hours=8)
            assert F(row["usep_sgd_per_kwh"]) == raw[(file.stem, t)] / 1000
            assert 0 <= F(row["load_kw"]) <= 300
            assert 0 <= F(row["pv_kw"]) <= 48
            all_rows.append(row)
            if 11 <= stamp.hour < 14:
                noon_rows.append(row)
    assert len(all_rows) == len(raw) == 1344
    b, ev, bess = CFG["building"], CFG["ev"], CFG["bess"]
    max_pv = max(F(row["pv_kw"]) for row in all_rows)
    min_noon_load = min(F(row["load_kw"]) for row in noon_rows)
    assert max_pv == 48 < min_noon_load
    assert all(F(row["pv_kw"]) < F(row["load_kw"]) for row in noon_rows)
    assert max(F(row["load_kw"]) for row in all_rows) == b["peak_load_kw"]
    gap = b["peak_load_kw"] - b["contract_capacity_kw"]
    assert gap == bess["discharge_max_kw"] == 60
    for n in CFG["simulation"]["n_slots_cases"]:
        assert F(1, 2) <= n * ev["discharge_max_kw"] / bess["discharge_max_kw"] <= 1
    assert max(F(row["load_kw"]) - F(row["pv_kw"]) for row in all_rows) > b["contract_capacity_kw"]
    # Analytic guarantee covers every allowed parking/initial-SoC draw, not just sampled seeds.
    shortest_stay = ev["departure_min_h"] - ev["arrival_max_h"]
    available = ev["eta_ch"] * ev["charge_max_kw"] * shortest_stay
    biggest_gap = ev["capacity_kwh"] * (ev["departure_target_soc"] - ev["initial_soc_min"])
    assert shortest_stay == 6 and available == F("32.4") > biggest_gap == 14
    for field, high in (("usep", F("0.13")), ("load", F("0.075")), ("pv", F("0.16"))):
        noise = CFG["forecast"]
        sigmas = [noise[f"{field}_sigma_at_h1"] + noise[f"{field}_sigma_increment"] * (h - 1)
                  for h in range(1, 13)]
        assert sigmas[-1] == high and all(x < y for x, y in zip(sigmas, sigmas[1:]))
    assert [len(recorded["splits"][key]) for key in ("train", "validation", "test")] == [14, 7, 7]
    print("PASS: 28 CSVs / 1344 source-matched prices, timestamps, exact offline rebuild and declared provenance")
    print(f"PASS: scale PV peak=48 < noon load min={float(min_noon_load):.6f} kW; BESS/gap=60/60; EV/BESS=0.5–1")
    print("PASS: all permitted arrival/SoC draws are charge-feasible; prediction SD increases over 12 leads; 14/7/7 split")


def main():
    check_metadata()
    check_economics()
    check_hand_case()
    check_boundaries()
    check_scale_and_data()
    print("ALL CHECKS PASSED — D reference/data only; A environment, SAC and C MILP remain integration work.")


if __name__ == "__main__":
    main()
