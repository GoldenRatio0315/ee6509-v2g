"""Rebuild the delivered days offline: real EMC USEP + assumed office/PV proxies.

Python 3.9+, standard library only. No download, interpolation or price repair.
The source CSV is kept unchanged in sources/.
"""
import csv
import io
import json
import random
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "sources" / "emc_usep_2026-10-01_2026-10-28.csv"
FIELDS = ["t", "start_local", "usep_sgd_per_kwh", "load_kw", "pv_kw"]


def values(node):
    """Unwrap annotated parameter leaves; retain lists and ordinary dictionaries."""
    if isinstance(node, dict):
        if {"value", "unit", "source", "status", "note"} <= node.keys():
            return node["value"]
        return {key: values(value) for key, value in node.items()}
    return node


def load_params():
    # params.yaml uses the JSON subset of YAML 1.2, so PyYAML is not required.
    return values(json.loads((HERE / "params.yaml").read_text(encoding="utf-8")))


def lerp(x, knots, levels):
    for left in range(len(knots) - 1):
        if knots[left] <= x <= knots[left + 1]:
            weight = (x - knots[left]) / (knots[left + 1] - knots[left])
            return levels[left] + weight * (levels[left + 1] - levels[left])
    raise ValueError("Hour outside the daily profile")


def decimal_text(value, places=None):
    number = Decimal(str(value))
    if places is not None:
        number = number.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)
    return format(number, "f")


def read_source(cfg):
    first = date.fromisoformat(cfg["data"]["first_date"])
    last = date.fromisoformat(cfg["data"]["last_date"])
    block = {}
    with SOURCE.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            assert row["INFORMATION TYPE"] == "USEP"
            day = datetime.strptime(row["DATE"], "%d-%b-%Y").date()
            period = int(row["PERIOD"])
            assert first <= day <= last and 1 <= period <= 48
            key = (day, period)
            assert key not in block, f"Duplicate EMC key: {key}"
            for col in ("USEP ($/MWh)", "SOLAR(MW)"):
                assert Decimal(row[col]).is_finite()
            block[key] = row
    expected = {(first + timedelta(days=d), t + 1)
                for d in range((last - first).days + 1) for t in range(48)}
    assert set(block) == expected, "Missing or extra source half-hours; no filling allowed"
    assert max(Decimal(row["SOLAR(MW)"]) for row in block.values()) == Decimal(
        str(cfg["pv"]["emc_solar_reference_mw"]))
    return block


def make_days(cfg=None):
    cfg = load_params() if cfg is None else cfg
    block = read_source(cfg)
    first = date.fromisoformat(cfg["data"]["first_date"])
    last = date.fromisoformat(cfg["data"]["last_date"])
    tz = timezone(timedelta(hours=8))  # Asia/Singapore, fixed UTC+08:00.
    days = {}
    for day_offset in range((last - first).days + 1):
        day = first + timedelta(days=day_offset)
        rng = random.Random(cfg["data"]["timeseries_seed"] + day_offset)
        b = cfg["building"]
        daily = rng.uniform(-b["daily_load_fluctuation"], b["daily_load_fluctuation"])
        weekend = b["weekend_load_multiplier"] if day.weekday() >= 5 else 1.0
        rows = []
        for t in range(cfg["simulation"]["steps_per_day"]):
            dt = cfg["simulation"]["delta_t_h"]
            midpoint = (t + 0.5) * dt
            shape = lerp(midpoint, b["load_profile_hours"], b["load_profile_fractions"])
            step_noise = rng.uniform(-b["step_load_fluctuation"], b["step_load_fluctuation"])
            load = min(b["peak_load_kw"], max(0, b["peak_load_kw"] * shape * weekend
                                            * (1 + daily + step_noise)))
            source = block[(day, t + 1)]
            usep = Decimal(source["USEP ($/MWh)"]) / cfg["tariff"]["usep_divisor"]
            pv = (Decimal(str(cfg["pv"]["installed_kwp"]))
                  * Decimal(str(cfg["pv"]["ac_peak_fraction"]))
                  * Decimal(source["SOLAR(MW)"])
                  / Decimal(str(cfg["pv"]["emc_solar_reference_mw"])))
            assert 0 <= pv <= cfg["pv"]["installed_kwp"]
            start = datetime.combine(day, datetime.min.time(), tz) + timedelta(hours=t * dt)
            places = cfg["data"]["power_decimal_places"]
            rows.append({"t": str(t), "start_local": start.isoformat(timespec="minutes"),
                         "usep_sgd_per_kwh": decimal_text(usep),
                         "load_kw": decimal_text(load, places), "pv_kw": decimal_text(pv, places)})
        days[day.isoformat()] = rows
    return days


def csv_text(rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def make_manifest(days, cfg):
    all_rows = [row for rows in days.values() for row in rows]
    noon = [row for row in all_rows if 11 <= datetime.fromisoformat(row["start_local"]).hour < 14]
    weekdays_noon = [row for row in noon if datetime.fromisoformat(row["start_local"]).weekday() < 5]
    prices = [Decimal(row["usep_sgd_per_kwh"]) for row in all_rows]
    pv_energy = sum(Decimal(row["pv_kw"]) for row in all_rows) * Decimal("0.5")
    split = {"train": [], "validation": [], "test": []}
    for day in days:
        label = ("train" if day <= cfg["data"]["train_last_date"] else
                 "validation" if day <= cfg["data"]["validation_last_date"] else "test")
        split[label].append(day)
    return {
        "version": "0.2", "source_id": "S01", "downloaded_date_local": "2026-11-10",
        "source_file": str(SOURCE.relative_to(HERE)), "csv_columns": FIELDS,
        "days": len(days), "rows_per_day": 48, "total_rows": len(all_rows),
        "provenance": {"usep_sgd_per_kwh": "Actual publicly downloaded final EMC half-hour USEP, divided by 1000",
                       "load_kw": "Synthetic office shape and bounded noise, A07; not measured building load",
                       "pv_kw": "National EMC SOLAR forecast shape scaled to an assumed rooftop; S01/S10/A06; not measured rooftop generation"},
        "processing": {"prices_filled_or_interpolated": False, "price_spikes_clipped": False,
                       "solar_series_kind": "historical PSO forecast", "load_rng": "Python random.Random(20261110 + day_offset)",
                       "pv_scale_kw_per_mw": float(Decimal("48") / Decimal("1788.87")),
                       "pv_scale_note": "Fixed whole-block scenario calibration, not a fitted predictor. Frozen for every split."},
        "summary": {"usep_min_sgd_per_kwh": float(min(prices)),
                    "usep_max_sgd_per_kwh": float(max(prices)),
                    "usep_mean_sgd_per_kwh": float(sum(prices) / len(prices)),
                    "negative_price_rows": sum(price < 0 for price in prices),
                    "load_max_kw": max(float(row["load_kw"]) for row in all_rows),
                    "pv_max_kw": max(float(row["pv_kw"]) for row in all_rows),
                    "noon_load_min_all_days_kw": min(float(row["load_kw"]) for row in noon),
                    "noon_load_min_weekdays_kw": min(float(row["load_kw"]) for row in weekdays_noon),
                    "pv_daily_mean_kwh": float(pv_energy / len(days)),
                    "no_storage_net_peak_kw": max(float(row["load_kw"]) - float(row["pv_kw"]) for row in all_rows)},
        "splits": split,
        "quality_note": "No source rows were filled. Real price spikes are retained. Other columns are explicitly documented model inputs; these are not co-measured office/PV observations.",
    }


def main():
    cfg = load_params()
    days = make_days(cfg)
    target = HERE / "timeseries"
    target.mkdir(exist_ok=True)
    for day, rows in days.items():
        (target / f"{day}.csv").write_text(csv_text(rows), encoding="utf-8")
    manifest = make_manifest(days, cfg)
    (HERE / "data_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False)
                                           + "\n", encoding="utf-8")
    print(f"Wrote {len(days)} days / {manifest['total_rows']} rows; no price filling or clipping.")
    print(json.dumps(manifest["summary"], indent=2))


if __name__ == "__main__":
    main()
