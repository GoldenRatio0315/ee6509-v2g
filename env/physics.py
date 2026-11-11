"""Pure AC-side battery dynamics and step settlement, in kW, kWh and SGD."""

from dataclasses import dataclass, fields
import math
from numbers import Integral, Real
from typing import Mapping

import numpy as np


P_TOL = 1e-9
E_TOL = 1e-9
POWER_INPUT_TOL = 1e-6


def _finite_real(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    try:
        result = float(value)
    except (OverflowError, ValueError):
        raise ValueError(f"{name} must be a finite real number") from None
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result


def _nonnegative(value, name):
    result = _finite_real(value, name)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _time_step(dt):
    result = _finite_real(dt, "dt")
    if result <= 0:
        raise ValueError("dt must be positive")
    return result


@dataclass(frozen=True)
class BatterySpec:
    capacity_kwh: float
    charge_max_kw: float
    discharge_max_kw: float
    soc_min: float
    soc_max: float
    eta_ch: float
    eta_dis: float
    wear_sgd_per_kwh: float

    def __post_init__(self):
        for field in fields(self):
            object.__setattr__(self, field.name,
                               _finite_real(getattr(self, field.name), field.name))
        if self.capacity_kwh <= 0:
            raise ValueError("capacity_kwh must be positive")
        if self.charge_max_kw <= 0 or self.discharge_max_kw <= 0:
            raise ValueError("charge and discharge ratings must be positive")
        if not 0 <= self.soc_min < self.soc_max <= 1:
            raise ValueError("SoC bounds must satisfy 0 <= soc_min < soc_max <= 1")
        if not 0 < self.eta_ch <= 1 or not 0 < self.eta_dis <= 1:
            raise ValueError("efficiencies must lie in (0, 1]")
        if self.wear_sgd_per_kwh < 0:
            raise ValueError("wear_sgd_per_kwh must be nonnegative")

    @classmethod
    def from_params(cls, group: Mapping):
        """Read one EV/BESS group after the parameter ``value`` leaves are unwrapped."""
        if not isinstance(group, Mapping):
            raise ValueError("battery parameters must be a mapping")
        try:
            return cls(**{field.name: group[field.name] for field in fields(cls)})
        except KeyError as error:
            raise ValueError(f"missing battery parameter: {error.args[0]}") from None

    @property
    def energy_min(self):
        return self.capacity_kwh * self.soc_min

    @property
    def energy_max(self):
        return self.capacity_kwh * self.soc_max


def _energy(value, spec, name):
    if not isinstance(spec, BatterySpec):
        raise ValueError("spec must be a BatterySpec")
    result = _finite_real(value, name)
    if not spec.energy_min <= result <= spec.energy_max:
        raise ValueError(f"infeasible_input: {name} outside physical energy bounds")
    return result


def feasible_interval(energy, target, remaining_steps, spec, dt=0.5):
    """Return (lo, hi, lo_correction), retaining enough energy to meet the target.

    ``remaining_steps`` includes this step. Input energy and target must already
    lie within physical bounds. Only a reversed interval within P_TOL is folded
    to its upper endpoint; the signed correction is returned for the caller's log.
    """
    energy = _energy(energy, spec, "energy")
    target = _energy(target, spec, "target")
    dt = _time_step(dt)
    charge_step = spec.eta_ch * dt
    if charge_step == 0:
        raise ValueError("dt and charging efficiency produce an unrepresentable time step")
    if (isinstance(remaining_steps, (bool, np.bool_))
            or not isinstance(remaining_steps, Integral) or remaining_steps < 1):
        raise ValueError("remaining_steps must be a positive integer")
    try:
        future_charge = (spec.eta_ch * spec.charge_max_kw
                         * int(remaining_steps - 1) * dt)
    except OverflowError:
        raise ValueError("remaining_steps is too large") from None
    lower_energy = max(spec.energy_min, target - future_charge)

    def inverse(delta):
        if delta >= 0:
            return delta / charge_step
        return spec.eta_dis * delta / dt

    lo = max(-spec.discharge_max_kw, inverse(lower_energy - energy))
    hi = min(spec.charge_max_kw, inverse(spec.energy_max - energy))
    if not math.isfinite(lo) or not math.isfinite(hi):
        raise ValueError("non-finite feasible interval")
    if lo > hi:
        if lo - hi > P_TOL:
            raise ValueError("infeasible_input: target is unreachable within remaining steps")
        return hi, hi, hi - lo
    return lo, hi, 0.0


def advance_energy(energy, power, spec, dt=0.5):
    """Apply signed AC power and return (next energy, signed energy correction).

    The caller checks terminal reachability and logs any input-power clipping.
    This function checks nameplate power and physical energy limits. Only an
    energy overshoot within E_TOL is snapped back to the violated boundary.
    """
    energy = _energy(energy, spec, "energy")
    power = _finite_real(power, "power")
    dt = _time_step(dt)
    if power > spec.charge_max_kw + P_TOL or power < -spec.discharge_max_kw - P_TOL:
        raise ValueError("infeasible_input: power exceeds the battery rating")
    raw = (energy + spec.eta_ch * max(power, 0.0) * dt
           - max(-power, 0.0) * dt / spec.eta_dis)
    if not math.isfinite(raw):
        raise ValueError("non-finite next energy")
    if raw < spec.energy_min:
        if spec.energy_min - raw > E_TOL:
            raise ValueError("infeasible_input: next energy below physical minimum")
        return spec.energy_min, spec.energy_min - raw
    if raw > spec.energy_max:
        if raw - spec.energy_max > E_TOL:
            raise ValueError("infeasible_input: next energy above physical maximum")
        return spec.energy_max, spec.energy_max - raw
    return raw, 0.0


def settle_cost(*, load_kw, pv_kw, usep, power, bess, ev, peak_before_kw,
                contract_capacity_kw, ucc_daily, adder, dt=0.5):
    """Settle one step; ``power`` is [BESS, EV_1, ...] at the AC bus.

    No state is mutated. Purchase and export revenue retain price signs, wear is
    charged on AC throughput, and capacity cost is charged only on a new excess
    peak. Fixed daily charges are intentionally outside this step settlement.
    """
    if not isinstance(bess, BatterySpec) or not isinstance(ev, BatterySpec):
        raise ValueError("bess and ev must be BatterySpec instances")
    load_kw = _nonnegative(load_kw, "load_kw")
    pv_kw = _nonnegative(pv_kw, "pv_kw")
    usep = _finite_real(usep, "usep")
    peak_before_kw = _nonnegative(peak_before_kw, "peak_before_kw")
    contract_capacity_kw = _nonnegative(contract_capacity_kw, "contract_capacity_kw")
    ucc_daily = _nonnegative(ucc_daily, "ucc_daily")
    adder = _nonnegative(adder, "adder")
    dt = _time_step(dt)
    try:
        raw_power = np.asarray(power)
    except (TypeError, ValueError):
        raise ValueError("power must be a finite one-dimensional numeric vector") from None
    if raw_power.ndim != 1 or raw_power.size < 1 or raw_power.dtype.kind not in "iuf":
        raise ValueError("power must be a finite one-dimensional numeric vector with a BESS entry")
    power = raw_power.astype(np.float64, copy=True)
    if not np.all(np.isfinite(power)):
        raise ValueError("power must contain only finite values")
    with np.errstate(over="ignore", invalid="ignore"):
        ev_total = float(np.sum(power[1:]))
        ev_throughput = float(np.sum(np.abs(power[1:])))
    p_grid = load_kw - pv_kw + float(power[0]) + ev_total
    buy_price = usep + adder
    peak_after = max(peak_before_kw, p_grid)
    purchase = buy_price * max(p_grid, 0.0) * dt
    revenue = usep * max(-p_grid, 0.0) * dt
    peak_increment = ucc_daily * (
        max(0.0, peak_after - contract_capacity_kw)
        - max(0.0, peak_before_kw - contract_capacity_kw))
    bess_wear = bess.wear_sgd_per_kwh * abs(float(power[0])) * dt
    ev_wear = ev.wear_sgd_per_kwh * ev_throughput * dt
    result = {
        "p_grid_kw": p_grid,
        "p_bess_kw": float(power[0]),
        "buy_price": buy_price,
        "sell_price": usep,
        "peak_before_kw": peak_before_kw,
        "peak_after_kw": peak_after,
        "purchase_sgd": purchase,
        "revenue_sgd": revenue,
        "peak_increment_sgd": peak_increment,
        "bess_wear_sgd": bess_wear,
        "ev_wear_sgd": ev_wear,
        "step_cost_sgd": purchase - revenue + peak_increment + bess_wear + ev_wear,
    }
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError("non-finite settlement result")
    result["p_ev_kw"] = power[1:].copy()
    return result
