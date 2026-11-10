"""V2G 楼宇混合储能仿真环境 —— 接口骨架。

负责人：A。本文件给出所有人共用的接口（类名、方法签名、info 字段），
具体实现由 A 按技术方案第 3、5 节和 data/D/D_model_and_parameters.md
第 4.2、8 节完成。B、C 可以先按这里的签名写代码。

已实现：参数与日数据读取（load_params / load_day）。
待实现：标注了 NotImplementedError 的方法。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import gymnasium as gym
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "D"


def load_params(path: Path = DATA_DIR / "params.yaml") -> dict:
    """读取 D 的参数文件，返回 {分组: {参数名: value}}，去掉单位、来源等注释字段。"""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    params = {}
    for group, items in raw.items():
        if isinstance(items, dict):
            params[group] = {k: v["value"] for k, v in items.items() if isinstance(v, dict) and "value" in v}
    return params


def load_day(date: str, timeseries_dir: Path = DATA_DIR / "timeseries") -> dict:
    """读取一天的时序数据，返回 48 长度的 numpy 数组：usep（S$/kWh）、load（kW）、pv（kW）。"""
    with open(Path(timeseries_dir) / f"{date}.csv", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 48, f"{date} 应为 48 行，实际 {len(rows)} 行"
    return {
        "usep": np.array([float(r["usep_sgd_per_kwh"]) for r in rows]),
        "load": np.array([float(r["load_kw"]) for r in rows]),
        "pv": np.array([float(r["pv_kw"]) for r in rows]),
    }


class V2GEnv(gym.Env):
    """一天 = 48 步，每步 30 min。

    动作顺序：[BESS, 槽位 1, …, 槽位 N]；功率以楼宇交流侧计，充电为正、放电为负。
    """

    def __init__(self, n_slots: int = 5, params: dict | None = None):
        self.params = params or load_params()
        self.n_slots = n_slots
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(n_slots + 1,), dtype=np.float32)
        # 观测空间由 A 按技术方案 3.1 节确定：外部时序 3×20、全局状态 4、EV 槽位 4×N
        self.observation_space = None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """options["date"]：数据日期，如 "2026-10-22"。

        同一 (seed, date) 必须生成完全相同的车辆计划和预测噪声。
        BESS 每天按 params 的 initial_soc 重置，当日峰值 P̂ 置 0。
        返回 (obs, info)。
        """
        super().reset(seed=seed)
        raise NotImplementedError("A 实现")

    def feasible_intervals(self) -> np.ndarray:
        """返回本步各设备可行功率区间，形状 (N+1, 2)，列为 [P_lo, P_hi]（kW）。

        包括额定功率、SoC 上下限、EV 离场约束、BESS 日末约束（日末电量不低于日初）；空槽为 [0, 0]。
        区间为空时抛出 ValueError("infeasible_input")。
        """
        raise NotImplementedError("A 实现")

    def step(self, action: np.ndarray):
        """SAC 使用：action ∈ [-1, 1]^(N+1)，按 P = P_lo + (a+1)/2·(P_hi−P_lo) 映射为功率后结算。"""
        intervals = self.feasible_intervals()
        a = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        power = intervals[:, 0] + (a + 1.0) / 2.0 * (intervals[:, 1] - intervals[:, 0])
        return self._settle(power)

    def step_power(self, power: np.ndarray, tol: float = 1e-6):
        """MILP 回放使用：直接给各设备实际功率（kW），先检查是否在可行区间内再结算。"""
        intervals = self.feasible_intervals()
        p = np.asarray(power, dtype=float)
        if np.any(p < intervals[:, 0] - tol) or np.any(p > intervals[:, 1] + tol):
            raise ValueError(f"功率超出可行区间：power={p}, intervals={intervals.tolist()}")
        return self._settle(np.clip(p, intervals[:, 0], intervals[:, 1]))

    def _settle(self, power: np.ndarray):
        """按技术方案 3.3–3.4 节更新电量、计算 P_grid 和单步成本，返回 (obs, r, terminated, truncated, info)。

        info 至少包含：t, p_grid_kw, p_bess_kw, p_ev_kw(长度 N), e_bess_kwh, e_ev_kwh(长度 N),
        intervals, buy_price, sell_price, purchase_sgd, revenue_sgd, peak_increment_sgd,
        bess_wear_sgd, ev_wear_sgd, step_cost_sgd，以及本步离场车辆是否达标。
        """
        raise NotImplementedError("A 实现")
