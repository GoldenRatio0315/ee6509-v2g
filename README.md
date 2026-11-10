# EE6509 · 楼宇 V2G 混合储能调度

用 SAC 强化学习，按 30 min 步长调度楼宇固定储能（BESS）和 5–10 辆 V2G 电动车，使全天运行总成本最低，同时保证每辆车离场时达到目标电量。

**技术方案**（以在线文档为准）：<https://claude.ai/artifact/KeCSxpz4FCjPWHAhVHMVxy>

## 目录与负责人

| 目录 | 负责人 | 内容 |
| --- | --- | --- |
| `data/D/` | D | 参数 `params.yaml`、28 天时序数据 `timeseries/`、模型说明、手算案例、校验脚本 |
| `env/` | A | 仿真环境 `V2GEnv`（接口骨架已给出，见 `env/v2g_env.py`） |
| `sac/` | B | SAC 训练代码和模型 |
| `baselines/` | C | 两种普通方案：到场就充、低价充高价放 |
| `milp/` | C | MILP 离线最优参考方案 |
| `analysis/` | C、G | 统一运行入口、绘图、结果分析 |
| `results/` | 所有人 | 运行输出（不提交到仓库） |

E 的背景资料和 F 的测试记录先放在各自的共享文档里，需要进仓库时放 `docs/`。

## 快速开始

```bash
pip install -r requirements.txt
python data/D/check_model.py      # D 的参数与手算校验，应输出 ALL CHECKS PASSED
```

## 接口约定

各人分头写的部分按下面的约定交接，最后才能直接拼在一起。

**参数和数据（D → 所有人）**

- `data/D/params.yaml`：JSON 子集写成的 YAML，用 `json.load` 就能读；每个参数取 `value` 字段。
- `data/D/timeseries/YYYY-MM-DD.csv`：每天 48 行，列为 `t, start_local, usep_sgd_per_kwh, load_kw, pv_kw`。电价已经是 S$/kWh。
- 日期划分：10-01 至 10-14 训练，10-15 至 10-21 验证，10-22 至 10-28 测试。车辆计划和预测噪声的随机种子：训练 0–99，测试 100–119。

**环境（A → B、C、F）**，Gymnasium 接口：

```python
env = V2GEnv(n_slots=5)
obs, info = env.reset(seed=s, options={"date": "2026-10-22"})
obs, r, terminated, truncated, info = env.step(a)          # a ∈ [-1, 1]^(N+1)，给 SAC
obs, r, terminated, truncated, info = env.step_power(p)    # p 为各设备实际功率 kW，给 MILP 回放
```

- 动作顺序：`[BESS, 槽位 1, …, 槽位 N]`；功率充电为正、放电为负。
- `info` 里带本步各设备实际功率、可行区间、电量、成本分项和 EV 离场达标情况。
- 同一 `seed` 和日期下，车辆计划和预测噪声必须完全相同，所有策略共用。

**运行结果（A、B、C → F、G）**：每次运行输出一个 CSV，每行一步，列为

```
strategy, date, seed, t, p_grid_kw, p_bess_kw, p_ev_1_kw … p_ev_N_kw,
e_bess_kwh, e_ev_1_kwh … e_ev_N_kwh, buy_price, sell_price,
purchase_sgd, revenue_sgd, peak_increment_sgd, bess_wear_sgd, ev_wear_sgd, step_cost_sgd
```

**两条硬约束**：EV 离场约束和 BESS 日末约束（日末电量不低于日初）都写进动作投影，见技术方案 3.5 节和 `data/D/D_model_and_parameters.md` 第 4.2 节。

## 交付与验收

| 人员 | 交付 | 验收标准 |
| --- | --- | --- |
| A | 环境代码、使用说明、随机动作跑 2026-10-01 一天的结果 CSV | 回放 `hand_case.csv`（4 步当作一个完整日）与 `hand_case_expected.csv` 逐列一致：总成本 96.10157925 S$、峰值 252 kW、车末电量 26 kWh、储能末电量 60 kWh |
| C | 两种普通方案、MILP、统一运行入口、对比图 | 每个测试日每个种子满足 MILP ≤ 低价充高价放 ≤ 到场就充；MILP 功率放进 A 的环境逐步结算，与 MILP 目标值相差不超过 0.01 S$ |
| B | SAC 代码、训练好的模型、训练曲线 | 目标：测试日成本落在 MILP 与规则策略之间 |

## 协作方式

- 每人只改自己负责的目录；需要改别人的代码或 `data/` 时，先在群里说。
- 新功能开分支，写完提 Pull Request 合并到 `main`。
- 改动定稿决策（技术方案第 2 节）前，先在在线文档里评论。
