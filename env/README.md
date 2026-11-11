# V2G 环境接口（A）

物理和费用依据 `data/D/D_model_and_parameters.md` v0.3、`params.yaml` 和技术方案第 3、5 节。30 分钟一步，正式日为 48 步；AC 侧正功率充电，负功率放电。240 kW 是计费阈值，没有额外并网硬上限。内部 float64，观测 float32；环境不归一化、不缩放 reward。

## 调用与复现

`V2GEnv(n_slots=5, params=None, split="train", timeseries_dir=...)`；请使用关键词参数。`params` 为 `load_params()` 返回的去注释值字典。N 是正整数，正式实验使用 5、10，手算测试可用 1。

本接口按 D v0.3 的定稿模型实现，正式实验的 `params` 应使用 D v0.3 定稿值，环境只读取其中一部分字段。N 只由构造参数 `n_slots` 决定，修改 `params["simulation"]["n_slots"]` 不会改变槽位数。`simulation.bess_terminal_rule="end_ge_start"`、`simulation.reverse_export_allowed=True`、`tariff.gst_multiplier=1` 是固定规则，不是当前环境支持的模型切换开关；构造函数目前不会拒绝这些字段的非定稿值，但修改它们不会切换对应行为。负荷生成、PV 缩放和 `timeseries_seed` 等生成 CSV 的参数也不会重新生成真值；`pv.installed_kwp` 只影响观测中 PV 预测的裁剪上限。非定稿参数的实验须另行验证，不属于当前验收范围。日期统一传入 `YYYY-MM-DD`；不依赖不同 Python 版本对其他 ISO 日期写法的宽松解析。

电价、负荷、PV 真值从 D 的固定日 CSV 读取，seed 只控制车辆和预测噪声。技术方案 §5、§9.1 的 `V2GEnv(params)`、`reset(seed=s)` 和“按种子生成电价/负荷/光伏”等流程示意落实到代码时，以根 README 和本节的具体接口为准。

- `reset(seed=100, options={"date": "2026-10-22"})`：直接按 seed、日期及流编号派生场景，与之前调用历史无关。显式日期可来自任一 split。
- `reset(seed=0)` 或 `reset()`：从构造时指定 split 的日期抽样，并从对应种子池抽场景种子；不传 seed 时沿用 RNG，不重新播种。显式日期但省略 seed 时也抽取场景种子。实际日期、场景种子和参数标识写入 info。默认池：train/validation 为 0–99，test 为 100–119；显式 seed+date 直接采用传入 seed。
- `reset(options={"scenario": scenario})`：注入只读 Scenario；可同时传 seed，仅重置环境 RNG，不改变注入数据。scenario 与 date 互斥。

train 为 10-01 至 10-14；validation 为 10-15 至 10-21；test 为 10-22 至 10-28（2026）。验证种子池沿用训练池 0–99，验证日期独立；测试日期和种子均留出。B 的标准化仅在训练数据拟合。测试日期/种子应显式指定。

完整测试集需逐一枚举 7 天 × 20 个种子，并同时显式传入 seed 和 date；此时 split 不影响场景。下面只生成 N=5 的 140 个场景，各策略随后注入同一场景并运行至终止；N=10 遍历同一组 (date, seed)。单独循环 `reset(seed=s)` 会抽样日期和场景种子，不能代替完整测试网格。只给 date 也会抽样场景种子，种子池由构造时的 split 决定，不按该日期所属集合切换。

```python
from env import V2GEnv, load_params

sim = V2GEnv(n_slots=5, params=load_params(), split="test")
test_scenarios = []
for day in range(22, 29):
    date = f"2026-10-{day:02d}"
    for seed in range(100, 120):
        obs, meta = sim.reset(seed=seed, options={"date": date})
        test_scenarios.append(sim.export_scenario())
```

`step(action)` 接收 `(N+1,)` 有限向量，顺序 `[BESS, EV_1, …, EV_N]`，裁至 [-1,1] 后仿射映射到当前可行区间，并将浮点舍入造成的区间外偏差（≤1e-9 kW）夹回区间、记录 power 修正。`feasible_intervals()` 返回 `(N+1,2)` 的 `[lo,hi]`，包含终端可达性约束，空槽 `[0,0]`。

`step_power(power, tol=1e-6)` 接收相同顺序的实际 kW，拒绝超出区间超过 tol 的输入，仅裁去容差内误差。两种 step 共用结算；形状错误、NaN、Inf 均报错；尚未 reset 或已经终止时拒绝 step。返回 `(obs, reward, terminated, truncated, info)`。

## 观测

| 键 | shape / dtype | 含义与单位 |
|---|---|---|
| timeseries | (3,20) / float32 | 行顺序 USEP (S$/kWh)、load (kW)、PV (kW)；列为含当前的 8 步历史，再接 h=1…12 预测 |
| global | (4,) / float32 | BESS SoC、当前已结算的进口峰值 kW、sin(2πt/48)、cos(2πt/48) |
| ev | (N,4) / float32 | 有车标记、SoC、距离场剩余步数（含当前）、max(0,目标电量−当前电量) kWh；空槽全零 |
| ev_present | (N,) / int8 | 1 有车、0 空槽，Gym MultiBinary |

B 使用 `key_padding_mask = ev_present < 0.5`；全空车队须单独处理 attention，避免全部屏蔽产生 NaN。EV 行按物理槽位编号，不按 SoC 重排。SoC 均为 0–1 比例。timeseries 的 USEP 上下界为 ±inf，load/PV 下界 0、上界 inf；global 界为 [0,0,-1,-1] 至 [1,inf,1,1]；ev 界为 [0,0,0,0] 至 [1,1,48,40]（末项按容量参数设置）。返回数组不与内部状态共享可写内存。

历史日初补当天首值；预测越过日末补当天末值，绝不读取另一日期。预测仅在观测中加入固定场景噪声，结算始终使用 CSV 真值。price/load/PV 的相对噪声标准差为 `.02+.01(h−1)`、`.02+.005(h−1)`、`.05+.01(h−1)`；价格保留负值，load 裁至非负，PV 裁至 [0,60]。同场景同 t 的预测不受动作或观测读取次数影响。

正式日 t=48 返回合法终止观测：外生索引夹到 [0,T−1]，保留真实 BESS 末电量和峰值，所有车清槽；`terminated=True, truncated=False`，不返回下日初始状态，也不另外套 TimeLimit。短 Scenario 同样以自身 T 终止，时钟仍用实际半小时刻度。

## 状态顺序与费用

到达 a 的车参与 a 步动作，占用区间 a≤t<d。每步先计算区间、更新能量并计费，再检查 d=t+1 的离场目标、记录结果、清槽，最后加入下一步到达车辆并构造观测。BESS 终端目标为场景初始电量。不可达输入明确报 `infeasible_input`，不能降低目标或扩大功率。

`reward = -step_cost_sgd`；`step_cost = purchase - revenue + peak_increment + bess_wear + ev_wear`。购价 USEP+.03，售价 USEP；负价格时 purchase/revenue 均允许为负。容量费只收峰值增量，磨损按 AC 吞吐量；不逐步舍入。

末步 `info["episode_summary"]` 包含可变成本、固定 PV 费、固定合同费、总成本、峰值、BESS 末电量和离场记录/达标率。仅 T=48 的完整日加入 PV 11.83561644、合同 113.68 S$ 各一次；短 fixture 两项均为 0。无车日 `departure_success_rate=None`。`peak_grid_kw` 是计费参考峰值，即初始运行峰值与当日最大进口功率的较大者；正式参数初始峰值为 0。

## info 与 CSV

reset info：`date, scenario_seed, params_id, t`。step info 另含：`p_grid_kw, p_bess_kw, p_ev_kw (N), intervals (N+1,2), e_bess_before_kwh, e_ev_before_kwh (N), e_bess_kwh, e_ev_kwh (N), buy_price, sell_price, purchase_sgd, revenue_sgd, peak_before_kw, peak_after_kw, peak_increment_sgd, bess_wear_sgd, ev_wear_sgd, step_cost_sgd, departures, corrections`。info.t 是刚结算的步编号。末能量记录在清槽之前，因此离场车辆的最终能量仍可读取；next_obs 中对应槽位才变为空。空槽在 info 和 CSV 中的功率及电量均记 0（占位，不代表在场车辆电池耗尽）；step info.ev_present 记录本步动作前占用状态，辨别空槽应使用该字段或共享 Scenario 的 a≤t<d，不应单凭 SoC 推断。

`departures` 为含 `slot`（0 起）、`departure_t, energy_kwh, target_kwh, success` 的记录列表。`corrections` 为记录列表，每项 `{device, kind, delta, unit}`；device 为 0=BESS、1…N=EV，kind 为 interval_lo/power/energy，delta 为修正后减修正前，unit 为 kW 或 kWh，仅非零修正入表。power 既可表示 step 的仿射舍入修正（≤1e-9 kW），也可表示 step_power 的入参容差修正（默认≤1e-6 kW）。回放已修正功率不会再次产生同一舍入修正，因此 step 与 step_power 对账比较电量和费用，不要求 corrections 列表相同。

CSV 由 `env.recording` 的单一列定义和 `info_to_row(info, strategy=...)` 生成，供 C 复用。严格保留根 README 列顺序：`strategy,date,seed,t,p_grid_kw,p_bess_kw`，然后所有 `p_ev_{i}_kw`、`e_bess_kwh`、所有 `e_ev_{i}_kwh`（i 从 1 起），最后 `buy_price,sell_price,purchase_sgd,revenue_sgd,peak_increment_sgd,bess_wear_sgd,ev_wear_sgd,step_cost_sgd`。CSV 的 seed 为 info.scenario_seed；params_id、峰值诊断等留在 info 和日汇总，不改变既有 CSV。画 SoC 曲线和分析时必须按 Scenario 占用区间屏蔽空槽，`occupancy_mask()` 可生成该掩码。CSV 电量均为本步末、离场清槽前；固定费不混入逐步 CSV。

## Scenario 与数值约定

Scenario 包含 `date, seed, params_id, truth (3,T), vehicles (每槽 None 或 arrival/departure/initial_kwh/target_kwh), bess_initial_kwh, initial_peak_kw, noise (3,T+1,12)`；arrival/departure 是整数步编号，电量单位 kWh；T 从 truth 长度取得且限制 1…48。字段冻结且数组数值只读；NumPy 的 shape/dtype 等元数据仍可由持有者修改。注入时环境重新校验并持有独立场景快照，因此调用方随后修改数组元数据不会影响正在运行的环境。`export_scenario()` 直接导出当前同一场景的 JSON 兼容副本，不重新抽样；参数标识为`json.dumps(params,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode("utf-8")` 的 SHA-256，注入时须匹配。

每个槽位分别由 SeedSequence([seed,date_ordinal,stream_id,slot]) 派生，所以 N=10 前五槽与 N=5 相同；预测随机流独立于车辆。工作日每槽一车，周末无车；到离场用连续截断正态拒绝采样后 `floor(hour/.5+.5)` 量化。到达 SoC 为 Uniform(.3,.7)。数据中的 load 是合成曲线、PV 是全国预测的缩放代理，不是楼宇实测；两者不随车辆 seed 重生成。

功率区间容差 1e-9 kW，电量与终端目标容差 1e-9 kWh；仅容差内反转区间可折叠到 hi（物理上限一侧，lo=hi），仅容差内电量越界可贴回边界，并记录修正。MILP 输入功率容差单独为 1e-6 kW。手算逐列对照绝对误差 ≤1e-9，日累计账目 ≤1e-8。物理约束、seed、shape 和有限性在入口验证。场景和记录中的物理数据只接受实数，不接受复数、布尔或字符串数组；CSV 的 seed/t 必须是非布尔非负整数，不能通过类型转换截断。

运行依赖使用 `env/requirements.txt`；Python 3.9–3.12。验证范围为物理/环境一致性，SAC 性能和 C 的 MILP 目标值联调另行开展。

## 运行和交接

从仓库根目录运行（建议单独虚拟环境）：

```bash
python -m pip install -r env/requirements.txt
python -m unittest discover -s env/tests -v
python -m env.validate --output results/env_validation.json
python -m env.demo --date 2026-10-01 --seed 0 --n-slots 5 --action-seed 6509
```

`env.demo` 将逐步 CSV、对应的 `.scenario.json` 和 `.summary.json` 一起写入 `results/env_demo/`，可用 `--output-dir` 指定目录。场景 seed 控制车辆/预测，action_seed 仅控制随机动作。示例是环境运行演示，不用于声称调度策略性能。运行产物按根 `.gitignore` 留在本地。

```python
import numpy as np
from env import V2GEnv

sim = V2GEnv(n_slots=5)
obs, meta = sim.reset(seed=100, options={"date": "2026-10-22"})
scenario = sim.export_scenario()  # 所有策略共用并保存此副本
obs, reward, terminated, truncated, info = sim.step(np.zeros(6))
```

零归一化动作是可行区间中点，不一定代表零 kW。C 的策略需先计算实际功率时，可读 `feasible_intervals()` 并使用 `step_power()`。完整优化功率表可复用：

```python
from env.replay import replay_power_schedule
infos, summary = replay_power_schedule(scenario, powers)  # powers.shape == (T, N+1)
variable_cost = summary["variable_cost_sgd"]
```

MILP 应用相同参数、AC 正负号、场景、终端约束及成本口径；其目标若不含固定费，就与 variable_cost 比较。MILP 求解器和目标值由 C 提供，当前环境验证不代表已完成 MILP 最优解对账。根 README 中策略成本的预期排序不作为环境验收条件。

外部功率表应保留 float64 原值或可无损往返的文本精度，不预先截断为固定小数位。前面步骤的舍入会累积为电量偏差，可能在后续贴边步骤超过默认 `tol=1e-6 kW`；报错步骤可能晚于误差产生步骤，应先排查舍入和转存。检查求解器可行性精度，必要时在离场目标、BESS 日末目标和 SoC 边界这些电量约束上保留数值裕量；只在功率上留裕量不能控制累积电量误差。保持组内验收容差，修改验收容差须另行确认。`tol=0` 的逐位回放结论适用于同一场景下环境自身输出的功率，不保证外部独立算法算出的贴边值零误差通过。默认容差内的实际修正仍记入 `corrections`。

B/G 绘图时可调用 `from env.recording import occupancy_mask`，对 CSV 能量列使用 `mask = occupancy_mask(scenario)`；用 `np.where(mask, energy, np.nan)` 只在绘图副本中屏蔽空槽，保留离场前最后一行。不要将 CSV 的零占位绘成车辆 SoC 跌至零。

模块职责：`physics.py` 为物理与费用纯函数；`data.py` 校验数据时间/数值；`scenario.py` 生成或注入可复现场景；`v2g_env.py` 管理状态；`recording.py` 定义共享 CSV 与占用掩码；`replay.py` 回放实际功率；`demo.py` 演示；`validate.py` 运行完整验收矩阵。
