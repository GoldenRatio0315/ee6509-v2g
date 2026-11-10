# milp（C）

MILP 离线最优参考方案。每天单独求解，用当天真实序列和同一份车辆计划；建模口径见 `data/D/D_model_and_parameters.md` 第 8 节“给 C”（已含 BESS 日末约束）。求得功率后用 `env.step_power` 回放核账。
