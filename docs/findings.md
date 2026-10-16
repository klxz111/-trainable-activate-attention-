# 冒烟判读（2026-10-17）

三臂各 2000 步（seq 512、4 层 256 维、micro-batch 16 × accum 2、eval 每 250 步，数据 data/ts200m）。
所有数字取自 `runs/smoke_*/metrics.jsonl`（每臂 8 个 eval 点），终值 = step 2000 行；吞吐为该行 tok_per_s。

| 臂 | 2k 步 val loss | min_frac（逐层，step 2000） | tok/s | 判定 |
|---|---|---|---|---|
| dense | 2.5896 | — | 73,262 | 基线 |
| aa4 | 2.6054 | 0.1728 / 0.1804 / 0.0912 / 0.0882 | 44,448（dense 的 60.7%） | 判据 1/2/4 过，判据 3 压线过（ruling 后实质 PASS） |
| moa4 | 2.6123 | 0.1134 / 0.0923 / 0.0868 / 0.1054 | 28,847（dense 的 39.4%） | 判据 3 FAIL（ruling 后修正为 PASS，见下节） |

## 四条判据逐条判读

1. **三臂 2000 步 val loss 均下降且无 NaN — PASS**
   dense 3.7367→2.5896、aa4 3.7136→2.6054、moa4 3.7261→2.6123，三臂 8 个 eval 点逐点单调下降，全程无 NaN/Inf，无 OOM。
2. **aa4 每层 min_frac ≥ 0.05（router 无塌缩）— PASS（压线）**
   终值逐层 [0.1728, 0.1804, 0.0912, 0.0882]；全程（8 个 eval 点）逐层最低 [0.1728, 0.1756, 0.0912, 0.0503]，`collapsed_layers` 恒为空。layer 3 全程最低 0.0503，仅高出阈值 0.0003。
   风险注记：aa4 min_frac 最低 0.0503 压线；moa4 曾在 1250-1500 步塌缩后自恢复——扫描阶段 collapsed_layers 必须逐点检查。
3. **aa4 与 moa4 吞吐 ≥ dense 的 60% — PASS（2026-10-17 按 Controller Ruling 语义修正；原判 FAIL，原始判定见下文标注与 Ruling 节）**
   实测：aa4 44,448 / 73,262 = 60.7%；moa4 28,847 / 73,262 = **39.4%**（均低于 60% 字面阈值，moa4 远低——此即原始 FAIL 判定的依据，予以保留）。
   修正说明：60% 阈值基于条件执行假设；全量计算实现下按"实测 ≥ 理论 FLOPs 比"判，aa4 0.607≥0.56、moa4 0.394≥0.24，无病态开销；真吞吐命题归 Stage D。
4. **aa4 val loss ≤ dense + 0.15 — PASS**
   2.6054 − 2.5896 = **+0.0158**，远小于 0.15 阈值（容量路径未拖后腿）。

## 结论

**冒烟通过（判据 3 按 ruling 语义修正后），可进扫描；附带条件：扫描全程监控 collapsed_layers 与 min_frac。**

> 原结论（2026-10-17 首判，基于判据 3 字面 60% 阈值，保留可见）：判据 3 FAIL → 冒烟未全过，暂不进扫描（proposal §8 要求全过）。处置（按 brief）：吞吐差 → profile gather 路径——moa4（shared_kv=false）需为 4 条路径各持独立 KV 投影，top-1 gather 与逐 token 路径分发开销使吞吐跌至 dense 的 39.4%。该处置经 Ruling 语义修正后不再作为进入扫描的前置条件；条件执行级真吞吐命题移交 Stage D 兑现。

## Controller Ruling (2026-10-17)

- **背景**：Task 4 的实现是"全路径计算再 gather"（E≤16 全量计算，条件执行归 Stage D）。全量计算下 E=4 的理论计算量：aa4 = 投影 10d² + 4×attn-core vs dense = 4d² + 1×attn-core（seq 512 下理论吞吐比 ≈ 0.56）；moa4 = 16d² + 4×attn-core（理论比 ≈ 0.24）。proposal §3.2 原文："These parameter quantities are only approximate indicators of compute."
- **裁决**：判据 3 的意图是"路由开销可控"（无病态开销），而非"条件执行级吞吐"（那是 Stage D 命题）。重新判定标准 = 实测吞吐比 ≥ 理论全量 FLOPs 比。实测：aa4 60.7% ≥ 0.56 ✓、moa4 39.4% ≥ 0.24 ✓ → **判据 3 实质 PASS（无病态开销）**。
- **历史可见性**：首判 FAIL（依据 60% 字面阈值）的原文保留于上文判据 3 行与原结论引文块，未删除、未改写。

## 备注

- moa4 在 step 1250–1500 曾出现 layer 2/3 塌缩（min_frac 最低 0.0148 / 0.0032，`collapsed_layers=[2,3]`），1750 步起自行恢复为空。判据 2 只约束 aa4，不计 FAIL；但提示 STE top-1 路由在无共享 KV 时稳定性较差，正式扫描需持续监控 `collapsed_layers`。
- aa4 的判据 2（0.0503 vs 0.05）压线；判据 3 在 ruling 语义下实测 0.607 vs 理论 0.56，余量同样有限，扫描配置若有变动需重验。
- ste 模式 mean_max_weight 恒 1.0，无信息量，未纳入判读；塌缩判读只看 min_frac / collapsed_layers。
- aa4 的 STE 梯度对照（soft 模式 500 步）本判据未触发（判据 4 PASS）。
