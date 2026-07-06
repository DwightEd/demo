# NTS 法向逃逸验证执行规划（2026-07-03）

> 前置文档：`2026-06-26-nts-evidence-gates.md`（gate 实现计划）、
> 桌面《论文设计提案_NTS_SGFS_2026-06-26.md》（方法与 kill 条件）、`../../guides/DATA.md`（数据地图）。
> 本文档是把提案的闸门 0–3 落到可执行状态的施工图：先修代码，再出判决，判决决定论文主故事。

---

## 0. 判决逻辑（不变）

```
闸门0 (Mahal 诚实地板) → 闸门1 (几何可估性) → 闸门2 (NTS vs REMA, 命门) → 闸门3 (曲率)
                                                      ↓ 过
                                          gate_localize (首错步定位判决, 新增)
```

任一 kill 立即止损，按提案 §5 降级预案执行。**在闸门 2 出判决之前不投入任何
own-generation 或干预工程。**

## 1. 数据与环境事实（本次核实）

- `full_*.npz` 由 `extract_features.py` 生成；**标签约定 `is_correct_strict`: 1=correct**
  （`_pb_record` 中 `correct = final_answer_correct or label==-1`）。
  `gold_error_step`: ProcessBench 约定，-1=全对，≥0=首错步索引。
- `stepvec (T,8,4096)` 是**步内 token 云的 exp-pool 位置向量 h_t（fp16），不是位移**。
  未中心化的 h_t 方向统计因表征各向异性饱和（κ→1），这是
  `validate_phase_instability.py` 全指标 AUC≈0.5 的信号侧根因（标签反转是数值侧根因，
  两者叠加得到 0.4–0.5）。→ 一切方向/谱统计必须作用在 **Δh 或中心化/白化后的向量** 上，
  NTS 的 reducer(去 massive 维+白化)+Δh 正是这个修复。
- 数据只在服务器 `/gz-data/research/demo/data/`；本地改代码、服务器跑判决。
- 步级标签 y=1 仅标 `gold_error_step` 那一步；**首错步之后的步是被污染的负类**，
  评估必须可选地掩掉（本次实现，默认开启）。
- NTS/REMA 按构造在 t=0 无分数（无前一步锚点）→ `gold_error_step==0` 的链的错误步
  不进入步级评估。可选修复：用 `qvec`（题面池化向量）作 h_{-1} 锚点（Phase 3）。

## 2. Phase 0 — 代码修复（本地，先于一切判决）

| # | 文件 | 问题 | 修复 |
|---|---|---|---|
| F1 | `validate_phase_instability.py:546` | `label==0` 当 correct，方向反了 | `==1` 为 correct；输出注明约定 |
| F2 | `data_loading.py` / `_gpu` / `_fast` / `_cache` / `_sliding` | 同 F1（`is_correct_strict==0` 当 correct） | 统一改为 `==1`，集中成一个 helper |
| F3 | `analyze_results.py:418-419` | n_correct/n_error 统计反了 | 同上 |
| F4 | `nts/gates/gate1_estimability.py:fold_angle` | null 对照 = 行置换+全局旋转，是**几何恒等变换**（kNN/切空间在等距变换下不变），null≡real，闸门必然误杀 | 改为**逐列独立置换**（破坏维度间联合结构、保留边际分布），这才是提案要求的 structure-destroyed null |
| F5 | `nts/data/loader.py` + `nts/core/types.py` + gates | 无 post-error 掩码 | ChainData/Flat 增加 `eval_mask`（t≤ges 或链全对）；gate1(c)/2/3 的 y、残差化、bootstrap 全部只在 mask 内 |
| F6 | `validate_phase_instability.py` | 只能算 raw h_t 方向统计（饱和） | 增加 `--vector_mode raw/center/delta`（默认 delta）；增加长度基线 AUROC(T alone) 与每指标 vs 长度的 Spearman |
| F7 | `data_loading*.py` 全家（sliding 除外） | **[致命]** 用绝对 token 索引+开区间切 hidden 分片；分片第0行=绝对位置 a0（`extract_features.py:431`），步区间是绝对闭区间 → 所有步几何算在错位窗口（错开 prompt 长度）且丢每步最后一个 token。`nts/data/loader.py` 一直是对的 | 统一换算"绝对闭区间→分片相对半开区间"，已修 data_loading / _fast / _cache / _minimal / _gpu×2 |
| F8 | `gate2` | NTS 在 t=0 恒 NaN 而 REMA/κ 有值 → 头对头 AUROC 算在不同样本集；ges=0 的错误步只从 NTS 中消失 | gate2 增加共同支撑集掩码（所有被比信号同为有限值），报告掩码数量 |
| F9 | `validate_local.py` | mock 数据按 0=correct 生成，与真实数据相反 → 反转的消费端在本地测试互相自洽"通过"，掩盖方向错误 | mock 改 1=correct 并生成 gold_error_step，可端到端测出方向错误 |
| F10 | `diagnose_results.py` | 自称"诊断反转"却按反转约定判定（恒真检查） | 改按 1=correct + gold_error_step 锚点交叉验证，能检出反转缓存 |
| F11 | `run_geometry_validation.py:49`、`inspect_omnimath_data.py`、3 份 md 文档 | 同 F1 方向反转/错误约定固化 | 已修；DATA.md 顶部加约定红线 |

**审查覆盖缺口（诚实声明）**：多智能体审查因会话限额中断，`nts/geom+signals` 数学镜头（切/法分解、
TwoNN、Ledoit-Wolf、REMA 同 bank 差分）未产出结构化判决——主会话已人工通读这些文件未见问题
（位移 Δh、锚点 h_{t-1}、bank 仅正确链、GroupKFold 按题分折均符合提案规格），但建议在服务器跑
gate 前先 `python -m pytest tests/ -q`（11 个合成数据单测）作为最后防线。

**服务器端必做（在跑任何判决之前）**：
```bash
# 1) 旧缓存标签反转 + 窗口错位，必须删除重建
rm -rf /gz-data/research/demo/data/hidden/cache/
# 2) 单测
cd /gz-data/research/demo && python -m pytest tests/ -q
# 3) 快速方向自检（应打印 label check ≈ 1.000）
python validate_phase_instability.py --dataset gsm8k --vector_mode delta
```

## 3. Phase 1 — 闸门判决（服务器，gsm8k + math 双数据集）

```bash
cd /gz-data/research/demo
python scripts/run_gate.py gate=gate0 data=gsm8k     # 0.5天档
python scripts/run_gate.py gate=gate1 data=gsm8k     # ID曲线也顺带输出选层依据
python scripts/run_gate.py gate=gate2 data=gsm8k     # ★命门
python scripts/run_gate.py gate=gate3 data=gsm8k
# math 重复一遍；gate_cloud / gate_ntc 便宜，顺带跑
```

判读表（对照提案 §7）：

| 闸门 | kill 条件（代码已实现） | kill 后动作 |
|---|---|---|
| 0 | honest=min(bucket, len-resid) ≤ 0.60 | 先查标签/数据污染，隐状态几何路线整体存疑 |
| 1 | 列置换 null 的跨折主角 ≤ real+0.05，或法向 SNR<1 | 局部切空间不可估 → NTS 无定义，降级预案 3 |
| 2 | ALL 与 cbw 两块的 [REMA+κ+logN+speed+rep] 之上增量均 ns | 换皮指控坐实 → NTS 放弃/重构，SGFS 连带取消 |
| 3 | 曲率残差在裸法向之上增量 ns | 撤"曲率去偏"机制叙事，保留裸法向 |

跑完把 `outputs/nts_gates/*.json` 拉回本地归档到 `AAAI2027/结果/`。

## 4. Phase 2 — gate_localize：首错步定位判决（新增，闸门2过后立即跑）

**动机**：链级 AUROC 已封顶 ~0.83 且是红海；ProcessBench 首错步定位是
提案 P5 的实证兑现，也是"步级+零/弱监督+定位"故事能否成立的判决实验。
定位不需要前兆信号（event_study 证明 gsm8k 信号与错误同步），同步就够。

**状态：已实现**（`nts/gates/gate_localize.py` + `config/gate/gate_localize.yaml`，
运行 `python scripts/run_gate.py gate=gate_localize data=gsm8k`）。

**设计**（`nts/gates/gate_localize.py`）：
- 分数：crossfit 的 NTS resid（主）；基线：REMA、−κ(resultant)、步长度、random、
  uniform-middle。
- 错误链定位：`pred = argmax_t score`（掩码内）；报 exact hit率、±1 hit率、
  相对随机基线的提升（随机基线 hit率 = E[1/T_valid]）。
- ProcessBench 风格 F1：正确链上按训练折正确链 max-score 的 95 分位定阈值 →
  预测"无错/有错+位置"，报 correct-side accuracy 与 error-side accuracy 的调和平均
  （ProcessBench 官方指标形状）。
- 按 T 分桶报告（定位难度随 T 变化，防"长链撞运气"质疑）。

**通过标准**（预登记）：NTS exact hit率显著高于 random 与 −κ（按题 bootstrap CI），
且 F1 至少可与 GeoReason 报告的无监督基线段位可比。达不到 → 定位故事降级为
辅助证据，主故事回到链级 NTS 检测。

## 4.5 Phase 2' — 谱流验证（2026-07-03 追加；闸门 0–3 全 KILL 后的新主线候选）

闸门判决（gsm8k+math）：REMA≈0.52 → off-manifold 家族整体无信号，NTS/SGFS 按预案 3 撤下；
唯一存活信号 = 步内 token 云方向集中度 κ（步级 0.779/bucket 0.715；定位 exact 0.449 vs 随机 0.267）。
新主线候选："推理的谱流"——窗口化 Gram 谱 + 一阶矩 + 时间序作为统一对象，
κ 是一阶矩投影、α（Spectral Geometry of Thought 斜率）/PR/eff_rank 是谱形状投影。
`spectral_flow.py` 验证 S1–S5（检测增量 / 步内再收缩 / 链内秩定位（构造上免长度混杂）/
层剖面 / 相位形状），运行：`python spectral_flow.py --dataset gsm8k`。

## 5. Phase 3 — 条件推进（闸门 2 过才做）

1. **跨数据集迁移**：gsm8k bank/阈值 → math/omnimath zero-shot，掉幅 ≤0.10（提案关b）。
2. **层扫描**：8 个 sv-layers + hidden [10,14,18,22] 全报，选层用 gate1 的 ID 曲线，
   不声称层无关。
3. **q-anchor**：用 qvec 作 h_{-1}，补上 t=0 的位移，消除"首步错误盲区"。
4. **层×token 扩展**（回答 2026-07-03 讨论）：
   - token 维：`nts_cloud`（步内 token 云的 off-correct-subspace 能量）已实现，
     与步级 NTS 做互补性分析（provides 步内混乱 vs 步间逃逸两个正交轴）；
   - 层维：把"每层一个标量"的 33 层 κ 剖面 / 4 层法向能量剖面作为特征**剖面**报告，
     但注意本项目已证伪跨层低秩协调——层维只做剖面描述与选层依据，不做跨层耦合 claim；
   - 聚合原则：方差分解（步内 token 离散度 ÷ 步间位移离散度 = 步凝聚指数），
     解耦原则：先单轴出判决再谈组合，组合一律走 oof_logit + 消融阶梯，不手搓乘积公式。

## 6. 评估协议规范（写论文前的红线）

- 标签断言：所有 loader 载入后 assert `(is_correct_strict==1) == (gold_error_step<0)`。
- post-error 掩码默认开启，论文注明；不掩码版本进附录。
- 首步错误链在步级评估中的处理方式必须显式声明（当前：排除；q-anchor 后：纳入）。
- 一切 AUROC 与 gate0 的 honest floor 同表呈现；0.83 封顶结论如实引用。
- 每个组合信号必须有消融阶梯（单指标→双指标→全组合）。
