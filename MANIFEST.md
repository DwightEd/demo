# Artifact Manifest

固定文件始终指向当前版本；时间戳文件保留审计轨迹。

| 时间（Asia/Shanghai） | 阶段 | 时间戳 artifact | 固定副本 | 状态 |
|---|---|---|---|---|
| 2026-07-13 01:02 | paper-analysis | research-reports/phenomenology-of-hallucinations-20260713-010210.md | research-reports/PHENOMENOLOGY_LATEST.md | completed |
| 2026-07-13 01:02 | superseded idea draft | research-reports/idea-discovery-20260713-010210.md | - | superseded by unified geometry revision |
| 2026-07-13 01:02 | superseded experiment draft | refine-logs/experiment-plan-20260713-010210.md | - | superseded by unified geometry revision |
| 2026-07-13 02:13 | idea-discovery revision | research-reports/idea-discovery-20260713-021336.md | research-reports/IDEA_DISCOVERY_LATEST.md | completed, same-family multi-agent review |
| 2026-07-13 02:13 | experiment-bridge revision | refine-logs/experiment-plan-20260713-021336.md | refine-logs/EXPERIMENT_PLAN.md; refine-logs/EXPERIMENT_PLAN_LATEST.md | drafted; real GPU validation pending |
| 2026-07-13 02:13 | experiment tracker | refine-logs/experiment-tracker-20260713-021336.md | refine-logs/EXPERIMENT_TRACKER.md | active |
| 2026-07-13 02:13 | method specification | prompt_control_flow/METHOD_LAYER_TIME_GEOMETRY.md | - | implementation-aligned |
| 2026-07-13 02:13 | engineering entrypoint | md/guides/PROJECT_README.md | - | updated |

## Code entrypoints

- prompt_control_flow/cli/extract_mechanisms.py --geometry_only
- prompt_control_flow/cli/audit_layer_time_geometry.py
- prompt_control_flow/layer_time_geometry.py
- tests/test_layer_time_geometry.py
- tests/test_teacher_forcing_trace.py

## External skills installation

Project-local install manifest:

- D:/projects/research/.aris/installed-skills-codex.txt
- 80 Codex skills under D:/projects/research/.agents/skills
- shared references under D:/projects/research/.agents/skills/shared-references
- helper tools under D:/projects/research/.aris/tools
