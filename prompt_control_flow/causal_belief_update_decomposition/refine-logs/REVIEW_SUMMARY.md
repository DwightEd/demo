# Review Summary

**Problem**：从长度污染的几何指标转向可识别、可干预的信念更新机制。

**Final name**：Causal Belief Update Decomposition (CBUD)。

**Status**：方向通过内部方法审计，但外部 reviewer 因工具超时未完成；在 factorial patch 前，MLP 结论严格限定为 observational update signature。

关键收紧：使用 progress/error 计量真实更新；强制 block reconstruction；强制 primary-layer preregistration；旧 routing artifact 保持兼容。
