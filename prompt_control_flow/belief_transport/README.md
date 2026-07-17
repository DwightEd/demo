# Belief Transport Validation

This folder contains the self-contained Stage-1 validator for
constraint-supported belief transport.

- `world.py`: finite hypothesis worlds and strictly reducing conditions
- `belief.py`: exact conditioning, entropy, support margin, Fisher-Rao geometry
- `model_capture.py`: memory-bounded residual boundary hooks
- `extraction.py`: prompt rendering, length-bucketed GPU extraction, compact logits
- `artifact.py`: strict trace schema and shard merge
- `decoder.py`: problem-grouped soft-belief cross-fitting
- `audit.py`: conditional information, operator nulls, bootstrap gates, reports
- `METHOD.md`: hypotheses, equations, protocol, commands, and stopping rules

Top-level entry points are:

```text
build_belief_wind_tunnel.py
extract_belief_transport.py
merge_belief_transport.py
audit_belief_transport.py
```
