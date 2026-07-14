"""Output-conditioned geometric predictive information (OC-GPI).

The package separates compact output-trace extraction from geometry loading and
cross-fitted conditional-increment audits.  Full vocabulary logits and full
attention tensors are deliberately never persisted.
"""

from .schema import (
    OCGPI_TRACE_SCHEMA_VERSION,
    CompactTraceItem,
    TraceArtifact,
)

__all__ = [
    "OCGPI_TRACE_SCHEMA_VERSION",
    "CompactTraceItem",
    "TraceArtifact",
]
