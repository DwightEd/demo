from __future__ import annotations

from datetime import datetime
from pathlib import Path


def versioned_paths(path: Path) -> tuple[Path, ...]:
    """Return permanent-history then latest paths when a latest artifact exists."""
    if not path.exists():
        return (path,)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archived = path.with_name(f"{path.stem}_{timestamp}{path.suffix}")
    counter = 2
    while archived.exists():
        archived = path.with_name(f"{path.stem}_{timestamp}_{counter}{path.suffix}")
        counter += 1
    return archived, path
