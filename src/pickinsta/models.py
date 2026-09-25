"""Domain models shared by Pickinsta pipeline stages."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ImageScore:
    """Scores and selection metadata associated with one image."""

    path: Path
    source_path: Optional[Path] = None
    technical: dict = field(default_factory=dict)
    vision: dict = field(default_factory=dict)
    final_score: float = 0.0
    one_line: str = ""
    burst_group: Optional[list[Path]] = None
    burst_selected_by: str = ""
