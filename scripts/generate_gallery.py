#!/usr/bin/env python3
"""Generate static HTML galleries from pickinsta selection output folders.

Usage:
    python scripts/generate_gallery.py ~/Photos/td6_selected/2ab/Session_1_Pit_Lane_Entry
    python scripts/generate_gallery.py ~/Photos/td6_selected  # recursively generates galleries + folder indexes
"""

import sys
from pathlib import Path

from pickinsta.ig_image_selector import generate_gallery, generate_gallery_index


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <output_folder>")
        print("  Generates index.html in each folder containing selection_report.json")
        print("  Parent folders get a directory listing linking to galleries")
        sys.exit(1)

    root = Path(sys.argv[1]).resolve()

    if (root / "selection_report.json").exists():
        result = generate_gallery(root)
        if result:
            print(f"  {result}")
    else:
        generated = generate_gallery_index(root)
        if not generated:
            print(f"No selection_report.json found under {root}")
            sys.exit(1)
        print(f"Generated {len(generated)} index/gallery pages under {root}")
        for p in generated:
            print(f"  {p}")


if __name__ == "__main__":
    main()
