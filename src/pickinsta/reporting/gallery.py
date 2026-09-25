"""HTML gallery rendering for selection and deduplication outputs."""

import json
from html import escape
from pathlib import Path
from typing import Optional

from pickinsta.reporting.gallery_data import build_gallery_data as _gallery_build_data
from pickinsta.reporting.templates import (
    DEDUP_GALLERY_TEMPLATE,
    GALLERY_HTML_TEMPLATE,
    INDEX_HTML_TEMPLATE,
    SHARED_HEADER_CSS,
)


def _json_for_script(value: object) -> str:
    """Serialize JSON without allowing data to terminate the script element."""
    return (
        json.dumps(value, indent=None)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )


def _generate_dedup_gallery(output_folder: Path, items: list[dict]) -> None:
    """Generate a simplified gallery for dedup-only mode."""
    from urllib.parse import quote as _q

    title = escape(output_folder.name)
    tiles = ""
    for i, item in enumerate(items):
        burst_badge = (
            f'<span class="tile-burst">&#x1F4F7;{item["burst_count"]}</span>'
            if item.get("burst_count", 0) > 1
            else ""
        )
        tiles += (
            f'<div class="tile" data-idx="{i}" onclick="sel({i})">'
            f'<img src="{_q(item["cropped"])}" loading="lazy" alt="">'
            f"{burst_badge}"
            f"</div>\n"
        )

    json_items = [
        {
            "filename": it["filename"],
            "cropped": _q(it["cropped"]),
            "hd": _q(it["hd"]),
            "full": _q(it["full"]),
            "exif": it.get("exif", {}),
        }
        for it in items
    ]

    html = DEDUP_GALLERY_TEMPLATE.format(
        title=title,
        count=len(items),
        tiles=tiles,
        json_data=_json_for_script(json_items),
        shared_header_css=SHARED_HEADER_CSS,
    )
    (output_folder / "index.html").write_text(html, encoding="utf-8")


def generate_gallery(
    output_folder: Path,
    input_folder: Optional[Path] = None,
    gallery_root: Optional[Path] = None,
) -> Optional[Path]:
    """Generate an index.html gallery in the output folder. Returns path or None."""
    data = _gallery_build_data(output_folder, input_folder)
    if not data:
        return None

    title = escape(output_folder.name)

    if data:
        avg_final = sum(d["final_score"] for d in data) / len(data)
        avg_tech = sum(d["technical_composite"] for d in data) / len(data)
        avg_vision = sum(d["vision_total"] for d in data) / len(data)
        stats = [
            ("Images", str(len(data))),
            ("Top score", f"{data[0]['final_score']:.3f}"),
            ("Avg final", f"{avg_final:.3f}"),
            ("Avg technical", f"{avg_tech:.3f}"),
            ("Avg vision", f"{avg_vision:.1f}/60"),
            ("Uncertain crops", str(sum(1 for d in data if d["uncertain_crop"]))),
        ]
    else:
        stats = [("Images", "0")]

    summary_html = "\n".join(
        f'<div class="stat"><div class="stat-label">{label}</div>'
        f'<div class="stat-value">{value}</div></div>'
        for label, value in stats
    )
    tiles_html = ""
    for i, d in enumerate(data):
        uncertain_badge = (
            '<span class="tile-uncertain" title="Uncertain crop">'
            '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">'
            '<path d="M6 2v14a2 2 0 002 2h14"/><path d="M18 22V8a2 2 0 00-2-2H2"/></svg></span>'
            if d.get("uncertain_crop")
            else ""
        )
        burst = d.get("burst")
        if not isinstance(burst, dict) or "count" not in burst or "selected_by" not in burst:
            burst = None
        burst_badge = (
            f'<span class="tile-burst" title="Best of {burst["count"]} (by {burst["selected_by"]})">'
            f"&#x1F4F7;{burst['count']}</span>"
            if burst
            else ""
        )
        tiles_html += (
            f'<div class="tile" data-idx="{i}" onclick="selectImage({i})">'
            f'<img src="{d["output_cropped"]}" loading="lazy" alt="">'
            f'<div class="tile-overlay"></div>'
            f'<span class="tile-rank">#{d["rank"]}</span>'
            f"{uncertain_badge}"
            f"{burst_badge}"
            f'<span class="tile-score">{d["final_score"]:.3f}</span>'
            f"</div>\n"
        )

    # Breadcrumb
    breadcrumb = ""
    if gallery_root and output_folder != gallery_root:
        try:
            parts = output_folder.relative_to(gallery_root).parts
            crumbs = ['<a href="' + "../" * len(parts) + '">Home</a>']
            for i, part in enumerate(parts[:-1]):
                depth = len(parts) - i - 1
                href = "../" * depth
                crumbs.append(f'<a href="{href}">{escape(part)}</a>')
            sep = '<span class="sep">/</span>'
            breadcrumb = f'<span class="breadcrumb">{sep.join(crumbs)}{sep}</span>'
        except ValueError:
            pass

    html = GALLERY_HTML_TEMPLATE.format(
        title=title,
        summary_stats=summary_html,
        tiles=tiles_html,
        json_data=_json_for_script(data),
        breadcrumb=breadcrumb,
        shared_header_css=SHARED_HEADER_CSS,
    )
    gallery_path = output_folder / "index.html"
    gallery_path.write_text(html, encoding="utf-8")
    return gallery_path


def generate_gallery_index(root: Path) -> list[Path]:
    """Generate index.html files for all directories, recursively.

    Leaf directories with selection_report.json get a gallery.
    Parent directories get a folder listing linking to children.
    Returns list of all generated index.html paths.
    """
    from urllib.parse import quote as _quote

    # First, generate all leaf galleries
    generated: list[Path] = []
    reports = sorted(root.rglob("selection_report.json"))
    gallery_dirs: set[Path] = set()
    for rpt in reports:
        result = generate_gallery(rpt.parent, gallery_root=root)
        if result:
            generated.append(result)
            gallery_dirs.add(rpt.parent)

    # Now generate index pages for every ancestor directory
    # that contains galleries (directly or nested)
    index_dirs: set[Path] = set()
    for gdir in gallery_dirs:
        parent = gdir.parent
        while parent >= root:
            index_dirs.add(parent)
            if parent == root:
                break
            parent = parent.parent

    for idx_dir in sorted(index_dirs):
        # Collect immediate children that either have a gallery or have their own index
        children = []
        for child in sorted(idx_dir.iterdir()):
            if not child.is_dir():
                continue
            # Count only reports that produced a valid gallery. Malformed reports
            # must not create dead links in ancestor indexes.
            child_reports = sorted(
                report
                for report in child.rglob("selection_report.json")
                if report.parent in gallery_dirs
            )
            if not child_reports:
                continue
            total_images = 0
            for cr in child_reports:
                try:
                    data = json.loads(cr.read_text(encoding="utf-8"))
                    total_images += len(data)
                except Exception:
                    pass
            # Find a thumbnail from the first gallery
            thumb = ""
            first_report = child_reports[0]
            try:
                data = json.loads(first_report.read_text(encoding="utf-8"))
                if data:
                    cropped = data[0].get("output_cropped") or data[0].get("output", "")
                    if cropped:
                        rel_path = first_report.parent.relative_to(idx_dir)
                        thumb = str(rel_path / cropped)
            except Exception:
                pass

            children.append(
                {
                    "name": child.name,
                    "path": child.name + "/",
                    "galleries": len(child_reports),
                    "images": total_images,
                    "thumb": _quote(thumb) if thumb else "",
                }
            )

        if not children:
            continue

        rows_html = ""
        for c in children:
            thumb_html = (
                f'<img class="folder-thumb" src="{c["thumb"]}" loading="lazy" alt="">'
                if c["thumb"]
                else ""
            )
            sub = f"{c['galleries']} sessions, " if c["galleries"] > 1 else ""
            rows_html += (
                f'<li><a class="folder-link" href="{_quote(c["path"])}">'
                f"{thumb_html}"
                f'<span class="folder-name">{escape(c["name"])}</span>'
                f'<span class="folder-count">{sub}{c["images"]} images</span>'
                f"</a></li>\n"
            )

        # Breadcrumb
        breadcrumb = ""
        if idx_dir != root:
            parts = idx_dir.relative_to(root).parts
            crumbs = ['<a href="' + "../" * len(parts) + '">Home</a>']
            for i, part in enumerate(parts[:-1]):
                depth = len(parts) - i - 1
                href = "../" * depth
                crumbs.append(f'<a href="{href}">{escape(part)}</a>')
            sep = '<span class="sep">/</span>'
            breadcrumb = f'<span class="breadcrumb">{sep.join(crumbs)}{sep}</span>'

        html = INDEX_HTML_TEMPLATE.format(
            title=escape(idx_dir.name),
            breadcrumb=breadcrumb,
            rows=rows_html,
            shared_header_css=SHARED_HEADER_CSS,
        )
        idx_path = idx_dir / "index.html"
        idx_path.write_text(html, encoding="utf-8")
        generated.append(idx_path)

    return generated
