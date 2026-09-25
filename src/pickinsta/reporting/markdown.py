"""Markdown selection-report rendering."""

from pathlib import Path
from typing import Optional

from pickinsta.infrastructure.filesystem import atomic_write_text
from pickinsta.models import ImageScore


def escape_markdown(value: object) -> str:
    """Escape a value for a Markdown table cell."""
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", " ").strip()


def write_markdown_report(
    report_path: Path,
    *,
    input_folder: Path,
    output_folder: Path,
    scorer: str,
    top_n: int,
    selected_report: list[dict],
    analyzed_items: list[ImageScore],
    run_summary: Optional[dict] = None,
) -> None:
    """Write a Markdown report with selections and all analyzed results."""
    lines = [
        "# pickinsta Selection Report",
        "",
        f"- Input: `{input_folder}`",
        f"- Output: `{output_folder}`",
        f"- Scorer: `{scorer}`",
        f"- Top N requested: `{top_n}`",
        f"- Images analyzed in vision stage: `{len(analyzed_items)}`",
    ]
    if run_summary is not None:
        lines.extend(
            [
                f"- Run status: `{run_summary.get('status', 'unknown')}`",
                f"- Processed: `{run_summary.get('processed', 0)}`",
                f"- Skipped: `{run_summary.get('skipped', 0)}`",
                f"- Failed artifacts: `{run_summary.get('failed', 0)}`",
            ]
        )
        issues = run_summary.get("issues", [])
        if issues:
            lines.extend(["", "## Run Issues", ""])
            for issue in issues:
                lines.append(
                    f"- `{escape_markdown(issue.get('stage', 'unknown'))}` / "
                    f"`{escape_markdown(issue.get('artifact', 'unknown'))}`: "
                    f"{escape_markdown(issue.get('reason', 'unknown error'))}"
                )

    lines.extend(
        [
            "",
            "## Top Selected Outputs",
            "",
            "| Rank | Filename | Final | Tech | Vision | Output | Summary |",
            "|---:|---|---:|---:|---:|---|---|",
        ]
    )
    for row in selected_report:
        lines.append(
            "| "
            f"{row.get('rank', '')} | "
            f"{escape_markdown(row.get('filename', ''))} | "
            f"{row.get('final_score', '')} | "
            f"{row.get('technical_composite', '')} | "
            f"{row.get('vision_total', '')} | "
            f"{escape_markdown(row.get('output_cropped', ''))} | "
            f"{escape_markdown(row.get('one_line', ''))} |"
        )

    title = (
        "Claude Analysis (All Images Analyzed)"
        if scorer == "claude"
        else "Vision Analysis (All Images Analyzed)"
    )
    lines.extend(
        [
            "",
            f"## {title}",
            "",
            "| Rank | Filename | Final | Tech | Vision | Subject | Lighting | Color | Emotion | Scroll | Crop | Summary |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for idx, item in enumerate(analyzed_items, start=1):
        vision = item.vision or {}
        display_name = item.source_path.name if item.source_path else item.path.name
        lines.append(
            "| "
            f"{idx} | {escape_markdown(display_name)} | {item.final_score:.4f} | "
            f"{item.technical.get('composite', 0):.4f} | {vision.get('total', 0)} | "
            f"{vision.get('subject_clarity', '')} | {vision.get('lighting', '')} | "
            f"{vision.get('color_pop', '')} | {vision.get('emotion', '')} | "
            f"{vision.get('scroll_stop', '')} | {vision.get('crop_4x5', '')} | "
            f"{escape_markdown(item.one_line)} |"
        )

    atomic_write_text(report_path, "\n".join(lines) + "\n")
