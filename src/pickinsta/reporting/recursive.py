"""Recursive-run summary reporting."""

from pathlib import Path

from pickinsta.infrastructure.filesystem import atomic_write_json, atomic_write_text
from pickinsta.reporting.markdown import escape_markdown


def write_recursive_summary_report(
    *,
    root_input: Path,
    output_root: Path,
    scorer: str,
    top_n: int,
    folder_summaries: list[dict],
) -> tuple[Path, Path]:
    """Write JSON and Markdown summaries for a recursive run."""
    json_path = output_root / "selection_report_recursive.json"
    md_path = output_root / "selection_report_recursive.md"
    total_folders = len(folder_summaries)
    total_selected = sum(int(item.get("selected_count", 0)) for item in folder_summaries)
    total_uncertain = sum(int(item.get("uncertain_crops", 0)) for item in folder_summaries)
    avg_top_score = (
        sum(float(item.get("top_score", 0.0)) for item in folder_summaries) / total_folders
        if total_folders
        else 0.0
    )
    atomic_write_json(
        json_path,
        {
            "input_root": str(root_input),
            "output_root": str(output_root),
            "scorer": scorer,
            "top_n": top_n,
            "folders_processed": total_folders,
            "total_selected": total_selected,
            "total_uncertain_crops": total_uncertain,
            "avg_top_score": round(avg_top_score, 4),
            "folders": folder_summaries,
        },
    )
    lines = [
        "# pickinsta Recursive Report",
        "",
        f"- Input root: `{root_input}`",
        f"- Output root: `{output_root}`",
        f"- Scorer: `{scorer}`",
        f"- Top N per folder: `{top_n}`",
        f"- Folders processed: `{total_folders}`",
        f"- Total selected outputs: `{total_selected}`",
        f"- Total uncertain crops: `{total_uncertain}`",
        f"- Average top score: `{avg_top_score:.4f}`",
        "",
        "| Folder | Selected | Top score | Uncertain crops | Output |",
        "|---|---:|---:|---:|---|",
    ]
    for item in folder_summaries:
        rel = item.get("relative_folder", ".")
        out_rel = item.get("output_relative", rel)
        lines.append(
            f"| {escape_markdown(rel)} | {item.get('selected_count', 0)} | "
            f"{float(item.get('top_score', 0.0)):.4f} | "
            f"{item.get('uncertain_crops', 0)} | {escape_markdown(out_rel)} |"
        )
    lines.extend(["", "## Folder Reports", ""])
    for item in folder_summaries:
        rel = item.get("relative_folder", ".")
        report_rel = item.get("report_relative")
        lines.append(f"- `{rel}` -> `{item.get('output_relative')}`")
        if report_rel:
            lines.append(f"  - Report: `{report_rel}`")
    atomic_write_text(md_path, "\n".join(lines) + "\n")
    return json_path, md_path
