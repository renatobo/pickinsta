"""Report and gallery generation for pickinsta."""

from pickinsta.reporting.gallery import generate_gallery, generate_gallery_index
from pickinsta.reporting.gallery_data import build_gallery_data
from pickinsta.reporting.markdown import escape_markdown, write_markdown_report
from pickinsta.reporting.recursive import write_recursive_summary_report

__all__ = [
    "build_gallery_data",
    "escape_markdown",
    "generate_gallery",
    "generate_gallery_index",
    "write_markdown_report",
    "write_recursive_summary_report",
]
