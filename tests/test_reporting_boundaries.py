"""Boundary tests for reporting modules extracted from the CLI orchestrator."""

import hashlib
import json
from pathlib import Path

from pickinsta import ig_image_selector as selector
from pickinsta.reporting import gallery, gallery_data, markdown, recursive, templates


def test_selector_reexports_markdown_report_writer() -> None:
    assert selector.write_markdown_report is markdown.write_markdown_report


def test_selector_reexports_recursive_report_writer() -> None:
    assert selector._write_recursive_summary_report is recursive.write_recursive_summary_report


def test_selector_uses_extracted_gallery_data_builder() -> None:
    assert selector._gallery_build_data is gallery_data.build_gallery_data


def test_gallery_renderer_uses_extracted_document_templates() -> None:
    assert gallery.GALLERY_HTML_TEMPLATE is templates.GALLERY_HTML_TEMPLATE
    assert gallery.DEDUP_GALLERY_TEMPLATE is templates.DEDUP_GALLERY_TEMPLATE
    assert gallery.INDEX_HTML_TEMPLATE is templates.INDEX_HTML_TEMPLATE


def test_extracted_templates_preserve_exact_gallery_output(tmp_path: Path) -> None:
    output = tmp_path / "session"
    output.mkdir()
    (output / "selection_report.json").write_text(
        json.dumps(
            [
                {
                    "rank": 1,
                    "filename": "source.jpg",
                    "final_score": 0.75,
                    "technical_composite": 0.5,
                    "vision_total": 45,
                    "output_cropped": "01 cropped.jpg",
                }
            ]
        ),
        encoding="utf-8",
    )

    rendered = gallery.generate_gallery(output)

    assert rendered is not None
    assert hashlib.sha256(rendered.read_bytes()).hexdigest() == (
        "3b1e2f5c0529da63b60e889661c626fc03e11c7d3c49abea5ccf81137952afd5"
    )


def test_extracted_templates_preserve_exact_dedup_output(tmp_path: Path) -> None:
    output = tmp_path / "dedup"
    output.mkdir()

    gallery._generate_dedup_gallery(
        output,
        [
            {
                "filename": "source.jpg",
                "cropped": "crop.jpg",
                "hd": "hd.jpg",
                "full": "full.jpg",
                "burst_count": 1,
                "exif": {},
            }
        ],
    )

    assert hashlib.sha256((output / "index.html").read_bytes()).hexdigest() == (
        "d003da58455584a42a36127761a8f8888507b8ab2ef7053f1148318d3c2bbf97"
    )


def test_gallery_data_builder_preserves_report_fields(tmp_path: Path) -> None:
    (tmp_path / "selection_report.json").write_text(
        json.dumps(
            [
                {
                    "rank": 1,
                    "filename": "photo one.jpg",
                    "final_score": 0.9,
                    "technical_composite": 0.8,
                    "vision_total": 50,
                    "output_cropped": "01 cropped.jpg",
                    "output_hd": "01 hd.jpg",
                    "uncertain_crop": True,
                }
            ]
        ),
        encoding="utf-8",
    )

    assert gallery_data.build_gallery_data(tmp_path) == [
        {
            "rank": 1,
            "filename": "photo one.jpg",
            "final_score": 0.9,
            "technical_composite": 0.8,
            "vision_total": 50,
            "one_line": "",
            "output_cropped": "01%20cropped.jpg",
            "output_hd": "01%20hd.jpg",
            "output_full": "",
            "uncertain_crop": True,
            "vision_detail": {},
            "yolo": None,
            "exif": {},
            "burst": None,
            "yolo_debug_img": None,
        }
    ]


def test_recursive_report_module_writes_complete_atomic_outputs(tmp_path: Path) -> None:
    summaries = [
        {
            "relative_folder": "session|one",
            "output_relative": "session-one",
            "report_relative": "session-one/selection_report.json",
            "selected_count": 2,
            "uncertain_crops": 1,
            "top_score": 0.875,
        }
    ]

    json_path, markdown_path = recursive.write_recursive_summary_report(
        root_input=tmp_path / "input",
        output_root=tmp_path,
        scorer="clip",
        top_n=10,
        folder_summaries=summaries,
    )

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["folders_processed"] == 1
    assert payload["total_selected"] == 2
    assert payload["total_uncertain_crops"] == 1
    assert payload["avg_top_score"] == 0.875
    rendered = markdown_path.read_text(encoding="utf-8")
    assert "session\\|one" in rendered
    assert "session-one/selection_report.json" in rendered
    assert not list(tmp_path.glob(".selection_report_recursive.*"))
