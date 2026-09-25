"""Edge-case coverage for report parsing and rendering."""

import json
from pathlib import Path

from PIL import Image

from pickinsta.models import ImageScore
from pickinsta.reporting import gallery, gallery_data, markdown, recursive


def _write_report(folder: Path, rows: object) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "selection_report.json").write_text(json.dumps(rows), encoding="utf-8")


def _valid_row(**overrides: object) -> dict:
    row = {
        "rank": 1,
        "filename": "source.jpg",
        "final_score": 0.75,
        "technical_composite": 0.5,
        "vision_total": 45,
        "output_cropped": "01 cropped.jpg",
    }
    row.update(overrides)
    return row


def test_gallery_data_treats_missing_or_malformed_reports_as_empty(tmp_path: Path) -> None:
    assert gallery_data.build_gallery_data(tmp_path) == []

    (tmp_path / "selection_report.json").write_text("{not json", encoding="utf-8")
    assert gallery_data.build_gallery_data(tmp_path) == []

    (tmp_path / "selection_report.json").write_text('{"rank": 1}', encoding="utf-8")
    assert gallery_data.build_gallery_data(tmp_path) == []


def test_gallery_data_skips_bad_rows_and_tolerates_bad_cache_and_exif(
    tmp_path: Path,
) -> None:
    source = tmp_path / "input"
    output = tmp_path / "output"
    source.mkdir()
    _write_report(output, [None, "bad", {}, _valid_row(output_full=None)])
    (source / "source.jpg.pickinsta.json").write_text("broken", encoding="utf-8")
    (source / "source.jpg").write_bytes(b"not an image")

    rows = gallery_data.build_gallery_data(output, source)

    assert len(rows) == 1
    assert rows[0]["output_full"] == ""
    assert rows[0]["vision_detail"] == {}
    assert rows[0]["exif"] == {}


def test_gallery_data_tolerates_unreadable_inferred_input_report(tmp_path: Path) -> None:
    _write_report(tmp_path, [_valid_row()])
    (tmp_path / "selection_report.md").write_bytes(b"\xff\xfe invalid utf-8")

    assert len(gallery_data.build_gallery_data(tmp_path)) == 1


def test_gallery_data_skips_rows_with_non_numeric_scores(tmp_path: Path) -> None:
    _write_report(
        tmp_path,
        [_valid_row(final_score="not-a-score"), _valid_row(rank=2, final_score="0.8")],
    )

    rows = gallery_data.build_gallery_data(tmp_path)

    assert [row["rank"] for row in rows] == [2]
    assert rows[0]["final_score"] == 0.8


def test_gallery_data_ignores_incomplete_vision_cache(tmp_path: Path) -> None:
    source = tmp_path / "input"
    output = tmp_path / "output"
    source.mkdir()
    _write_report(output, [_valid_row()])
    (source / "source.jpg.pickinsta.json").write_text(
        json.dumps({"vision": "wrong shape"}), encoding="utf-8"
    )

    assert gallery_data.build_gallery_data(output, source)[0]["vision_detail"] == {}


def test_exif_conversion_failure_degrades_to_empty_metadata(tmp_path: Path, monkeypatch) -> None:
    source_file = tmp_path / "source.jpg"
    source_file.write_bytes(b"placeholder")

    class FakeImage:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def _getexif(self):
            return {33434: 0}  # ExposureTime=0 cannot be converted to a shutter fraction.

    monkeypatch.setattr(Image, "open", lambda _path: FakeImage())

    assert gallery_data._exif(tmp_path, source_file.name) == {}


def test_exif_extracts_supported_camera_fields(tmp_path: Path, monkeypatch) -> None:
    source_file = tmp_path / "source.jpg"
    source_file.write_bytes(b"placeholder")

    class FakeImage:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def _getexif(self):
            return {
                271: "Canon",
                272: "Canon R5",
                42036: "RF 24-70mm",
                37386: 50,
                33437: 2.8,
                33434: 1 / 250,
                34855: 400,
                36867: "2026:07:18 12:00:00",
            }

    monkeypatch.setattr(Image, "open", lambda _path: FakeImage())

    assert gallery_data._exif(tmp_path, source_file.name) == {
        "camera": "Canon R5",
        "lens": "RF 24-70mm",
        "focal": "50mm",
        "aperture": "f/2.8",
        "shutter": "1/250s",
        "iso": "ISO 400",
        "date": "2026:07:18 12:00:00",
    }


def test_gallery_escapes_html_and_embeds_untrusted_json_safely(tmp_path: Path) -> None:
    output = tmp_path / 'session <unsafe> & "quoted"'
    attack = "</script><script>alert('x')</script>"
    _write_report(output, [_valid_row(filename=attack, one_line=attack)])

    result = gallery.generate_gallery(output)

    assert result is not None
    html = result.read_text(encoding="utf-8")
    assert "session &lt;unsafe&gt; &amp; &quot;quoted&quot;" in html
    assert attack not in html
    assert "\\u003c/script\\u003e" in html
    assert html.count("</script>") == 1


def test_gallery_returns_none_without_usable_rows_and_writes_nothing(tmp_path: Path) -> None:
    _write_report(tmp_path, [None, {}, "invalid"])

    assert gallery.generate_gallery(tmp_path) is None
    assert not (tmp_path / "index.html").exists()


def test_gallery_ignores_malformed_optional_burst_metadata(tmp_path: Path) -> None:
    _write_report(tmp_path, [_valid_row(burst={"count": 3})])

    result = gallery.generate_gallery(tmp_path)

    assert result is not None
    assert 'title="Best of' not in result.read_text(encoding="utf-8")


def test_gallery_index_ignores_malformed_reports_and_links_nested_galleries(
    tmp_path: Path,
) -> None:
    valid = tmp_path / "year <2026>" / "session & one"
    invalid = tmp_path / "broken"
    _write_report(valid, [_valid_row()])
    invalid.mkdir()
    (invalid / "selection_report.json").write_text("not json", encoding="utf-8")

    generated = gallery.generate_gallery_index(tmp_path)

    assert valid / "index.html" in generated
    assert tmp_path / "year <2026>" / "index.html" in generated
    assert tmp_path / "index.html" in generated
    assert invalid / "index.html" not in generated
    root_html = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert "year &lt;2026&gt;" in root_html
    assert "broken" not in root_html
    nested_html = (tmp_path / "year <2026>" / "index.html").read_text(encoding="utf-8")
    assert "session &amp; one" in nested_html
    assert 'href="session%20%26%20one/"' in nested_html


def test_recursive_empty_summary_has_zero_aggregates(tmp_path: Path) -> None:
    json_path, md_path = recursive.write_recursive_summary_report(
        root_input=tmp_path / "input",
        output_root=tmp_path,
        scorer="clip",
        top_n=5,
        folder_summaries=[],
    )

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["folders_processed"] == 0
    assert payload["total_selected"] == 0
    assert payload["avg_top_score"] == 0.0
    assert "Average top score: `0.0000`" in md_path.read_text(encoding="utf-8")


def test_recursive_summary_aggregates_multiple_folders_and_escapes_tables(
    tmp_path: Path,
) -> None:
    summaries = [
        {
            "relative_folder": "one|first",
            "output_relative": "out|one",
            "selected_count": 2,
            "uncertain_crops": 1,
            "top_score": 0.9,
        },
        {
            "relative_folder": "two",
            "output_relative": "out/two",
            "selected_count": 1,
            "uncertain_crops": 0,
            "top_score": 0.5,
        },
    ]

    json_path, md_path = recursive.write_recursive_summary_report(
        root_input=tmp_path / "input",
        output_root=tmp_path,
        scorer="ollama",
        top_n=3,
        folder_summaries=summaries,
    )

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["total_selected"] == 3
    assert payload["total_uncertain_crops"] == 1
    assert payload["avg_top_score"] == 0.7
    rendered = md_path.read_text(encoding="utf-8")
    assert "one\\|first" in rendered
    assert "out\\|one" in rendered


def test_degraded_markdown_includes_manifest_counts_and_sanitized_issues(
    tmp_path: Path,
) -> None:
    report = tmp_path / "selection_report.md"
    analyzed = [
        ImageScore(
            path=Path("work.jpg"),
            source_path=Path("source.jpg"),
            technical={},
            final_score=0.25,
            one_line="line one\nline two",
        )
    ]

    markdown.write_markdown_report(
        report,
        input_folder=tmp_path / "input",
        output_folder=tmp_path / "output",
        scorer="clip",
        top_n=1,
        selected_report=[],
        analyzed_items=analyzed,
        run_summary={
            "status": "degraded",
            "processed": 1,
            "skipped": 2,
            "failed": 3,
            "issues": [{"stage": "copy|full", "artifact": "x.jpg", "reason": "disk\nfull"}],
        },
    )

    content = report.read_text(encoding="utf-8")
    assert "Run status: `degraded`" in content
    assert "Processed: `1`" in content
    assert "Skipped: `2`" in content
    assert "Failed artifacts: `3`" in content
    assert "`copy\\|full` / `x.jpg`: disk full" in content
    assert "line one line two" in content
