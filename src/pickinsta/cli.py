"""Command-line parsing and dispatch for :mod:`pickinsta`."""

from __future__ import annotations

import argparse
import importlib
import sys
from dataclasses import dataclass
from types import ModuleType
from typing import Callable, Sequence

from pickinsta.config import DEFAULT_CLAUDE_MODEL, resolve_claude_model


class _HelpOnErrorArgumentParser(argparse.ArgumentParser):
    """ArgumentParser that prints full help text on parse errors."""

    def error(self, message: str) -> None:
        self.print_help(sys.stderr)
        self.exit(2, f"\n{self.prog}: error: {message}\n")


@dataclass(frozen=True)
class CliCollaborators:
    """Facade operations invoked after command-line validation."""

    run_pipeline: Callable[..., object]
    run_pipeline_recursive: Callable[..., object]
    run_dedup_only: Callable[..., object]

    @classmethod
    def from_selector(cls, selector: ModuleType) -> CliCollaborators:
        return cls(
            run_pipeline=selector.run_pipeline,
            run_pipeline_recursive=selector.run_pipeline_recursive,
            run_dedup_only=selector.run_dedup_only,
        )


def build_parser() -> _HelpOnErrorArgumentParser:
    """Build the public command-line parser."""
    parser = _HelpOnErrorArgumentParser(
        prog="pickinsta",
        description="Select the best Instagram cover images from an event photo dump.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s ./input --top 10 --scorer clip
  %(prog)s ./input --output ./selected --scorer claude --top 5
  %(prog)s ./input --output ./selected --scorer claude --all
  %(prog)s ./input --output ./selected --scorer claude --all --claude-crop-first
  %(prog)s ./input --output ./selected --scorer clip --all
        """,
    )
    parser.add_argument("input", help="Path to folder containing event photos")
    parser.add_argument(
        "--output", "-o", default="selected", help="Output folder (default: selected)"
    )
    parser.add_argument(
        "--work",
        "-w",
        default=None,
        help="Work folder for intermediate files (default: <input>_work next to input)",
    )
    parser.add_argument(
        "--top",
        "-n",
        type=int,
        default=10,
        metavar="N",
        help="Number of top images to output (default: 10, must be >= 1)",
    )
    parser.add_argument(
        "--scorer",
        "-s",
        choices=["clip", "claude", "ollama"],
        default="clip",
        help="Vision scorer: 'clip' (free/local), 'claude' (API), or 'ollama' (self-hosted)",
    )
    parser.add_argument(
        "--vision-pct",
        type=float,
        default=0.5,
        metavar="[0-1]",
        help=(
            "Fraction of technically-scored images to send to vision scoring "
            "(default: 0.5, range: 0.0-1.0)"
        ),
    )
    parser.add_argument(
        "--claude-model",
        default=resolve_claude_model(),
        help=(
            f"Claude model id (default from ANTHROPIC_MODEL/CLAUDE_MODEL or {DEFAULT_CLAUDE_MODEL})"
        ),
    )
    parser.add_argument(
        "--all",
        "--claude-all",
        dest="score_all",
        action="store_true",
        help="Score all Stage 2 images (ignore --vision-pct).",
    )
    parser.add_argument(
        "--claude-crop-first",
        dest="claude_crop_first",
        action="store_true",
        help=(
            "For --scorer claude, pre-crop candidate images to 1080x1440 before "
            "Claude scoring to better align ranking with final crop quality."
        ),
    )
    parser.add_argument(
        "--rescore",
        action="store_true",
        help="Force re-scoring all images, ignoring cached vision scores.",
    )
    parser.add_argument(
        "--dedup-only",
        dest="dedup_only",
        action="store_true",
        help=(
            "Dedup-only mode: select best shot per burst, output all unique images "
            "as full/hd/cropped. No scoring, no ranking, no debug files."
        ),
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help=(
            "Process each leaf subfolder under the input folder, mirror the folder "
            "structure in the output, and build a recursive summary report."
        ),
    )
    return parser


def _validate(parser: _HelpOnErrorArgumentParser, args: argparse.Namespace) -> None:
    if args.recursive and args.dedup_only:
        parser.error("--recursive cannot be combined with --dedup-only")
    if args.top < 1:
        parser.error(f"--top must be >= 1, got {args.top}")
    if not 0.0 <= args.vision_pct <= 1.0:
        parser.error(f"--vision-pct must be between 0.0 and 1.0, got {args.vision_pct}")


def _default_collaborators() -> CliCollaborators:
    selector = importlib.import_module("pickinsta.ig_image_selector")
    return CliCollaborators.from_selector(selector)


def main(
    argv: Sequence[str] | None = None,
    *,
    collaborators: CliCollaborators | None = None,
) -> None:
    """Parse ``argv`` and dispatch the selected pipeline operation."""
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate(parser, args)
    operations = collaborators or _default_collaborators()

    if args.dedup_only:
        operations.run_dedup_only(
            input_folder=args.input, output_folder=args.output, work_folder=args.work
        )
        return

    runner = operations.run_pipeline_recursive if args.recursive else operations.run_pipeline
    runner(
        input_folder=args.input,
        output_folder=args.output,
        work_folder=args.work,
        top_n=args.top,
        scorer=args.scorer,
        vision_candidates_pct=args.vision_pct,
        claude_model=args.claude_model,
        score_all=args.score_all,
        claude_crop_first=args.claude_crop_first,
        rescore=args.rescore,
    )
