# Documentation Index

This directory holds the operator and reference docs for `pickinsta`.

## Core References

- [`composition-rules.md`](/home/renatobo/devel/pickinsta/docs/composition-rules.md)
  Technical scoring weights, composition heuristics, and crop logic reference.

- [`troubleshooting.md`](/home/renatobo/devel/pickinsta/docs/troubleshooting.md)
  Setup, dependency, and runtime troubleshooting across CLIP, Claude, YOLO, and Ollama workflows.

- [`ollama-server-setup.md`](/home/renatobo/devel/pickinsta/docs/ollama-server-setup.md)
  Practical setup guide for running Ollama locally or on a remote Linux/macOS host, including `pickinsta` environment variables and tuning knobs.

## Benchmarking and Analysis

- [`model-quality-speed-comparison.md`](/home/renatobo/devel/pickinsta/docs/model-quality-speed-comparison.md)
  Manual comparison notes for model quality and throughput tradeoffs.

- [`ollama-model-speed-benchmark-report-serverone.md`](/home/renatobo/devel/pickinsta/docs/ollama-model-speed-benchmark-report-serverone.md)
  Example benchmark results captured on a specific host.

## Assets

- [`assets/pipeline-high-level.svg`](/home/renatobo/devel/pickinsta/docs/assets/pipeline-high-level.svg)
  Pipeline diagram used by the main README.

## Suggested Reading Order

1. Start with [`README.md`](/home/renatobo/devel/pickinsta/README.md) for install and normal CLI usage.
2. Use [`composition-rules.md`](/home/renatobo/devel/pickinsta/docs/composition-rules.md) when changing scoring or crop behavior.
3. Use [`ollama-server-setup.md`](/home/renatobo/devel/pickinsta/docs/ollama-server-setup.md) when deploying self-hosted vision scoring.
4. Use [`troubleshooting.md`](/home/renatobo/devel/pickinsta/docs/troubleshooting.md) when installs, model downloads, or scorer calls fail.

