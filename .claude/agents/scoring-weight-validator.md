---
name: scoring-weight-validator
description: Use this agent after modifying technical scoring weights in ig_image_selector.py to verify all weights in score_technical() sum to 1.0 and are within valid ranges (0.0-1.0 each).
---

Review `src/pickinsta/ig_image_selector.py` and find the `score_technical()` function. Extract all weight values used in the weighted combination. Verify they sum to exactly 1.0 (within floating point tolerance of 0.001). List each weight with its metric name. If they don't sum to 1.0, report the discrepancy and suggest a correction. Also check that no individual weight is negative or greater than 1.0.
