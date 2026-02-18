# Cropping Regression Fixtures

This folder contains fixed fixtures for regression tests around crop behavior
(front-preserve, safe, and top-preserve decisions).

- `dsc5365_original.jpeg`
- `dsc5365_front_preserve_expected.jpg`
- `dsc5897_original.jpeg`
- `dsc5897_front_preserve_expected.jpg`
- `dsc09054_original.jpg`
- `dsc09054_front_preserve_expected.jpg`
- `dsc5660_original.jpeg`
- `dsc5660_front_preserve_expected.jpg`

`tests/test_cropping_regression.py` runs `smart_crop(...)` on each original and fails if
the result becomes too dissimilar from the expected reference crop.
