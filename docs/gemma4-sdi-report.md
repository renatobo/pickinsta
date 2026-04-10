# Model Benchmark Report (Speed + Quality)

- Generated: `2026-04-10T11:17:53`
- Input folder: `/Users/renatobo/development/pickinsta/input`
- Candidates scored per run: `42`
- Runs per variant: `1`
- Warmup enabled: `True` (1 image + 10s wait)
- Ollama base URL: `http://localhost:11434`
- Ollama concurrency: `2`
- Ollama max retries: `2`
- Ollama keep_alive: `15m`

## Findings

### gemma4:e4b-it-q8_0 — Q8 quantization does not help

The Q8 variant (`gemma4:e4b-it-q8_0`) was benchmarked against the base `gemma4:e4b` across all supported prompt variants with the format-schema fix applied. Results are conclusive:

| Model | Best SDI | Best unique tuples | Avg sec/img (best variant) |
|---|---:|---:|---:|
| `gemma4:e4b` | **0.19** | **19** | 22.69 |
| `gemma4:e4b-it-q8_0` | 0.09 | 12 | 33.64 |

- **Q8 is ~48% slower** (33.6 vs 22.7 s/img at best) with no throughput benefit on this hardware.
- **Q8 scores worse on every prompt variant**: best SDI is 0.09 vs 0.19 for the base model — half the score differentiation.
- The Q8 build uses 11 GB vs 9.6 GB and still underperforms; the additional memory footprint has no payoff.

**Verdict: do not use `gemma4:e4b-it-q8_0`.** The base `gemma4:e4b` with `claude+system` is strictly better on both speed and quality.

### Prompt variant — claude and system dropped

`claude` (SDI 0.00) and `system` (SDI 0.00) were removed from the supported variants in v1.1.1.
Only `default` (SDI 0.17) and `claude+system` (SDI 0.19, **default**) remain.

### Format-schema fix (v1.1.0) resolved tier-collapse

The original `gemma4-benchmark-report.md` concluded that tier-collapse was a model capacity issue requiring a larger model. That diagnosis was wrong: the root cause was the Ollama `format` schema parameter, which silently collapses all criteria to a single tier on Gemma 4 regardless of prompt. Removing the schema (now applied to all Gemma 4 models) raised SDI from 0.00 to 0.17–0.19 on the same 4B model.

---

## Speed Summary

| Variant | Scorer | Model | YOLO | Avg sec/img | Avg imgs/min | Avg duration (s) | Avg failures/run | SDI | Unique tuples | Speed vs fastest |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| gemma4:e4b \| yolo=off \| prompt=claude+system | ollama | gemma4:e4b | off | 22.69 | 2.64 | 952.96 | 0.00 | 0.19 | 19 | 1.00x |
| gemma4:e4b \| yolo=off \| prompt=system | ollama | gemma4:e4b | off | 22.81 | 2.63 | 958.17 | 0.00 | 0.00 | 5 | 1.01x |
| gemma4:e4b \| yolo=off | ollama | gemma4:e4b | off | 24.17 | 2.48 | 1015.25 | 0.00 | 0.17 | 18 | 1.07x |
| gemma4:e4b-it-q8_0 \| yolo=off \| prompt=system | ollama | gemma4:e4b-it-q8_0 | off | 27.75 | 2.16 | 1165.44 | 0.00 | 0.00 | 3 | 1.22x |
| gemma4:e4b \| yolo=off \| prompt=claude | ollama | gemma4:e4b | off | 28.34 | 2.12 | 1190.10 | 0.00 | 0.00 | 8 | 1.25x |
| gemma4:e4b-it-q8_0 \| yolo=off \| prompt=claude+system | ollama | gemma4:e4b-it-q8_0 | off | 33.64 | 1.78 | 1412.82 | 0.00 | 0.09 | 12 | 1.48x |
| gemma4:e4b-it-q8_0 \| yolo=off | ollama | gemma4:e4b-it-q8_0 | off | 34.25 | 1.75 | 1438.47 | 0.00 | 0.04 | 10 | 1.51x |
| gemma4:e4b-it-q8_0 \| yolo=off \| prompt=claude | ollama | gemma4:e4b-it-q8_0 | off | 34.79 | 1.72 | 1461.28 | 0.00 | 0.00 | 6 | 1.53x |

## Per-run Timing

| Variant | Run | Duration (s) | Sec/img | Imgs/min | Failures |
|---|---:|---:|---:|---:|---:|
| gemma4:e4b \| yolo=off | 1 | 1015.25 | 24.17 | 2.48 | 0 |
| gemma4:e4b \| yolo=off \| prompt=claude | 1 | 1190.10 | 28.34 | 2.12 | 0 |
| gemma4:e4b \| yolo=off \| prompt=system | 1 | 958.17 | 22.81 | 2.63 | 0 |
| gemma4:e4b \| yolo=off \| prompt=claude+system | 1 | 952.96 | 22.69 | 2.64 | 0 |
| gemma4:e4b-it-q8_0 \| yolo=off | 1 | 1438.47 | 34.25 | 1.75 | 0 |
| gemma4:e4b-it-q8_0 \| yolo=off \| prompt=claude | 1 | 1461.28 | 34.79 | 1.72 | 0 |
| gemma4:e4b-it-q8_0 \| yolo=off \| prompt=system | 1 | 1165.44 | 27.75 | 2.16 | 0 |
| gemma4:e4b-it-q8_0 \| yolo=off \| prompt=claude+system | 1 | 1412.82 | 33.64 | 1.78 | 0 |

## Image-by-Image Score Comparison (Run 1)

| Image | gemma4:e4b \| yolo=off final | gemma4:e4b \| yolo=off rank | gemma4:e4b \| yolo=off \| prompt=claude final | gemma4:e4b \| yolo=off \| prompt=claude rank | gemma4:e4b \| yolo=off \| prompt=system final | gemma4:e4b \| yolo=off \| prompt=system rank | gemma4:e4b \| yolo=off \| prompt=claude+system final | gemma4:e4b \| yolo=off \| prompt=claude+system rank | gemma4:e4b-it-q8_0 \| yolo=off final | gemma4:e4b-it-q8_0 \| yolo=off rank | gemma4:e4b-it-q8_0 \| yolo=off \| prompt=claude final | gemma4:e4b-it-q8_0 \| yolo=off \| prompt=claude rank | gemma4:e4b-it-q8_0 \| yolo=off \| prompt=system final | gemma4:e4b-it-q8_0 \| yolo=off \| prompt=system rank | gemma4:e4b-it-q8_0 \| yolo=off \| prompt=claude+system final | gemma4:e4b-it-q8_0 \| yolo=off \| prompt=claude+system rank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1-Around the Pits - 1-Around the Pits - CVR_0012_Mar1025_702AM_CaliPhoto.jpg | 0.0494 | 41 | 0.0494 | 41 | 0.1642 | 41 | 0.0599 | 39 | 0.0494 | 41 | 0.0599 | 40 | 0.1642 | 40 | 0.0494 | 40 |
| 1-Around the Pits - 1-Around the Pits - CVR_0025_Mar1025_728AM_CaliPhoto.jpg | 0.7245 | 17 | 0.1801 | 33 | 0.3507 | 20 | 0.1801 | 33 | 0.1801 | 28 | 0.1801 | 29 | 0.1801 | 25 | 0.1801 | 29 |
| 1-Around the Pits - CVR_0098_Mar1025_802AM_CaliPhoto.jpg | 0.5897 | 21 | 0.5897 | 20 | 0.4003 | 13 | 0.5897 | 21 | 0.2090 | 20 | 0.2090 | 16 | 0.2090 | 4 | 0.0756 | 34 |
| 1-Around the Pits - CVR_0792_Mar1025_832AM_CaliPhoto.jpg | 0.8307 | 1 | 0.8307 | 2 | 0.1927 | 32 | 0.7607 | 9 | 0.7607 | 6 | 0.7607 | 6 | 0.1927 | 11 | 0.1927 | 21 |
| 2-Group Photo - Raffle - CVR_7677_Mar1025_1249PM_CaliPhoto.jpg | 0.0683 | 35 | 0.1840 | 31 | 0.3574 | 18 | 0.0683 | 35 | 0.0683 | 36 | 0.0683 | 36 | 0.1840 | 21 | 0.0473 | 41 |
| 2-Group Photo - Raffle - CVR_7682_Mar1025_1249PM_CaliPhoto.jpg | 0.0644 | 36 | 0.0556 | 38 | 0.3484 | 22 | 0.0661 | 36 | 0.1787 | 30 | 0.1787 | 30 | 0.1787 | 28 | 0.0451 | 42 |
| AB Group - Session 1 (Turn 16 Entry) - CVR_0148_Mar1025_815AM_CaliPhoto.jpg | 0.7457 | 15 | 0.7457 | 11 | 0.1875 | 34 | 0.7457 | 11 | 0.7457 | 9 | 0.7457 | 8 | 0.1875 | 17 | 0.1875 | 23 |
| AB Group - Session 1 (Turn 16 Entry) - CVR_0230_Mar1025_817AM_CaliPhoto.jpg | 0.8058 | 4 | 0.7358 | 13 | 0.1840 | 36 | 0.1840 | 32 | 0.5326 | 12 | 0.1840 | 24 | 0.1840 | 20 | 0.1840 | 26 |
| AB Group - Session 2 (Turn 11) - CVR_1047_Mar1025_908AM_CaliPhoto.jpg | 0.7508 | 14 | 0.7275 | 16 | 0.7275 | 3 | 0.7508 | 10 | 0.1811 | 27 | 0.1811 | 28 | 0.7275 | 1 | 0.1811 | 28 |
| AB Group - Session 4 (Turn 16) - CVR_6545_Mar1025_1122AM_CaliPhoto.jpg | 0.7775 | 9 | 0.8475 | 1 | 0.5660 | 5 | 0.7775 | 8 | 0.7775 | 5 | 0.8475 | 2 | 0.1986 | 10 | 0.8475 | 1 |
| C Group - Session 2 Back Straight Speed Pans - CVR_2628_Mar1025_943AM_CaliPhoto.jpg | 0.3929 | 25 | 0.3929 | 23 | 0.3929 | 14 | 0.3929 | 28 | 0.2047 | 21 | 0.5799 | 11 | 0.3929 | 2 | 0.3929 | 13 |
| C Group - Session 2 Back Straight Speed Pans - CVR_2850_Mar1025_946AM_CaliPhoto.jpg | 0.7898 | 7 | 0.7898 | 6 | 0.7898 | 1 | 0.7898 | 5 | 0.2029 | 23 | 0.2029 | 17 | 0.2029 | 6 | 0.7898 | 2 |
| C Group - Session 3 (Turn 9 and 8) - CVR_4982_Mar1025_1040AM_CaliPhoto.jpg | 0.5115 | 24 | 0.7093 | 18 | 0.5115 | 10 | 0.5115 | 25 | 0.0609 | 38 | 0.7093 | 9 | 0.1748 | 31 | 0.5115 | 11 |
| DSC-5436-NaraMedia.jpeg | 0.7136 | 19 | 0.5429 | 21 | 0.3652 | 17 | 0.7136 | 15 | 0.3652 | 15 | 0.1885 | 21 | 0.1885 | 16 | 0.5429 | 10 |
| IG - renatobo - March 10 DROC Track Day-26 HDR.jpeg | 0.7030 | 20 | 0.3378 | 27 | 0.3378 | 25 | 0.7030 | 16 | 0.1726 | 33 | 0.7030 | 10 | 0.1726 | 36 | 0.1726 | 31 |
| IG - renatobo - March 10 DROC Track Day-28.jpeg | 0.0428 | 42 | 0.0428 | 42 | 0.1733 | 39 | 0.0533 | 40 | 0.1733 | 32 | 0.1733 | 34 | 0.1733 | 35 | 0.1733 | 30 |
| IG - renatobo - March 10 DROC Track Day-58 HDR.jpeg | 0.0555 | 38 | 0.0555 | 39 | 0.1785 | 38 | 0.0643 | 38 | 0.0555 | 40 | 0.0660 | 37 | 0.1785 | 29 | 0.0590 | 39 |
| IG - renatobo - lineup 1st session - March 10 DROC Track Day-01 HDR.jpeg | 0.3197 | 31 | 0.6729 | 19 | 0.3197 | 28 | 0.6729 | 19 | 0.4823 | 14 | 0.4823 | 15 | 0.1620 | 42 | 0.1620 | 33 |
| IG - renatobo - lineup 1st session - March 10 DROC Track Day-02 HDR.jpeg | 0.7772 | 10 | 0.7772 | 9 | 0.5098 | 11 | 0.6955 | 17 | 0.1740 | 31 | 0.1740 | 33 | 0.1740 | 33 | 0.7772 | 5 |
| IG - renatobo - lineup 1st session - March 10 DROC Track Day-12 HDR.jpeg | 0.7837 | 8 | 0.7837 | 8 | 0.3442 | 23 | 0.7837 | 6 | 0.7837 | 4 | 0.1763 | 31 | 0.1763 | 30 | 0.7837 | 3 |
| IG cali_carnivores - DSC00013.jpg | 0.7704 | 12 | 0.2002 | 28 | 0.2002 | 31 | 0.6537 | 20 | 0.7937 | 2 | 0.2002 | 19 | 0.2002 | 9 | 0.7820 | 4 |
| IG cali_carnivores - DSC09850.jpg | 0.3890 | 26 | 0.3890 | 24 | 0.2024 | 30 | 0.7184 | 14 | 0.2024 | 24 | 0.7884 | 5 | 0.2024 | 7 | 0.2024 | 19 |
| IG cali_carnivores - DSC09857.jpg | 0.7143 | 18 | 0.7376 | 12 | 0.5341 | 7 | 0.5341 | 23 | 0.5341 | 11 | 0.1847 | 23 | 0.1847 | 19 | 0.1847 | 25 |
| IG desmo.donna - IMG-1015-Donna.jpeg | 0.7320 | 16 | 0.7320 | 15 | 0.3552 | 19 | 0.7320 | 13 | 0.3552 | 16 | 0.1827 | 27 | 0.1827 | 24 | 0.7320 | 9 |
| IG desmo.donna - IMG-1080-Donna.jpeg | 0.8028 | 5 | 0.7328 | 14 | 0.5302 | 8 | 0.8028 | 2 | 0.7328 | 10 | 0.1830 | 26 | 0.1830 | 23 | 0.1830 | 27 |
| IG desmo.donna - IMG-1151-Donna.jpeg | 0.1669 | 33 | 0.0610 | 36 | 0.3281 | 26 | 0.0505 | 41 | 0.0400 | 42 | 0.0505 | 42 | 0.1669 | 38 | 0.0610 | 37 |
| IG guardiansvoice - IMG-4129-Delenian.jpeg | 0.5219 | 23 | 0.7224 | 17 | 0.5219 | 9 | 0.5219 | 24 | 0.1793 | 29 | 0.5219 | 13 | 0.1793 | 27 | 0.3494 | 14 |
| IG guardiansvoice - IMG-4160-Delenian.jpeg | 0.0643 | 37 | 0.0538 | 40 | 0.3413 | 24 | 0.0643 | 37 | 0.0643 | 37 | 0.1746 | 32 | 0.1746 | 32 | 0.0643 | 36 |
| IG kamiumitv - IMG-3981-Bao.jpeg | 0.8151 | 3 | 0.8151 | 4 | 0.1873 | 35 | 0.7451 | 12 | 0.1873 | 26 | 0.1873 | 22 | 0.1873 | 18 | 0.1873 | 24 |
| IG luckie.moto - 20250310-075123-luckie.moto.jpg | 0.7512 | 13 | 0.1894 | 29 | 0.5450 | 6 | 0.5450 | 22 | 0.7512 | 8 | 0.8212 | 4 | 0.1894 | 13 | 0.7512 | 7 |
| IG m92663m - IMG-6019-Mark Momot.jpeg | 0.0495 | 40 | 0.0705 | 35 | 0.1891 | 33 | 0.0390 | 42 | 0.1891 | 25 | 0.0600 | 39 | 0.1891 | 15 | 0.1891 | 22 |
| IG martangelenos - PXL_20250310_132329430.jpg | 0.1733 | 32 | 0.3391 | 26 | 0.0638 | 42 | 0.1733 | 34 | 0.3391 | 18 | 0.0533 | 41 | 0.1733 | 34 | 0.3391 | 15 |
| IG naramedia_official - DSC-5365-NaraMedia.jpeg | 0.8284 | 2 | 0.7934 | 5 | 0.2042 | 29 | 0.7817 | 7 | 0.2042 | 22 | 0.8634 | 1 | 0.2042 | 5 | 0.2042 | 18 |
| IG naramedia_official - DSC-5897-NaraMedia.jpeg | 0.8018 | 6 | 0.8251 | 3 | 0.7551 | 2 | 0.8018 | 3 | 0.7551 | 7 | 0.8251 | 3 | 0.1908 | 12 | 0.7551 | 6 |
| IG naramedia_official - DSC-5992-NaraMedia.jpeg | 0.7755 | 11 | 0.7871 | 7 | 0.5737 | 4 | 0.7988 | 4 | 0.7871 | 3 | 0.2020 | 18 | 0.2020 | 8 | 0.2020 | 20 |
| IG ocredevil - DROC-Track-Day-3-10-2-filippo pagan.jpeg | 0.0544 | 39 | 0.1678 | 34 | 0.1678 | 40 | 0.3296 | 30 | 0.3296 | 19 | 0.0614 | 38 | 0.1678 | 37 | 0.3296 | 16 |
| IG renatobo - IMG_5013.jpeg | 0.5316 | 22 | 0.1836 | 32 | 0.1836 | 37 | 0.8045 | 1 | 0.8045 | 1 | 0.1836 | 25 | 0.1836 | 22 | 0.7345 | 8 |
| IG renatobo - IMG_5014.jpeg | 0.3198 | 30 | 0.4824 | 22 | 0.3198 | 27 | 0.4357 | 27 | 0.4824 | 13 | 0.4824 | 14 | 0.1620 | 41 | 0.3198 | 17 |
| IMG_4984.jpeg | 0.3496 | 29 | 0.3496 | 25 | 0.3496 | 21 | 0.4754 | 26 | 0.3496 | 17 | 0.5221 | 12 | 0.1794 | 26 | 0.4848 | 12 |
| IMG_5012.jpeg | 0.1665 | 34 | 0.7558 | 10 | 0.4927 | 12 | 0.6858 | 18 | 0.1665 | 34 | 0.7558 | 7 | 0.1665 | 39 | 0.1665 | 32 |
| caliphotovideo - Group picture - DROC Track Day March 10 202500001.jpg | 0.3665 | 28 | 0.1893 | 30 | 0.3665 | 16 | 0.1893 | 31 | 0.0601 | 39 | 0.0706 | 35 | 0.1893 | 14 | 0.0706 | 35 |
| caliphotovideo - Group picture - DROC Track Day March 10 202500002.jpg | 0.3687 | 27 | 0.0607 | 37 | 0.3687 | 15 | 0.3687 | 29 | 0.0712 | 35 | 0.1906 | 20 | 0.3687 | 3 | 0.0607 | 38 |

## Ranked Quality Details (All Images)

_Each section below is from timed run 1 for that variant._

### gemma4:e4b | yolo=off

| Rank | Image | Final | Tech | Vision | Subject | Light | Color | Emotion | Scroll | Crop | Failed | One line |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 1-Around the Pits - CVR_0792_Mar1025_832AM_CaliPhoto.jpg | 0.8307 | 0.6688 | 54 | 9 | 9 | 9 | 9 | 9 | 9 | no | Vision scoring summary |
| 2 | IG naramedia_official - DSC-5365-NaraMedia.jpeg | 0.8284 | 0.7780 | 51 | 9 | 8 | 8 | 9 | 9 | 8 | no | Strong, low-angle shot with excellent subject isolation and dynamic posing, making it highly engaging for a motorcycle enthusiast feed. |
| 3 | IG kamiumitv - IMG-3981-Bao.jpeg | 0.8151 | 0.6171 | 54 | 9 | 9 | 9 | 9 | 9 | 9 | no | Vision scoring summary |
| 4 | AB Group - Session 1 (Turn 16 Entry) - CVR_0230_Mar1025_817AM_CaliPhoto.jpg | 0.8058 | 0.5860 | 54 | 9 | 9 | 9 | 9 | 9 | 9 | no | Vision scoring summary |
| 5 | IG desmo.donna - IMG-1080-Donna.jpeg | 0.8028 | 0.5760 | 54 | 9 | 9 | 9 | 9 | 9 | 9 | no | Vision scoring summary |
| 6 | IG naramedia_official - DSC-5897-NaraMedia.jpeg | 0.8018 | 0.6504 | 52 | 9 | 8 | 8 | 9 | 9 | 9 | no | Excellent action shot with strong leading lines and high energy, perfect for a Ducati enthusiast feed. |
| 7 | C Group - Session 2 Back Straight Speed Pans - CVR_2850_Mar1025_946AM_CaliPhoto.jpg | 0.7898 | 0.7661 | 48 | 8 | 7 | 8 | 9 | 8 | 8 | no | Excellent dynamic shot with strong motion blur and good color contrast, making it highly engaging for a motorcycle enthusiast feed. |
| 8 | IG - renatobo - lineup 1st session - March 10 DROC Track Day-12 HDR.jpeg | 0.7837 | 0.5123 | 54 | 9 | 9 | 9 | 9 | 9 | 9 | no | Vision scoring summary |
| 9 | AB Group - Session 4 (Turn 16) - CVR_6545_Mar1025_1122AM_CaliPhoto.jpg | 0.7775 | 0.7249 | 48 | 8 | 7 | 8 | 9 | 8 | 8 | no | Strong action shot with excellent color contrast against the muted background, making it highly engaging for a motorcycle enthusiast feed. |
| 10 | IG - renatobo - lineup 1st session - March 10 DROC Track Day-02 HDR.jpeg | 0.7772 | 0.4907 | 54 | 9 | 9 | 9 | 9 | 9 | 9 | no | Vision scoring summary |
| 11 | IG naramedia_official - DSC-5992-NaraMedia.jpeg | 0.7755 | 0.7571 | 47 | 8 | 6 | 7 | 8 | 8 | 8 | no | Strong motion and subject placement on a clean track provide excellent Instagram potential, though the lighting is slightly flat. |
| 12 | IG cali_carnivores - DSC00013.jpg | 0.7704 | 0.7401 | 47 | 8 | 7 | 8 | 8 | 8 | 8 | no | Strong, vibrant subject against a clean background, making it highly clickable, though the angle is slightly low and the background is a bit plain. |
| 13 | IG luckie.moto - 20250310-075123-luckie.moto.jpg | 0.7512 | 0.6374 | 48 | 8 | 8 | 8 | 8 | 8 | 8 | no | Vision scoring summary |
| 14 | AB Group - Session 2 (Turn 11) - CVR_1047_Mar1025_908AM_CaliPhoto.jpg | 0.7508 | 0.5583 | 50 | 9 | 7 | 8 | 9 | 9 | 8 | no | Excellent action shot with strong leading lines and vibrant color contrast, making it highly engaging for a Ducati enthusiast feed. |
| 15 | AB Group - Session 1 (Turn 16 Entry) - CVR_0148_Mar1025_815AM_CaliPhoto.jpg | 0.7457 | 0.6190 | 48 | 8 | 8 | 8 | 8 | 8 | 8 | no | Vision scoring summary |
| 16 | IG desmo.donna - IMG-1015-Donna.jpeg | 0.7320 | 0.5732 | 48 | 8 | 8 | 8 | 8 | 8 | 8 | no | Vision scoring summary |
| 17 | 1-Around the Pits - 1-Around the Pits - CVR_0025_Mar1025_728AM_CaliPhoto.jpg | 0.7245 | 0.5483 | 48 | 8 | 8 | 8 | 8 | 8 | 8 | no | Vision scoring summary |
| 18 | IG cali_carnivores - DSC09857.jpg | 0.7143 | 0.5921 | 46 | 8 | 7 | 7 | 8 | 8 | 8 | no | Strong action shot with good leading lines, though the background is somewhat distracting and the lighting isn't dramatic enough for a perfect score. |
| 19 | DSC-5436-NaraMedia.jpeg | 0.7136 | 0.6287 | 45 | 7 | 6 | 7 | 7 | 8 | 8 | no | Strong leading lines and good subject placement make this highly scroll-stopping, though the midday light is slightly flat. |
| 20 | IG - renatobo - March 10 DROC Track Day-26 HDR.jpeg | 0.7030 | 0.4768 | 48 | 8 | 8 | 8 | 8 | 8 | 8 | no | Vision scoring summary |
| 21 | 1-Around the Pits - CVR_0098_Mar1025_802AM_CaliPhoto.jpg | 0.5897 | 0.8237 | 42 | 7 | 7 | 7 | 7 | 7 | 7 | no | Vision scoring summary |
| 22 | IG renatobo - IMG_5013.jpeg | 0.5316 | 0.5817 | 42 | 7 | 7 | 7 | 7 | 7 | 7 | no | Vision scoring summary |
| 23 | IG guardiansvoice - IMG-4129-Delenian.jpeg | 0.5219 | 0.5413 | 42 | 7 | 7 | 7 | 7 | 7 | 7 | no | Vision scoring summary |
| 24 | C Group - Session 3 (Turn 9 and 8) - CVR_4982_Mar1025_1040AM_CaliPhoto.jpg | 0.5115 | 0.4978 | 42 | 7 | 7 | 7 | 7 | 7 | 7 | no | Vision scoring summary |
| 25 | C Group - Session 2 Back Straight Speed Pans - CVR_2628_Mar1025_943AM_CaliPhoto.jpg | 0.3929 | 0.7827 | 36 | 6 | 6 | 6 | 6 | 6 | 6 | no | Vision scoring summary |
| 26 | IG cali_carnivores - DSC09850.jpg | 0.3890 | 0.7612 | 36 | 6 | 6 | 6 | 6 | 6 | 6 | no | Vision scoring summary |
| 27 | caliphotovideo - Group picture - DROC Track Day March 10 202500002.jpg | 0.3687 | 0.6484 | 36 | 6 | 6 | 6 | 6 | 6 | 6 | no | Vision scoring summary |
| 28 | caliphotovideo - Group picture - DROC Track Day March 10 202500001.jpg | 0.3665 | 0.6360 | 36 | 6 | 6 | 6 | 6 | 6 | 6 | no | Vision scoring summary |
| 29 | IMG_4984.jpeg | 0.3496 | 0.5421 | 36 | 6 | 6 | 6 | 6 | 6 | 6 | no | Vision scoring summary |
| 30 | IG renatobo - IMG_5014.jpeg | 0.3198 | 0.3765 | 36 | 6 | 6 | 6 | 6 | 6 | 6 | no | Vision scoring summary |
| 31 | IG - renatobo - lineup 1st session - March 10 DROC Track Day-01 HDR.jpeg | 0.3197 | 0.3764 | 36 | 6 | 6 | 6 | 6 | 6 | 6 | no | Vision scoring summary |
| 32 | IG martangelenos - PXL_20250310_132329430.jpg | 0.1733 | 0.4839 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 33 | IG desmo.donna - IMG-1151-Donna.jpeg | 0.1669 | 0.4230 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 34 | IMG_5012.jpeg | 0.1665 | 0.4195 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 35 | 2-Group Photo - Raffle - CVR_7677_Mar1025_1249PM_CaliPhoto.jpg | 0.0683 | 0.5853 | 24 | 4 | 4 | 4 | 4 | 4 | 4 | no | Vision scoring summary |
| 36 | 2-Group Photo - Raffle - CVR_7682_Mar1025_1249PM_CaliPhoto.jpg | 0.0644 | 0.5357 | 23 | 4 | 4 | 4 | 4 | 4 | 3 | no | Vision scoring summary |
| 37 | IG guardiansvoice - IMG-4160-Delenian.jpeg | 0.0643 | 0.4960 | 24 | 4 | 4 | 4 | 4 | 4 | 4 | no | Vision scoring summary |
| 38 | IG - renatobo - March 10 DROC Track Day-58 HDR.jpeg | 0.0555 | 0.5335 | 18 | 3 | 3 | 3 | 3 | 3 | 3 | no | Vision scoring summary |
| 39 | IG ocredevil - DROC-Track-Day-3-10-2-filippo pagan.jpeg | 0.0544 | 0.4312 | 20 | 3 | 4 | 4 | 4 | 4 | 1 | no | Vision scoring summary |
| 40 | IG m92663m - IMG-6019-Mark Momot.jpeg | 0.0495 | 0.6338 | 12 | 2 | 2 | 2 | 2 | 2 | 2 | no | Vision scoring summary |
| 41 | 1-Around the Pits - 1-Around the Pits - CVR_0012_Mar1025_702AM_CaliPhoto.jpg | 0.0494 | 0.3975 | 18 | 3 | 3 | 3 | 3 | 3 | 3 | no | Vision scoring summary |
| 42 | IG - renatobo - March 10 DROC Track Day-28.jpeg | 0.0428 | 0.4833 | 12 | 2 | 2 | 2 | 2 | 2 | 2 | no | Vision scoring summary |

### gemma4:e4b | yolo=off | prompt=claude

| Rank | Image | Final | Tech | Vision | Subject | Light | Color | Emotion | Scroll | Crop | Failed | One line |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 | AB Group - Session 4 (Turn 16) - CVR_6545_Mar1025_1122AM_CaliPhoto.jpg | 0.8475 | 0.7249 | 54 | 9 | 9 | 9 | 9 | 9 | 9 | no | Vision scoring summary |
| 2 | 1-Around the Pits - CVR_0792_Mar1025_832AM_CaliPhoto.jpg | 0.8307 | 0.6688 | 54 | 9 | 9 | 9 | 9 | 9 | 9 | no | Vision scoring summary |
| 3 | IG naramedia_official - DSC-5897-NaraMedia.jpeg | 0.8251 | 0.6504 | 54 | 9 | 9 | 9 | 9 | 9 | 9 | no | Vision scoring summary |
| 4 | IG kamiumitv - IMG-3981-Bao.jpeg | 0.8151 | 0.6171 | 54 | 9 | 9 | 9 | 9 | 9 | 9 | no | Vision scoring summary |
| 5 | IG naramedia_official - DSC-5365-NaraMedia.jpeg | 0.7934 | 0.7780 | 48 | 8 | 8 | 8 | 8 | 8 | 8 | no | Vision scoring summary |
| 6 | C Group - Session 2 Back Straight Speed Pans - CVR_2850_Mar1025_946AM_CaliPhoto.jpg | 0.7898 | 0.7661 | 48 | 8 | 8 | 8 | 8 | 8 | 8 | no | Vision scoring summary |
| 7 | IG naramedia_official - DSC-5992-NaraMedia.jpeg | 0.7871 | 0.7571 | 48 | 8 | 8 | 8 | 8 | 8 | 8 | no | Vision scoring summary |
| 8 | IG - renatobo - lineup 1st session - March 10 DROC Track Day-12 HDR.jpeg | 0.7837 | 0.5123 | 54 | 9 | 9 | 9 | 9 | 9 | 9 | no | Vision scoring summary |
| 9 | IG - renatobo - lineup 1st session - March 10 DROC Track Day-02 HDR.jpeg | 0.7772 | 0.4907 | 54 | 9 | 9 | 9 | 9 | 9 | 9 | no | Vision scoring summary |
| 10 | IMG_5012.jpeg | 0.7558 | 0.4195 | 54 | 9 | 9 | 9 | 9 | 9 | 9 | no | Vision scoring summary |
| 11 | AB Group - Session 1 (Turn 16 Entry) - CVR_0148_Mar1025_815AM_CaliPhoto.jpg | 0.7457 | 0.6190 | 48 | 8 | 8 | 8 | 8 | 8 | 8 | no | Vision scoring summary |
| 12 | IG cali_carnivores - DSC09857.jpg | 0.7376 | 0.5921 | 48 | 8 | 8 | 8 | 8 | 8 | 8 | no | Vision scoring summary |
| 13 | AB Group - Session 1 (Turn 16 Entry) - CVR_0230_Mar1025_817AM_CaliPhoto.jpg | 0.7358 | 0.5860 | 48 | 8 | 8 | 8 | 8 | 8 | 8 | no | Vision scoring summary |
| 14 | IG desmo.donna - IMG-1080-Donna.jpeg | 0.7328 | 0.5760 | 48 | 8 | 8 | 8 | 8 | 8 | 8 | no | Vision scoring summary |
| 15 | IG desmo.donna - IMG-1015-Donna.jpeg | 0.7320 | 0.5732 | 48 | 8 | 8 | 8 | 8 | 8 | 8 | no | Vision scoring summary |
| 16 | AB Group - Session 2 (Turn 11) - CVR_1047_Mar1025_908AM_CaliPhoto.jpg | 0.7275 | 0.5583 | 48 | 8 | 8 | 8 | 8 | 8 | 8 | no | Vision scoring summary |
| 17 | IG guardiansvoice - IMG-4129-Delenian.jpeg | 0.7224 | 0.5413 | 48 | 8 | 8 | 8 | 8 | 8 | 8 | no | Vision scoring summary |
| 18 | C Group - Session 3 (Turn 9 and 8) - CVR_4982_Mar1025_1040AM_CaliPhoto.jpg | 0.7093 | 0.4978 | 48 | 8 | 8 | 8 | 8 | 8 | 8 | no | Vision scoring summary |
| 19 | IG - renatobo - lineup 1st session - March 10 DROC Track Day-01 HDR.jpeg | 0.6729 | 0.3764 | 48 | 8 | 8 | 8 | 8 | 8 | 8 | no | Vision scoring summary |
| 20 | 1-Around the Pits - CVR_0098_Mar1025_802AM_CaliPhoto.jpg | 0.5897 | 0.8237 | 42 | 7 | 7 | 7 | 7 | 7 | 7 | no | Vision scoring summary |
| 21 | DSC-5436-NaraMedia.jpeg | 0.5429 | 0.6287 | 42 | 7 | 7 | 7 | 7 | 7 | 7 | no | Vision scoring summary |
| 22 | IG renatobo - IMG_5014.jpeg | 0.4824 | 0.3765 | 42 | 7 | 7 | 7 | 7 | 7 | 7 | no | Vision scoring summary |
| 23 | C Group - Session 2 Back Straight Speed Pans - CVR_2628_Mar1025_943AM_CaliPhoto.jpg | 0.3929 | 0.7827 | 36 | 6 | 6 | 6 | 6 | 6 | 6 | no | Vision scoring summary |
| 24 | IG cali_carnivores - DSC09850.jpg | 0.3890 | 0.7612 | 36 | 6 | 6 | 6 | 6 | 6 | 6 | no | Vision scoring summary |
| 25 | IMG_4984.jpeg | 0.3496 | 0.5421 | 36 | 6 | 6 | 6 | 6 | 6 | 6 | no | Vision scoring summary |
| 26 | IG martangelenos - PXL_20250310_132329430.jpg | 0.3391 | 0.4839 | 36 | 6 | 6 | 6 | 6 | 6 | 6 | no | Vision scoring summary |
| 27 | IG - renatobo - March 10 DROC Track Day-26 HDR.jpeg | 0.3378 | 0.4768 | 36 | 6 | 6 | 6 | 6 | 6 | 6 | no | Vision scoring summary |
| 28 | IG cali_carnivores - DSC00013.jpg | 0.2002 | 0.7401 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 29 | IG luckie.moto - 20250310-075123-luckie.moto.jpg | 0.1894 | 0.6374 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 30 | caliphotovideo - Group picture - DROC Track Day March 10 202500001.jpg | 0.1893 | 0.6360 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 31 | 2-Group Photo - Raffle - CVR_7677_Mar1025_1249PM_CaliPhoto.jpg | 0.1840 | 0.5853 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 32 | IG renatobo - IMG_5013.jpeg | 0.1836 | 0.5817 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 33 | 1-Around the Pits - 1-Around the Pits - CVR_0025_Mar1025_728AM_CaliPhoto.jpg | 0.1801 | 0.5483 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 34 | IG ocredevil - DROC-Track-Day-3-10-2-filippo pagan.jpeg | 0.1678 | 0.4312 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 35 | IG m92663m - IMG-6019-Mark Momot.jpeg | 0.0705 | 0.6338 | 24 | 4 | 4 | 4 | 4 | 4 | 4 | no | Vision scoring summary |
| 36 | IG desmo.donna - IMG-1151-Donna.jpeg | 0.0610 | 0.4230 | 24 | 4 | 4 | 4 | 4 | 4 | 4 | no | Vision scoring summary |
| 37 | caliphotovideo - Group picture - DROC Track Day March 10 202500002.jpg | 0.0607 | 0.6484 | 18 | 3 | 3 | 3 | 3 | 3 | 3 | no | Vision scoring summary |
| 38 | 2-Group Photo - Raffle - CVR_7682_Mar1025_1249PM_CaliPhoto.jpg | 0.0556 | 0.5357 | 18 | 3 | 3 | 3 | 3 | 3 | 3 | no | Vision scoring summary |
| 39 | IG - renatobo - March 10 DROC Track Day-58 HDR.jpeg | 0.0555 | 0.5335 | 18 | 3 | 3 | 3 | 3 | 3 | 3 | no | Vision scoring summary |
| 40 | IG guardiansvoice - IMG-4160-Delenian.jpeg | 0.0538 | 0.4960 | 18 | 3 | 3 | 3 | 3 | 3 | 3 | no | Vision scoring summary |
| 41 | 1-Around the Pits - 1-Around the Pits - CVR_0012_Mar1025_702AM_CaliPhoto.jpg | 0.0494 | 0.3975 | 18 | 3 | 3 | 3 | 3 | 3 | 3 | no | Vision scoring summary |
| 42 | IG - renatobo - March 10 DROC Track Day-28.jpeg | 0.0428 | 0.4833 | 12 | 2 | 2 | 2 | 2 | 2 | 2 | no | Vision scoring summary |

### gemma4:e4b | yolo=off | prompt=system

| Rank | Image | Final | Tech | Vision | Subject | Light | Color | Emotion | Scroll | Crop | Failed | One line |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 | C Group - Session 2 Back Straight Speed Pans - CVR_2850_Mar1025_946AM_CaliPhoto.jpg | 0.7898 | 0.7661 | 48 | 8 | 8 | 8 | 8 | 8 | 8 | no | Vision scoring summary |
| 2 | IG naramedia_official - DSC-5897-NaraMedia.jpeg | 0.7551 | 0.6504 | 48 | 8 | 8 | 8 | 8 | 8 | 8 | no | Vision scoring summary |
| 3 | AB Group - Session 2 (Turn 11) - CVR_1047_Mar1025_908AM_CaliPhoto.jpg | 0.7275 | 0.5583 | 48 | 8 | 8 | 8 | 8 | 8 | 8 | no | Vision scoring summary |
| 4 | IG naramedia_official - DSC-5992-NaraMedia.jpeg | 0.5737 | 0.7571 | 42 | 7 | 7 | 7 | 7 | 7 | 7 | no | Vision scoring summary |
| 5 | AB Group - Session 4 (Turn 16) - CVR_6545_Mar1025_1122AM_CaliPhoto.jpg | 0.5660 | 0.7249 | 42 | 7 | 7 | 7 | 7 | 7 | 7 | no | Vision scoring summary |
| 6 | IG luckie.moto - 20250310-075123-luckie.moto.jpg | 0.5450 | 0.6374 | 42 | 7 | 7 | 7 | 7 | 7 | 7 | no | Vision scoring summary |
| 7 | IG cali_carnivores - DSC09857.jpg | 0.5341 | 0.5921 | 42 | 7 | 7 | 7 | 7 | 7 | 7 | no | Vision scoring summary |
| 8 | IG desmo.donna - IMG-1080-Donna.jpeg | 0.5302 | 0.5760 | 42 | 7 | 7 | 7 | 7 | 7 | 7 | no | Vision scoring summary |
| 9 | IG guardiansvoice - IMG-4129-Delenian.jpeg | 0.5219 | 0.5413 | 42 | 7 | 7 | 7 | 7 | 7 | 7 | no | Vision scoring summary |
| 10 | C Group - Session 3 (Turn 9 and 8) - CVR_4982_Mar1025_1040AM_CaliPhoto.jpg | 0.5115 | 0.4978 | 42 | 7 | 7 | 7 | 7 | 7 | 7 | no | Vision scoring summary |
| 11 | IG - renatobo - lineup 1st session - March 10 DROC Track Day-02 HDR.jpeg | 0.5098 | 0.4907 | 42 | 7 | 7 | 7 | 7 | 7 | 7 | no | Vision scoring summary |
| 12 | IMG_5012.jpeg | 0.4927 | 0.4195 | 42 | 7 | 7 | 7 | 7 | 7 | 7 | no | Vision scoring summary |
| 13 | 1-Around the Pits - CVR_0098_Mar1025_802AM_CaliPhoto.jpg | 0.4003 | 0.8237 | 36 | 6 | 6 | 6 | 6 | 6 | 6 | no | Vision scoring summary |
| 14 | C Group - Session 2 Back Straight Speed Pans - CVR_2628_Mar1025_943AM_CaliPhoto.jpg | 0.3929 | 0.7827 | 36 | 6 | 6 | 6 | 6 | 6 | 6 | no | Vision scoring summary |
| 15 | caliphotovideo - Group picture - DROC Track Day March 10 202500002.jpg | 0.3687 | 0.6484 | 36 | 6 | 6 | 6 | 6 | 6 | 6 | no | Vision scoring summary |
| 16 | caliphotovideo - Group picture - DROC Track Day March 10 202500001.jpg | 0.3665 | 0.6360 | 36 | 6 | 6 | 6 | 6 | 6 | 6 | no | Vision scoring summary |
| 17 | DSC-5436-NaraMedia.jpeg | 0.3652 | 0.6287 | 36 | 6 | 6 | 6 | 6 | 6 | 6 | no | Vision scoring summary |
| 18 | 2-Group Photo - Raffle - CVR_7677_Mar1025_1249PM_CaliPhoto.jpg | 0.3574 | 0.5853 | 36 | 6 | 6 | 6 | 6 | 6 | 6 | no | Vision scoring summary |
| 19 | IG desmo.donna - IMG-1015-Donna.jpeg | 0.3552 | 0.5732 | 36 | 6 | 6 | 6 | 6 | 6 | 6 | no | Vision scoring summary |
| 20 | 1-Around the Pits - 1-Around the Pits - CVR_0025_Mar1025_728AM_CaliPhoto.jpg | 0.3507 | 0.5483 | 36 | 6 | 6 | 6 | 6 | 6 | 6 | no | Vision scoring summary |
| 21 | IMG_4984.jpeg | 0.3496 | 0.5421 | 36 | 6 | 6 | 6 | 6 | 6 | 6 | no | Vision scoring summary |
| 22 | 2-Group Photo - Raffle - CVR_7682_Mar1025_1249PM_CaliPhoto.jpg | 0.3484 | 0.5357 | 36 | 6 | 6 | 6 | 6 | 6 | 6 | no | Vision scoring summary |
| 23 | IG - renatobo - lineup 1st session - March 10 DROC Track Day-12 HDR.jpeg | 0.3442 | 0.5123 | 36 | 6 | 6 | 6 | 6 | 6 | 6 | no | Vision scoring summary |
| 24 | IG guardiansvoice - IMG-4160-Delenian.jpeg | 0.3413 | 0.4960 | 36 | 6 | 6 | 6 | 6 | 6 | 6 | no | Vision scoring summary |
| 25 | IG - renatobo - March 10 DROC Track Day-26 HDR.jpeg | 0.3378 | 0.4768 | 36 | 6 | 6 | 6 | 6 | 6 | 6 | no | Vision scoring summary |
| 26 | IG desmo.donna - IMG-1151-Donna.jpeg | 0.3281 | 0.4230 | 36 | 6 | 6 | 6 | 6 | 6 | 6 | no | Vision scoring summary |
| 27 | IG renatobo - IMG_5014.jpeg | 0.3198 | 0.3765 | 36 | 6 | 6 | 6 | 6 | 6 | 6 | no | Vision scoring summary |
| 28 | IG - renatobo - lineup 1st session - March 10 DROC Track Day-01 HDR.jpeg | 0.3197 | 0.3764 | 36 | 6 | 6 | 6 | 6 | 6 | 6 | no | Vision scoring summary |
| 29 | IG naramedia_official - DSC-5365-NaraMedia.jpeg | 0.2042 | 0.7780 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 30 | IG cali_carnivores - DSC09850.jpg | 0.2024 | 0.7612 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 31 | IG cali_carnivores - DSC00013.jpg | 0.2002 | 0.7401 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 32 | 1-Around the Pits - CVR_0792_Mar1025_832AM_CaliPhoto.jpg | 0.1927 | 0.6688 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 33 | IG m92663m - IMG-6019-Mark Momot.jpeg | 0.1891 | 0.6338 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 34 | AB Group - Session 1 (Turn 16 Entry) - CVR_0148_Mar1025_815AM_CaliPhoto.jpg | 0.1875 | 0.6190 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 35 | IG kamiumitv - IMG-3981-Bao.jpeg | 0.1873 | 0.6171 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 36 | AB Group - Session 1 (Turn 16 Entry) - CVR_0230_Mar1025_817AM_CaliPhoto.jpg | 0.1840 | 0.5860 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 37 | IG renatobo - IMG_5013.jpeg | 0.1836 | 0.5817 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 38 | IG - renatobo - March 10 DROC Track Day-58 HDR.jpeg | 0.1785 | 0.5335 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 39 | IG - renatobo - March 10 DROC Track Day-28.jpeg | 0.1733 | 0.4833 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 40 | IG ocredevil - DROC-Track-Day-3-10-2-filippo pagan.jpeg | 0.1678 | 0.4312 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 41 | 1-Around the Pits - 1-Around the Pits - CVR_0012_Mar1025_702AM_CaliPhoto.jpg | 0.1642 | 0.3975 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 42 | IG martangelenos - PXL_20250310_132329430.jpg | 0.0638 | 0.4839 | 24 | 4 | 4 | 4 | 4 | 4 | 4 | no | Vision scoring summary |

### gemma4:e4b | yolo=off | prompt=claude+system

| Rank | Image | Final | Tech | Vision | Subject | Light | Color | Emotion | Scroll | Crop | Failed | One line |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 | IG renatobo - IMG_5013.jpeg | 0.8045 | 0.5817 | 54 | 9 | 9 | 9 | 9 | 9 | 9 | no | Vision scoring summary |
| 2 | IG desmo.donna - IMG-1080-Donna.jpeg | 0.8028 | 0.5760 | 54 | 9 | 9 | 9 | 9 | 9 | 9 | no | Vision scoring summary |
| 3 | IG naramedia_official - DSC-5897-NaraMedia.jpeg | 0.8018 | 0.6504 | 52 | 9 | 8 | 8 | 9 | 9 | 9 | no | Excellent dynamic shot with strong motion blur and vibrant subject contrast against the track, making it highly engaging for an enthusiast feed. |
| 4 | IG naramedia_official - DSC-5992-NaraMedia.jpeg | 0.7988 | 0.7571 | 49 | 8 | 7 | 8 | 9 | 9 | 8 | no | Excellent dynamic shot with strong leading lines and subject placement, making it highly engaging for a performance-focused audience. |
| 5 | C Group - Session 2 Back Straight Speed Pans - CVR_2850_Mar1025_946AM_CaliPhoto.jpg | 0.7898 | 0.7661 | 48 | 8 | 8 | 8 | 8 | 8 | 8 | no | Vision scoring summary |
| 6 | IG - renatobo - lineup 1st session - March 10 DROC Track Day-12 HDR.jpeg | 0.7837 | 0.5123 | 54 | 9 | 9 | 9 | 9 | 9 | 9 | no | Vision scoring summary |
| 7 | IG naramedia_official - DSC-5365-NaraMedia.jpeg | 0.7817 | 0.7780 | 47 | 8 | 7 | 8 | 8 | 8 | 8 | no | Strong composition with good color contrast and a powerful stance, making it highly suitable for an enthusiast feed. |
| 8 | AB Group - Session 4 (Turn 16) - CVR_6545_Mar1025_1122AM_CaliPhoto.jpg | 0.7775 | 0.7249 | 48 | 8 | 7 | 8 | 9 | 8 | 8 | no | Excellent action shot with strong color contrast and dynamic composition, perfect for an enthusiast feed. |
| 9 | 1-Around the Pits - CVR_0792_Mar1025_832AM_CaliPhoto.jpg | 0.7607 | 0.6688 | 48 | 8 | 8 | 8 | 8 | 8 | 8 | no | Vision scoring summary |
| 10 | AB Group - Session 2 (Turn 11) - CVR_1047_Mar1025_908AM_CaliPhoto.jpg | 0.7508 | 0.5583 | 50 | 9 | 7 | 8 | 9 | 9 | 8 | no | Excellent dynamic shot with strong leading lines and high energy, perfect for a Ducati enthusiast feed. |
| 11 | AB Group - Session 1 (Turn 16 Entry) - CVR_0148_Mar1025_815AM_CaliPhoto.jpg | 0.7457 | 0.6190 | 48 | 8 | 8 | 8 | 8 | 8 | 8 | no | Vision scoring summary |
| 12 | IG kamiumitv - IMG-3981-Bao.jpeg | 0.7451 | 0.6171 | 48 | 8 | 8 | 8 | 8 | 8 | 8 | no | Vision scoring summary |
| 13 | IG desmo.donna - IMG-1015-Donna.jpeg | 0.7320 | 0.5732 | 48 | 8 | 8 | 8 | 8 | 8 | 8 | no | Vision scoring summary |
| 14 | IG cali_carnivores - DSC09850.jpg | 0.7184 | 0.7612 | 42 | 7 | 6 | 7 | 7 | 7 | 8 | no | Good dynamic shot with strong leading lines on the track, though the midday light is slightly flat and the subject isn't perfectly framed for a vertical crop. |
| 15 | DSC-5436-NaraMedia.jpeg | 0.7136 | 0.6287 | 45 | 7 | 6 | 7 | 7 | 8 | 8 | no | Strong leading lines and good subject placement give this a high potential, though the midday light is slightly flat. |
| 16 | IG - renatobo - March 10 DROC Track Day-26 HDR.jpeg | 0.7030 | 0.4768 | 48 | 8 | 8 | 8 | 8 | 8 | 8 | no | Vision scoring summary |
| 17 | IG - renatobo - lineup 1st session - March 10 DROC Track Day-02 HDR.jpeg | 0.6955 | 0.4907 | 47 | 8 | 7 | 8 | 8 | 8 | 8 | no | Strong subject focus and good color contrast against the arid background, making it highly engaging for a motorcycle enthusiast feed. |
| 18 | IMG_5012.jpeg | 0.6858 | 0.4195 | 48 | 8 | 8 | 8 | 8 | 8 | 8 | no | Vision scoring summary |
| 19 | IG - renatobo - lineup 1st session - March 10 DROC Track Day-01 HDR.jpeg | 0.6729 | 0.3764 | 48 | 8 | 8 | 8 | 8 | 8 | 8 | no | Vision scoring summary |
| 20 | IG cali_carnivores - DSC00013.jpg | 0.6537 | 0.7401 | 37 | 8 | 6 | 7 | 8 | 8 | 8 | no | Strong subject focus and dynamic angle, though the lighting is flat and the composition could benefit from more negative space or leading lines. |
| 21 | 1-Around the Pits - CVR_0098_Mar1025_802AM_CaliPhoto.jpg | 0.5897 | 0.8237 | 42 | 7 | 7 | 7 | 7 | 7 | 7 | no | Vision scoring summary |
| 22 | IG luckie.moto - 20250310-075123-luckie.moto.jpg | 0.5450 | 0.6374 | 42 | 7 | 7 | 7 | 7 | 7 | 7 | no | Vision scoring summary |
| 23 | IG cali_carnivores - DSC09857.jpg | 0.5341 | 0.5921 | 42 | 7 | 7 | 7 | 7 | 7 | 7 | no | Vision scoring summary |
| 24 | IG guardiansvoice - IMG-4129-Delenian.jpeg | 0.5219 | 0.5413 | 42 | 7 | 7 | 7 | 7 | 7 | 7 | no | Vision scoring summary |
| 25 | C Group - Session 3 (Turn 9 and 8) - CVR_4982_Mar1025_1040AM_CaliPhoto.jpg | 0.5115 | 0.4978 | 42 | 7 | 7 | 7 | 7 | 7 | 7 | no | Vision scoring summary |
| 26 | IMG_4984.jpeg | 0.4754 | 0.5421 | 37 | 8 | 6 | 5 | 5 | 6 | 7 | no | The focus is strong on the bike's front end, but the indoor, cluttered background and low angle limit the dramatic impact needed for a top-tier Instagram shot. |
| 27 | IG renatobo - IMG_5014.jpeg | 0.4357 | 0.3765 | 37 | 7 | 5 | 6 | 6 | 6 | 7 | no | The bike is visible, but the harsh midday sun and large negative space dilute the impact; cropping vertically should focus on the bike's profile against the sky. |
| 28 | C Group - Session 2 Back Straight Speed Pans - CVR_2628_Mar1025_943AM_CaliPhoto.jpg | 0.3929 | 0.7827 | 36 | 6 | 6 | 6 | 6 | 6 | 6 | no | Vision scoring summary |
| 29 | caliphotovideo - Group picture - DROC Track Day March 10 202500002.jpg | 0.3687 | 0.6484 | 36 | 6 | 6 | 6 | 6 | 6 | 6 | no | Vision scoring summary |
| 30 | IG ocredevil - DROC-Track-Day-3-10-2-filippo pagan.jpeg | 0.3296 | 0.4312 | 36 | 6 | 6 | 6 | 6 | 6 | 6 | no | Vision scoring summary |
| 31 | caliphotovideo - Group picture - DROC Track Day March 10 202500001.jpg | 0.1893 | 0.6360 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 32 | AB Group - Session 1 (Turn 16 Entry) - CVR_0230_Mar1025_817AM_CaliPhoto.jpg | 0.1840 | 0.5860 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 33 | 1-Around the Pits - 1-Around the Pits - CVR_0025_Mar1025_728AM_CaliPhoto.jpg | 0.1801 | 0.5483 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 34 | IG martangelenos - PXL_20250310_132329430.jpg | 0.1733 | 0.4839 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 35 | 2-Group Photo - Raffle - CVR_7677_Mar1025_1249PM_CaliPhoto.jpg | 0.0683 | 0.5853 | 24 | 4 | 4 | 4 | 4 | 4 | 4 | no | Vision scoring summary |
| 36 | 2-Group Photo - Raffle - CVR_7682_Mar1025_1249PM_CaliPhoto.jpg | 0.0661 | 0.5357 | 24 | 4 | 4 | 4 | 4 | 4 | 4 | no | Vision scoring summary |
| 37 | IG guardiansvoice - IMG-4160-Delenian.jpeg | 0.0643 | 0.4960 | 24 | 4 | 4 | 4 | 4 | 4 | 4 | no | Vision scoring summary |
| 38 | IG - renatobo - March 10 DROC Track Day-58 HDR.jpeg | 0.0643 | 0.5335 | 23 | 4 | 4 | 4 | 4 | 4 | 3 | no | Vision scoring summary |
| 39 | 1-Around the Pits - 1-Around the Pits - CVR_0012_Mar1025_702AM_CaliPhoto.jpg | 0.0599 | 0.3975 | 24 | 4 | 4 | 4 | 4 | 4 | 4 | no | Vision scoring summary |
| 40 | IG - renatobo - March 10 DROC Track Day-28.jpeg | 0.0533 | 0.4833 | 18 | 3 | 3 | 3 | 3 | 3 | 3 | no | Vision scoring summary |
| 41 | IG desmo.donna - IMG-1151-Donna.jpeg | 0.0505 | 0.4230 | 18 | 3 | 3 | 3 | 3 | 3 | 3 | no | Vision scoring summary |
| 42 | IG m92663m - IMG-6019-Mark Momot.jpeg | 0.0390 | 0.6338 | 6 | 1 | 1 | 1 | 1 | 1 | 1 | no | Vision scoring summary |

### gemma4:e4b-it-q8_0 | yolo=off

| Rank | Image | Final | Tech | Vision | Subject | Light | Color | Emotion | Scroll | Crop | Failed | One line |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 | IG renatobo - IMG_5013.jpeg | 0.8045 | 0.5817 | 54 | 9 | 9 | 9 | 9 | 9 | 9 | no | Vision scoring summary |
| 2 | IG cali_carnivores - DSC00013.jpg | 0.7937 | 0.7401 | 49 | 9 | 7 | 8 | 8 | 8 | 9 | no | Strong, aggressive low-angle shot of a vibrant Ducati against a clean, bright backdrop, making it highly suitable for an enthusiast feed. |
| 3 | IG naramedia_official - DSC-5992-NaraMedia.jpeg | 0.7871 | 0.7571 | 48 | 8 | 8 | 8 | 8 | 8 | 8 | no | Vision scoring summary |
| 4 | IG - renatobo - lineup 1st session - March 10 DROC Track Day-12 HDR.jpeg | 0.7837 | 0.5123 | 54 | 9 | 9 | 9 | 9 | 9 | 9 | no | Vision scoring summary |
| 5 | AB Group - Session 4 (Turn 16) - CVR_6545_Mar1025_1122AM_CaliPhoto.jpg | 0.7775 | 0.7249 | 48 | 8 | 8 | 8 | 8 | 8 | 8 | no | Vision scoring summary |
| 6 | 1-Around the Pits - CVR_0792_Mar1025_832AM_CaliPhoto.jpg | 0.7607 | 0.6688 | 48 | 8 | 8 | 8 | 8 | 8 | 8 | no | Vision scoring summary |
| 7 | IG naramedia_official - DSC-5897-NaraMedia.jpeg | 0.7551 | 0.6504 | 48 | 8 | 8 | 8 | 8 | 8 | 8 | no | Vision scoring summary |
| 8 | IG luckie.moto - 20250310-075123-luckie.moto.jpg | 0.7512 | 0.6374 | 48 | 8 | 8 | 8 | 8 | 8 | 8 | no | Vision scoring summary |
| 9 | AB Group - Session 1 (Turn 16 Entry) - CVR_0148_Mar1025_815AM_CaliPhoto.jpg | 0.7457 | 0.6190 | 48 | 8 | 8 | 8 | 8 | 8 | 8 | no | Vision scoring summary |
| 10 | IG desmo.donna - IMG-1080-Donna.jpeg | 0.7328 | 0.5760 | 48 | 8 | 8 | 8 | 8 | 8 | 8 | no | Vision scoring summary |
| 11 | IG cali_carnivores - DSC09857.jpg | 0.5341 | 0.5921 | 42 | 7 | 7 | 7 | 7 | 7 | 7 | no | Vision scoring summary |
| 12 | AB Group - Session 1 (Turn 16 Entry) - CVR_0230_Mar1025_817AM_CaliPhoto.jpg | 0.5326 | 0.5860 | 42 | 7 | 7 | 7 | 7 | 7 | 7 | no | Vision scoring summary |
| 13 | IG renatobo - IMG_5014.jpeg | 0.4824 | 0.3765 | 42 | 7 | 7 | 7 | 7 | 7 | 7 | no | Vision scoring summary |
| 14 | IG - renatobo - lineup 1st session - March 10 DROC Track Day-01 HDR.jpeg | 0.4823 | 0.3764 | 42 | 7 | 7 | 7 | 7 | 7 | 7 | no | Vision scoring summary |
| 15 | DSC-5436-NaraMedia.jpeg | 0.3652 | 0.6287 | 36 | 6 | 6 | 6 | 6 | 6 | 6 | no | Vision scoring summary |
| 16 | IG desmo.donna - IMG-1015-Donna.jpeg | 0.3552 | 0.5732 | 36 | 6 | 6 | 6 | 6 | 6 | 6 | no | Vision scoring summary |
| 17 | IMG_4984.jpeg | 0.3496 | 0.5421 | 36 | 6 | 6 | 6 | 6 | 6 | 6 | no | Vision scoring summary |
| 18 | IG martangelenos - PXL_20250310_132329430.jpg | 0.3391 | 0.4839 | 36 | 6 | 6 | 6 | 6 | 6 | 6 | no | Vision scoring summary |
| 19 | IG ocredevil - DROC-Track-Day-3-10-2-filippo pagan.jpeg | 0.3296 | 0.4312 | 36 | 6 | 6 | 6 | 6 | 6 | 6 | no | Vision scoring summary |
| 20 | 1-Around the Pits - CVR_0098_Mar1025_802AM_CaliPhoto.jpg | 0.2090 | 0.8237 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 21 | C Group - Session 2 Back Straight Speed Pans - CVR_2628_Mar1025_943AM_CaliPhoto.jpg | 0.2047 | 0.7827 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 22 | IG naramedia_official - DSC-5365-NaraMedia.jpeg | 0.2042 | 0.7780 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 23 | C Group - Session 2 Back Straight Speed Pans - CVR_2850_Mar1025_946AM_CaliPhoto.jpg | 0.2029 | 0.7661 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 24 | IG cali_carnivores - DSC09850.jpg | 0.2024 | 0.7612 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 25 | IG m92663m - IMG-6019-Mark Momot.jpeg | 0.1891 | 0.6338 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 26 | IG kamiumitv - IMG-3981-Bao.jpeg | 0.1873 | 0.6171 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 27 | AB Group - Session 2 (Turn 11) - CVR_1047_Mar1025_908AM_CaliPhoto.jpg | 0.1811 | 0.5583 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 28 | 1-Around the Pits - 1-Around the Pits - CVR_0025_Mar1025_728AM_CaliPhoto.jpg | 0.1801 | 0.5483 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 29 | IG guardiansvoice - IMG-4129-Delenian.jpeg | 0.1793 | 0.5413 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 30 | 2-Group Photo - Raffle - CVR_7682_Mar1025_1249PM_CaliPhoto.jpg | 0.1787 | 0.5357 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 31 | IG - renatobo - lineup 1st session - March 10 DROC Track Day-02 HDR.jpeg | 0.1740 | 0.4907 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 32 | IG - renatobo - March 10 DROC Track Day-28.jpeg | 0.1733 | 0.4833 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 33 | IG - renatobo - March 10 DROC Track Day-26 HDR.jpeg | 0.1726 | 0.4768 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 34 | IMG_5012.jpeg | 0.1665 | 0.4195 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 35 | caliphotovideo - Group picture - DROC Track Day March 10 202500002.jpg | 0.0712 | 0.6484 | 24 | 4 | 4 | 4 | 4 | 4 | 4 | no | Vision scoring summary |
| 36 | 2-Group Photo - Raffle - CVR_7677_Mar1025_1249PM_CaliPhoto.jpg | 0.0683 | 0.5853 | 24 | 4 | 4 | 4 | 4 | 4 | 4 | no | Vision scoring summary |
| 37 | IG guardiansvoice - IMG-4160-Delenian.jpeg | 0.0643 | 0.4960 | 24 | 4 | 4 | 4 | 4 | 4 | 4 | no | Vision scoring summary |
| 38 | C Group - Session 3 (Turn 9 and 8) - CVR_4982_Mar1025_1040AM_CaliPhoto.jpg | 0.0609 | 0.4978 | 22 | 4 | 4 | 4 | 4 | 4 | 2 | no | Vision scoring summary |
| 39 | caliphotovideo - Group picture - DROC Track Day March 10 202500001.jpg | 0.0601 | 0.6360 | 18 | 3 | 3 | 3 | 3 | 3 | 3 | no | Vision scoring summary |
| 40 | IG - renatobo - March 10 DROC Track Day-58 HDR.jpeg | 0.0555 | 0.5335 | 18 | 3 | 3 | 3 | 3 | 3 | 3 | no | Vision scoring summary |
| 41 | 1-Around the Pits - 1-Around the Pits - CVR_0012_Mar1025_702AM_CaliPhoto.jpg | 0.0494 | 0.3975 | 18 | 3 | 3 | 3 | 3 | 3 | 3 | no | Vision scoring summary |
| 42 | IG desmo.donna - IMG-1151-Donna.jpeg | 0.0400 | 0.4230 | 12 | 2 | 2 | 2 | 2 | 2 | 2 | no | Vision scoring summary |

### gemma4:e4b-it-q8_0 | yolo=off | prompt=claude

| Rank | Image | Final | Tech | Vision | Subject | Light | Color | Emotion | Scroll | Crop | Failed | One line |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 | IG naramedia_official - DSC-5365-NaraMedia.jpeg | 0.8634 | 0.7780 | 54 | 9 | 9 | 9 | 9 | 9 | 9 | no | Vision scoring summary |
| 2 | AB Group - Session 4 (Turn 16) - CVR_6545_Mar1025_1122AM_CaliPhoto.jpg | 0.8475 | 0.7249 | 54 | 9 | 9 | 9 | 9 | 9 | 9 | no | Vision scoring summary |
| 3 | IG naramedia_official - DSC-5897-NaraMedia.jpeg | 0.8251 | 0.6504 | 54 | 9 | 9 | 9 | 9 | 9 | 9 | no | Vision scoring summary |
| 4 | IG luckie.moto - 20250310-075123-luckie.moto.jpg | 0.8212 | 0.6374 | 54 | 9 | 9 | 9 | 9 | 9 | 9 | no | Vision scoring summary |
| 5 | IG cali_carnivores - DSC09850.jpg | 0.7884 | 0.7612 | 48 | 8 | 8 | 8 | 8 | 8 | 8 | no | Vision scoring summary |
| 6 | 1-Around the Pits - CVR_0792_Mar1025_832AM_CaliPhoto.jpg | 0.7607 | 0.6688 | 48 | 8 | 8 | 8 | 8 | 8 | 8 | no | Vision scoring summary |
| 7 | IMG_5012.jpeg | 0.7558 | 0.4195 | 54 | 9 | 9 | 9 | 9 | 9 | 9 | no | Vision scoring summary |
| 8 | AB Group - Session 1 (Turn 16 Entry) - CVR_0148_Mar1025_815AM_CaliPhoto.jpg | 0.7457 | 0.6190 | 48 | 8 | 8 | 8 | 8 | 8 | 8 | no | Vision scoring summary |
| 9 | C Group - Session 3 (Turn 9 and 8) - CVR_4982_Mar1025_1040AM_CaliPhoto.jpg | 0.7093 | 0.4978 | 48 | 8 | 8 | 8 | 8 | 8 | 8 | no | Vision scoring summary |
| 10 | IG - renatobo - March 10 DROC Track Day-26 HDR.jpeg | 0.7030 | 0.4768 | 48 | 8 | 8 | 8 | 8 | 8 | 8 | no | Vision scoring summary |
| 11 | C Group - Session 2 Back Straight Speed Pans - CVR_2628_Mar1025_943AM_CaliPhoto.jpg | 0.5799 | 0.7827 | 42 | 7 | 7 | 7 | 7 | 7 | 7 | no | Vision scoring summary |
| 12 | IMG_4984.jpeg | 0.5221 | 0.5421 | 42 | 7 | 7 | 7 | 7 | 7 | 7 | no | Vision scoring summary |
| 13 | IG guardiansvoice - IMG-4129-Delenian.jpeg | 0.5219 | 0.5413 | 42 | 7 | 7 | 7 | 7 | 7 | 7 | no | Vision scoring summary |
| 14 | IG renatobo - IMG_5014.jpeg | 0.4824 | 0.3765 | 42 | 7 | 7 | 7 | 7 | 7 | 7 | no | Vision scoring summary |
| 15 | IG - renatobo - lineup 1st session - March 10 DROC Track Day-01 HDR.jpeg | 0.4823 | 0.3764 | 42 | 7 | 7 | 7 | 7 | 7 | 7 | no | Vision scoring summary |
| 16 | 1-Around the Pits - CVR_0098_Mar1025_802AM_CaliPhoto.jpg | 0.2090 | 0.8237 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 17 | C Group - Session 2 Back Straight Speed Pans - CVR_2850_Mar1025_946AM_CaliPhoto.jpg | 0.2029 | 0.7661 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 18 | IG naramedia_official - DSC-5992-NaraMedia.jpeg | 0.2020 | 0.7571 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 19 | IG cali_carnivores - DSC00013.jpg | 0.2002 | 0.7401 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 20 | caliphotovideo - Group picture - DROC Track Day March 10 202500002.jpg | 0.1906 | 0.6484 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 21 | DSC-5436-NaraMedia.jpeg | 0.1885 | 0.6287 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 22 | IG kamiumitv - IMG-3981-Bao.jpeg | 0.1873 | 0.6171 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 23 | IG cali_carnivores - DSC09857.jpg | 0.1847 | 0.5921 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 24 | AB Group - Session 1 (Turn 16 Entry) - CVR_0230_Mar1025_817AM_CaliPhoto.jpg | 0.1840 | 0.5860 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 25 | IG renatobo - IMG_5013.jpeg | 0.1836 | 0.5817 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 26 | IG desmo.donna - IMG-1080-Donna.jpeg | 0.1830 | 0.5760 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 27 | IG desmo.donna - IMG-1015-Donna.jpeg | 0.1827 | 0.5732 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 28 | AB Group - Session 2 (Turn 11) - CVR_1047_Mar1025_908AM_CaliPhoto.jpg | 0.1811 | 0.5583 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 29 | 1-Around the Pits - 1-Around the Pits - CVR_0025_Mar1025_728AM_CaliPhoto.jpg | 0.1801 | 0.5483 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 30 | 2-Group Photo - Raffle - CVR_7682_Mar1025_1249PM_CaliPhoto.jpg | 0.1787 | 0.5357 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 31 | IG - renatobo - lineup 1st session - March 10 DROC Track Day-12 HDR.jpeg | 0.1763 | 0.5123 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 32 | IG guardiansvoice - IMG-4160-Delenian.jpeg | 0.1746 | 0.4960 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 33 | IG - renatobo - lineup 1st session - March 10 DROC Track Day-02 HDR.jpeg | 0.1740 | 0.4907 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 34 | IG - renatobo - March 10 DROC Track Day-28.jpeg | 0.1733 | 0.4833 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 35 | caliphotovideo - Group picture - DROC Track Day March 10 202500001.jpg | 0.0706 | 0.6360 | 24 | 4 | 4 | 4 | 4 | 4 | 4 | no | Vision scoring summary |
| 36 | 2-Group Photo - Raffle - CVR_7677_Mar1025_1249PM_CaliPhoto.jpg | 0.0683 | 0.5853 | 24 | 4 | 4 | 4 | 4 | 4 | 4 | no | Vision scoring summary |
| 37 | IG - renatobo - March 10 DROC Track Day-58 HDR.jpeg | 0.0660 | 0.5335 | 24 | 4 | 4 | 4 | 4 | 4 | 4 | no | Vision scoring summary |
| 38 | IG ocredevil - DROC-Track-Day-3-10-2-filippo pagan.jpeg | 0.0614 | 0.4312 | 24 | 4 | 4 | 4 | 4 | 4 | 4 | no | Vision scoring summary |
| 39 | IG m92663m - IMG-6019-Mark Momot.jpeg | 0.0600 | 0.6338 | 18 | 3 | 3 | 3 | 3 | 3 | 3 | no | Vision scoring summary |
| 40 | 1-Around the Pits - 1-Around the Pits - CVR_0012_Mar1025_702AM_CaliPhoto.jpg | 0.0599 | 0.3975 | 24 | 4 | 4 | 4 | 4 | 4 | 4 | no | Vision scoring summary |
| 41 | IG martangelenos - PXL_20250310_132329430.jpg | 0.0533 | 0.4839 | 18 | 3 | 3 | 3 | 3 | 3 | 3 | no | Vision scoring summary |
| 42 | IG desmo.donna - IMG-1151-Donna.jpeg | 0.0505 | 0.4230 | 18 | 3 | 3 | 3 | 3 | 3 | 3 | no | Vision scoring summary |

### gemma4:e4b-it-q8_0 | yolo=off | prompt=system

| Rank | Image | Final | Tech | Vision | Subject | Light | Color | Emotion | Scroll | Crop | Failed | One line |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 | AB Group - Session 2 (Turn 11) - CVR_1047_Mar1025_908AM_CaliPhoto.jpg | 0.7275 | 0.5583 | 48 | 8 | 8 | 8 | 8 | 8 | 8 | no | Vision scoring summary |
| 2 | C Group - Session 2 Back Straight Speed Pans - CVR_2628_Mar1025_943AM_CaliPhoto.jpg | 0.3929 | 0.7827 | 36 | 6 | 6 | 6 | 6 | 6 | 6 | no | Vision scoring summary |
| 3 | caliphotovideo - Group picture - DROC Track Day March 10 202500002.jpg | 0.3687 | 0.6484 | 36 | 6 | 6 | 6 | 6 | 6 | 6 | no | Vision scoring summary |
| 4 | 1-Around the Pits - CVR_0098_Mar1025_802AM_CaliPhoto.jpg | 0.2090 | 0.8237 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 5 | IG naramedia_official - DSC-5365-NaraMedia.jpeg | 0.2042 | 0.7780 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 6 | C Group - Session 2 Back Straight Speed Pans - CVR_2850_Mar1025_946AM_CaliPhoto.jpg | 0.2029 | 0.7661 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 7 | IG cali_carnivores - DSC09850.jpg | 0.2024 | 0.7612 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 8 | IG naramedia_official - DSC-5992-NaraMedia.jpeg | 0.2020 | 0.7571 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 9 | IG cali_carnivores - DSC00013.jpg | 0.2002 | 0.7401 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 10 | AB Group - Session 4 (Turn 16) - CVR_6545_Mar1025_1122AM_CaliPhoto.jpg | 0.1986 | 0.7249 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 11 | 1-Around the Pits - CVR_0792_Mar1025_832AM_CaliPhoto.jpg | 0.1927 | 0.6688 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 12 | IG naramedia_official - DSC-5897-NaraMedia.jpeg | 0.1908 | 0.6504 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 13 | IG luckie.moto - 20250310-075123-luckie.moto.jpg | 0.1894 | 0.6374 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 14 | caliphotovideo - Group picture - DROC Track Day March 10 202500001.jpg | 0.1893 | 0.6360 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 15 | IG m92663m - IMG-6019-Mark Momot.jpeg | 0.1891 | 0.6338 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 16 | DSC-5436-NaraMedia.jpeg | 0.1885 | 0.6287 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 17 | AB Group - Session 1 (Turn 16 Entry) - CVR_0148_Mar1025_815AM_CaliPhoto.jpg | 0.1875 | 0.6190 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 18 | IG kamiumitv - IMG-3981-Bao.jpeg | 0.1873 | 0.6171 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 19 | IG cali_carnivores - DSC09857.jpg | 0.1847 | 0.5921 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 20 | AB Group - Session 1 (Turn 16 Entry) - CVR_0230_Mar1025_817AM_CaliPhoto.jpg | 0.1840 | 0.5860 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 21 | 2-Group Photo - Raffle - CVR_7677_Mar1025_1249PM_CaliPhoto.jpg | 0.1840 | 0.5853 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 22 | IG renatobo - IMG_5013.jpeg | 0.1836 | 0.5817 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 23 | IG desmo.donna - IMG-1080-Donna.jpeg | 0.1830 | 0.5760 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 24 | IG desmo.donna - IMG-1015-Donna.jpeg | 0.1827 | 0.5732 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 25 | 1-Around the Pits - 1-Around the Pits - CVR_0025_Mar1025_728AM_CaliPhoto.jpg | 0.1801 | 0.5483 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 26 | IMG_4984.jpeg | 0.1794 | 0.5421 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 27 | IG guardiansvoice - IMG-4129-Delenian.jpeg | 0.1793 | 0.5413 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 28 | 2-Group Photo - Raffle - CVR_7682_Mar1025_1249PM_CaliPhoto.jpg | 0.1787 | 0.5357 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 29 | IG - renatobo - March 10 DROC Track Day-58 HDR.jpeg | 0.1785 | 0.5335 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 30 | IG - renatobo - lineup 1st session - March 10 DROC Track Day-12 HDR.jpeg | 0.1763 | 0.5123 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 31 | C Group - Session 3 (Turn 9 and 8) - CVR_4982_Mar1025_1040AM_CaliPhoto.jpg | 0.1748 | 0.4978 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 32 | IG guardiansvoice - IMG-4160-Delenian.jpeg | 0.1746 | 0.4960 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 33 | IG - renatobo - lineup 1st session - March 10 DROC Track Day-02 HDR.jpeg | 0.1740 | 0.4907 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 34 | IG martangelenos - PXL_20250310_132329430.jpg | 0.1733 | 0.4839 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 35 | IG - renatobo - March 10 DROC Track Day-28.jpeg | 0.1733 | 0.4833 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 36 | IG - renatobo - March 10 DROC Track Day-26 HDR.jpeg | 0.1726 | 0.4768 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 37 | IG ocredevil - DROC-Track-Day-3-10-2-filippo pagan.jpeg | 0.1678 | 0.4312 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 38 | IG desmo.donna - IMG-1151-Donna.jpeg | 0.1669 | 0.4230 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 39 | IMG_5012.jpeg | 0.1665 | 0.4195 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 40 | 1-Around the Pits - 1-Around the Pits - CVR_0012_Mar1025_702AM_CaliPhoto.jpg | 0.1642 | 0.3975 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 41 | IG renatobo - IMG_5014.jpeg | 0.1620 | 0.3765 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 42 | IG - renatobo - lineup 1st session - March 10 DROC Track Day-01 HDR.jpeg | 0.1620 | 0.3764 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |

### gemma4:e4b-it-q8_0 | yolo=off | prompt=claude+system

| Rank | Image | Final | Tech | Vision | Subject | Light | Color | Emotion | Scroll | Crop | Failed | One line |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 | AB Group - Session 4 (Turn 16) - CVR_6545_Mar1025_1122AM_CaliPhoto.jpg | 0.8475 | 0.7249 | 54 | 9 | 9 | 9 | 9 | 9 | 9 | no | Vision scoring summary |
| 2 | C Group - Session 2 Back Straight Speed Pans - CVR_2850_Mar1025_946AM_CaliPhoto.jpg | 0.7898 | 0.7661 | 48 | 8 | 8 | 8 | 8 | 8 | 8 | no | Vision scoring summary |
| 3 | IG - renatobo - lineup 1st session - March 10 DROC Track Day-12 HDR.jpeg | 0.7837 | 0.5123 | 54 | 9 | 9 | 9 | 9 | 9 | 9 | no | Vision scoring summary |
| 4 | IG cali_carnivores - DSC00013.jpg | 0.7820 | 0.7401 | 48 | 9 | 7 | 8 | 8 | 8 | 8 | no | Strong, aggressive composition with excellent color contrast against the muted background, making it highly suitable for an enthusiast feed. |
| 5 | IG - renatobo - lineup 1st session - March 10 DROC Track Day-02 HDR.jpeg | 0.7772 | 0.4907 | 54 | 9 | 9 | 9 | 9 | 9 | 9 | no | Vision scoring summary |
| 6 | IG naramedia_official - DSC-5897-NaraMedia.jpeg | 0.7551 | 0.6504 | 48 | 8 | 8 | 8 | 8 | 8 | 8 | no | Vision scoring summary |
| 7 | IG luckie.moto - 20250310-075123-luckie.moto.jpg | 0.7512 | 0.6374 | 48 | 8 | 8 | 8 | 8 | 8 | 8 | no | Vision scoring summary |
| 8 | IG renatobo - IMG_5013.jpeg | 0.7345 | 0.5817 | 48 | 8 | 8 | 8 | 8 | 8 | 8 | no | Vision scoring summary |
| 9 | IG desmo.donna - IMG-1015-Donna.jpeg | 0.7320 | 0.5732 | 48 | 8 | 8 | 8 | 8 | 8 | 8 | no | Vision scoring summary |
| 10 | DSC-5436-NaraMedia.jpeg | 0.5429 | 0.6287 | 42 | 7 | 7 | 7 | 7 | 7 | 7 | no | Vision scoring summary |
| 11 | C Group - Session 3 (Turn 9 and 8) - CVR_4982_Mar1025_1040AM_CaliPhoto.jpg | 0.5115 | 0.4978 | 42 | 7 | 7 | 7 | 7 | 7 | 7 | no | Vision scoring summary |
| 12 | IMG_4984.jpeg | 0.4848 | 0.5421 | 38 | 8 | 6 | 6 | 5 | 6 | 7 | no | The bike is clearly visible, but the indoor setting and close-up angle limit the sense of power and dynamic composition needed for a top-tier Instagram cover. |
| 13 | C Group - Session 2 Back Straight Speed Pans - CVR_2628_Mar1025_943AM_CaliPhoto.jpg | 0.3929 | 0.7827 | 36 | 6 | 6 | 6 | 6 | 6 | 6 | no | Vision scoring summary |
| 14 | IG guardiansvoice - IMG-4129-Delenian.jpeg | 0.3494 | 0.5413 | 36 | 6 | 6 | 6 | 6 | 6 | 6 | no | Vision scoring summary |
| 15 | IG martangelenos - PXL_20250310_132329430.jpg | 0.3391 | 0.4839 | 36 | 6 | 6 | 6 | 6 | 6 | 6 | no | Vision scoring summary |
| 16 | IG ocredevil - DROC-Track-Day-3-10-2-filippo pagan.jpeg | 0.3296 | 0.4312 | 36 | 6 | 6 | 6 | 6 | 6 | 6 | no | Vision scoring summary |
| 17 | IG renatobo - IMG_5014.jpeg | 0.3198 | 0.3765 | 36 | 6 | 6 | 6 | 6 | 6 | 6 | no | Vision scoring summary |
| 18 | IG naramedia_official - DSC-5365-NaraMedia.jpeg | 0.2042 | 0.7780 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 19 | IG cali_carnivores - DSC09850.jpg | 0.2024 | 0.7612 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 20 | IG naramedia_official - DSC-5992-NaraMedia.jpeg | 0.2020 | 0.7571 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 21 | 1-Around the Pits - CVR_0792_Mar1025_832AM_CaliPhoto.jpg | 0.1927 | 0.6688 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 22 | IG m92663m - IMG-6019-Mark Momot.jpeg | 0.1891 | 0.6338 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 23 | AB Group - Session 1 (Turn 16 Entry) - CVR_0148_Mar1025_815AM_CaliPhoto.jpg | 0.1875 | 0.6190 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 24 | IG kamiumitv - IMG-3981-Bao.jpeg | 0.1873 | 0.6171 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 25 | IG cali_carnivores - DSC09857.jpg | 0.1847 | 0.5921 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 26 | AB Group - Session 1 (Turn 16 Entry) - CVR_0230_Mar1025_817AM_CaliPhoto.jpg | 0.1840 | 0.5860 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 27 | IG desmo.donna - IMG-1080-Donna.jpeg | 0.1830 | 0.5760 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 28 | AB Group - Session 2 (Turn 11) - CVR_1047_Mar1025_908AM_CaliPhoto.jpg | 0.1811 | 0.5583 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 29 | 1-Around the Pits - 1-Around the Pits - CVR_0025_Mar1025_728AM_CaliPhoto.jpg | 0.1801 | 0.5483 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 30 | IG - renatobo - March 10 DROC Track Day-28.jpeg | 0.1733 | 0.4833 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 31 | IG - renatobo - March 10 DROC Track Day-26 HDR.jpeg | 0.1726 | 0.4768 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 32 | IMG_5012.jpeg | 0.1665 | 0.4195 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 33 | IG - renatobo - lineup 1st session - March 10 DROC Track Day-01 HDR.jpeg | 0.1620 | 0.3764 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 34 | 1-Around the Pits - CVR_0098_Mar1025_802AM_CaliPhoto.jpg | 0.0756 | 0.8237 | 22 | 4 | 4 | 4 | 4 | 4 | 2 | no | Vision scoring summary |
| 35 | caliphotovideo - Group picture - DROC Track Day March 10 202500001.jpg | 0.0706 | 0.6360 | 24 | 4 | 4 | 4 | 4 | 4 | 4 | no | Vision scoring summary |
| 36 | IG guardiansvoice - IMG-4160-Delenian.jpeg | 0.0643 | 0.4960 | 24 | 4 | 4 | 4 | 4 | 4 | 4 | no | Vision scoring summary |
| 37 | IG desmo.donna - IMG-1151-Donna.jpeg | 0.0610 | 0.4230 | 24 | 4 | 4 | 4 | 4 | 4 | 4 | no | Vision scoring summary |
| 38 | caliphotovideo - Group picture - DROC Track Day March 10 202500002.jpg | 0.0607 | 0.6484 | 18 | 3 | 3 | 3 | 3 | 3 | 3 | no | Vision scoring summary |
| 39 | IG - renatobo - March 10 DROC Track Day-58 HDR.jpeg | 0.0590 | 0.5335 | 20 | 3 | 4 | 4 | 4 | 4 | 1 | no | Vision scoring summary |
| 40 | 1-Around the Pits - 1-Around the Pits - CVR_0012_Mar1025_702AM_CaliPhoto.jpg | 0.0494 | 0.3975 | 18 | 3 | 3 | 3 | 3 | 3 | 3 | no | Vision scoring summary |
| 41 | 2-Group Photo - Raffle - CVR_7677_Mar1025_1249PM_CaliPhoto.jpg | 0.0473 | 0.5853 | 12 | 2 | 2 | 2 | 2 | 2 | 2 | no | Vision scoring summary |
| 42 | 2-Group Photo - Raffle - CVR_7682_Mar1025_1249PM_CaliPhoto.jpg | 0.0451 | 0.5357 | 12 | 2 | 2 | 2 | 2 | 2 | 2 | no | Vision scoring summary |
