# Model Benchmark Report (Speed + Quality)

- Generated: `2026-04-09T11:03:59`
- Input folder: `/Users/renatobo/development/pickinsta/input`
- Candidates scored per run: `42`
- Runs per variant: `1`
- Warmup enabled: `True` (1 image + 10s wait)
- Ollama base URL: `http://localhost:11434`
- Ollama concurrency: `2`
- Ollama max retries: `2`
- Ollama keep_alive: `15m`

## Speed Summary

| Variant | Scorer | Model | YOLO | Avg sec/img | Avg imgs/min | Avg duration (s) | Avg failures/run | SDI | Unique tuples | Speed vs fastest |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| gemma4:e4b \| yolo=off \| prompt=freeform | ollama | gemma4:e4b | off | 27.88 | 2.15 | 1170.83 | 0.00 | 0.04 | 5 | 1.00x |

## Per-run Timing

| Variant | Run | Duration (s) | Sec/img | Imgs/min | Failures |
|---|---:|---:|---:|---:|---:|
| gemma4:e4b \| yolo=off \| prompt=freeform | 1 | 1170.83 | 27.88 | 2.15 | 0 |

## Image-by-Image Score Comparison (Run 1)

| Image | gemma4:e4b \| yolo=off \| prompt=freeform final | gemma4:e4b \| yolo=off \| prompt=freeform rank |
|---|---:|---:|
| 1-Around the Pits - 1-Around the Pits - CVR_0012_Mar1025_702AM_CaliPhoto.jpg | 0.1642 | 39 |
| 1-Around the Pits - 1-Around the Pits - CVR_0025_Mar1025_728AM_CaliPhoto.jpg | 0.1801 | 29 |
| 1-Around the Pits - CVR_0098_Mar1025_802AM_CaliPhoto.jpg | 0.1931 | 15 |
| 1-Around the Pits - CVR_0792_Mar1025_832AM_CaliPhoto.jpg | 0.2153 | 9 |
| 2-Group Photo - Raffle - CVR_7677_Mar1025_1249PM_CaliPhoto.jpg | 0.1840 | 24 |
| 2-Group Photo - Raffle - CVR_7682_Mar1025_1249PM_CaliPhoto.jpg | 0.1787 | 32 |
| AB Group - Session 1 (Turn 16 Entry) - CVR_0148_Mar1025_815AM_CaliPhoto.jpg | 0.1875 | 21 |
| AB Group - Session 1 (Turn 16 Entry) - CVR_0230_Mar1025_817AM_CaliPhoto.jpg | 0.1840 | 23 |
| AB Group - Session 2 (Turn 11) - CVR_1047_Mar1025_908AM_CaliPhoto.jpg | 0.1811 | 28 |
| AB Group - Session 4 (Turn 16) - CVR_6545_Mar1025_1122AM_CaliPhoto.jpg | 0.7775 | 2 |
| C Group - Session 2 Back Straight Speed Pans - CVR_2628_Mar1025_943AM_CaliPhoto.jpg | 0.2047 | 10 |
| C Group - Session 2 Back Straight Speed Pans - CVR_2850_Mar1025_946AM_CaliPhoto.jpg | 0.7898 | 1 |
| C Group - Session 3 (Turn 9 and 8) - CVR_4982_Mar1025_1040AM_CaliPhoto.jpg | 0.1748 | 34 |
| DSC-5436-NaraMedia.jpeg | 0.1885 | 20 |
| IG - renatobo - March 10 DROC Track Day-26 HDR.jpeg | 0.1726 | 37 |
| IG - renatobo - March 10 DROC Track Day-28.jpeg | 0.3390 | 7 |
| IG - renatobo - March 10 DROC Track Day-58 HDR.jpeg | 0.0660 | 40 |
| IG - renatobo - lineup 1st session - March 10 DROC Track Day-01 HDR.jpeg | 0.6729 | 4 |
| IG - renatobo - lineup 1st session - March 10 DROC Track Day-02 HDR.jpeg | 0.7072 | 3 |
| IG - renatobo - lineup 1st session - March 10 DROC Track Day-12 HDR.jpeg | 0.1763 | 33 |
| IG cali_carnivores - DSC00013.jpg | 0.2002 | 14 |
| IG cali_carnivores - DSC09850.jpg | 0.2024 | 12 |
| IG cali_carnivores - DSC09857.jpg | 0.3586 | 6 |
| IG desmo.donna - IMG-1015-Donna.jpeg | 0.1827 | 27 |
| IG desmo.donna - IMG-1080-Donna.jpeg | 0.1830 | 26 |
| IG desmo.donna - IMG-1151-Donna.jpeg | 0.0610 | 41 |
| IG guardiansvoice - IMG-4129-Delenian.jpeg | 0.1793 | 31 |
| IG guardiansvoice - IMG-4160-Delenian.jpeg | 0.1746 | 35 |
| IG kamiumitv - IMG-3981-Bao.jpeg | 0.1873 | 22 |
| IG luckie.moto - 20250310-075123-luckie.moto.jpg | 0.1894 | 18 |
| IG m92663m - IMG-6019-Mark Momot.jpeg | 0.1891 | 19 |
| IG martangelenos - PXL_20250310_132329430.jpg | 0.1733 | 36 |
| IG naramedia_official - DSC-5365-NaraMedia.jpeg | 0.2042 | 11 |
| IG naramedia_official - DSC-5897-NaraMedia.jpeg | 0.1908 | 16 |
| IG naramedia_official - DSC-5992-NaraMedia.jpeg | 0.2020 | 13 |
| IG ocredevil - DROC-Track-Day-3-10-2-filippo pagan.jpeg | 0.3296 | 8 |
| IG renatobo - IMG_5013.jpeg | 0.1836 | 25 |
| IG renatobo - IMG_5014.jpeg | 0.0607 | 42 |
| IMG_4984.jpeg | 0.1794 | 30 |
| IMG_5012.jpeg | 0.1665 | 38 |
| caliphotovideo - Group picture - DROC Track Day March 10 202500001.jpg | 0.3665 | 5 |
| caliphotovideo - Group picture - DROC Track Day March 10 202500002.jpg | 0.1906 | 17 |

## Ranked Quality Details (All Images)

_Each section below is from timed run 1 for that variant._

### gemma4:e4b | yolo=off | prompt=freeform

| Rank | Image | Final | Tech | Vision | Subject | Light | Color | Emotion | Scroll | Crop | Failed | One line |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 | C Group - Session 2 Back Straight Speed Pans - CVR_2850_Mar1025_946AM_CaliPhoto.jpg | 0.7898 | 0.7661 | 48 | 8 | 8 | 8 | 8 | 8 | 8 | no | Vision scoring summary |
| 2 | AB Group - Session 4 (Turn 16) - CVR_6545_Mar1025_1122AM_CaliPhoto.jpg | 0.7775 | 0.7249 | 48 | 8 | 8 | 8 | 8 | 8 | 8 | no | Vision scoring summary |
| 3 | IG - renatobo - lineup 1st session - March 10 DROC Track Day-02 HDR.jpeg | 0.7072 | 0.4907 | 48 | 8 | 8 | 8 | 8 | 8 | 8 | no | Vision scoring summary |
| 4 | IG - renatobo - lineup 1st session - March 10 DROC Track Day-01 HDR.jpeg | 0.6729 | 0.3764 | 48 | 8 | 8 | 8 | 8 | 8 | 8 | no | Vision scoring summary |
| 5 | caliphotovideo - Group picture - DROC Track Day March 10 202500001.jpg | 0.3665 | 0.6360 | 36 | 6 | 6 | 6 | 6 | 6 | 6 | no | Vision scoring summary |
| 6 | IG cali_carnivores - DSC09857.jpg | 0.3586 | 0.5921 | 36 | 6 | 6 | 6 | 6 | 6 | 6 | no | Vision scoring summary |
| 7 | IG - renatobo - March 10 DROC Track Day-28.jpeg | 0.3390 | 0.4833 | 36 | 6 | 6 | 6 | 6 | 6 | 6 | no | Vision scoring summary |
| 8 | IG ocredevil - DROC-Track-Day-3-10-2-filippo pagan.jpeg | 0.3296 | 0.4312 | 36 | 6 | 6 | 6 | 6 | 6 | 6 | no | Vision scoring summary |
| 9 | 1-Around the Pits - CVR_0792_Mar1025_832AM_CaliPhoto.jpg | 0.2153 | 0.8839 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 10 | C Group - Session 2 Back Straight Speed Pans - CVR_2628_Mar1025_943AM_CaliPhoto.jpg | 0.2047 | 0.7827 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 11 | IG naramedia_official - DSC-5365-NaraMedia.jpeg | 0.2042 | 0.7780 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 12 | IG cali_carnivores - DSC09850.jpg | 0.2024 | 0.7612 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 13 | IG naramedia_official - DSC-5992-NaraMedia.jpeg | 0.2020 | 0.7571 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 14 | IG cali_carnivores - DSC00013.jpg | 0.2002 | 0.7401 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 15 | 1-Around the Pits - CVR_0098_Mar1025_802AM_CaliPhoto.jpg | 0.1931 | 0.6723 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 16 | IG naramedia_official - DSC-5897-NaraMedia.jpeg | 0.1908 | 0.6504 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 17 | caliphotovideo - Group picture - DROC Track Day March 10 202500002.jpg | 0.1906 | 0.6484 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 18 | IG luckie.moto - 20250310-075123-luckie.moto.jpg | 0.1894 | 0.6374 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 19 | IG m92663m - IMG-6019-Mark Momot.jpeg | 0.1891 | 0.6338 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 20 | DSC-5436-NaraMedia.jpeg | 0.1885 | 0.6287 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 21 | AB Group - Session 1 (Turn 16 Entry) - CVR_0148_Mar1025_815AM_CaliPhoto.jpg | 0.1875 | 0.6190 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 22 | IG kamiumitv - IMG-3981-Bao.jpeg | 0.1873 | 0.6171 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 23 | AB Group - Session 1 (Turn 16 Entry) - CVR_0230_Mar1025_817AM_CaliPhoto.jpg | 0.1840 | 0.5860 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 24 | 2-Group Photo - Raffle - CVR_7677_Mar1025_1249PM_CaliPhoto.jpg | 0.1840 | 0.5853 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 25 | IG renatobo - IMG_5013.jpeg | 0.1836 | 0.5817 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 26 | IG desmo.donna - IMG-1080-Donna.jpeg | 0.1830 | 0.5760 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 27 | IG desmo.donna - IMG-1015-Donna.jpeg | 0.1827 | 0.5732 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 28 | AB Group - Session 2 (Turn 11) - CVR_1047_Mar1025_908AM_CaliPhoto.jpg | 0.1811 | 0.5583 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 29 | 1-Around the Pits - 1-Around the Pits - CVR_0025_Mar1025_728AM_CaliPhoto.jpg | 0.1801 | 0.5483 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 30 | IMG_4984.jpeg | 0.1794 | 0.5421 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 31 | IG guardiansvoice - IMG-4129-Delenian.jpeg | 0.1793 | 0.5413 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 32 | 2-Group Photo - Raffle - CVR_7682_Mar1025_1249PM_CaliPhoto.jpg | 0.1787 | 0.5357 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 33 | IG - renatobo - lineup 1st session - March 10 DROC Track Day-12 HDR.jpeg | 0.1763 | 0.5123 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 34 | C Group - Session 3 (Turn 9 and 8) - CVR_4982_Mar1025_1040AM_CaliPhoto.jpg | 0.1748 | 0.4978 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 35 | IG guardiansvoice - IMG-4160-Delenian.jpeg | 0.1746 | 0.4960 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 36 | IG martangelenos - PXL_20250310_132329430.jpg | 0.1733 | 0.4839 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 37 | IG - renatobo - March 10 DROC Track Day-26 HDR.jpeg | 0.1726 | 0.4768 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 38 | IMG_5012.jpeg | 0.1665 | 0.4195 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 39 | 1-Around the Pits - 1-Around the Pits - CVR_0012_Mar1025_702AM_CaliPhoto.jpg | 0.1642 | 0.3975 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 40 | IG - renatobo - March 10 DROC Track Day-58 HDR.jpeg | 0.0660 | 0.5335 | 24 | 4 | 4 | 4 | 4 | 4 | 4 | no | Vision scoring summary |
| 41 | IG desmo.donna - IMG-1151-Donna.jpeg | 0.0610 | 0.4230 | 24 | 4 | 4 | 4 | 4 | 4 | 4 | no | Vision scoring summary |
| 42 | IG renatobo - IMG_5014.jpeg | 0.0607 | 0.3765 | 25 | 3 | 7 | 4 | 3 | 4 | 4 | no | Vision scoring summary |
