# Model Benchmark Report (Speed + Quality)

- Generated: `2026-04-09T09:29:39`
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
| VladimirGav/gemma4-26b-16GB-VRAM \| yolo=off | ollama | VladimirGav/gemma4-26b-16GB-VRAM | off | 0.33 | 180.40 | 13.97 | 42.00 | 0.00 | 0 | 1.00x |
| gemma4:e4b \| yolo=off | ollama | gemma4:e4b | off | 23.81 | 2.52 | 999.97 | 0.00 | 0.00 | 3 | 71.59x |
| gemma4:e4b-it-q8_0 \| yolo=off | ollama | gemma4:e4b-it-q8_0 | off | 27.88 | 2.15 | 1171.03 | 0.00 | 0.01 | 2 | 83.83x |

## Per-run Timing

| Variant | Run | Duration (s) | Sec/img | Imgs/min | Failures |
|---|---:|---:|---:|---:|---:|
| gemma4:e4b \| yolo=off | 1 | 999.97 | 23.81 | 2.52 | 0 |
| gemma4:e4b-it-q8_0 \| yolo=off | 1 | 1171.03 | 27.88 | 2.15 | 0 |
| VladimirGav/gemma4-26b-16GB-VRAM \| yolo=off | 1 | 13.97 | 0.33 | 180.40 | 42 |

## Image-by-Image Score Comparison (Run 1)

| Image | gemma4:e4b \| yolo=off final | gemma4:e4b \| yolo=off rank | gemma4:e4b-it-q8_0 \| yolo=off final | gemma4:e4b-it-q8_0 \| yolo=off rank | VladimirGav/gemma4-26b-16GB-VRAM \| yolo=off final | VladimirGav/gemma4-26b-16GB-VRAM \| yolo=off rank |
|---|---:|---:|---:|---:|---:|---:|
| 1-Around the Pits - 1-Around the Pits - CVR_0012_Mar1025_702AM_CaliPhoto.jpg | 0.1642 | 41 | 0.1642 | 39 | 0.1193 | 40 |
| 1-Around the Pits - 1-Around the Pits - CVR_0025_Mar1025_728AM_CaliPhoto.jpg | 0.5236 | 1 | 0.1801 | 24 | 0.1645 | 25 |
| 1-Around the Pits - CVR_0098_Mar1025_802AM_CaliPhoto.jpg | 0.2090 | 13 | 0.2090 | 1 | 0.2471 | 1 |
| 1-Around the Pits - CVR_0792_Mar1025_832AM_CaliPhoto.jpg | 0.1927 | 19 | 0.1927 | 9 | 0.2007 | 9 |
| 2-Group Photo - Raffle - CVR_7677_Mar1025_1249PM_CaliPhoto.jpg | 0.1840 | 28 | 0.1840 | 19 | 0.1756 | 20 |
| 2-Group Photo - Raffle - CVR_7682_Mar1025_1249PM_CaliPhoto.jpg | 0.3484 | 7 | 0.1787 | 27 | 0.1607 | 28 |
| AB Group - Session 1 (Turn 16 Entry) - CVR_0148_Mar1025_815AM_CaliPhoto.jpg | 0.1875 | 24 | 0.1875 | 16 | 0.1857 | 16 |
| AB Group - Session 1 (Turn 16 Entry) - CVR_0230_Mar1025_817AM_CaliPhoto.jpg | 0.1840 | 27 | 0.1840 | 18 | 0.1758 | 19 |
| AB Group - Session 2 (Turn 11) - CVR_1047_Mar1025_908AM_CaliPhoto.jpg | 0.1811 | 32 | 0.1811 | 23 | 0.1675 | 24 |
| AB Group - Session 4 (Turn 16) - CVR_6545_Mar1025_1122AM_CaliPhoto.jpg | 0.1986 | 18 | 0.1986 | 8 | 0.2175 | 8 |
| C Group - Session 2 Back Straight Speed Pans - CVR_2628_Mar1025_943AM_CaliPhoto.jpg | 0.3929 | 2 | 0.2047 | 2 | 0.2348 | 2 |
| C Group - Session 2 Back Straight Speed Pans - CVR_2850_Mar1025_946AM_CaliPhoto.jpg | 0.2029 | 15 | 0.2029 | 4 | 0.2298 | 4 |
| C Group - Session 3 (Turn 9 and 8) - CVR_4982_Mar1025_1040AM_CaliPhoto.jpg | 0.1748 | 36 | 0.1748 | 30 | 0.1493 | 31 |
| DSC-5436-NaraMedia.jpeg | 0.1885 | 23 | 0.1885 | 15 | 0.1886 | 15 |
| IG - renatobo - March 10 DROC Track Day-26 HDR.jpeg | 0.3378 | 9 | 0.1726 | 35 | 0.1430 | 36 |
| IG - renatobo - March 10 DROC Track Day-28.jpeg | 0.1733 | 39 | 0.1733 | 34 | 0.1450 | 35 |
| IG - renatobo - March 10 DROC Track Day-58 HDR.jpeg | 0.1785 | 34 | 0.1785 | 28 | 0.1600 | 29 |
| IG - renatobo - lineup 1st session - March 10 DROC Track Day-01 HDR.jpeg | 0.1620 | 42 | 0.1620 | 41 | 0.1129 | 42 |
| IG - renatobo - lineup 1st session - March 10 DROC Track Day-02 HDR.jpeg | 0.1740 | 37 | 0.1740 | 32 | 0.1472 | 33 |
| IG - renatobo - lineup 1st session - March 10 DROC Track Day-12 HDR.jpeg | 0.1763 | 35 | 0.1763 | 29 | 0.1537 | 30 |
| IG cali_carnivores - DSC00013.jpg | 0.2002 | 17 | 0.2002 | 7 | 0.2220 | 7 |
| IG cali_carnivores - DSC09850.jpg | 0.3890 | 3 | 0.2024 | 5 | 0.2284 | 5 |
| IG cali_carnivores - DSC09857.jpg | 0.1847 | 26 | 0.1847 | 17 | 0.1776 | 18 |
| IG desmo.donna - IMG-1015-Donna.jpeg | 0.1827 | 31 | 0.1827 | 22 | 0.1720 | 23 |
| IG desmo.donna - IMG-1080-Donna.jpeg | 0.1830 | 30 | 0.1830 | 21 | 0.1728 | 22 |
| IG desmo.donna - IMG-1151-Donna.jpeg | 0.1669 | 40 | 0.1669 | 37 | 0.1269 | 38 |
| IG guardiansvoice - IMG-4129-Delenian.jpeg | 0.1793 | 33 | 0.1793 | 26 | 0.1624 | 27 |
| IG guardiansvoice - IMG-4160-Delenian.jpeg | 0.3413 | 8 | 0.1746 | 31 | 0.1488 | 32 |
| IG kamiumitv - IMG-3981-Bao.jpeg | 0.1873 | 25 | 0.0575 | 42 | 0.1851 | 17 |
| IG luckie.moto - 20250310-075123-luckie.moto.jpg | 0.1894 | 22 | 0.1894 | 12 | 0.1912 | 12 |
| IG m92663m - IMG-6019-Mark Momot.jpeg | 0.3661 | 5 | 0.1891 | 14 | 0.1901 | 14 |
| IG martangelenos - PXL_20250310_132329430.jpg | 0.1733 | 38 | 0.1733 | 33 | 0.1452 | 34 |
| IG naramedia_official - DSC-5365-NaraMedia.jpeg | 0.2042 | 14 | 0.2042 | 3 | 0.2334 | 3 |
| IG naramedia_official - DSC-5897-NaraMedia.jpeg | 0.1908 | 20 | 0.1908 | 10 | 0.1951 | 10 |
| IG naramedia_official - DSC-5992-NaraMedia.jpeg | 0.2020 | 16 | 0.2020 | 6 | 0.2271 | 6 |
| IG ocredevil - DROC-Track-Day-3-10-2-filippo pagan.jpeg | 0.3296 | 10 | 0.1678 | 36 | 0.1294 | 37 |
| IG renatobo - IMG_5013.jpeg | 0.1836 | 29 | 0.1836 | 20 | 0.1745 | 21 |
| IG renatobo - IMG_5014.jpeg | 0.3198 | 12 | 0.1620 | 40 | 0.1130 | 41 |
| IMG_4984.jpeg | 0.3496 | 6 | 0.1794 | 25 | 0.1626 | 26 |
| IMG_5012.jpeg | 0.3275 | 11 | 0.1665 | 38 | 0.1258 | 39 |
| caliphotovideo - Group picture - DROC Track Day March 10 202500001.jpg | 0.3665 | 4 | 0.1893 | 13 | 0.1908 | 13 |
| caliphotovideo - Group picture - DROC Track Day March 10 202500002.jpg | 0.1906 | 21 | 0.1906 | 11 | 0.1945 | 11 |

## Ranked Quality Details (All Images)

_Each section below is from timed run 1 for that variant._

### gemma4:e4b | yolo=off

| Rank | Image | Final | Tech | Vision | Subject | Light | Color | Emotion | Scroll | Crop | Failed | One line |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 1-Around the Pits - 1-Around the Pits - CVR_0025_Mar1025_728AM_CaliPhoto.jpg | 0.5236 | 0.5483 | 42 | 7 | 7 | 7 | 7 | 7 | 7 | no | Vision scoring summary |
| 2 | C Group - Session 2 Back Straight Speed Pans - CVR_2628_Mar1025_943AM_CaliPhoto.jpg | 0.3929 | 0.7827 | 36 | 6 | 6 | 6 | 6 | 6 | 6 | no | Vision scoring summary |
| 3 | IG cali_carnivores - DSC09850.jpg | 0.3890 | 0.7612 | 36 | 6 | 6 | 6 | 6 | 6 | 6 | no | Vision scoring summary |
| 4 | caliphotovideo - Group picture - DROC Track Day March 10 202500001.jpg | 0.3665 | 0.6360 | 36 | 6 | 6 | 6 | 6 | 6 | 6 | no | Vision scoring summary |
| 5 | IG m92663m - IMG-6019-Mark Momot.jpeg | 0.3661 | 0.6338 | 36 | 6 | 6 | 6 | 6 | 6 | 6 | no | Vision scoring summary |
| 6 | IMG_4984.jpeg | 0.3496 | 0.5421 | 36 | 6 | 6 | 6 | 6 | 6 | 6 | no | Vision scoring summary |
| 7 | 2-Group Photo - Raffle - CVR_7682_Mar1025_1249PM_CaliPhoto.jpg | 0.3484 | 0.5357 | 36 | 6 | 6 | 6 | 6 | 6 | 6 | no | Vision scoring summary |
| 8 | IG guardiansvoice - IMG-4160-Delenian.jpeg | 0.3413 | 0.4960 | 36 | 6 | 6 | 6 | 6 | 6 | 6 | no | Vision scoring summary |
| 9 | IG - renatobo - March 10 DROC Track Day-26 HDR.jpeg | 0.3378 | 0.4768 | 36 | 6 | 6 | 6 | 6 | 6 | 6 | no | Vision scoring summary |
| 10 | IG ocredevil - DROC-Track-Day-3-10-2-filippo pagan.jpeg | 0.3296 | 0.4312 | 36 | 6 | 6 | 6 | 6 | 6 | 6 | no | Vision scoring summary |
| 11 | IMG_5012.jpeg | 0.3275 | 0.4195 | 36 | 6 | 6 | 6 | 6 | 6 | 6 | no | Vision scoring summary |
| 12 | IG renatobo - IMG_5014.jpeg | 0.3198 | 0.3765 | 36 | 6 | 6 | 6 | 6 | 6 | 6 | no | Vision scoring summary |
| 13 | 1-Around the Pits - CVR_0098_Mar1025_802AM_CaliPhoto.jpg | 0.2090 | 0.8237 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 14 | IG naramedia_official - DSC-5365-NaraMedia.jpeg | 0.2042 | 0.7780 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 15 | C Group - Session 2 Back Straight Speed Pans - CVR_2850_Mar1025_946AM_CaliPhoto.jpg | 0.2029 | 0.7661 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 16 | IG naramedia_official - DSC-5992-NaraMedia.jpeg | 0.2020 | 0.7571 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 17 | IG cali_carnivores - DSC00013.jpg | 0.2002 | 0.7401 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 18 | AB Group - Session 4 (Turn 16) - CVR_6545_Mar1025_1122AM_CaliPhoto.jpg | 0.1986 | 0.7249 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 19 | 1-Around the Pits - CVR_0792_Mar1025_832AM_CaliPhoto.jpg | 0.1927 | 0.6688 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 20 | IG naramedia_official - DSC-5897-NaraMedia.jpeg | 0.1908 | 0.6504 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 21 | caliphotovideo - Group picture - DROC Track Day March 10 202500002.jpg | 0.1906 | 0.6484 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 22 | IG luckie.moto - 20250310-075123-luckie.moto.jpg | 0.1894 | 0.6374 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 23 | DSC-5436-NaraMedia.jpeg | 0.1885 | 0.6287 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 24 | AB Group - Session 1 (Turn 16 Entry) - CVR_0148_Mar1025_815AM_CaliPhoto.jpg | 0.1875 | 0.6190 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 25 | IG kamiumitv - IMG-3981-Bao.jpeg | 0.1873 | 0.6171 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 26 | IG cali_carnivores - DSC09857.jpg | 0.1847 | 0.5921 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 27 | AB Group - Session 1 (Turn 16 Entry) - CVR_0230_Mar1025_817AM_CaliPhoto.jpg | 0.1840 | 0.5860 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 28 | 2-Group Photo - Raffle - CVR_7677_Mar1025_1249PM_CaliPhoto.jpg | 0.1840 | 0.5853 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 29 | IG renatobo - IMG_5013.jpeg | 0.1836 | 0.5817 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 30 | IG desmo.donna - IMG-1080-Donna.jpeg | 0.1830 | 0.5760 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 31 | IG desmo.donna - IMG-1015-Donna.jpeg | 0.1827 | 0.5732 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 32 | AB Group - Session 2 (Turn 11) - CVR_1047_Mar1025_908AM_CaliPhoto.jpg | 0.1811 | 0.5583 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 33 | IG guardiansvoice - IMG-4129-Delenian.jpeg | 0.1793 | 0.5413 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 34 | IG - renatobo - March 10 DROC Track Day-58 HDR.jpeg | 0.1785 | 0.5335 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 35 | IG - renatobo - lineup 1st session - March 10 DROC Track Day-12 HDR.jpeg | 0.1763 | 0.5123 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 36 | C Group - Session 3 (Turn 9 and 8) - CVR_4982_Mar1025_1040AM_CaliPhoto.jpg | 0.1748 | 0.4978 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 37 | IG - renatobo - lineup 1st session - March 10 DROC Track Day-02 HDR.jpeg | 0.1740 | 0.4907 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 38 | IG martangelenos - PXL_20250310_132329430.jpg | 0.1733 | 0.4839 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 39 | IG - renatobo - March 10 DROC Track Day-28.jpeg | 0.1733 | 0.4833 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 40 | IG desmo.donna - IMG-1151-Donna.jpeg | 0.1669 | 0.4230 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 41 | 1-Around the Pits - 1-Around the Pits - CVR_0012_Mar1025_702AM_CaliPhoto.jpg | 0.1642 | 0.3975 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 42 | IG - renatobo - lineup 1st session - March 10 DROC Track Day-01 HDR.jpeg | 0.1620 | 0.3764 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |

### gemma4:e4b-it-q8_0 | yolo=off

| Rank | Image | Final | Tech | Vision | Subject | Light | Color | Emotion | Scroll | Crop | Failed | One line |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 1-Around the Pits - CVR_0098_Mar1025_802AM_CaliPhoto.jpg | 0.2090 | 0.8237 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 2 | C Group - Session 2 Back Straight Speed Pans - CVR_2628_Mar1025_943AM_CaliPhoto.jpg | 0.2047 | 0.7827 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 3 | IG naramedia_official - DSC-5365-NaraMedia.jpeg | 0.2042 | 0.7780 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 4 | C Group - Session 2 Back Straight Speed Pans - CVR_2850_Mar1025_946AM_CaliPhoto.jpg | 0.2029 | 0.7661 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 5 | IG cali_carnivores - DSC09850.jpg | 0.2024 | 0.7612 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 6 | IG naramedia_official - DSC-5992-NaraMedia.jpeg | 0.2020 | 0.7571 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 7 | IG cali_carnivores - DSC00013.jpg | 0.2002 | 0.7401 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 8 | AB Group - Session 4 (Turn 16) - CVR_6545_Mar1025_1122AM_CaliPhoto.jpg | 0.1986 | 0.7249 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 9 | 1-Around the Pits - CVR_0792_Mar1025_832AM_CaliPhoto.jpg | 0.1927 | 0.6688 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 10 | IG naramedia_official - DSC-5897-NaraMedia.jpeg | 0.1908 | 0.6504 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 11 | caliphotovideo - Group picture - DROC Track Day March 10 202500002.jpg | 0.1906 | 0.6484 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 12 | IG luckie.moto - 20250310-075123-luckie.moto.jpg | 0.1894 | 0.6374 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 13 | caliphotovideo - Group picture - DROC Track Day March 10 202500001.jpg | 0.1893 | 0.6360 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 14 | IG m92663m - IMG-6019-Mark Momot.jpeg | 0.1891 | 0.6338 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 15 | DSC-5436-NaraMedia.jpeg | 0.1885 | 0.6287 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 16 | AB Group - Session 1 (Turn 16 Entry) - CVR_0148_Mar1025_815AM_CaliPhoto.jpg | 0.1875 | 0.6190 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 17 | IG cali_carnivores - DSC09857.jpg | 0.1847 | 0.5921 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 18 | AB Group - Session 1 (Turn 16 Entry) - CVR_0230_Mar1025_817AM_CaliPhoto.jpg | 0.1840 | 0.5860 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 19 | 2-Group Photo - Raffle - CVR_7677_Mar1025_1249PM_CaliPhoto.jpg | 0.1840 | 0.5853 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 20 | IG renatobo - IMG_5013.jpeg | 0.1836 | 0.5817 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 21 | IG desmo.donna - IMG-1080-Donna.jpeg | 0.1830 | 0.5760 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 22 | IG desmo.donna - IMG-1015-Donna.jpeg | 0.1827 | 0.5732 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 23 | AB Group - Session 2 (Turn 11) - CVR_1047_Mar1025_908AM_CaliPhoto.jpg | 0.1811 | 0.5583 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 24 | 1-Around the Pits - 1-Around the Pits - CVR_0025_Mar1025_728AM_CaliPhoto.jpg | 0.1801 | 0.5483 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 25 | IMG_4984.jpeg | 0.1794 | 0.5421 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 26 | IG guardiansvoice - IMG-4129-Delenian.jpeg | 0.1793 | 0.5413 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 27 | 2-Group Photo - Raffle - CVR_7682_Mar1025_1249PM_CaliPhoto.jpg | 0.1787 | 0.5357 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 28 | IG - renatobo - March 10 DROC Track Day-58 HDR.jpeg | 0.1785 | 0.5335 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 29 | IG - renatobo - lineup 1st session - March 10 DROC Track Day-12 HDR.jpeg | 0.1763 | 0.5123 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 30 | C Group - Session 3 (Turn 9 and 8) - CVR_4982_Mar1025_1040AM_CaliPhoto.jpg | 0.1748 | 0.4978 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 31 | IG guardiansvoice - IMG-4160-Delenian.jpeg | 0.1746 | 0.4960 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 32 | IG - renatobo - lineup 1st session - March 10 DROC Track Day-02 HDR.jpeg | 0.1740 | 0.4907 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 33 | IG martangelenos - PXL_20250310_132329430.jpg | 0.1733 | 0.4839 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 34 | IG - renatobo - March 10 DROC Track Day-28.jpeg | 0.1733 | 0.4833 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 35 | IG - renatobo - March 10 DROC Track Day-26 HDR.jpeg | 0.1726 | 0.4768 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 36 | IG ocredevil - DROC-Track-Day-3-10-2-filippo pagan.jpeg | 0.1678 | 0.4312 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 37 | IG desmo.donna - IMG-1151-Donna.jpeg | 0.1669 | 0.4230 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 38 | IMG_5012.jpeg | 0.1665 | 0.4195 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 39 | 1-Around the Pits - 1-Around the Pits - CVR_0012_Mar1025_702AM_CaliPhoto.jpg | 0.1642 | 0.3975 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 40 | IG renatobo - IMG_5014.jpeg | 0.1620 | 0.3765 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 41 | IG - renatobo - lineup 1st session - March 10 DROC Track Day-01 HDR.jpeg | 0.1620 | 0.3764 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 42 | IG kamiumitv - IMG-3981-Bao.jpeg | 0.0575 | 0.6171 | 17 | 3 | 3 | 3 | 3 | 2 | 3 | no | Vision scoring summary |

### VladimirGav/gemma4-26b-16GB-VRAM | yolo=off

| Rank | Image | Final | Tech | Vision | Subject | Light | Color | Emotion | Scroll | Crop | Failed | One line |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 1-Around the Pits - CVR_0098_Mar1025_802AM_CaliPhoto.jpg | 0.2471 | 0.8237 |  |  |  |  |  |  |  | yes | Vision scoring failed — ranked by technical score only |
| 2 | C Group - Session 2 Back Straight Speed Pans - CVR_2628_Mar1025_943AM_CaliPhoto.jpg | 0.2348 | 0.7827 |  |  |  |  |  |  |  | yes | Vision scoring failed — ranked by technical score only |
| 3 | IG naramedia_official - DSC-5365-NaraMedia.jpeg | 0.2334 | 0.7780 |  |  |  |  |  |  |  | yes | Vision scoring failed — ranked by technical score only |
| 4 | C Group - Session 2 Back Straight Speed Pans - CVR_2850_Mar1025_946AM_CaliPhoto.jpg | 0.2298 | 0.7661 |  |  |  |  |  |  |  | yes | Vision scoring failed — ranked by technical score only |
| 5 | IG cali_carnivores - DSC09850.jpg | 0.2284 | 0.7612 |  |  |  |  |  |  |  | yes | Vision scoring failed — ranked by technical score only |
| 6 | IG naramedia_official - DSC-5992-NaraMedia.jpeg | 0.2271 | 0.7571 |  |  |  |  |  |  |  | yes | Vision scoring failed — ranked by technical score only |
| 7 | IG cali_carnivores - DSC00013.jpg | 0.2220 | 0.7401 |  |  |  |  |  |  |  | yes | Vision scoring failed — ranked by technical score only |
| 8 | AB Group - Session 4 (Turn 16) - CVR_6545_Mar1025_1122AM_CaliPhoto.jpg | 0.2175 | 0.7249 |  |  |  |  |  |  |  | yes | Ollama circuit breaker active — ranked by technical score only |
| 9 | 1-Around the Pits - CVR_0792_Mar1025_832AM_CaliPhoto.jpg | 0.2007 | 0.6688 |  |  |  |  |  |  |  | yes | Ollama circuit breaker active — ranked by technical score only |
| 10 | IG naramedia_official - DSC-5897-NaraMedia.jpeg | 0.1951 | 0.6504 |  |  |  |  |  |  |  | yes | Ollama circuit breaker active — ranked by technical score only |
| 11 | caliphotovideo - Group picture - DROC Track Day March 10 202500002.jpg | 0.1945 | 0.6484 |  |  |  |  |  |  |  | yes | Ollama circuit breaker active — ranked by technical score only |
| 12 | IG luckie.moto - 20250310-075123-luckie.moto.jpg | 0.1912 | 0.6374 |  |  |  |  |  |  |  | yes | Ollama circuit breaker active — ranked by technical score only |
| 13 | caliphotovideo - Group picture - DROC Track Day March 10 202500001.jpg | 0.1908 | 0.6360 |  |  |  |  |  |  |  | yes | Ollama circuit breaker active — ranked by technical score only |
| 14 | IG m92663m - IMG-6019-Mark Momot.jpeg | 0.1901 | 0.6338 |  |  |  |  |  |  |  | yes | Ollama circuit breaker active — ranked by technical score only |
| 15 | DSC-5436-NaraMedia.jpeg | 0.1886 | 0.6287 |  |  |  |  |  |  |  | yes | Ollama circuit breaker active — ranked by technical score only |
| 16 | AB Group - Session 1 (Turn 16 Entry) - CVR_0148_Mar1025_815AM_CaliPhoto.jpg | 0.1857 | 0.6190 |  |  |  |  |  |  |  | yes | Ollama circuit breaker active — ranked by technical score only |
| 17 | IG kamiumitv - IMG-3981-Bao.jpeg | 0.1851 | 0.6171 |  |  |  |  |  |  |  | yes | Ollama circuit breaker active — ranked by technical score only |
| 18 | IG cali_carnivores - DSC09857.jpg | 0.1776 | 0.5921 |  |  |  |  |  |  |  | yes | Ollama circuit breaker active — ranked by technical score only |
| 19 | AB Group - Session 1 (Turn 16 Entry) - CVR_0230_Mar1025_817AM_CaliPhoto.jpg | 0.1758 | 0.5860 |  |  |  |  |  |  |  | yes | Ollama circuit breaker active — ranked by technical score only |
| 20 | 2-Group Photo - Raffle - CVR_7677_Mar1025_1249PM_CaliPhoto.jpg | 0.1756 | 0.5853 |  |  |  |  |  |  |  | yes | Ollama circuit breaker active — ranked by technical score only |
| 21 | IG renatobo - IMG_5013.jpeg | 0.1745 | 0.5817 |  |  |  |  |  |  |  | yes | Ollama circuit breaker active — ranked by technical score only |
| 22 | IG desmo.donna - IMG-1080-Donna.jpeg | 0.1728 | 0.5760 |  |  |  |  |  |  |  | yes | Ollama circuit breaker active — ranked by technical score only |
| 23 | IG desmo.donna - IMG-1015-Donna.jpeg | 0.1720 | 0.5732 |  |  |  |  |  |  |  | yes | Ollama circuit breaker active — ranked by technical score only |
| 24 | AB Group - Session 2 (Turn 11) - CVR_1047_Mar1025_908AM_CaliPhoto.jpg | 0.1675 | 0.5583 |  |  |  |  |  |  |  | yes | Ollama circuit breaker active — ranked by technical score only |
| 25 | 1-Around the Pits - 1-Around the Pits - CVR_0025_Mar1025_728AM_CaliPhoto.jpg | 0.1645 | 0.5483 |  |  |  |  |  |  |  | yes | Ollama circuit breaker active — ranked by technical score only |
| 26 | IMG_4984.jpeg | 0.1626 | 0.5421 |  |  |  |  |  |  |  | yes | Ollama circuit breaker active — ranked by technical score only |
| 27 | IG guardiansvoice - IMG-4129-Delenian.jpeg | 0.1624 | 0.5413 |  |  |  |  |  |  |  | yes | Ollama circuit breaker active — ranked by technical score only |
| 28 | 2-Group Photo - Raffle - CVR_7682_Mar1025_1249PM_CaliPhoto.jpg | 0.1607 | 0.5357 |  |  |  |  |  |  |  | yes | Ollama circuit breaker active — ranked by technical score only |
| 29 | IG - renatobo - March 10 DROC Track Day-58 HDR.jpeg | 0.1600 | 0.5335 |  |  |  |  |  |  |  | yes | Ollama circuit breaker active — ranked by technical score only |
| 30 | IG - renatobo - lineup 1st session - March 10 DROC Track Day-12 HDR.jpeg | 0.1537 | 0.5123 |  |  |  |  |  |  |  | yes | Ollama circuit breaker active — ranked by technical score only |
| 31 | C Group - Session 3 (Turn 9 and 8) - CVR_4982_Mar1025_1040AM_CaliPhoto.jpg | 0.1493 | 0.4978 |  |  |  |  |  |  |  | yes | Ollama circuit breaker active — ranked by technical score only |
| 32 | IG guardiansvoice - IMG-4160-Delenian.jpeg | 0.1488 | 0.4960 |  |  |  |  |  |  |  | yes | Ollama circuit breaker active — ranked by technical score only |
| 33 | IG - renatobo - lineup 1st session - March 10 DROC Track Day-02 HDR.jpeg | 0.1472 | 0.4907 |  |  |  |  |  |  |  | yes | Ollama circuit breaker active — ranked by technical score only |
| 34 | IG martangelenos - PXL_20250310_132329430.jpg | 0.1452 | 0.4839 |  |  |  |  |  |  |  | yes | Ollama circuit breaker active — ranked by technical score only |
| 35 | IG - renatobo - March 10 DROC Track Day-28.jpeg | 0.1450 | 0.4833 |  |  |  |  |  |  |  | yes | Ollama circuit breaker active — ranked by technical score only |
| 36 | IG - renatobo - March 10 DROC Track Day-26 HDR.jpeg | 0.1430 | 0.4768 |  |  |  |  |  |  |  | yes | Ollama circuit breaker active — ranked by technical score only |
| 37 | IG ocredevil - DROC-Track-Day-3-10-2-filippo pagan.jpeg | 0.1294 | 0.4312 |  |  |  |  |  |  |  | yes | Ollama circuit breaker active — ranked by technical score only |
| 38 | IG desmo.donna - IMG-1151-Donna.jpeg | 0.1269 | 0.4230 |  |  |  |  |  |  |  | yes | Ollama circuit breaker active — ranked by technical score only |
| 39 | IMG_5012.jpeg | 0.1258 | 0.4195 |  |  |  |  |  |  |  | yes | Ollama circuit breaker active — ranked by technical score only |
| 40 | 1-Around the Pits - 1-Around the Pits - CVR_0012_Mar1025_702AM_CaliPhoto.jpg | 0.1193 | 0.3975 |  |  |  |  |  |  |  | yes | Ollama circuit breaker active — ranked by technical score only |
| 41 | IG renatobo - IMG_5014.jpeg | 0.1130 | 0.3765 |  |  |  |  |  |  |  | yes | Ollama circuit breaker active — ranked by technical score only |
| 42 | IG - renatobo - lineup 1st session - March 10 DROC Track Day-01 HDR.jpeg | 0.1129 | 0.3764 |  |  |  |  |  |  |  | yes | Ollama circuit breaker active — ranked by technical score only |
