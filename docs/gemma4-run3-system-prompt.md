# Model Benchmark Report (Speed + Quality)

- Generated: `2026-04-09T10:22:31`
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
| gemma4:e4b \| yolo=off \| prompt=system | ollama | gemma4:e4b | off | 43.73 | 1.37 | 1836.51 | 0.00 | 0.04 | 8 | 1.00x |

## Per-run Timing

| Variant | Run | Duration (s) | Sec/img | Imgs/min | Failures |
|---|---:|---:|---:|---:|---:|
| gemma4:e4b \| yolo=off \| prompt=system | 1 | 1836.51 | 43.73 | 1.37 | 0 |

## Image-by-Image Score Comparison (Run 1)

| Image | gemma4:e4b \| yolo=off \| prompt=system final | gemma4:e4b \| yolo=off \| prompt=system rank |
|---|---:|---:|
| 1-Around the Pits - 1-Around the Pits - CVR_0012_Mar1025_702AM_CaliPhoto.jpg | 0.0494 | 42 |
| 1-Around the Pits - 1-Around the Pits - CVR_0025_Mar1025_728AM_CaliPhoto.jpg | 0.5236 | 8 |
| 1-Around the Pits - CVR_0098_Mar1025_802AM_CaliPhoto.jpg | 0.1931 | 26 |
| 1-Around the Pits - CVR_0792_Mar1025_832AM_CaliPhoto.jpg | 0.2153 | 21 |
| 2-Group Photo - Raffle - CVR_7677_Mar1025_1249PM_CaliPhoto.jpg | 0.3574 | 14 |
| 2-Group Photo - Raffle - CVR_7682_Mar1025_1249PM_CaliPhoto.jpg | 0.3484 | 16 |
| AB Group - Session 1 (Turn 16 Entry) - CVR_0148_Mar1025_815AM_CaliPhoto.jpg | 0.5406 | 7 |
| AB Group - Session 1 (Turn 16 Entry) - CVR_0230_Mar1025_817AM_CaliPhoto.jpg | 0.7941 | 2 |
| AB Group - Session 2 (Turn 11) - CVR_1047_Mar1025_908AM_CaliPhoto.jpg | 0.7975 | 1 |
| AB Group - Session 4 (Turn 16) - CVR_6545_Mar1025_1122AM_CaliPhoto.jpg | 0.1986 | 25 |
| C Group - Session 2 Back Straight Speed Pans - CVR_2628_Mar1025_943AM_CaliPhoto.jpg | 0.3929 | 11 |
| C Group - Session 2 Back Straight Speed Pans - CVR_2850_Mar1025_946AM_CaliPhoto.jpg | 0.2029 | 23 |
| C Group - Session 3 (Turn 9 and 8) - CVR_4982_Mar1025_1040AM_CaliPhoto.jpg | 0.1748 | 36 |
| DSC-5436-NaraMedia.jpeg | 0.5429 | 6 |
| IG - renatobo - March 10 DROC Track Day-26 HDR.jpeg | 0.1726 | 37 |
| IG - renatobo - March 10 DROC Track Day-28.jpeg | 0.0638 | 41 |
| IG - renatobo - March 10 DROC Track Day-58 HDR.jpeg | 0.3480 | 17 |
| IG - renatobo - lineup 1st session - March 10 DROC Track Day-01 HDR.jpeg | 0.4823 | 10 |
| IG - renatobo - lineup 1st session - March 10 DROC Track Day-02 HDR.jpeg | 0.7072 | 4 |
| IG - renatobo - lineup 1st session - March 10 DROC Track Day-12 HDR.jpeg | 0.5149 | 9 |
| IG cali_carnivores - DSC00013.jpg | 0.3852 | 12 |
| IG cali_carnivores - DSC09850.jpg | 0.2024 | 24 |
| IG cali_carnivores - DSC09857.jpg | 0.1847 | 31 |
| IG desmo.donna - IMG-1015-Donna.jpeg | 0.1827 | 34 |
| IG desmo.donna - IMG-1080-Donna.jpeg | 0.1830 | 33 |
| IG desmo.donna - IMG-1151-Donna.jpeg | 0.1669 | 39 |
| IG guardiansvoice - IMG-4129-Delenian.jpeg | 0.1793 | 35 |
| IG guardiansvoice - IMG-4160-Delenian.jpeg | 0.3413 | 18 |
| IG kamiumitv - IMG-3981-Bao.jpeg | 0.7451 | 3 |
| IG luckie.moto - 20250310-075123-luckie.moto.jpg | 0.1894 | 28 |
| IG m92663m - IMG-6019-Mark Momot.jpeg | 0.1891 | 30 |
| IG martangelenos - PXL_20250310_132329430.jpg | 0.0638 | 40 |
| IG naramedia_official - DSC-5365-NaraMedia.jpeg | 0.2042 | 22 |
| IG naramedia_official - DSC-5897-NaraMedia.jpeg | 0.1908 | 27 |
| IG naramedia_official - DSC-5992-NaraMedia.jpeg | 0.5737 | 5 |
| IG ocredevil - DROC-Track-Day-3-10-2-filippo pagan.jpeg | 0.1678 | 38 |
| IG renatobo - IMG_5013.jpeg | 0.1836 | 32 |
| IG renatobo - IMG_5014.jpeg | 0.3198 | 20 |
| IMG_4984.jpeg | 0.3496 | 15 |
| IMG_5012.jpeg | 0.3275 | 19 |
| caliphotovideo - Group picture - DROC Track Day March 10 202500001.jpg | 0.1893 | 29 |
| caliphotovideo - Group picture - DROC Track Day March 10 202500002.jpg | 0.3687 | 13 |

## Ranked Quality Details (All Images)

_Each section below is from timed run 1 for that variant._

### gemma4:e4b | yolo=off | prompt=system

| Rank | Image | Final | Tech | Vision | Subject | Light | Color | Emotion | Scroll | Crop | Failed | One line |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 | AB Group - Session 2 (Turn 11) - CVR_1047_Mar1025_908AM_CaliPhoto.jpg | 0.7975 | 0.5583 | 54 | 9 | 9 | 9 | 9 | 9 | 9 | no | Vision scoring summary |
| 2 | AB Group - Session 1 (Turn 16 Entry) - CVR_0230_Mar1025_817AM_CaliPhoto.jpg | 0.7941 | 0.5860 | 53 | 10 | 6 | 9 | 10 | 9 | 9 | no | Vision scoring summary |
| 3 | IG kamiumitv - IMG-3981-Bao.jpeg | 0.7451 | 0.6171 | 48 | 8 | 8 | 8 | 8 | 8 | 8 | no | Vision scoring summary |
| 4 | IG - renatobo - lineup 1st session - March 10 DROC Track Day-02 HDR.jpeg | 0.7072 | 0.4907 | 48 | 8 | 8 | 8 | 8 | 8 | 8 | no | Vision scoring summary |
| 5 | IG naramedia_official - DSC-5992-NaraMedia.jpeg | 0.5737 | 0.7571 | 42 | 7 | 7 | 7 | 7 | 7 | 7 | no | Vision scoring summary |
| 6 | DSC-5436-NaraMedia.jpeg | 0.5429 | 0.6287 | 42 | 7 | 7 | 7 | 7 | 7 | 7 | no | Vision scoring summary |
| 7 | AB Group - Session 1 (Turn 16 Entry) - CVR_0148_Mar1025_815AM_CaliPhoto.jpg | 0.5406 | 0.6190 | 42 | 7 | 7 | 7 | 7 | 7 | 7 | no | Vision scoring summary |
| 8 | 1-Around the Pits - 1-Around the Pits - CVR_0025_Mar1025_728AM_CaliPhoto.jpg | 0.5236 | 0.5483 | 42 | 7 | 7 | 7 | 7 | 7 | 7 | no | Vision scoring summary |
| 9 | IG - renatobo - lineup 1st session - March 10 DROC Track Day-12 HDR.jpeg | 0.5149 | 0.5123 | 42 | 7 | 7 | 7 | 7 | 7 | 7 | no | Vision scoring summary |
| 10 | IG - renatobo - lineup 1st session - March 10 DROC Track Day-01 HDR.jpeg | 0.4823 | 0.3764 | 42 | 7 | 7 | 7 | 7 | 7 | 7 | no | Vision scoring summary |
| 11 | C Group - Session 2 Back Straight Speed Pans - CVR_2628_Mar1025_943AM_CaliPhoto.jpg | 0.3929 | 0.7827 | 36 | 6 | 6 | 6 | 6 | 6 | 6 | no | Vision scoring summary |
| 12 | IG cali_carnivores - DSC00013.jpg | 0.3852 | 0.7401 | 36 | 6 | 6 | 6 | 6 | 6 | 6 | no | Vision scoring summary |
| 13 | caliphotovideo - Group picture - DROC Track Day March 10 202500002.jpg | 0.3687 | 0.6484 | 36 | 6 | 6 | 6 | 6 | 6 | 6 | no | Vision scoring summary |
| 14 | 2-Group Photo - Raffle - CVR_7677_Mar1025_1249PM_CaliPhoto.jpg | 0.3574 | 0.5853 | 36 | 6 | 6 | 6 | 6 | 6 | 6 | no | Vision scoring summary |
| 15 | IMG_4984.jpeg | 0.3496 | 0.5421 | 36 | 6 | 6 | 6 | 6 | 6 | 6 | no | Vision scoring summary |
| 16 | 2-Group Photo - Raffle - CVR_7682_Mar1025_1249PM_CaliPhoto.jpg | 0.3484 | 0.5357 | 36 | 6 | 6 | 6 | 6 | 6 | 6 | no | Vision scoring summary |
| 17 | IG - renatobo - March 10 DROC Track Day-58 HDR.jpeg | 0.3480 | 0.5335 | 36 | 6 | 6 | 6 | 6 | 6 | 6 | no | Vision scoring summary |
| 18 | IG guardiansvoice - IMG-4160-Delenian.jpeg | 0.3413 | 0.4960 | 36 | 6 | 6 | 6 | 6 | 6 | 6 | no | Vision scoring summary |
| 19 | IMG_5012.jpeg | 0.3275 | 0.4195 | 36 | 6 | 6 | 6 | 6 | 6 | 6 | no | Vision scoring summary |
| 20 | IG renatobo - IMG_5014.jpeg | 0.3198 | 0.3765 | 36 | 6 | 6 | 6 | 6 | 6 | 6 | no | Vision scoring summary |
| 21 | 1-Around the Pits - CVR_0792_Mar1025_832AM_CaliPhoto.jpg | 0.2153 | 0.8839 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 22 | IG naramedia_official - DSC-5365-NaraMedia.jpeg | 0.2042 | 0.7780 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 23 | C Group - Session 2 Back Straight Speed Pans - CVR_2850_Mar1025_946AM_CaliPhoto.jpg | 0.2029 | 0.7661 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 24 | IG cali_carnivores - DSC09850.jpg | 0.2024 | 0.7612 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 25 | AB Group - Session 4 (Turn 16) - CVR_6545_Mar1025_1122AM_CaliPhoto.jpg | 0.1986 | 0.7249 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 26 | 1-Around the Pits - CVR_0098_Mar1025_802AM_CaliPhoto.jpg | 0.1931 | 0.6723 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 27 | IG naramedia_official - DSC-5897-NaraMedia.jpeg | 0.1908 | 0.6504 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 28 | IG luckie.moto - 20250310-075123-luckie.moto.jpg | 0.1894 | 0.6374 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 29 | caliphotovideo - Group picture - DROC Track Day March 10 202500001.jpg | 0.1893 | 0.6360 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 30 | IG m92663m - IMG-6019-Mark Momot.jpeg | 0.1891 | 0.6338 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 31 | IG cali_carnivores - DSC09857.jpg | 0.1847 | 0.5921 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 32 | IG renatobo - IMG_5013.jpeg | 0.1836 | 0.5817 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 33 | IG desmo.donna - IMG-1080-Donna.jpeg | 0.1830 | 0.5760 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 34 | IG desmo.donna - IMG-1015-Donna.jpeg | 0.1827 | 0.5732 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 35 | IG guardiansvoice - IMG-4129-Delenian.jpeg | 0.1793 | 0.5413 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 36 | C Group - Session 3 (Turn 9 and 8) - CVR_4982_Mar1025_1040AM_CaliPhoto.jpg | 0.1748 | 0.4978 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 37 | IG - renatobo - March 10 DROC Track Day-26 HDR.jpeg | 0.1726 | 0.4768 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 38 | IG ocredevil - DROC-Track-Day-3-10-2-filippo pagan.jpeg | 0.1678 | 0.4312 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 39 | IG desmo.donna - IMG-1151-Donna.jpeg | 0.1669 | 0.4230 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 40 | IG martangelenos - PXL_20250310_132329430.jpg | 0.0638 | 0.4839 | 24 | 4 | 4 | 4 | 4 | 4 | 4 | no | Vision scoring summary |
| 41 | IG - renatobo - March 10 DROC Track Day-28.jpeg | 0.0638 | 0.4833 | 24 | 4 | 4 | 4 | 4 | 4 | 4 | no | Vision scoring summary |
| 42 | 1-Around the Pits - 1-Around the Pits - CVR_0012_Mar1025_702AM_CaliPhoto.jpg | 0.0494 | 0.3975 | 18 | 3 | 3 | 3 | 3 | 3 | 3 | no | Vision scoring summary |
