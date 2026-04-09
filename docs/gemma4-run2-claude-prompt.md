# Model Benchmark Report (Speed + Quality)

- Generated: `2026-04-09T09:51:04`
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
| gemma4:e4b \| yolo=off \| prompt=claude | ollama | gemma4:e4b | off | 28.41 | 2.11 | 1193.24 | 0.00 | 0.00 | 8 | 1.00x |

## Per-run Timing

| Variant | Run | Duration (s) | Sec/img | Imgs/min | Failures |
|---|---:|---:|---:|---:|---:|
| gemma4:e4b \| yolo=off \| prompt=claude | 1 | 1193.24 | 28.41 | 2.11 | 0 |

## Image-by-Image Score Comparison (Run 1)

| Image | gemma4:e4b \| yolo=off \| prompt=claude final | gemma4:e4b \| yolo=off \| prompt=claude rank |
|---|---:|---:|
| 1-Around the Pits - 1-Around the Pits - CVR_0012_Mar1025_702AM_CaliPhoto.jpg | 0.0599 | 39 |
| 1-Around the Pits - 1-Around the Pits - CVR_0025_Mar1025_728AM_CaliPhoto.jpg | 0.7245 | 19 |
| 1-Around the Pits - CVR_0098_Mar1025_802AM_CaliPhoto.jpg | 0.7617 | 14 |
| 1-Around the Pits - CVR_0792_Mar1025_832AM_CaliPhoto.jpg | 0.8307 | 2 |
| 2-Group Photo - Raffle - CVR_7677_Mar1025_1249PM_CaliPhoto.jpg | 0.0683 | 36 |
| 2-Group Photo - Raffle - CVR_7682_Mar1025_1249PM_CaliPhoto.jpg | 0.0556 | 40 |
| AB Group - Session 1 (Turn 16 Entry) - CVR_0148_Mar1025_815AM_CaliPhoto.jpg | 0.8157 | 3 |
| AB Group - Session 1 (Turn 16 Entry) - CVR_0230_Mar1025_817AM_CaliPhoto.jpg | 0.8058 | 5 |
| AB Group - Session 2 (Turn 11) - CVR_1047_Mar1025_908AM_CaliPhoto.jpg | 0.7681 | 13 |
| AB Group - Session 4 (Turn 16) - CVR_6545_Mar1025_1122AM_CaliPhoto.jpg | 0.7775 | 12 |
| C Group - Session 2 Back Straight Speed Pans - CVR_2628_Mar1025_943AM_CaliPhoto.jpg | 0.5799 | 22 |
| C Group - Session 2 Back Straight Speed Pans - CVR_2850_Mar1025_946AM_CaliPhoto.jpg | 0.7898 | 9 |
| C Group - Session 3 (Turn 9 and 8) - CVR_4982_Mar1025_1040AM_CaliPhoto.jpg | 0.7093 | 20 |
| DSC-5436-NaraMedia.jpeg | 0.5429 | 23 |
| IG - renatobo - March 10 DROC Track Day-26 HDR.jpeg | 0.5064 | 24 |
| IG - renatobo - March 10 DROC Track Day-28.jpeg | 0.0323 | 42 |
| IG - renatobo - March 10 DROC Track Day-58 HDR.jpeg | 0.1785 | 32 |
| IG - renatobo - lineup 1st session - March 10 DROC Track Day-01 HDR.jpeg | 0.7429 | 18 |
| IG - renatobo - lineup 1st session - March 10 DROC Track Day-02 HDR.jpeg | 0.7072 | 21 |
| IG - renatobo - lineup 1st session - March 10 DROC Track Day-12 HDR.jpeg | 0.1763 | 33 |
| IG cali_carnivores - DSC00013.jpg | 0.8520 | 1 |
| IG cali_carnivores - DSC09850.jpg | 0.7884 | 10 |
| IG cali_carnivores - DSC09857.jpg | 0.8076 | 4 |
| IG desmo.donna - IMG-1015-Donna.jpeg | 0.1827 | 29 |
| IG desmo.donna - IMG-1080-Donna.jpeg | 0.8028 | 7 |
| IG desmo.donna - IMG-1151-Donna.jpeg | 0.0610 | 37 |
| IG guardiansvoice - IMG-4129-Delenian.jpeg | 0.1793 | 31 |
| IG guardiansvoice - IMG-4160-Delenian.jpeg | 0.0538 | 41 |
| IG kamiumitv - IMG-3981-Bao.jpeg | 0.7451 | 17 |
| IG luckie.moto - 20250310-075123-luckie.moto.jpg | 0.1894 | 28 |
| IG m92663m - IMG-6019-Mark Momot.jpeg | 0.0705 | 35 |
| IG martangelenos - PXL_20250310_132329430.jpg | 0.1733 | 34 |
| IG naramedia_official - DSC-5365-NaraMedia.jpeg | 0.7934 | 8 |
| IG naramedia_official - DSC-5897-NaraMedia.jpeg | 0.7551 | 16 |
| IG naramedia_official - DSC-5992-NaraMedia.jpeg | 0.7871 | 11 |
| IG ocredevil - DROC-Track-Day-3-10-2-filippo pagan.jpeg | 0.3296 | 27 |
| IG renatobo - IMG_5013.jpeg | 0.8045 | 6 |
| IG renatobo - IMG_5014.jpeg | 0.4824 | 25 |
| IMG_4984.jpeg | 0.1794 | 30 |
| IMG_5012.jpeg | 0.7558 | 15 |
| caliphotovideo - Group picture - DROC Track Day March 10 202500001.jpg | 0.3665 | 26 |
| caliphotovideo - Group picture - DROC Track Day March 10 202500002.jpg | 0.0607 | 38 |

## Ranked Quality Details (All Images)

_Each section below is from timed run 1 for that variant._

### gemma4:e4b | yolo=off | prompt=claude

| Rank | Image | Final | Tech | Vision | Subject | Light | Color | Emotion | Scroll | Crop | Failed | One line |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 | IG cali_carnivores - DSC00013.jpg | 0.8520 | 0.7401 | 54 | 9 | 9 | 9 | 9 | 9 | 9 | no | Vision scoring summary |
| 2 | 1-Around the Pits - CVR_0792_Mar1025_832AM_CaliPhoto.jpg | 0.8307 | 0.6688 | 54 | 9 | 9 | 9 | 9 | 9 | 9 | no | Vision scoring summary |
| 3 | AB Group - Session 1 (Turn 16 Entry) - CVR_0148_Mar1025_815AM_CaliPhoto.jpg | 0.8157 | 0.6190 | 54 | 9 | 9 | 9 | 9 | 9 | 9 | no | Vision scoring summary |
| 4 | IG cali_carnivores - DSC09857.jpg | 0.8076 | 0.5921 | 54 | 9 | 9 | 9 | 9 | 9 | 9 | no | Vision scoring summary |
| 5 | AB Group - Session 1 (Turn 16 Entry) - CVR_0230_Mar1025_817AM_CaliPhoto.jpg | 0.8058 | 0.5860 | 54 | 9 | 9 | 9 | 9 | 9 | 9 | no | Vision scoring summary |
| 6 | IG renatobo - IMG_5013.jpeg | 0.8045 | 0.5817 | 54 | 9 | 9 | 9 | 9 | 9 | 9 | no | Vision scoring summary |
| 7 | IG desmo.donna - IMG-1080-Donna.jpeg | 0.8028 | 0.5760 | 54 | 9 | 9 | 9 | 9 | 9 | 9 | no | Vision scoring summary |
| 8 | IG naramedia_official - DSC-5365-NaraMedia.jpeg | 0.7934 | 0.7780 | 48 | 8 | 8 | 8 | 8 | 8 | 8 | no | Vision scoring summary |
| 9 | C Group - Session 2 Back Straight Speed Pans - CVR_2850_Mar1025_946AM_CaliPhoto.jpg | 0.7898 | 0.7661 | 48 | 8 | 8 | 8 | 8 | 8 | 8 | no | Vision scoring summary |
| 10 | IG cali_carnivores - DSC09850.jpg | 0.7884 | 0.7612 | 48 | 8 | 8 | 8 | 8 | 8 | 8 | no | Vision scoring summary |
| 11 | IG naramedia_official - DSC-5992-NaraMedia.jpeg | 0.7871 | 0.7571 | 48 | 8 | 8 | 8 | 8 | 8 | 8 | no | Vision scoring summary |
| 12 | AB Group - Session 4 (Turn 16) - CVR_6545_Mar1025_1122AM_CaliPhoto.jpg | 0.7775 | 0.7249 | 48 | 8 | 8 | 8 | 8 | 8 | 8 | no | Vision scoring summary |
| 13 | AB Group - Session 2 (Turn 11) - CVR_1047_Mar1025_908AM_CaliPhoto.jpg | 0.7681 | 0.6938 | 48 | 8 | 8 | 8 | 8 | 8 | 8 | no | Vision scoring summary |
| 14 | 1-Around the Pits - CVR_0098_Mar1025_802AM_CaliPhoto.jpg | 0.7617 | 0.6723 | 48 | 8 | 8 | 8 | 8 | 8 | 8 | no | Vision scoring summary |
| 15 | IMG_5012.jpeg | 0.7558 | 0.4195 | 54 | 9 | 9 | 9 | 9 | 9 | 9 | no | Vision scoring summary |
| 16 | IG naramedia_official - DSC-5897-NaraMedia.jpeg | 0.7551 | 0.6504 | 48 | 8 | 8 | 8 | 8 | 8 | 8 | no | Vision scoring summary |
| 17 | IG kamiumitv - IMG-3981-Bao.jpeg | 0.7451 | 0.6171 | 48 | 8 | 8 | 8 | 8 | 8 | 8 | no | Vision scoring summary |
| 18 | IG - renatobo - lineup 1st session - March 10 DROC Track Day-01 HDR.jpeg | 0.7429 | 0.3764 | 54 | 9 | 9 | 9 | 9 | 9 | 9 | no | Vision scoring summary |
| 19 | 1-Around the Pits - 1-Around the Pits - CVR_0025_Mar1025_728AM_CaliPhoto.jpg | 0.7245 | 0.5483 | 48 | 8 | 8 | 8 | 8 | 8 | 8 | no | Vision scoring summary |
| 20 | C Group - Session 3 (Turn 9 and 8) - CVR_4982_Mar1025_1040AM_CaliPhoto.jpg | 0.7093 | 0.4978 | 48 | 8 | 8 | 8 | 8 | 8 | 8 | no | Vision scoring summary |
| 21 | IG - renatobo - lineup 1st session - March 10 DROC Track Day-02 HDR.jpeg | 0.7072 | 0.4907 | 48 | 8 | 8 | 8 | 8 | 8 | 8 | no | Vision scoring summary |
| 22 | C Group - Session 2 Back Straight Speed Pans - CVR_2628_Mar1025_943AM_CaliPhoto.jpg | 0.5799 | 0.7827 | 42 | 7 | 7 | 7 | 7 | 7 | 7 | no | Vision scoring summary |
| 23 | DSC-5436-NaraMedia.jpeg | 0.5429 | 0.6287 | 42 | 7 | 7 | 7 | 7 | 7 | 7 | no | Vision scoring summary |
| 24 | IG - renatobo - March 10 DROC Track Day-26 HDR.jpeg | 0.5064 | 0.4768 | 42 | 7 | 7 | 7 | 7 | 7 | 7 | no | Vision scoring summary |
| 25 | IG renatobo - IMG_5014.jpeg | 0.4824 | 0.3765 | 42 | 7 | 7 | 7 | 7 | 7 | 7 | no | Vision scoring summary |
| 26 | caliphotovideo - Group picture - DROC Track Day March 10 202500001.jpg | 0.3665 | 0.6360 | 36 | 6 | 6 | 6 | 6 | 6 | 6 | no | Vision scoring summary |
| 27 | IG ocredevil - DROC-Track-Day-3-10-2-filippo pagan.jpeg | 0.3296 | 0.4312 | 36 | 6 | 6 | 6 | 6 | 6 | 6 | no | Vision scoring summary |
| 28 | IG luckie.moto - 20250310-075123-luckie.moto.jpg | 0.1894 | 0.6374 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 29 | IG desmo.donna - IMG-1015-Donna.jpeg | 0.1827 | 0.5732 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 30 | IMG_4984.jpeg | 0.1794 | 0.5421 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 31 | IG guardiansvoice - IMG-4129-Delenian.jpeg | 0.1793 | 0.5413 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 32 | IG - renatobo - March 10 DROC Track Day-58 HDR.jpeg | 0.1785 | 0.5335 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 33 | IG - renatobo - lineup 1st session - March 10 DROC Track Day-12 HDR.jpeg | 0.1763 | 0.5123 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 34 | IG martangelenos - PXL_20250310_132329430.jpg | 0.1733 | 0.4839 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 35 | IG m92663m - IMG-6019-Mark Momot.jpeg | 0.0705 | 0.6338 | 24 | 4 | 4 | 4 | 4 | 4 | 4 | no | Vision scoring summary |
| 36 | 2-Group Photo - Raffle - CVR_7677_Mar1025_1249PM_CaliPhoto.jpg | 0.0683 | 0.5853 | 24 | 4 | 4 | 4 | 4 | 4 | 4 | no | Vision scoring summary |
| 37 | IG desmo.donna - IMG-1151-Donna.jpeg | 0.0610 | 0.4230 | 24 | 4 | 4 | 4 | 4 | 4 | 4 | no | Vision scoring summary |
| 38 | caliphotovideo - Group picture - DROC Track Day March 10 202500002.jpg | 0.0607 | 0.6484 | 18 | 3 | 3 | 3 | 3 | 3 | 3 | no | Vision scoring summary |
| 39 | 1-Around the Pits - 1-Around the Pits - CVR_0012_Mar1025_702AM_CaliPhoto.jpg | 0.0599 | 0.3975 | 24 | 4 | 4 | 4 | 4 | 4 | 4 | no | Vision scoring summary |
| 40 | 2-Group Photo - Raffle - CVR_7682_Mar1025_1249PM_CaliPhoto.jpg | 0.0556 | 0.5357 | 18 | 3 | 3 | 3 | 3 | 3 | 3 | no | Vision scoring summary |
| 41 | IG guardiansvoice - IMG-4160-Delenian.jpeg | 0.0538 | 0.4960 | 18 | 3 | 3 | 3 | 3 | 3 | 3 | no | Vision scoring summary |
| 42 | IG - renatobo - March 10 DROC Track Day-28.jpeg | 0.0323 | 0.4833 | 6 | 1 | 1 | 1 | 1 | 1 | 1 | no | Vision scoring summary |
