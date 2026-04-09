# Model Benchmark Report (Speed + Quality)

- Generated: `2026-04-08T22:45:48`
- Input folder: `/Users/renatobo/development/pickinsta/input`
- Candidates scored per run: `21`
- Runs per variant: `1`
- Warmup enabled: `True` (1 image + 10s wait)
- Ollama base URL: `http://localhost:11434`
- Ollama concurrency: `2`
- Ollama max retries: `2`
- Ollama keep_alive: `15m`

## Speed Summary

| Variant | Scorer | Model | YOLO | Avg sec/img | Avg imgs/min | Avg duration (s) | Avg failures/run | Speed vs fastest |
|---|---|---|---|---:|---:|---:|---:|---:|
| gemma4:e4b \| yolo=off | ollama | gemma4:e4b | off | 27.11 | 2.21 | 569.29 | 0.00 | 1.00x |

## Per-run Timing

| Variant | Run | Duration (s) | Sec/img | Imgs/min | Failures |
|---|---:|---:|---:|---:|---:|
| gemma4:e4b \| yolo=off | 1 | 569.29 | 27.11 | 2.21 | 0 |

## Image-by-Image Score Comparison (Run 1)

| Image | gemma4:e4b \| yolo=off final | gemma4:e4b \| yolo=off rank |
|---|---:|---:|
| 1-Around the Pits - CVR_0098_Mar1025_802AM_CaliPhoto.jpg | 0.1931 | 10 |
| 1-Around the Pits - CVR_0792_Mar1025_832AM_CaliPhoto.jpg | 0.1927 | 11 |
| 2-Group Photo - Raffle - CVR_7677_Mar1025_1249PM_CaliPhoto.jpg | 0.1840 | 20 |
| AB Group - Session 1 (Turn 16 Entry) - CVR_0148_Mar1025_815AM_CaliPhoto.jpg | 0.2145 | 5 |
| AB Group - Session 1 (Turn 16 Entry) - CVR_0230_Mar1025_817AM_CaliPhoto.jpg | 0.1840 | 19 |
| AB Group - Session 4 (Turn 16) - CVR_6545_Mar1025_1122AM_CaliPhoto.jpg | 0.7775 | 1 |
| C Group - Session 2 Back Straight Speed Pans - CVR_2628_Mar1025_943AM_CaliPhoto.jpg | 0.2047 | 6 |
| C Group - Session 2 Back Straight Speed Pans - CVR_2850_Mar1025_946AM_CaliPhoto.jpg | 0.5759 | 2 |
| DSC-5436-NaraMedia.jpeg | 0.5429 | 4 |
| IG cali_carnivores - DSC00013.jpg | 0.5696 | 3 |
| IG cali_carnivores - DSC09850.jpg | 0.2024 | 8 |
| IG cali_carnivores - DSC09857.jpg | 0.1847 | 18 |
| IG kamiumitv - IMG-3981-Bao.jpeg | 0.1873 | 17 |
| IG luckie.moto - 20250310-075123-luckie.moto.jpg | 0.1894 | 14 |
| IG m92663m - IMG-6019-Mark Momot.jpeg | 0.1891 | 16 |
| IG naramedia_official - DSC-5365-NaraMedia.jpeg | 0.2042 | 7 |
| IG naramedia_official - DSC-5897-NaraMedia.jpeg | 0.1908 | 12 |
| IG naramedia_official - DSC-5992-NaraMedia.jpeg | 0.2020 | 9 |
| IG renatobo - IMG_5013.jpeg | 0.1836 | 21 |
| caliphotovideo - Group picture - DROC Track Day March 10 202500001.jpg | 0.1893 | 15 |
| caliphotovideo - Group picture - DROC Track Day March 10 202500002.jpg | 0.1906 | 13 |

## Ranked Quality Details (All Images)

_Each section below is from timed run 1 for that variant._

### gemma4:e4b | yolo=off

| Rank | Image | Final | Tech | Vision | Subject | Light | Color | Emotion | Scroll | Crop | Failed | One line |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 | AB Group - Session 4 (Turn 16) - CVR_6545_Mar1025_1122AM_CaliPhoto.jpg | 0.7775 | 0.7249 | 48 | 8 | 8 | 8 | 8 | 8 | 8 | no | Vision scoring summary |
| 2 | C Group - Session 2 Back Straight Speed Pans - CVR_2850_Mar1025_946AM_CaliPhoto.jpg | 0.5759 | 0.7661 | 42 | 7 | 7 | 7 | 7 | 7 | 7 | no | Vision scoring summary |
| 3 | IG cali_carnivores - DSC00013.jpg | 0.5696 | 0.7401 | 42 | 7 | 7 | 7 | 7 | 7 | 7 | no | Vision scoring summary |
| 4 | DSC-5436-NaraMedia.jpeg | 0.5429 | 0.6287 | 42 | 7 | 7 | 7 | 7 | 7 | 7 | no | Vision scoring summary |
| 5 | AB Group - Session 1 (Turn 16 Entry) - CVR_0148_Mar1025_815AM_CaliPhoto.jpg | 0.2145 | 0.8766 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 6 | C Group - Session 2 Back Straight Speed Pans - CVR_2628_Mar1025_943AM_CaliPhoto.jpg | 0.2047 | 0.7827 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 7 | IG naramedia_official - DSC-5365-NaraMedia.jpeg | 0.2042 | 0.7780 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 8 | IG cali_carnivores - DSC09850.jpg | 0.2024 | 0.7612 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 9 | IG naramedia_official - DSC-5992-NaraMedia.jpeg | 0.2020 | 0.7571 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 10 | 1-Around the Pits - CVR_0098_Mar1025_802AM_CaliPhoto.jpg | 0.1931 | 0.6723 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 11 | 1-Around the Pits - CVR_0792_Mar1025_832AM_CaliPhoto.jpg | 0.1927 | 0.6688 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 12 | IG naramedia_official - DSC-5897-NaraMedia.jpeg | 0.1908 | 0.6504 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 13 | caliphotovideo - Group picture - DROC Track Day March 10 202500002.jpg | 0.1906 | 0.6484 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 14 | IG luckie.moto - 20250310-075123-luckie.moto.jpg | 0.1894 | 0.6374 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 15 | caliphotovideo - Group picture - DROC Track Day March 10 202500001.jpg | 0.1893 | 0.6360 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 16 | IG m92663m - IMG-6019-Mark Momot.jpeg | 0.1891 | 0.6338 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 17 | IG kamiumitv - IMG-3981-Bao.jpeg | 0.1873 | 0.6171 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 18 | IG cali_carnivores - DSC09857.jpg | 0.1847 | 0.5921 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 19 | AB Group - Session 1 (Turn 16 Entry) - CVR_0230_Mar1025_817AM_CaliPhoto.jpg | 0.1840 | 0.5860 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 20 | 2-Group Photo - Raffle - CVR_7677_Mar1025_1249PM_CaliPhoto.jpg | 0.1840 | 0.5853 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 21 | IG renatobo - IMG_5013.jpeg | 0.1836 | 0.5817 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
