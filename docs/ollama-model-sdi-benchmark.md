# Model Benchmark Report (Speed + Quality)

- Generated: `2026-04-09T13:23:48`
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
| claude-haiku-4-5-20251001 \| scorer=claude | claude | claude-haiku-4-5-20251001 | off | 0.02 | 2819.02 | 0.89 | 0.00 | 1.05 | 39 | 1.00x |
| blaifa/InternVL3_5:4B \| yolo=off | ollama | blaifa/InternVL3_5:4B | off | 6.09 | 9.85 | 255.73 | 0.00 | 1.15 | 27 | 286.07x |
| blaifa/InternVL3_5:8b \| yolo=off | ollama | blaifa/InternVL3_5:8b | off | 9.95 | 6.03 | 417.88 | 0.00 | 0.84 | 23 | 467.47x |
| openbmb/minicpm-v4.5:8b \| yolo=off | ollama | openbmb/minicpm-v4.5:8b | off | 11.26 | 5.33 | 473.03 | 0.00 | 1.91 | 16 | 529.16x |
| gemma4:e4b \| yolo=off | ollama | gemma4:e4b | off | 24.11 | 2.49 | 1012.66 | 0.00 | 0.22 | 16 | 1132.82x |
| qwen3-vl:8b \| yolo=off | ollama | qwen3-vl:8b | off | 82.84 | 0.72 | 3479.39 | 0.00 | 1.27 | 34 | 3892.25x |

## Per-run Timing

| Variant | Run | Duration (s) | Sec/img | Imgs/min | Failures |
|---|---:|---:|---:|---:|---:|
| qwen3-vl:8b \| yolo=off | 1 | 3479.39 | 82.84 | 0.72 | 0 |
| blaifa/InternVL3_5:8b \| yolo=off | 1 | 417.88 | 9.95 | 6.03 | 0 |
| blaifa/InternVL3_5:4B \| yolo=off | 1 | 255.73 | 6.09 | 9.85 | 0 |
| openbmb/minicpm-v4.5:8b \| yolo=off | 1 | 473.03 | 11.26 | 5.33 | 0 |
| gemma4:e4b \| yolo=off | 1 | 1012.66 | 24.11 | 2.49 | 0 |
| claude-haiku-4-5-20251001 \| scorer=claude | 1 | 0.89 | 0.02 | 2819.02 | 0 |

## Image-by-Image Score Comparison (Run 1)

| Image | qwen3-vl:8b \| yolo=off final | qwen3-vl:8b \| yolo=off rank | blaifa/InternVL3_5:8b \| yolo=off final | blaifa/InternVL3_5:8b \| yolo=off rank | blaifa/InternVL3_5:4B \| yolo=off final | blaifa/InternVL3_5:4B \| yolo=off rank | openbmb/minicpm-v4.5:8b \| yolo=off final | openbmb/minicpm-v4.5:8b \| yolo=off rank | gemma4:e4b \| yolo=off final | gemma4:e4b \| yolo=off rank | claude-haiku-4-5-20251001 \| scorer=claude final | claude-haiku-4-5-20251001 \| scorer=claude rank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1-Around the Pits - 1-Around the Pits - CVR_0012_Mar1025_702AM_CaliPhoto.jpg | 0.4034 | 18 | 0.4127 | 36 | 0.0336 | 42 | 0.2359 | 34 | 0.0599 | 37 | 0.0564 | 41 |
| 1-Around the Pits - 1-Around the Pits - CVR_0025_Mar1025_728AM_CaliPhoto.jpg | 0.1104 | 35 | 0.6916 | 25 | 0.6916 | 20 | 0.7245 | 7 | 0.3507 | 25 | 0.3997 | 18 |
| 1-Around the Pits - CVR_0098_Mar1025_802AM_CaliPhoto.jpg | 0.8434 | 2 | 0.9017 | 5 | 0.7214 | 15 | 0.7617 | 3 | 0.7617 | 10 | 0.5627 | 11 |
| 1-Around the Pits - CVR_0792_Mar1025_832AM_CaliPhoto.jpg | 0.1273 | 29 | 0.9652 | 1 | 0.9652 | 1 | 0.0800 | 35 | 0.8952 | 1 | 0.6695 | 3 |
| 2-Group Photo - Raffle - CVR_7677_Mar1025_1249PM_CaliPhoto.jpg | 0.1840 | 27 | 0.3784 | 37 | 0.0666 | 36 | 0.3178 | 26 | 0.0683 | 34 | 0.3224 | 25 |
| 2-Group Photo - Raffle - CVR_7682_Mar1025_1249PM_CaliPhoto.jpg | 0.5672 | 14 | 0.3694 | 38 | 0.0714 | 33 | 0.3059 | 29 | 0.1787 | 30 | 0.1542 | 35 |
| AB Group - Session 1 (Turn 16 Entry) - CVR_0148_Mar1025_815AM_CaliPhoto.jpg | 0.1084 | 36 | 0.7086 | 20 | 0.8857 | 8 | 0.4124 | 21 | 0.7457 | 11 | 0.6059 | 8 |
| AB Group - Session 1 (Turn 16 Entry) - CVR_0230_Mar1025_817AM_CaliPhoto.jpg | 0.8058 | 7 | 0.7006 | 22 | 0.8758 | 11 | 0.4065 | 22 | 0.7358 | 15 | 0.5793 | 10 |
| AB Group - Session 2 (Turn 11) - CVR_1047_Mar1025_908AM_CaliPhoto.jpg | 0.2841 | 23 | 0.8675 | 8 | 0.8675 | 12 | 0.4015 | 23 | 0.7391 | 13 | 0.6193 | 6 |
| AB Group - Session 4 (Turn 16) - CVR_6545_Mar1025_1122AM_CaliPhoto.jpg | 0.3341 | 20 | 0.9175 | 4 | 0.9175 | 6 | 0.4315 | 20 | 0.7775 | 8 | 0.6126 | 7 |
| C Group - Session 2 Back Straight Speed Pans - CVR_2628_Mar1025_943AM_CaliPhoto.jpg | 0.3515 | 19 | 0.7479 | 12 | 0.9348 | 2 | 0.7715 | 2 | 0.3929 | 24 | 0.4419 | 15 |
| C Group - Session 2 Back Straight Speed Pans - CVR_2850_Mar1025_946AM_CaliPhoto.jpg | 0.1202 | 30 | 0.7439 | 14 | 0.9298 | 3 | 0.4389 | 19 | 0.7898 | 6 | 0.4669 | 13 |
| C Group - Session 3 (Turn 9 and 8) - CVR_4982_Mar1025_1040AM_CaliPhoto.jpg | 0.8377 | 3 | 0.6795 | 27 | 0.8493 | 13 | 0.3906 | 24 | 0.0574 | 38 | 0.0784 | 36 |
| DSC-5436-NaraMedia.jpeg | 0.3053 | 21 | 0.7109 | 19 | 0.7109 | 16 | 0.7253 | 6 | 0.5429 | 21 | 0.4142 | 17 |
| IG - renatobo - March 10 DROC Track Day-26 HDR.jpeg | 0.4878 | 17 | 0.6744 | 32 | 0.4114 | 31 | 0.0740 | 36 | 0.5064 | 23 | 0.3098 | 26 |
| IG - renatobo - March 10 DROC Track Day-28.jpeg | 0.0988 | 38 | 0.6760 | 31 | 0.0480 | 40 | 0.2617 | 32 | 0.0533 | 40 | 0.1855 | 33 |
| IG - renatobo - March 10 DROC Track Day-58 HDR.jpeg | 0.5667 | 15 | 0.0748 | 41 | 0.0503 | 38 | 0.3054 | 30 | 0.0450 | 41 | 0.0608 | 40 |
| IG - renatobo - lineup 1st session - March 10 DROC Track Day-01 HDR.jpeg | 0.7312 | 9 | 0.8129 | 11 | 0.6503 | 27 | 0.6146 | 15 | 0.3197 | 28 | 0.1865 | 32 |
| IG - renatobo - lineup 1st session - March 10 DROC Track Day-02 HDR.jpeg | 0.5938 | 13 | 0.6778 | 29 | 0.6778 | 24 | 0.6489 | 13 | 0.7072 | 18 | 0.5004 | 12 |
| IG - renatobo - lineup 1st session - March 10 DROC Track Day-12 HDR.jpeg | 0.6269 | 11 | 0.8537 | 10 | 0.6829 | 23 | 0.7137 | 8 | 0.7837 | 7 | 0.3862 | 20 |
| IG cali_carnivores - DSC00013.jpg | 0.0963 | 39 | 0.7376 | 15 | 0.7376 | 14 | 0.0736 | 37 | 0.7704 | 9 | 0.6350 | 5 |
| IG cali_carnivores - DSC09850.jpg | 0.2760 | 24 | 0.9284 | 2 | 0.9284 | 4 | 0.4800 | 16 | 0.5747 | 19 | 0.2392 | 29 |
| IG cali_carnivores - DSC09857.jpg | 0.1141 | 33 | 0.7021 | 21 | 0.8776 | 10 | 0.4496 | 17 | 0.7376 | 14 | 0.4216 | 16 |
| IG desmo.donna - IMG-1015-Donna.jpeg | 0.6042 | 12 | 0.4736 | 35 | 0.6976 | 19 | 0.0660 | 41 | 0.7320 | 17 | 0.2194 | 31 |
| IG desmo.donna - IMG-1080-Donna.jpeg | 0.8145 | 5 | 0.6982 | 24 | 0.6982 | 18 | 0.6745 | 10 | 0.7328 | 16 | 0.3837 | 21 |
| IG desmo.donna - IMG-1151-Donna.jpeg | 0.5588 | 16 | 0.0663 | 42 | 0.0453 | 41 | 0.2436 | 33 | 0.0400 | 42 | 0.2791 | 28 |
| IG guardiansvoice - IMG-4129-Delenian.jpeg | 0.1136 | 34 | 0.6899 | 26 | 0.6899 | 22 | 0.4404 | 18 | 0.5219 | 22 | 0.2324 | 30 |
| IG guardiansvoice - IMG-4160-Delenian.jpeg | 0.0678 | 41 | 0.6790 | 28 | 0.0486 | 39 | 0.2964 | 31 | 0.0643 | 35 | 0.2993 | 27 |
| IG kamiumitv - IMG-3981-Bao.jpeg | 0.3018 | 22 | 0.8851 | 7 | 0.8851 | 9 | 0.0680 | 40 | 0.7451 | 12 | 0.5961 | 9 |
| IG luckie.moto - 20250310-075123-luckie.moto.jpg | 0.1144 | 31 | 0.7130 | 17 | 0.4997 | 29 | 0.0689 | 39 | 0.5450 | 20 | 0.3737 | 22 |
| IG m92663m - IMG-6019-Mark Momot.jpeg | 0.1891 | 26 | 0.1523 | 40 | 0.0670 | 35 | 0.3295 | 25 | 0.0635 | 36 | 0.1646 | 34 |
| IG martangelenos - PXL_20250310_132329430.jpg | 0.0673 | 42 | 0.6761 | 30 | 0.1571 | 32 | 0.0725 | 38 | 0.1733 | 31 | 0.0725 | 38 |
| IG naramedia_official - DSC-5365-NaraMedia.jpeg | 0.6814 | 10 | 0.7467 | 13 | 0.5367 | 28 | 0.7351 | 4 | 0.8284 | 2 | 0.6720 | 2 |
| IG naramedia_official - DSC-5897-NaraMedia.jpeg | 0.8135 | 6 | 0.8951 | 6 | 0.8951 | 7 | 0.7318 | 5 | 0.8018 | 4 | 0.6508 | 4 |
| IG naramedia_official - DSC-5992-NaraMedia.jpeg | 0.8455 | 1 | 0.9271 | 3 | 0.9271 | 5 | 0.7988 | 1 | 0.7988 | 5 | 0.4513 | 14 |
| IG ocredevil - DROC-Track-Day-3-10-2-filippo pagan.jpeg | 0.0842 | 40 | 0.3506 | 39 | 0.4206 | 30 | 0.0492 | 42 | 0.0544 | 39 | 0.0527 | 42 |
| IG renatobo - IMG_5013.jpeg | 0.8162 | 4 | 0.6996 | 23 | 0.6996 | 17 | 0.6762 | 9 | 0.8045 | 3 | 0.3917 | 19 |
| IG renatobo - IMG_5014.jpeg | 0.1620 | 28 | 0.6504 | 34 | 0.6504 | 26 | 0.6730 | 11 | 0.3198 | 27 | 0.3548 | 23 |
| IMG_4984.jpeg | 0.2234 | 25 | 0.8626 | 9 | 0.6901 | 21 | 0.6643 | 12 | 0.3496 | 26 | 0.6760 | 1 |
| IMG_5012.jpeg | 0.1064 | 37 | 0.6607 | 33 | 0.6607 | 25 | 0.6275 | 14 | 0.1665 | 32 | 0.3415 | 24 |
| caliphotovideo - Group picture - DROC Track Day March 10 202500001.jpg | 0.1144 | 32 | 0.7126 | 18 | 0.0584 | 37 | 0.3075 | 28 | 0.1893 | 29 | 0.0689 | 39 |
| caliphotovideo - Group picture - DROC Track Day March 10 202500002.jpg | 0.7778 | 8 | 0.7156 | 16 | 0.0694 | 34 | 0.3112 | 27 | 0.0712 | 33 | 0.0782 | 37 |

## Ranked Quality Details (All Images)

_Each section below is from timed run 1 for that variant._

### qwen3-vl:8b | yolo=off

| Rank | Image | Final | Tech | Vision | Subject | Light | Color | Emotion | Scroll | Crop | Failed | One line |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 | IG naramedia_official - DSC-5992-NaraMedia.jpeg | 0.8455 | 0.7571 | 53 | 9 | 9 | 9 | 9 | 9 | 8 | no | <think> Got it, let's evaluate this motorcycle photo for Instagram cover potential. |
| 2 | 1-Around the Pits - CVR_0098_Mar1025_802AM_CaliPhoto.jpg | 0.8434 | 0.6723 | 55 | 10 | 9 | 8 | 10 | 9 | 9 | no | must be one concise sentence. "A Ducati rider in full gear poses on a sleek motorcycle against a scenic Southern California backdrop with mountains and clear skies. |
| 3 | C Group - Session 3 (Turn 9 and 8) - CVR_4982_Mar1025_1040AM_CaliPhoto.jpg | 0.8377 | 0.4978 | 59 | 10 | 9 | 10 | 10 | 10 | 10 | no | <think> Got it, let's evaluate this motorcycle photo for Instagram cover potential. |
| 4 | IG renatobo - IMG_5013.jpeg | 0.8162 | 0.5817 | 55 | 10 | 9 | 9 | 10 | 9 | 8 | no | <think> Got it, let's evaluate this photo for Instagram cover potential. |
| 5 | IG desmo.donna - IMG-1080-Donna.jpeg | 0.8145 | 0.5760 | 55 | 10 | 9 | 9 | 10 | 9 | 8 | no | <think> Got it, let's evaluate this motorcycle photo for Instagram cover potential. |
| 6 | IG naramedia_official - DSC-5897-NaraMedia.jpeg | 0.8135 | 0.6504 | 53 | 9 | 8 | 9 | 10 | 9 | 8 | no | <think> Got it, let's evaluate this motorcycle photo for Instagram cover potential. |
| 7 | AB Group - Session 1 (Turn 16 Entry) - CVR_0230_Mar1025_817AM_CaliPhoto.jpg | 0.8058 | 0.5860 | 54 | 9 | 9 | 9 | 10 | 9 | 8 | no | Must be one concise sentence. "A Ducati rider leans into a turn on a racetrack with mountains in the background, capturing speed and precision. |
| 8 | caliphotovideo - Group picture - DROC Track Day March 10 202500002.jpg | 0.7778 | 0.6484 | 50 | 8 | 9 | 8 | 9 | 8 | 8 | no | <think> Got it, let's evaluate this photo for Instagram cover potential. |
| 9 | IG - renatobo - lineup 1st session - March 10 DROC Track Day-01 HDR.jpeg | 0.7312 | 0.3764 | 53 | 9 | 9 | 9 | 9 | 9 | 8 | no | <think> Got it, let's evaluate this motorcycle photo for Instagram cover potential. |
| 10 | IG naramedia_official - DSC-5365-NaraMedia.jpeg | 0.6814 | 0.7780 | 53 | 10 | 8 | 9 | 10 | 9 | 7 | no | must be one concise sentence. "A sharp Ducati 848 motorcycle photo with vibrant colors and clear branding, perfect for a motorcycle enthusiast's Instagram cover. |
| 11 | IG - renatobo - lineup 1st session - March 10 DROC Track Day-12 HDR.jpeg | 0.6269 | 0.5123 | 54 | 10 | 9 | 9 | 10 | 9 | 7 | no | <think> Got it, let's evaluate this motorcycle photo for Instagram cover potential. |
| 12 | IG desmo.donna - IMG-1015-Donna.jpeg | 0.6042 | 0.5732 | 50 | 9 | 8 | 9 | 8 | 9 | 7 | no | Sunset-lit Ducati event with vibrant red bikes and enthusiasts under a branded tent. |
| 13 | IG - renatobo - lineup 1st session - March 10 DROC Track Day-02 HDR.jpeg | 0.5938 | 0.4907 | 51 | 10 | 8 | 9 | 8 | 9 | 7 | no | must be one concise sentence. "A Ducati rider in full gear poses on a red 21-numbered bike against a desert backdrop, showcasing vibrant colors and dynamic motorcycle culture. |
| 14 | 2-Group Photo - Raffle - CVR_7682_Mar1025_1249PM_CaliPhoto.jpg | 0.5672 | 0.5357 | 47 | 8 | 9 | 8 | 7 | 8 | 7 | no | <think> Got it, let's evaluate this photo for Instagram cover potential. |
| 15 | IG - renatobo - March 10 DROC Track Day-58 HDR.jpeg | 0.5667 | 0.5335 | 47 | 8 | 8 | 7 | 8 | 9 | 7 | no | <think> Got it, let's evaluate this image for Instagram cover potential. |
| 16 | IG desmo.donna - IMG-1151-Donna.jpeg | 0.5588 | 0.4230 | 49 | 8 | 9 | 8 | 9 | 8 | 7 | no | <think> Got it, let's evaluate this photo for Instagram cover potential. |
| 17 | IG - renatobo - March 10 DROC Track Day-26 HDR.jpeg | 0.4878 | 0.4768 | 40 | 0 | 9 | 7 | 8 | 9 | 7 | no | <think> Got it, let's evaluate this image for Instagram cover potential. |
| 18 | 1-Around the Pits - 1-Around the Pits - CVR_0012_Mar1025_702AM_CaliPhoto.jpg | 0.4034 | 0.3975 | 33 | 0 | 8 | 6 | 7 | 5 | 7 | no | A group of motorcycle enthusiasts gathered in a room for a presentation, with no visible motorcycle. |
| 19 | C Group - Session 2 Back Straight Speed Pans - CVR_2628_Mar1025_943AM_CaliPhoto.jpg | 0.3515 | 0.7827 | 10 | 10 | 9 | 9 | 10 | 9 | 8 | no | must be one concise sentence. "Dynamic Ducati rider leaning into a turn on a desert track, capturing speed and precision. |
| 20 | AB Group - Session 4 (Turn 16) - CVR_6545_Mar1025_1122AM_CaliPhoto.jpg | 0.3341 | 0.7249 | 10 | 10 | 9 | 9 | 10 | 9 | 8 | no | <think> Got it, let's evaluate this motorcycle photo for Instagram cover potential. |
| 21 | DSC-5436-NaraMedia.jpeg | 0.3053 | 0.6287 | 10 | 9 | 10 | 9 | 10 | 9 | 8 | no | <think> Got it, let's evaluate this motorcycle photo for Instagram cover potential. |
| 22 | IG kamiumitv - IMG-3981-Bao.jpeg | 0.3018 | 0.6171 | 10 | 10 | 9 | 10 | 10 | 9 | 8 | no | <think> Got it, let's break this down. |
| 23 | AB Group - Session 2 (Turn 11) - CVR_1047_Mar1025_908AM_CaliPhoto.jpg | 0.2841 | 0.5583 | 10 | 10 | 9 | 9 | 9 | 9 | 8 | no | <think> Got it, let's break this down. |
| 24 | IG cali_carnivores - DSC09850.jpg | 0.2760 | 0.7612 | 10 | 10 | 9 | 8 | 10 | 9 | 7 | no | must be exactly one concise sentence. "A Ducati rider leans into a turn on a desert track, capturing speed and precision in Southern California. |
| 25 | IMG_4984.jpeg | 0.2234 | 0.5421 | 10 | 9 | 8 | 9 | 9 | 8 | 7 | no | Close-up of a Ducati motorcycle's front with thermal tech patch, showcasing vibrant red and white colors in a garage setting. |
| 26 | IG m92663m - IMG-6019-Mark Momot.jpeg | 0.1891 | 0.6338 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | <think> Got it, let's evaluate this image for Instagram cover potential. |
| 27 | 2-Group Photo - Raffle - CVR_7677_Mar1025_1249PM_CaliPhoto.jpg | 0.1840 | 0.5853 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | <think> Got it, let's evaluate this image for Instagram cover potential. |
| 28 | IG renatobo - IMG_5014.jpeg | 0.1620 | 0.3765 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | <think> Got it, let's evaluate this motorcycle photo for Instagram cover potential. |
| 29 | 1-Around the Pits - CVR_0792_Mar1025_832AM_CaliPhoto.jpg | 0.1273 | 0.8839 | 50 | 10 | 9 | 9 | 10 | 8 | 4 | no | <think> Got it, let's evaluate this motorcycle photo for Instagram cover potential. |
| 30 | C Group - Session 2 Back Straight Speed Pans - CVR_2850_Mar1025_946AM_CaliPhoto.jpg | 0.1202 | 0.7661 | 49 | 9 | 9 | 9 | 9 | 9 | 4 | no | <think> Got it, let's evaluate this motorcycle photo for Instagram cover potential. |
| 31 | IG luckie.moto - 20250310-075123-luckie.moto.jpg | 0.1144 | 0.6374 | 49 | 10 | 9 | 8 | 10 | 8 | 4 | no | <think> Got it, let's evaluate this motorcycle photo for Instagram cover potential. |
| 32 | caliphotovideo - Group picture - DROC Track Day March 10 202500001.jpg | 0.1144 | 0.6360 | 49 | 10 | 9 | 8 | 10 | 8 | 4 | no | <think> Got it, let's evaluate this photo for Instagram cover potential. |
| 33 | IG cali_carnivores - DSC09857.jpg | 0.1141 | 0.5921 | 50 | 10 | 9 | 9 | 10 | 8 | 4 | no | <think> Got it, let's break this down. |
| 34 | IG guardiansvoice - IMG-4129-Delenian.jpeg | 0.1136 | 0.5413 | 51 | 10 | 9 | 9 | 10 | 9 | 4 | no | <think> Got it, let's evaluate this motorcycle photo for Instagram cover potential. |
| 35 | 1-Around the Pits - 1-Around the Pits - CVR_0025_Mar1025_728AM_CaliPhoto.jpg | 0.1104 | 0.5483 | 49 | 10 | 9 | 9 | 9 | 8 | 4 | no | <think> Got it, let's evaluate this motorcycle photo for Instagram cover potential. |
| 36 | AB Group - Session 1 (Turn 16 Entry) - CVR_0148_Mar1025_815AM_CaliPhoto.jpg | 0.1084 | 0.6190 | 46 | 9 | 8 | 8 | 9 | 8 | 4 | no | <think> Got it, let's evaluate this motorcycle photo for Instagram cover potential. |
| 37 | IMG_5012.jpeg | 0.1064 | 0.4195 | 50 | 10 | 9 | 8 | 10 | 9 | 4 | no | <think> Got it, let's evaluate this motorcycle photo for Instagram cover potential. |
| 38 | IG - renatobo - March 10 DROC Track Day-28.jpeg | 0.0988 | 0.4833 | 44 | 9 | 8 | 7 | 9 | 7 | 4 | no | <think> Got it, let's evaluate this photo for Instagram cover potential. |
| 39 | IG cali_carnivores - DSC00013.jpg | 0.0963 | 0.7401 | 36 | 9 | 9 | 6 | 2 | 6 | 4 | no | <think> Got it, let's evaluate this motorcycle photo for Instagram cover potential. |
| 40 | IG ocredevil - DROC-Track-Day-3-10-2-filippo pagan.jpeg | 0.0842 | 0.4312 | 37 | 5 | 9 | 6 | 7 | 6 | 4 | no | <think> Got it, let's break this down. |
| 41 | IG guardiansvoice - IMG-4160-Delenian.jpeg | 0.0678 | 0.4960 | 26 | 3 | 4 | 4 | 7 | 4 | 4 | no | <think> Got it, let's break this down. |
| 42 | IG martangelenos - PXL_20250310_132329430.jpg | 0.0673 | 0.4839 | 26 | 0 | 5 | 7 | 6 | 4 | 4 | no | A sunset-lit Chuckwalla building with no visible motorcycle, lacking subject relevance for a Ducati-focused account. |

### blaifa/InternVL3_5:8b | yolo=off

| Rank | Image | Final | Tech | Vision | Subject | Light | Color | Emotion | Scroll | Crop | Failed | One line |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 1-Around the Pits - CVR_0792_Mar1025_832AM_CaliPhoto.jpg | 0.9652 | 0.8839 | 60 | 9 | 8 | 8 | 7 | 8 | 8 | no | The motorcycle and rider are the clear focal point with good contrast against the background, balanced lighting, vibrant colors, and a composition that conveys motion and aspiration. The image is likely to stop fast-scro |
| 2 | IG cali_carnivores - DSC09850.jpg | 0.9284 | 0.7612 | 60 | 8 | 7 | 7 | 8 | 9 | 8 | no | The motorcycle is the clear focal point with good contrast against the background, balanced lighting, and dynamic composition that conveys motion. The image likely stops fast-scrolling due to its strong leading lines and |
| 3 | IG naramedia_official - DSC-5992-NaraMedia.jpeg | 0.9271 | 0.7571 | 60 | 9 | 8 | 8 | 8 | 7 | 8 | no | The motorcycle is the clear focal point with good contrast against the background, balanced lighting, and dynamic composition that conveys motion. The Ducati branding adds to its appeal. |
| 4 | AB Group - Session 4 (Turn 16) - CVR_6545_Mar1025_1122AM_CaliPhoto.jpg | 0.9175 | 0.7249 | 60 | 9 | 8 | 8 | 9 | 9 | 8 | no | The image effectively highlights the Ducati motorcycle with a clear focal point, balanced lighting, and dynamic composition that conveys motion and aspiration. The orange bike contrasts well against the background, makin |
| 5 | 1-Around the Pits - CVR_0098_Mar1025_802AM_CaliPhoto.jpg | 0.9017 | 0.6723 | 60 | 8 | 7 | 7 | 8 | 9 | 8 | no | The motorcycle and rider are clear focal points with good contrast against the background, balanced lighting, vibrant colors, and a composition that conveys motion and aspiration. The image is likely to stop fast-scrolli |
| 6 | IG naramedia_official - DSC-5897-NaraMedia.jpeg | 0.8951 | 0.6504 | 60 | 9 | 8 | 8 | 9 | 9 | 8 | no | This image effectively highlights the Ducati motorcycle with clear subject focus, balanced lighting, vibrant colors, and dynamic composition that conveys motion and power. It's visually engaging and suitable for stopping |
| 7 | IG kamiumitv - IMG-3981-Bao.jpeg | 0.8851 | 0.6171 | 60 | 9 | 8 | 9 | 8 | 9 | 8 | no | The motorcycle is the clear focal point with a sharp contrast against the background, balanced lighting, and vibrant colors that convey power and aspiration. The composition would likely stop fast-scrolling due to its dr |
| 8 | AB Group - Session 2 (Turn 11) - CVR_1047_Mar1025_908AM_CaliPhoto.jpg | 0.8675 | 0.5583 | 60 | 9 | 8 | 8 | 9 | 9 | 8 | no | The image effectively highlights the Ducati motorcycle with clear subject focus, balanced lighting, vibrant colors, and dynamic composition that conveys motion and power. It's visually engaging and suitable for stopping |
| 9 | IMG_4984.jpeg | 0.8626 | 0.5421 | 60 | 9 | 8 | 8 | 7 | 8 | 8 | no | The motorcycle is the clear focal point with good contrast against the background, balanced lighting, and vibrant colors. The composition is engaging and suitable for stopping fast-scrolling on Instagram. |
| 10 | IG - renatobo - lineup 1st session - March 10 DROC Track Day-12 HDR.jpeg | 0.8537 | 0.5123 | 60 | 9 | 8 | 8 | 7 | 8 | 8 | no | The image effectively highlights the Ducati motorcycle with a clear focal point, balanced lighting, and vibrant colors. The composition is engaging and suitable for stopping fast-scrolling on Instagram. |
| 11 | IG - renatobo - lineup 1st session - March 10 DROC Track Day-01 HDR.jpeg | 0.8129 | 0.3764 | 60 | 8 | 7 | 9 | 8 | 9 | 8 | no | The image effectively highlights the motorcycles with a clear focal point, balanced lighting, and vibrant colors. The composition is engaging and suitable for stopping fast-scrolling on Instagram. |
| 12 | C Group - Session 2 Back Straight Speed Pans - CVR_2628_Mar1025_943AM_CaliPhoto.jpg | 0.7479 | 0.7827 | 60 | 9 | 8 | 7 | 8 | 8 | 7 | no | The motorcycle is the clear focal point with good contrast against the background, balanced lighting, and dynamic composition that conveys motion and power. The image likely stops fast-scrolling due to its strong visual |
| 13 | IG naramedia_official - DSC-5365-NaraMedia.jpeg | 0.7467 | 0.7780 | 60 | 9 | 8 | 8 | 7 | 8 | 7 | no | The Ducati motorcycle is the clear focal point with strong contrast against the background, balanced lighting, and vibrant colors. The composition conveys power and aspiration, making it engaging for fast scrolling. |
| 14 | C Group - Session 2 Back Straight Speed Pans - CVR_2850_Mar1025_946AM_CaliPhoto.jpg | 0.7439 | 0.7661 | 60 | 9 | 8 | 8 | 8 | 7 | 7 | no | The motorcycle is the clear focal point with good contrast against the background, balanced lighting, and dynamic composition that conveys motion. The Ducati branding adds to its appeal. |
| 15 | IG cali_carnivores - DSC00013.jpg | 0.7376 | 0.7401 | 60 | 9 | 8 | 8 | 7 | 8 | 7 | no | The Ducati motorcycle is the clear focal point with strong contrast against the background, balanced lighting, and vibrant color. The composition conveys power and aspiration, making it engaging for fast scrolling. |
| 16 | caliphotovideo - Group picture - DROC Track Day March 10 202500002.jpg | 0.7156 | 0.6484 | 60 | 7 | 8 | 6 | 7 | 9 | 7 | no | The image effectively captures a group with a clear focal point on the Ducati car, balanced lighting, and strong composition that would likely stop fast-scrolling. However, cropping to portrait might cut off some element |
| 17 | IG luckie.moto - 20250310-075123-luckie.moto.jpg | 0.7130 | 0.6374 | 60 | 8 | 7 | 9 | 7 | 8 | 7 | no | The motorcycles are the clear focal point with good contrast against the background, vibrant colors enhance visual appeal, and the composition is engaging. However, cropping to portrait might cut off some details. |
| 18 | caliphotovideo - Group picture - DROC Track Day March 10 202500001.jpg | 0.7126 | 0.6360 | 60 | 7 | 8 | 6 | 7 | 9 | 7 | no | The image effectively captures a group with clear composition and lighting, but lacks strong motorcycle focus or dynamic emotion. It has potential as an Instagram cover due to its balanced layout. |
| 19 | DSC-5436-NaraMedia.jpeg | 0.7109 | 0.6287 | 60 | 8 | 7 | 7 | 9 | 8 | 7 | no | The image effectively highlights the motorcycles with a clear focal point and dynamic composition, though lighting could be improved for better contrast. |
| 20 | AB Group - Session 1 (Turn 16 Entry) - CVR_0148_Mar1025_815AM_CaliPhoto.jpg | 0.7086 | 0.6190 | 60 | 9 | 8 | 7 | 8 | 8 | 7 | no | The motorcycle is the clear focal point with good contrast against the background, balanced lighting, and dynamic composition that conveys motion. The Ducati branding adds to its appeal. |
| 21 | IG cali_carnivores - DSC09857.jpg | 0.7021 | 0.5921 | 60 | 9 | 8 | 7 | 8 | 8 | 7 | no | The motorcycle is the clear focal point with good contrast against the background, balanced lighting, and dynamic composition that conveys motion and power. The image likely stops fast-scrolling due to its strong visual |
| 22 | AB Group - Session 1 (Turn 16 Entry) - CVR_0230_Mar1025_817AM_CaliPhoto.jpg | 0.7006 | 0.5860 | 60 | 9 | 8 | 7 | 8 | 8 | 7 | no | The motorcycle is the clear focal point with good contrast against the background, balanced lighting, and dynamic composition that conveys motion and power. The image likely stops fast-scrolling due to its strong visual |
| 23 | IG renatobo - IMG_5013.jpeg | 0.6996 | 0.5817 | 60 | 8 | 7 | 7 | 6 | 7 | 7 | no | The image effectively highlights the motorcycles and riders with good subject clarity, balanced lighting, and vibrant colors. However, it could better convey motion or power to enhance emotion. |
| 24 | IG desmo.donna - IMG-1080-Donna.jpeg | 0.6982 | 0.5760 | 60 | 8 | 9 | 8 | 7 | 8 | 7 | no | The image effectively highlights the Ducati motorcycle with good lighting and color contrast, but could improve composition for Instagram's 3:4 grid. |
| 25 | 1-Around the Pits - 1-Around the Pits - CVR_0025_Mar1025_728AM_CaliPhoto.jpg | 0.6916 | 0.5483 | 60 | 8 | 7 | 9 | 7 | 8 | 7 | no | The image effectively highlights the Ducati motorcycles with strong color contrast and composition, though lighting could be improved for a more dramatic effect. |
| 26 | IG guardiansvoice - IMG-4129-Delenian.jpeg | 0.6899 | 0.5413 | 60 | 8 | 9 | 7 | 7 | 8 | 7 | no | The image effectively highlights the motorcycles with good lighting and composition, but could improve in color contrast and emotional impact. |
| 27 | C Group - Session 3 (Turn 9 and 8) - CVR_4982_Mar1025_1040AM_CaliPhoto.jpg | 0.6795 | 0.4978 | 60 | 8 | 7 | 7 | 9 | 8 | 7 | no | The image effectively captures motion and power with a clear focus on the motorcycles, though lighting could be improved for better contrast. |
| 28 | IG guardiansvoice - IMG-4160-Delenian.jpeg | 0.6790 | 0.4960 | 60 | 6 | 7 | 8 | 7 | 8 | 7 | no | The image has a clear subject with good color contrast and composition, but lacks strong lighting and distinct motorcycle branding. |
| 29 | IG - renatobo - lineup 1st session - March 10 DROC Track Day-02 HDR.jpeg | 0.6778 | 0.4907 | 60 | 8 | 7 | 9 | 8 | 8 | 7 | no | The motorcycle and rider are clear focal points with good contrast against the background, vibrant colors enhance visual appeal, and the composition is engaging. The Ducati branding adds to the subject clarity and emotio |
| 30 | IG martangelenos - PXL_20250310_132329430.jpg | 0.6761 | 0.4839 | 60 | 7 | 8 | 6 | 7 | 8 | 7 | no | The image effectively highlights the motorcycle against a scenic backdrop with good lighting and composition, though the Ducati isn't clearly identifiable. |
| 31 | IG - renatobo - March 10 DROC Track Day-28.jpeg | 0.6760 | 0.4833 | 60 | 7 | 8 | 6 | 7 | 9 | 7 | no | The image effectively captures a group in front of a building, with good lighting and composition that could stop fast-scrolling. However, the motorcycle isn't clearly visible as a Ducati. |
| 32 | IG - renatobo - March 10 DROC Track Day-26 HDR.jpeg | 0.6744 | 0.4768 | 60 | 7 | 8 | 6 | 7 | 8 | 7 | no | The image effectively highlights the rider and scooter with good lighting, but lacks a clear Ducati motorcycle which would enhance brand relevance. |
| 33 | IMG_5012.jpeg | 0.6607 | 0.4195 | 60 | 8 | 7 | 8 | 9 | 8 | 7 | no | The image effectively highlights the motorcycles with clear subjects against a simple background, good lighting, and vibrant colors. The composition conveys motion and power, making it engaging for viewers. |
| 34 | IG renatobo - IMG_5014.jpeg | 0.6504 | 0.3765 | 60 | 8 | 7 | 9 | 7 | 8 | 7 | no | The motorcycle is the clear focal point with good contrast against the background, and the orange bike contrasts well with the blue sky. The lighting is balanced but not dramatic. The composition could be improved for a |
| 35 | IG desmo.donna - IMG-1015-Donna.jpeg | 0.4736 | 0.5732 | 36 | 8 | 9 | 7 | 7 | 8 | 7 | no | The image effectively highlights the Ducati motorcycles with good lighting and composition, but could improve in color contrast and emotional impact. |
| 36 | 1-Around the Pits - 1-Around the Pits - CVR_0012_Mar1025_702AM_CaliPhoto.jpg | 0.4127 | 0.3975 | 34 | 6 | 7 | 8 | 7 | 9 | 7 | no | The image effectively captures a group setting with good lighting and color contrast, but lacks a clear focus on a Ducati motorcycle as the main subject. |
| 37 | 2-Group Photo - Raffle - CVR_7677_Mar1025_1249PM_CaliPhoto.jpg | 0.3784 | 0.5853 | 39 | 6 | 7 | 8 | 5 | 7 | 6 | no | The image has a clear subject with good lighting and color contrast, but lacks motion or power conveyed typical of Ducati motorcycles. It could be more engaging for Instagram. |
| 38 | 2-Group Photo - Raffle - CVR_7682_Mar1025_1249PM_CaliPhoto.jpg | 0.3694 | 0.5357 | 39 | 6 | 7 | 8 | 5 | 7 | 6 | no | The image has good color contrast and lighting but lacks a clear motorcycle focal point, which affects overall composition. |
| 39 | IG ocredevil - DROC-Track-Day-3-10-2-filippo pagan.jpeg | 0.3506 | 0.4312 | 39 | 6 | 7 | 8 | 5 | 7 | 6 | no | The image captures two individuals on scooters with a clear focus, good lighting, and vibrant colors. However, it lacks the motorcycle subject needed for Ducati branding. |
| 40 | IG m92663m - IMG-6019-Mark Momot.jpeg | 0.1523 | 0.6338 | 21 | 3 | 7 | 6 | 4 | 8 | 5 | no | The image lacks a motorcycle as the focal point, but it has good lighting and composition that could stop fast-scrolling. However, it doesn't convey motion or power effectively. |
| 41 | IG - renatobo - March 10 DROC Track Day-58 HDR.jpeg | 0.0748 | 0.5335 | 29 | 3 | 7 | 6 | 4 | 5 | 4 | no | The image lacks a clear motorcycle subject and has a busy background, but the lighting is good. It doesn't strongly convey emotion or have a dramatic composition. |
| 42 | IG desmo.donna - IMG-1151-Donna.jpeg | 0.0663 | 0.4230 | 27 | 3 | 7 | 6 | 4 | 5 | 4 | no | The image lacks a motorcycle, focusing instead on people which doesn't align with Ducati content. Lighting is good but the composition and color contrast could be improved for better engagement. |

### blaifa/InternVL3_5:4B | yolo=off

| Rank | Image | Final | Tech | Vision | Subject | Light | Color | Emotion | Scroll | Crop | Failed | One line |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 1-Around the Pits - CVR_0792_Mar1025_832AM_CaliPhoto.jpg | 0.9652 | 0.8839 | 60 | 9 | 8 | 10 | 9 | 9 | 8 | no | The image effectively highlights the Ducati motorcycle with a clear subject, vibrant colors contrasting against the background, and conveys power and motion well. |
| 2 | C Group - Session 2 Back Straight Speed Pans - CVR_2628_Mar1025_943AM_CaliPhoto.jpg | 0.9348 | 0.7827 | 60 | 8 | 7 | 9 | 9 | 8 | 8 | no | The image effectively conveys motion and power with a clear Ducati focus, vibrant colors contrasting against the desert background, and strong composition for Instagram. |
| 3 | C Group - Session 2 Back Straight Speed Pans - CVR_2850_Mar1025_946AM_CaliPhoto.jpg | 0.9298 | 0.7661 | 60 | 9 | 8 | 9 | 10 | 9 | 8 | no | The image effectively showcases the Ducati motorcycle with a clear subject, vibrant colors, and strong emotion of motion, making it suitable for an Instagram cover. |
| 4 | IG cali_carnivores - DSC09850.jpg | 0.9284 | 0.7612 | 60 | 8 | 7 | 9 | 9 | 8 | 8 | no | The image effectively highlights the Ducati motorcycle with clear subject clarity, vibrant colors, and a dynamic composition that conveys power and motion. |
| 5 | IG naramedia_official - DSC-5992-NaraMedia.jpeg | 0.9271 | 0.7571 | 60 | 9 | 8 | 7 | 9 | 8 | 8 | no | The image effectively highlights the Ducati motorcycle with clear subject clarity and strong emotion, making it suitable for an Instagram cover. |
| 6 | AB Group - Session 4 (Turn 16) - CVR_6545_Mar1025_1122AM_CaliPhoto.jpg | 0.9175 | 0.7249 | 60 | 9 | 8 | 7 | 10 | 9 | 8 | no | The image effectively showcases the Ducati motorcycle with a clear focus, strong emotion through motion, and good color contrast against the background. |
| 7 | IG naramedia_official - DSC-5897-NaraMedia.jpeg | 0.8951 | 0.6504 | 60 | 9 | 8 | 7 | 10 | 9 | 8 | no | The image effectively showcases the Ducati motorcycle with strong emotion and clarity, making it suitable for an Instagram cover. |
| 8 | AB Group - Session 1 (Turn 16 Entry) - CVR_0148_Mar1025_815AM_CaliPhoto.jpg | 0.8857 | 0.6190 | 60 | 8 | 7 | 9 | 9 | 8 | 8 | no | The image effectively conveys motion and power with a clear Ducati focus, strong color contrast against the background, and balanced composition. |
| 9 | IG kamiumitv - IMG-3981-Bao.jpeg | 0.8851 | 0.6171 | 60 | 9 | 8 | 10 | 9 | 9 | 8 | no | The image effectively highlights the Ducati motorcycle with strong color contrast against a clear blue sky, conveying power and aspiration. |
| 10 | IG cali_carnivores - DSC09857.jpg | 0.8776 | 0.5921 | 60 | 9 | 8 | 7 | 10 | 9 | 8 | no | The image effectively conveys motion and power with a clear Ducati focus, strong lighting, and balanced composition. |
| 11 | AB Group - Session 1 (Turn 16 Entry) - CVR_0230_Mar1025_817AM_CaliPhoto.jpg | 0.8758 | 0.5860 | 60 | 9 | 8 | 7 | 10 | 9 | 8 | no | The image effectively conveys motion and power with a clear Ducati focus, strong lighting, and balanced composition. |
| 12 | AB Group - Session 2 (Turn 11) - CVR_1047_Mar1025_908AM_CaliPhoto.jpg | 0.8675 | 0.5583 | 60 | 9 | 8 | 9 | 10 | 9 | 8 | no | The image effectively showcases the Ducati motorcycle with high clarity, vibrant colors, and a dynamic composition that conveys power and motion. |
| 13 | C Group - Session 3 (Turn 9 and 8) - CVR_4982_Mar1025_1040AM_CaliPhoto.jpg | 0.8493 | 0.4978 | 60 | 8 | 7 | 9 | 10 | 9 | 8 | no | The image effectively conveys motion and power with clear Ducati identification, vibrant colors against a scenic background, and strong composition for Instagram. |
| 14 | IG cali_carnivores - DSC00013.jpg | 0.7376 | 0.7401 | 60 | 9 | 8 | 9 | 9 | 8 | 7 | no | The image effectively highlights the Ducati motorcycle with strong color contrast and clear subject placement, conveying power and aspiration. |
| 15 | 1-Around the Pits - CVR_0098_Mar1025_802AM_CaliPhoto.jpg | 0.7214 | 0.6723 | 60 | 8 | 9 | 7 | 8 | 8 | 7 | no | The image effectively highlights the Ducati motorcycle with clear subject clarity and strong lighting, conveying power and aspiration. |
| 16 | DSC-5436-NaraMedia.jpeg | 0.7109 | 0.6287 | 60 | 8 | 9 | 7 | 9 | 8 | 7 | no | The image effectively conveys motion and power with clear Ducati identification, strong lighting, and balanced composition. |
| 17 | IG renatobo - IMG_5013.jpeg | 0.6996 | 0.5817 | 60 | 8 | 9 | 7 | 7 | 8 | 7 | no | The image effectively captures the Ducati motorcycles and riders, with good lighting and clear subject focus against a contrasting background. |
| 18 | IG desmo.donna - IMG-1080-Donna.jpeg | 0.6982 | 0.5760 | 60 | 8 | 9 | 9 | 7 | 8 | 7 | no | The image effectively highlights the Ducati motorcycle with strong lighting and color contrast, but could benefit from a more dynamic composition. |
| 19 | IG desmo.donna - IMG-1015-Donna.jpeg | 0.6976 | 0.5732 | 60 | 8 | 9 | 7 | 7 | 8 | 7 | no | The image effectively showcases the Ducati motorcycles with clear branding and appealing lighting, though some background elements could be more focused. |
| 20 | 1-Around the Pits - 1-Around the Pits - CVR_0025_Mar1025_728AM_CaliPhoto.jpg | 0.6916 | 0.5483 | 60 | 8 | 9 | 7 | 8 | 8 | 7 | no | The image effectively showcases the Ducati motorcycle with strong lighting and clear subject focus, but could benefit from a more centered composition. |
| 21 | IMG_4984.jpeg | 0.6901 | 0.5421 | 60 | 9 | 8 | 7 | 8 | 8 | 7 | no | The image effectively highlights the Ducati motorcycle with clear subject clarity and strong emotion, but could benefit from improved color contrast and better lighting balance. |
| 22 | IG guardiansvoice - IMG-4129-Delenian.jpeg | 0.6899 | 0.5413 | 60 | 8 | 9 | 7 | 8 | 8 | 7 | no | The image effectively captures the Ducati motorcycles against a golden hour backdrop, with clear subject clarity and strong emotion conveyed through the setting sun and motorcycle arrangement. |
| 23 | IG - renatobo - lineup 1st session - March 10 DROC Track Day-12 HDR.jpeg | 0.6829 | 0.5123 | 60 | 8 | 7 | 9 | 8 | 8 | 7 | no | The image effectively highlights the Ducati motorcycle with vibrant colors and a clear subject against a contrasting background, conveying power and aspiration. |
| 24 | IG - renatobo - lineup 1st session - March 10 DROC Track Day-02 HDR.jpeg | 0.6778 | 0.4907 | 60 | 8 | 7 | 9 | 8 | 8 | 7 | no | The image effectively highlights the Ducati motorcycle with clear subject clarity and vibrant colors, conveying power and aspiration. |
| 25 | IMG_5012.jpeg | 0.6607 | 0.4195 | 60 | 8 | 9 | 7 | 8 | 8 | 7 | no | The image effectively showcases the Ducati motorcycles with clear subject clarity and strong lighting, conveying power and aspiration. |
| 26 | IG renatobo - IMG_5014.jpeg | 0.6504 | 0.3765 | 60 | 8 | 7 | 9 | 7 | 8 | 7 | no | The image effectively highlights the Ducati motorcycle against a clear sky, with strong color contrast and good lighting. The composition is engaging but could benefit from better placement for Instagram's grid. |
| 27 | IG - renatobo - lineup 1st session - March 10 DROC Track Day-01 HDR.jpeg | 0.6503 | 0.3764 | 60 | 8 | 9 | 7 | 8 | 8 | 7 | no | The image effectively showcases a group of motorcycles, with Ducati bikes identifiable, against a clear blue sky. The lighting is excellent, and the composition conveys power and aspiration. |
| 28 | IG naramedia_official - DSC-5365-NaraMedia.jpeg | 0.5367 | 0.7780 | 26 | 9 | 8 | 7 | 8 | 8 | 8 | no | The Ducati motorcycle stands out clearly against the background, with good lighting and color contrast. The composition is strong, likely to stop scrolling. |
| 29 | IG luckie.moto - 20250310-075123-luckie.moto.jpg | 0.4997 | 0.6374 | 55 | 8 | 9 | 7 | 7 | 8 | 6 | no | The image effectively showcases a lineup of motorcycles, with Ducati bikes identifiable, against a clear blue sky. The lighting is excellent, and the colors pop well, though some cropping could improve composition. |
| 30 | IG ocredevil - DROC-Track-Day-3-10-2-filippo pagan.jpeg | 0.4206 | 0.4312 | 49 | 6 | 7 | 8 | 5 | 7 | 6 | no | The image has good lighting and color pop, but the motorcycle is not clearly visible due to the focus on people. The composition could be improved for better impact. |
| 31 | IG - renatobo - March 10 DROC Track Day-26 HDR.jpeg | 0.4114 | 0.4768 | 23 | 7 | 8 | 6 | 7 | 7 | 8 | no | The image has good subject clarity and lighting, with a balanced composition that would likely stop scrolling on Instagram. The rider's pose adds emotion, but the motorcycle is not clearly identifiable as a Ducati. |
| 32 | IG martangelenos - PXL_20250310_132329430.jpg | 0.1571 | 0.4839 | 10 | 3 | 8 | 7 | 2 | 4 | 6 | no | The image has good lighting and color pop but lacks a clear motorcycle subject, making it less effective for an Instagram cover. |
| 33 | 2-Group Photo - Raffle - CVR_7682_Mar1025_1249PM_CaliPhoto.jpg | 0.0714 | 0.5357 | 27 | 3 | 6 | 7 | 4 | 5 | 4 | no | The image lacks a clear motorcycle subject and has moderate lighting, but the colors are vibrant and there's potential for emotion with people. |
| 34 | caliphotovideo - Group picture - DROC Track Day March 10 202500002.jpg | 0.0694 | 0.6484 | 23 | 3 | 7 | 6 | 2 | 4 | 1 | no | The image lacks a clear motorcycle subject and has low emotion, but the lighting is good. |
| 35 | IG m92663m - IMG-6019-Mark Momot.jpeg | 0.0670 | 0.6338 | 22 | 1 | 8 | 7 | 2 | 3 | 1 | no | The image lacks a clear motorcycle subject, focusing instead on an entrance sign with palm trees, making it less suitable for a Ducati-focused Instagram account. |
| 36 | 2-Group Photo - Raffle - CVR_7677_Mar1025_1249PM_CaliPhoto.jpg | 0.0666 | 0.5853 | 23 | 2 | 7 | 6 | 3 | 4 | 1 | no | The image focuses on a person holding a mug, not the motorcycle, making it unsuitable for a Ducati-focused Instagram cover. |
| 37 | caliphotovideo - Group picture - DROC Track Day March 10 202500001.jpg | 0.0584 | 0.6360 | 17 | 2 | 7 | 3 | 1 | 4 | 2 | no | The image is not focused on a motorcycle, making it unsuitable for a Ducati-focused Instagram account. The lighting and color pop are decent but the overall composition lacks clarity and emotion. |
| 38 | IG - renatobo - March 10 DROC Track Day-58 HDR.jpeg | 0.0503 | 0.5335 | 15 | 2 | 7 | 6 | 3 | 4 | 3 | no | The image does not focus on a motorcycle, so it doesn't meet the Ducati-focused criteria. The lighting is good, but the subject (a person with food) isn't clear or related to motorcycles. |
| 39 | IG guardiansvoice - IMG-4160-Delenian.jpeg | 0.0486 | 0.4960 | 15 | 3 | 7 | 6 | 2 | 4 | 3 | no | The image lacks a clear motorcycle focus and strong emotion, but the lighting is decent. |
| 40 | IG - renatobo - March 10 DROC Track Day-28.jpeg | 0.0480 | 0.4833 | 15 | 3 | 7 | 6 | 2 | 4 | 3 | no | The image features a group in front of a building with the text 'CHUCKWALLA', but it lacks a clear motorcycle subject and strong emotion, making it less suitable for an Instagram cover. |
| 41 | IG desmo.donna - IMG-1151-Donna.jpeg | 0.0453 | 0.4230 | 15 | 2 | 7 | 6 | 3 | 4 | 3 | no | The image lacks a motorcycle, focusing instead on people, which reduces subject clarity and relevance for a Ducati-focused account. |
| 42 | 1-Around the Pits - 1-Around the Pits - CVR_0012_Mar1025_702AM_CaliPhoto.jpg | 0.0336 | 0.3975 | 9 | 2 | 3 | 4 | 1 | 2 | 1 | no | The image does not focus on a motorcycle, lacks clear lighting and color contrast, and the composition is not engaging for Instagram. |

### openbmb/minicpm-v4.5:8b | yolo=off

| Rank | Image | Final | Tech | Vision | Subject | Light | Color | Emotion | Scroll | Crop | Failed | One line |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 | IG naramedia_official - DSC-5992-NaraMedia.jpeg | 0.7988 | 0.7571 | 49 | 9 | 8 | 7 | 8 | 9 | 10 | no | Dynamic composition and clear subject make this a strong cover photo for Ducati enthusiasts. |
| 2 | C Group - Session 2 Back Straight Speed Pans - CVR_2628_Mar1025_943AM_CaliPhoto.jpg | 0.7715 | 0.7827 | 46 | 9 | 8 | 7 | 8 | 9 | 10 | no | Dynamic Ducati action with clear subject and strong composition for a motorcycle enthusiast account. |
| 3 | 1-Around the Pits - CVR_0098_Mar1025_802AM_CaliPhoto.jpg | 0.7617 | 0.6723 | 48 | 9 | 8 | 7 | 6 | 10 | 8 | no | Strong subject clarity and composition make this a compelling cover photo for Ducati enthusiasts. |
| 4 | IG naramedia_official - DSC-5365-NaraMedia.jpeg | 0.7351 | 0.7780 | 43 | 9 | 8 | 7 | 6 | 10 | 8 | no | High clarity and lighting with strong brand presence make this a compelling cover photo. |
| 5 | IG naramedia_official - DSC-5897-NaraMedia.jpeg | 0.7318 | 0.6504 | 46 | 9 | 8 | 7 | 8 | 9 | 10 | no | Dynamic composition, clear Ducati branding, and strong motion conveyance make this a compelling cover photo. |
| 6 | DSC-5436-NaraMedia.jpeg | 0.7253 | 0.6287 | 46 | 9 | 8 | 7 | 8 | 9 | 10 | no | Dynamic composition, strong leading lines, and clear subject make this a compelling cover photo for Ducati enthusiasts. |
| 7 | 1-Around the Pits - 1-Around the Pits - CVR_0025_Mar1025_728AM_CaliPhoto.jpg | 0.7245 | 0.5483 | 48 | 9 | 8 | 7 | 6 | 10 | 8 | no | Strong subject clarity and composition with Ducati branding enhance appeal for enthusiast accounts. |
| 8 | IG - renatobo - lineup 1st session - March 10 DROC Track Day-12 HDR.jpeg | 0.7137 | 0.5123 | 48 | 9 | 8 | 7 | 6 | 10 | 8 | no | Strong subject clarity and scroll stop potential, with good lighting and color pop; composition works well for Instagram cover. |
| 9 | IG renatobo - IMG_5013.jpeg | 0.6762 | 0.5817 | 43 | 9 | 8 | 7 | 6 | 8 | 10 | no | Strong subject clarity and composition, with good lighting and color contrast, but could convey more emotion. |
| 10 | IG desmo.donna - IMG-1080-Donna.jpeg | 0.6745 | 0.5760 | 43 | 9 | 8 | 7 | 6 | 8 | 10 | no | Strong subject clarity and composition, with good lighting and color contrast, but lower emotion due to static pose. |
| 11 | IG renatobo - IMG_5014.jpeg | 0.6730 | 0.3765 | 48 | 9 | 8 | 7 | 6 | 10 | 8 | no | Strong subject clarity and scroll-stopping potential with a Ducati in clear light. |
| 12 | IMG_4984.jpeg | 0.6643 | 0.5421 | 43 | 9 | 8 | 7 | 6 | 8 | 10 | no | Strong subject focus and composition with Ducati branding enhance appeal for enthusiast accounts. |
| 13 | IG - renatobo - lineup 1st session - March 10 DROC Track Day-02 HDR.jpeg | 0.6489 | 0.4907 | 43 | 9 | 8 | 7 | 6 | 8 | 10 | no | Strong subject clarity and composition, with good lighting and color contrast, but lower emotion due to static pose. |
| 14 | IMG_5012.jpeg | 0.6275 | 0.4195 | 43 | 9 | 8 | 7 | 6 | 8 | 10 | no | Strong subject clarity and composition, with good lighting and color contrast, but could better convey Ducati brand emotion. |
| 15 | IG - renatobo - lineup 1st session - March 10 DROC Track Day-01 HDR.jpeg | 0.6146 | 0.3764 | 43 | 9 | 8 | 7 | 6 | 8 | 10 | no | Strong subject clarity and composition, with Ducati branding enhancing appeal. |
| 16 | IG cali_carnivores - DSC09850.jpg | 0.4800 | 0.7612 | 49 | 9 | 8 | 7 | 10 | 9 | 6 | no | Dynamic action shot with clear subject and strong composition, but could be improved for portrait cropping. |
| 17 | IG cali_carnivores - DSC09857.jpg | 0.4496 | 0.5921 | 49 | 9 | 8 | 7 | 10 | 9 | 6 | no | Dynamic action shot with clear subject and strong composition, but could use more lead room for portrait crop. |
| 18 | IG guardiansvoice - IMG-4129-Delenian.jpeg | 0.4404 | 0.5413 | 49 | 9 | 10 | 8 | 7 | 9 | 6 | no | Golden hour lighting and clear subject make this a strong cover photo for Ducati enthusiasts. |
| 19 | C Group - Session 2 Back Straight Speed Pans - CVR_2850_Mar1025_946AM_CaliPhoto.jpg | 0.4389 | 0.7661 | 43 | 9 | 8 | 7 | 8 | 10 | 6 | no | Dynamic composition and clear Ducati branding make this a strong cover photo. |
| 20 | AB Group - Session 4 (Turn 16) - CVR_6545_Mar1025_1122AM_CaliPhoto.jpg | 0.4315 | 0.7249 | 43 | 9 | 8 | 7 | 8 | 10 | 6 | no | Dynamic composition and clear subject make it strong for Ducati-focused content. |
| 21 | AB Group - Session 1 (Turn 16 Entry) - CVR_0148_Mar1025_815AM_CaliPhoto.jpg | 0.4124 | 0.6190 | 43 | 9 | 8 | 7 | 8 | 10 | 6 | no | Dynamic action shot with clear subject and strong composition, ideal for Ducati enthusiast account. |
| 22 | AB Group - Session 1 (Turn 16 Entry) - CVR_0230_Mar1025_817AM_CaliPhoto.jpg | 0.4065 | 0.5860 | 43 | 9 | 8 | 7 | 8 | 10 | 6 | no | Dynamic composition with clear subject and strong color contrast makes it ideal for Ducati-focused Instagram cover. |
| 23 | AB Group - Session 2 (Turn 11) - CVR_1047_Mar1025_908AM_CaliPhoto.jpg | 0.4015 | 0.5583 | 43 | 9 | 8 | 7 | 8 | 10 | 6 | no | Dynamic composition and clear subject make it strong for Ducati-focused content. |
| 24 | C Group - Session 3 (Turn 9 and 8) - CVR_4982_Mar1025_1040AM_CaliPhoto.jpg | 0.3906 | 0.4978 | 43 | 9 | 8 | 7 | 8 | 10 | 6 | no | Dynamic composition and clear subject make it strong for Ducati-focused account. |
| 25 | IG m92663m - IMG-6019-Mark Momot.jpeg | 0.3295 | 0.6338 | 19 | 2 | 6 | 3 | 1 | 0 | 7 | no | Image is of a sign for Chuckwalla and not the motorcycle; composition lacks subject clarity and emotional impact. |
| 26 | 2-Group Photo - Raffle - CVR_7677_Mar1025_1249PM_CaliPhoto.jpg | 0.3178 | 0.5853 | 19 | 2 | 6 | 3 | 1 | 0 | 7 | no | Focus is on a mug instead of motorcycle; composition and context are off. |
| 27 | caliphotovideo - Group picture - DROC Track Day March 10 202500002.jpg | 0.3112 | 0.6484 | 10 | 2 | 7 | 6 | 3 | 8 | 9 | no | Strong composition and branding make up for lack of clear motorcycle focus. |
| 28 | caliphotovideo - Group picture - DROC Track Day March 10 202500001.jpg | 0.3075 | 0.6360 | 10 | 2 | 6 | 7 | 3 | 8 | 9 | no | Strong group shot with clear branding and vibrant colors, but lacks a distinct Ducati motorcycle as the focal point. |
| 29 | 2-Group Photo - Raffle - CVR_7682_Mar1025_1249PM_CaliPhoto.jpg | 0.3059 | 0.5357 | 19 | 2 | 6 | 3 | 1 | 0 | 7 | no | Focus is on people, not motorcycle; composition and lighting are average. |
| 30 | IG - renatobo - March 10 DROC Track Day-58 HDR.jpeg | 0.3054 | 0.5335 | 19 | 2 | 6 | 3 | 1 | 0 | 7 | no | photo is of a person at an event with food, not motorcycle; lacks clarity and emotion |
| 31 | IG guardiansvoice - IMG-4160-Delenian.jpeg | 0.2964 | 0.4960 | 19 | 2 | 6 | 3 | 1 | 0 | 7 | no | Lacks motorcycle focus; strong composition but misses key elements for Ducati account. |
| 32 | IG - renatobo - March 10 DROC Track Day-28.jpeg | 0.2617 | 0.4833 | 10 | 2 | 7 | 6 | 3 | 8 | 9 | no | Great lighting and composition for a Ducati enthusiast account, but the subject is not clearly identifiable as such. |
| 33 | IG desmo.donna - IMG-1151-Donna.jpeg | 0.2436 | 0.4230 | 10 | 2 | 7 | 6 | 3 | 8 | 9 | no | Great lighting and composition for a Ducati enthusiast account, but lacks clear motorcycle focus. |
| 34 | 1-Around the Pits - 1-Around the Pits - CVR_0012_Mar1025_702AM_CaliPhoto.jpg | 0.2359 | 0.3975 | 10 | 3 | 6 | 7 | 2 | 8 | 9 | no | The image is a group shot with no clear motorcycle focus, but the Ducati logo in corner adds brand relevance. |
| 35 | 1-Around the Pits - CVR_0792_Mar1025_832AM_CaliPhoto.jpg | 0.0800 | 0.8839 | 23 | 9 | 8 | 7 | 6 | 10 | 3 | no | Strong subject clarity and lighting with a Ducati logo, but composition struggles for Instagram's portrait crop. |
| 36 | IG - renatobo - March 10 DROC Track Day-26 HDR.jpeg | 0.0740 | 0.4768 | 30 | 7 | 8 | 6 | 3 | 4 | 2 | no | The image lacks a clear Ducati focus and has limited emotional impact, but the lighting is strong. |
| 37 | IG cali_carnivores - DSC00013.jpg | 0.0736 | 0.7401 | 23 | 9 | 8 | 7 | 6 | 10 | 3 | no | Strong subject clarity and lighting with Ducati branding boost scores; tight crop limits Instagram grid potential. |
| 38 | IG martangelenos - PXL_20250310_132329430.jpg | 0.0725 | 0.4839 | 29 | 2 | 3 | 6 | 7 | 8 | 1 | no | The image is not suitable for a Ducati enthusiast Instagram cover due to the wrong subject and composition. |
| 39 | IG luckie.moto - 20250310-075123-luckie.moto.jpg | 0.0689 | 0.6374 | 23 | 8 | 7 | 9 | 6 | 10 | 3 | no | Strong subject clarity and color pop make it ideal for Ducati enthusiasts; lighting is good but not golden hour. |
| 40 | IG kamiumitv - IMG-3981-Bao.jpeg | 0.0680 | 0.6171 | 23 | 9 | 8 | 7 | 6 | 10 | 3 | no | Strong subject clarity and lighting with high color pop; composition works for scroll but may need adjustment for ideal crop. |
| 41 | IG desmo.donna - IMG-1015-Donna.jpeg | 0.0660 | 0.5732 | 23 | 8 | 9 | 7 | 6 | 10 | 3 | no | Strong lighting and clear subject make it a good cover candidate, but composition could be improved for Instagram's grid. |
| 42 | IG ocredevil - DROC-Track-Day-3-10-2-filippo pagan.jpeg | 0.0492 | 0.4312 | 17 | 2 | 6 | 3 | 1 | 4 | 0 | no | Poor subject focus, distracting background, and lack of motorcycle branding or emotion. |

### gemma4:e4b | yolo=off

| Rank | Image | Final | Tech | Vision | Subject | Light | Color | Emotion | Scroll | Crop | Failed | One line |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 1-Around the Pits - CVR_0792_Mar1025_832AM_CaliPhoto.jpg | 0.8952 | 0.8839 | 54 | 9 | 9 | 9 | 9 | 9 | 9 | no | Vision scoring summary |
| 2 | IG naramedia_official - DSC-5365-NaraMedia.jpeg | 0.8284 | 0.7780 | 51 | 9 | 8 | 8 | 9 | 9 | 8 | no | Excellent composition with strong subject focus and dynamic angle, perfect for a high-impact Instagram feed. |
| 3 | IG renatobo - IMG_5013.jpeg | 0.8045 | 0.5817 | 54 | 9 | 9 | 9 | 9 | 9 | 9 | no | Vision scoring summary |
| 4 | IG naramedia_official - DSC-5897-NaraMedia.jpeg | 0.8018 | 0.6504 | 52 | 9 | 8 | 8 | 9 | 9 | 9 | no | Excellent dynamic shot with strong motion blur and subject focus, perfect for a high-impact Instagram feed. |
| 5 | IG naramedia_official - DSC-5992-NaraMedia.jpeg | 0.7988 | 0.7571 | 49 | 8 | 7 | 8 | 9 | 9 | 8 | no | Strong motion blur and vibrant bike color against a neutral track background make this highly engaging for a Ducati enthusiast feed. |
| 6 | C Group - Session 2 Back Straight Speed Pans - CVR_2850_Mar1025_946AM_CaliPhoto.jpg | 0.7898 | 0.7661 | 48 | 8 | 8 | 8 | 8 | 8 | 8 | no | Vision scoring summary |
| 7 | IG - renatobo - lineup 1st session - March 10 DROC Track Day-12 HDR.jpeg | 0.7837 | 0.5123 | 54 | 9 | 9 | 9 | 9 | 9 | 9 | no | Vision scoring summary |
| 8 | AB Group - Session 4 (Turn 16) - CVR_6545_Mar1025_1122AM_CaliPhoto.jpg | 0.7775 | 0.7249 | 48 | 8 | 7 | 8 | 9 | 8 | 8 | no | Strong action shot with good contrast and dynamic angle, making it highly engaging for a motorcycle enthusiast feed. |
| 9 | IG cali_carnivores - DSC00013.jpg | 0.7704 | 0.7401 | 47 | 8 | 6 | 7 | 8 | 8 | 8 | no | Strong, aggressive composition with excellent subject presence, though the lighting is flat and the background is unremarkable for maximum impact. |
| 10 | 1-Around the Pits - CVR_0098_Mar1025_802AM_CaliPhoto.jpg | 0.7617 | 0.6723 | 48 | 8 | 8 | 8 | 8 | 8 | 8 | no | Vision scoring summary |
| 11 | AB Group - Session 1 (Turn 16 Entry) - CVR_0148_Mar1025_815AM_CaliPhoto.jpg | 0.7457 | 0.6190 | 48 | 8 | 6 | 7 | 9 | 8 | 8 | no | Strong action shot with excellent rider posture and leading lines, though the background is slightly distracting for a perfect Instagram cover. |
| 12 | IG kamiumitv - IMG-3981-Bao.jpeg | 0.7451 | 0.6171 | 48 | 8 | 8 | 8 | 8 | 8 | 8 | no | Vision scoring summary |
| 13 | AB Group - Session 2 (Turn 11) - CVR_1047_Mar1025_908AM_CaliPhoto.jpg | 0.7391 | 0.5583 | 49 | 8 | 7 | 8 | 9 | 9 | 8 | no | Excellent dynamic shot with strong leading lines and vibrant color contrast, perfect for an action-oriented Ducati feed. |
| 14 | IG cali_carnivores - DSC09857.jpg | 0.7376 | 0.5921 | 48 | 8 | 8 | 8 | 8 | 8 | 8 | no | Vision scoring summary |
| 15 | AB Group - Session 1 (Turn 16 Entry) - CVR_0230_Mar1025_817AM_CaliPhoto.jpg | 0.7358 | 0.5860 | 48 | 8 | 8 | 8 | 8 | 8 | 8 | no | Vision scoring summary |
| 16 | IG desmo.donna - IMG-1080-Donna.jpeg | 0.7328 | 0.5760 | 48 | 8 | 8 | 8 | 8 | 8 | 8 | no | Vision scoring summary |
| 17 | IG desmo.donna - IMG-1015-Donna.jpeg | 0.7320 | 0.5732 | 48 | 8 | 8 | 8 | 8 | 8 | 8 | no | Vision scoring summary |
| 18 | IG - renatobo - lineup 1st session - March 10 DROC Track Day-02 HDR.jpeg | 0.7072 | 0.4907 | 48 | 8 | 7 | 8 | 8 | 8 | 9 | no | Strong composition with excellent color contrast and a dynamic, slightly low angle that emphasizes the bike's power, making it highly Instagrammable. |
| 19 | IG cali_carnivores - DSC09850.jpg | 0.5747 | 0.7612 | 42 | 7 | 7 | 7 | 7 | 7 | 7 | no | Vision scoring summary |
| 20 | IG luckie.moto - 20250310-075123-luckie.moto.jpg | 0.5450 | 0.6374 | 42 | 7 | 7 | 7 | 7 | 7 | 7 | no | Vision scoring summary |
| 21 | DSC-5436-NaraMedia.jpeg | 0.5429 | 0.6287 | 42 | 7 | 7 | 7 | 7 | 7 | 7 | no | Vision scoring summary |
| 22 | IG guardiansvoice - IMG-4129-Delenian.jpeg | 0.5219 | 0.5413 | 42 | 7 | 7 | 7 | 7 | 7 | 7 | no | Vision scoring summary |
| 23 | IG - renatobo - March 10 DROC Track Day-26 HDR.jpeg | 0.5064 | 0.4768 | 42 | 7 | 7 | 7 | 7 | 7 | 7 | no | Vision scoring summary |
| 24 | C Group - Session 2 Back Straight Speed Pans - CVR_2628_Mar1025_943AM_CaliPhoto.jpg | 0.3929 | 0.7827 | 36 | 6 | 6 | 6 | 6 | 6 | 6 | no | Vision scoring summary |
| 25 | 1-Around the Pits - 1-Around the Pits - CVR_0025_Mar1025_728AM_CaliPhoto.jpg | 0.3507 | 0.5483 | 36 | 6 | 6 | 6 | 6 | 6 | 6 | no | Vision scoring summary |
| 26 | IMG_4984.jpeg | 0.3496 | 0.5421 | 36 | 6 | 6 | 6 | 6 | 6 | 6 | no | Vision scoring summary |
| 27 | IG renatobo - IMG_5014.jpeg | 0.3198 | 0.3765 | 36 | 6 | 6 | 6 | 6 | 6 | 6 | no | Vision scoring summary |
| 28 | IG - renatobo - lineup 1st session - March 10 DROC Track Day-01 HDR.jpeg | 0.3197 | 0.3764 | 36 | 6 | 6 | 6 | 6 | 6 | 6 | no | Vision scoring summary |
| 29 | caliphotovideo - Group picture - DROC Track Day March 10 202500001.jpg | 0.1893 | 0.6360 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 30 | 2-Group Photo - Raffle - CVR_7682_Mar1025_1249PM_CaliPhoto.jpg | 0.1787 | 0.5357 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 31 | IG martangelenos - PXL_20250310_132329430.jpg | 0.1733 | 0.4839 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 32 | IMG_5012.jpeg | 0.1665 | 0.4195 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 33 | caliphotovideo - Group picture - DROC Track Day March 10 202500002.jpg | 0.0712 | 0.6484 | 24 | 4 | 4 | 4 | 4 | 4 | 4 | no | Vision scoring summary |
| 34 | 2-Group Photo - Raffle - CVR_7677_Mar1025_1249PM_CaliPhoto.jpg | 0.0683 | 0.5853 | 24 | 4 | 4 | 4 | 4 | 4 | 4 | no | Vision scoring summary |
| 35 | IG guardiansvoice - IMG-4160-Delenian.jpeg | 0.0643 | 0.4960 | 24 | 4 | 4 | 4 | 4 | 4 | 4 | no | Vision scoring summary |
| 36 | IG m92663m - IMG-6019-Mark Momot.jpeg | 0.0635 | 0.6338 | 20 | 3 | 4 | 4 | 4 | 4 | 1 | no | Vision scoring summary |
| 37 | 1-Around the Pits - 1-Around the Pits - CVR_0012_Mar1025_702AM_CaliPhoto.jpg | 0.0599 | 0.3975 | 24 | 4 | 4 | 4 | 4 | 4 | 4 | no | Vision scoring summary |
| 38 | C Group - Session 3 (Turn 9 and 8) - CVR_4982_Mar1025_1040AM_CaliPhoto.jpg | 0.0574 | 0.4978 | 20 | 3 | 4 | 4 | 4 | 4 | 1 | no | Vision scoring summary |
| 39 | IG ocredevil - DROC-Track-Day-3-10-2-filippo pagan.jpeg | 0.0544 | 0.4312 | 20 | 3 | 4 | 4 | 4 | 4 | 1 | no | Vision scoring summary |
| 40 | IG - renatobo - March 10 DROC Track Day-28.jpeg | 0.0533 | 0.4833 | 18 | 3 | 3 | 3 | 3 | 3 | 3 | no | Vision scoring summary |
| 41 | IG - renatobo - March 10 DROC Track Day-58 HDR.jpeg | 0.0450 | 0.5335 | 12 | 2 | 2 | 2 | 2 | 2 | 2 | no | Vision scoring summary |
| 42 | IG desmo.donna - IMG-1151-Donna.jpeg | 0.0400 | 0.4230 | 12 | 2 | 2 | 2 | 2 | 2 | 2 | no | Vision scoring summary |

### claude-haiku-4-5-20251001 | scorer=claude

| Rank | Image | Final | Tech | Vision | Subject | Light | Color | Emotion | Scroll | Crop | Failed | One line |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 | IMG_4984.jpeg | 0.6760 | 0.5421 | 44 | 9 | 6 | 7 | 7 | 7 | 8 | no | Strong front-facing Ducati Panigale V4R identity with tire warmer detail adds authenticity, but flat garage lighting and static presentation limit the aspirational energy needed for a standout cover. |
| 2 | IG naramedia_official - DSC-5365-NaraMedia.jpeg | 0.6720 | 0.7780 | 52 | 10 | 8 | 9 | 9 | 9 | 7 | no | Ducati 848 Corse in iconic red-white-black tricolor dominates the frame with a dramatic low-angle 3/4 front shot against clean desert sky — powerful brand identity but centered placement and wide format hurt portrait crop potential |
| 3 | 1-Around the Pits - CVR_0792_Mar1025_832AM_CaliPhoto.jpg | 0.6695 | 0.8839 | 49 | 10 | 7 | 9 | 8 | 8 | 7 | no | Red Ducati Panigale V4 pops hard against the desert mountain backdrop but flat midday light and dead-center framing flatten the drama that a low angle or golden-hour shot would deliver |
| 4 | IG naramedia_official - DSC-5897-NaraMedia.jpeg | 0.6508 | 0.6504 | 53 | 10 | 8 | 9 | 10 | 9 | 7 | no | A aggressive corner-carving Ducati Panigale V4 in full Corse livery with dramatic lean angle and motion blur creates an instinctive scroll-stop, but the centered-landscape composition limits portrait crop flexibility. |
| 5 | IG cali_carnivores - DSC00013.jpg | 0.6350 | 0.7401 | 49 | 10 | 7 | 9 | 8 | 8 | 7 | no | Bold red Panigale V4 pops hard against the pale sky with strong front-quarter presence, but flat midday light and near-centered framing limit drama and portrait-crop flexibility. |
| 6 | AB Group - Session 2 (Turn 11) - CVR_1047_Mar1025_908AM_CaliPhoto.jpg | 0.6193 | 0.5583 | 52 | 10 | 7 | 9 | 10 | 9 | 7 | no | Aggressive cornering angle on a Ducati Panigale V4 in signature Ducati red delivers raw speed and emotion, but flat midday light and centered composition slightly limit crop flexibility and depth drama. |
| 7 | AB Group - Session 4 (Turn 16) - CVR_6545_Mar1025_1122AM_CaliPhoto.jpg | 0.6126 | 0.7249 | 47 | 9 | 6 | 8 | 9 | 8 | 7 | no | Aggressive lean angle on a Ducati Hypermotard in full Dainese livery creates visceral tension, but flat midday desert light robs the chrome and bodywork of the drama this action deserves. |
| 8 | AB Group - Session 1 (Turn 16 Entry) - CVR_0148_Mar1025_815AM_CaliPhoto.jpg | 0.6059 | 0.6190 | 49 | 9 | 7 | 9 | 9 | 8 | 7 | no | Red Ducati Panigale carving hard against dramatic desert mountains delivers strong brand energy, but slightly centered framing and midday flat light limit scroll-stop ceiling |
| 9 | IG kamiumitv - IMG-3981-Bao.jpeg | 0.5961 | 0.6171 | 48 | 9 | 7 | 9 | 8 | 8 | 7 | no | Two red Ducatis (Streetfighter V4 + Panigale V4) against clean SoCal blue sky is a brand-lover's dream pairing, but midday flat light and tight dual-bike framing reduce portrait crop flexibility |
| 10 | AB Group - Session 1 (Turn 16 Entry) - CVR_0230_Mar1025_817AM_CaliPhoto.jpg | 0.5793 | 0.5860 | 47 | 9 | 6 | 8 | 9 | 8 | 7 | no | Aggressive cornering stance on a red Ducati Panigale (#34) with dramatic mountain backdrop delivers strong emotion and brand identity, but flat midday light and slightly cluttered background prevent a perfect score. |
| 11 | 1-Around the Pits - CVR_0098_Mar1025_802AM_CaliPhoto.jpg | 0.5627 | 0.6723 | 43 | 9 | 6 | 7 | 7 | 7 | 7 | no | Strong Ducati Panigale V4 subject presence with bold orange-on-black livery, but flat midday lighting and cluttered pit-lane background reduce the dramatic impact this bike deserves. |
| 12 | IG - renatobo - lineup 1st session - March 10 DROC Track Day-02 HDR.jpeg | 0.5004 | 0.4907 | 41 | 8 | 6 | 8 | 6 | 6 | 7 | no | Iconic Ducati Panigale V4 Corse livery with number 21 pops beautifully against the desert backdrop, but flat midday light, standing rider pose, and cluttered pit-lane context rob it of the drama and motion needed to truly stop a scroll. |
| 13 | C Group - Session 2 Back Straight Speed Pans - CVR_2850_Mar1025_946AM_CaliPhoto.jpg | 0.4669 | 0.7661 | 47 | 9 | 7 | 9 | 8 | 8 | 6 | no | Striking red Ducati Panigale V4R in full Ducati Corse livery pops brilliantly against the desert sky, but flat midday light and wide landscape framing hurt portrait crop viability and dramatic impact. |
| 14 | IG naramedia_official - DSC-5992-NaraMedia.jpeg | 0.4513 | 0.7571 | 45 | 8 | 7 | 8 | 9 | 7 | 6 | no | Aggressive Panigale lean angle with red-on-desert contrast delivers strong emotion, but wide landscape framing and centered subject hurt crop flexibility and thumbnail impact |
| 15 | C Group - Session 2 Back Straight Speed Pans - CVR_2628_Mar1025_943AM_CaliPhoto.jpg | 0.4419 | 0.7827 | 43 | 8 | 6 | 8 | 8 | 7 | 6 | no | Red Ducati Panigale leaning hard into a desert track corner with a second bike in tow creates compelling racing drama, but flat midday light and a busy mid-ground undercut the impact at thumbnail size. |
| 16 | IG cali_carnivores - DSC09857.jpg | 0.4216 | 0.5921 | 45 | 9 | 6 | 7 | 9 | 8 | 6 | no | Ducati Streetfighter V4 in aggressive lean with motion blur creates visceral energy, but harsh midday light and tight framing limit portrait crop potential |
| 17 | DSC-5436-NaraMedia.jpeg | 0.4142 | 0.6287 | 43 | 8 | 6 | 8 | 8 | 7 | 6 | no | Three Ducati Panigales in desert race context delivers strong brand energy and motion, but flat midday light, wide composition, and portrait-crop challenges with three subjects sharing the frame limit Instagram cover potential. |
| 18 | 1-Around the Pits - 1-Around the Pits - CVR_0025_Mar1025_728AM_CaliPhoto.jpg | 0.3997 | 0.5483 | 43 | 8 | 7 | 8 | 7 | 7 | 6 | no | Strong Ducati Panigale V4 lineup with vivid red pop but cluttered pit-lane background and flat midday light limit scroll-stopping impact and portrait crop flexibility. |
| 19 | IG renatobo - IMG_5013.jpeg | 0.3917 | 0.5817 | 41 | 8 | 6 | 7 | 7 | 7 | 6 | no | Two Ducati Panigales fist-bumping at Chuckwalla is a compelling moment but flat midday light, eye-level angle, and cluttered background undercut the drama. |
| 20 | IG - renatobo - lineup 1st session - March 10 DROC Track Day-12 HDR.jpeg | 0.3862 | 0.5123 | 42 | 8 | 6 | 8 | 7 | 7 | 6 | no | Red Ducati Hypermotard SP livery pops against desert backdrop but flat midday light, eye-level camera angle, and second rider competing for attention dilute the impact. |
| 21 | IG desmo.donna - IMG-1080-Donna.jpeg | 0.3837 | 0.5760 | 40 | 7 | 8 | 8 | 5 | 6 | 6 | no | Stunning red Ducati Panigale with golden-hour light and great color pop, but upright camera angle, people in frame, and cluttered paddock background kill the emotional impact and scroll-stop potential. |
| 22 | IG luckie.moto - 20250310-075123-luckie.moto.jpg | 0.3737 | 0.6374 | 37 | 7 | 6 | 7 | 5 | 6 | 6 | no | Strong red Ducati V4 anchors a visually rich lineup but the side-on standing-height angle, flat midday light, and crowded multi-bike composition dilute impact — a low 3/4 solo shot at golden hour would transform this. |
| 23 | IG renatobo - IMG_5014.jpeg | 0.3548 | 0.3765 | 41 | 7 | 6 | 8 | 7 | 7 | 6 | no | Red Panigale against desert sky and Chuckwalla signage creates authentic track-day storytelling, but harsh midday backlight silhouettes the bike and rider, losing Ducati detail and brand impact. |
| 24 | IMG_5012.jpeg | 0.3415 | 0.4195 | 38 | 8 | 6 | 7 | 5 | 6 | 6 | no | Two Ducati Panigales in white and red at Chuckwalla have great brand appeal but flat midday light, eye-level shooting angle, and a cluttered background with a third bike dilute the visual impact needed for a strong cover shot. |
| 25 | 2-Group Photo - Raffle - CVR_7677_Mar1025_1249PM_CaliPhoto.jpg | 0.3224 | 0.5853 | 31 | 6 | 5 | 5 | 4 | 5 | 6 | no | Charming pit crew mug moment but no motorcycle in frame makes this a weak cover candidate for a Ducati account despite the DOC branding in corner |
| 26 | IG - renatobo - March 10 DROC Track Day-26 HDR.jpeg | 0.3098 | 0.4768 | 32 | 6 | 7 | 5 | 4 | 4 | 6 | no | A smiling guy on an electric kick scooter at dusk has warm light and decent clarity but lacks motorcycle content, power, or aspirational energy for a Ducati enthusiast account. |
| 27 | IG guardiansvoice - IMG-4160-Delenian.jpeg | 0.2993 | 0.4960 | 30 | 5 | 6 | 5 | 4 | 4 | 6 | no | Fun track-day vibe with the DROC map and Ducati branding, but no motorcycle in frame kills the core appeal for a moto account — this works better as a rider lifestyle post than a cover. |
| 28 | IG desmo.donna - IMG-1151-Donna.jpeg | 0.2791 | 0.4230 | 29 | 5 | 6 | 5 | 3 | 4 | 6 | no | Three smiling women in Ducati/DROC gear at a track day is community-friendly content but lacks a motorcycle as focal point, dramatic angle, or scroll-stopping visual tension needed for a strong cover. |
| 29 | IG cali_carnivores - DSC09850.jpg | 0.2392 | 0.7612 | 39 | 7 | 6 | 7 | 8 | 6 | 5 | no | Ducati Streetfighter wheelie on desert track delivers raw power, but small subject size, flat midday light, and wide landscape framing hurt thumbnail impact and portrait crop viability. |
| 30 | IG guardiansvoice - IMG-4129-Delenian.jpeg | 0.2324 | 0.5413 | 43 | 6 | 9 | 8 | 7 | 8 | 5 | no | Stunning Chuckwalla golden-hour sunset with red Ducati Panigale and Aprilia RSV4 creates compelling scene, but multiple bikes and people fragment focal point and portrait crop struggles to isolate a single hero subject |
| 31 | IG desmo.donna - IMG-1015-Donna.jpeg | 0.2194 | 0.5732 | 39 | 7 | 8 | 8 | 5 | 6 | 5 | no | Golden-hour Ducati demo day atmosphere is authentic and on-brand, but static parking-lot staging, eye-level perspective, and cluttered multi-bike scene dilute impact — a low 3/4 angle isolating the foreground Panigale V4 against the sunset would unlock this shot's potential. |
| 32 | IG - renatobo - lineup 1st session - March 10 DROC Track Day-01 HDR.jpeg | 0.1865 | 0.3764 | 36 | 6 | 6 | 7 | 6 | 6 | 5 | no | Strong Ducati group energy and desert backdrop add brand appeal, but crowded grid staging with no single focal hero, flat midday light, and competing subjects hurt scroll-stop power and portrait crop viability. |
| 33 | IG - renatobo - March 10 DROC Track Day-28.jpeg | 0.1855 | 0.4833 | 33 | 4 | 7 | 6 | 6 | 5 | 5 | no | Fun group shot at Chuckwalla with great golden-hour sky, but no motorcycle visible and the building/signage dominates — works as club lifestyle content but fails as a motorcycle cover photo. |
| 34 | IG m92663m - IMG-6019-Mark Momot.jpeg | 0.1646 | 0.6338 | 24 | 1 | 6 | 6 | 2 | 4 | 5 | no | No motorcycle present — this is an entrance gate photo that scores near-zero on bike-centric criteria despite decent desert scenery and leading lines. |
| 35 | 2-Group Photo - Raffle - CVR_7682_Mar1025_1249PM_CaliPhoto.jpg | 0.1542 | 0.5357 | 24 | 4 | 5 | 4 | 3 | 3 | 5 | no | No motorcycle visible — this registration/check-in event photo lacks the bike, speed, and drama needed for a compelling Ducati Instagram cover. |
| 36 | C Group - Session 3 (Turn 9 and 8) - CVR_4982_Mar1025_1040AM_CaliPhoto.jpg | 0.0784 | 0.4978 | 32 | 5 | 5 | 5 | 7 | 6 | 4 | no | Multi-bike racing action has energy but no single focal point dominates, background mountain clutter competes, and no clear Ducati presence weakens brand relevance |
| 37 | caliphotovideo - Group picture - DROC Track Day March 10 202500002.jpg | 0.0782 | 0.6484 | 28 | 3 | 5 | 5 | 6 | 5 | 4 | no | Large enthusiastic group shot on a desert track with DROC/Ducati branding has community energy but no motorcycle as focal point, flat midday light, and poor portrait crop potential. |
| 38 | IG martangelenos - PXL_20250310_132329430.jpg | 0.0725 | 0.4839 | 29 | 3 | 7 | 6 | 4 | 5 | 4 | no | Stunning desert sunset backdrop but the Chuckwalla track facility sign dominates — no motorcycle is identifiable as the hero subject, making this a location shot rather than a bike feature. |
| 39 | caliphotovideo - Group picture - DROC Track Day March 10 202500001.jpg | 0.0689 | 0.6360 | 23 | 3 | 5 | 4 | 4 | 4 | 3 | no | Large group photo at a Ducati club track day — great community content but lacks a hero motorcycle focal point, flat midday light, and portrait crop would sacrifice most of the group. |
| 40 | IG - renatobo - March 10 DROC Track Day-58 HDR.jpeg | 0.0608 | 0.5335 | 21 | 3 | 4 | 3 | 4 | 3 | 4 | no | No motorcycle present — this is a backyard BBQ party photo with zero riding/bike content, making it a poor fit for a Ducati Instagram cover despite the DROC watermark. |
| 41 | 1-Around the Pits - 1-Around the Pits - CVR_0012_Mar1025_702AM_CaliPhoto.jpg | 0.0564 | 0.3975 | 22 | 3 | 5 | 4 | 3 | 3 | 4 | no | Indoor club meeting with no motorcycle visible — people-focused gathering shot misses every motorcycle content criterion for a Ducati enthusiast cover |
| 42 | IG ocredevil - DROC-Track-Day-3-10-2-filippo pagan.jpeg | 0.0527 | 0.4312 | 19 | 3 | 5 | 3 | 2 | 2 | 4 | no | Two men on electric kick scooters in a parking lot — no motorcycle is the primary subject, making this unsuitable for a Ducati/motorcycle enthusiast Instagram cover. |
