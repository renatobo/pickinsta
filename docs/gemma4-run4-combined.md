# Model Benchmark Report (Speed + Quality)

- Generated: `2026-04-09T10:43:25`
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
| gemma4:e4b \| yolo=off \| prompt=claude+system | ollama | gemma4:e4b | off | 24.12 | 2.49 | 1013.03 | 0.00 | 0.23 | 20 | 1.00x |

## Per-run Timing

| Variant | Run | Duration (s) | Sec/img | Imgs/min | Failures |
|---|---:|---:|---:|---:|---:|
| gemma4:e4b \| yolo=off \| prompt=claude+system | 1 | 1013.03 | 24.12 | 2.49 | 0 |

## Image-by-Image Score Comparison (Run 1)

| Image | gemma4:e4b \| yolo=off \| prompt=claude+system final | gemma4:e4b \| yolo=off \| prompt=claude+system rank |
|---|---:|---:|
| 1-Around the Pits - 1-Around the Pits - CVR_0012_Mar1025_702AM_CaliPhoto.jpg | 0.0599 | 38 |
| 1-Around the Pits - 1-Around the Pits - CVR_0025_Mar1025_728AM_CaliPhoto.jpg | 0.7245 | 18 |
| 1-Around the Pits - CVR_0098_Mar1025_802AM_CaliPhoto.jpg | 0.3730 | 27 |
| 1-Around the Pits - CVR_0792_Mar1025_832AM_CaliPhoto.jpg | 0.7490 | 10 |
| 2-Group Photo - Raffle - CVR_7677_Mar1025_1249PM_CaliPhoto.jpg | 0.0683 | 35 |
| 2-Group Photo - Raffle - CVR_7682_Mar1025_1249PM_CaliPhoto.jpg | 0.0346 | 42 |
| AB Group - Session 1 (Turn 16 Entry) - CVR_0148_Mar1025_815AM_CaliPhoto.jpg | 0.7457 | 11 |
| AB Group - Session 1 (Turn 16 Entry) - CVR_0230_Mar1025_817AM_CaliPhoto.jpg | 0.8092 | 2 |
| AB Group - Session 2 (Turn 11) - CVR_1047_Mar1025_908AM_CaliPhoto.jpg | 0.7391 | 12 |
| AB Group - Session 4 (Turn 16) - CVR_6545_Mar1025_1122AM_CaliPhoto.jpg | 0.7775 | 7 |
| C Group - Session 2 Back Straight Speed Pans - CVR_2628_Mar1025_943AM_CaliPhoto.jpg | 0.3929 | 25 |
| C Group - Session 2 Back Straight Speed Pans - CVR_2850_Mar1025_946AM_CaliPhoto.jpg | 0.7898 | 4 |
| C Group - Session 3 (Turn 9 and 8) - CVR_4982_Mar1025_1040AM_CaliPhoto.jpg | 0.5115 | 24 |
| DSC-5436-NaraMedia.jpeg | 0.7253 | 17 |
| IG - renatobo - March 10 DROC Track Day-26 HDR.jpeg | 0.7030 | 20 |
| IG - renatobo - March 10 DROC Track Day-28.jpeg | 0.0428 | 41 |
| IG - renatobo - March 10 DROC Track Day-58 HDR.jpeg | 0.1785 | 31 |
| IG - renatobo - lineup 1st session - March 10 DROC Track Day-01 HDR.jpeg | 0.6729 | 21 |
| IG - renatobo - lineup 1st session - March 10 DROC Track Day-02 HDR.jpeg | 0.7072 | 19 |
| IG - renatobo - lineup 1st session - March 10 DROC Track Day-12 HDR.jpeg | 0.7837 | 6 |
| IG cali_carnivores - DSC00013.jpg | 0.7704 | 8 |
| IG cali_carnivores - DSC09850.jpg | 0.3890 | 26 |
| IG cali_carnivores - DSC09857.jpg | 0.7376 | 13 |
| IG desmo.donna - IMG-1015-Donna.jpeg | 0.7320 | 16 |
| IG desmo.donna - IMG-1080-Donna.jpeg | 0.7328 | 15 |
| IG desmo.donna - IMG-1151-Donna.jpeg | 0.0505 | 40 |
| IG guardiansvoice - IMG-4129-Delenian.jpeg | 0.5219 | 23 |
| IG guardiansvoice - IMG-4160-Delenian.jpeg | 0.0643 | 37 |
| IG kamiumitv - IMG-3981-Bao.jpeg | 0.7335 | 14 |
| IG luckie.moto - 20250310-075123-luckie.moto.jpg | 0.7512 | 9 |
| IG m92663m - IMG-6019-Mark Momot.jpeg | 0.0670 | 36 |
| IG martangelenos - PXL_20250310_132329430.jpg | 0.0568 | 39 |
| IG naramedia_official - DSC-5365-NaraMedia.jpeg | 0.8284 | 1 |
| IG naramedia_official - DSC-5897-NaraMedia.jpeg | 0.7901 | 3 |
| IG naramedia_official - DSC-5992-NaraMedia.jpeg | 0.7871 | 5 |
| IG ocredevil - DROC-Track-Day-3-10-2-filippo pagan.jpeg | 0.3296 | 28 |
| IG renatobo - IMG_5013.jpeg | 0.1836 | 30 |
| IG renatobo - IMG_5014.jpeg | 0.3198 | 29 |
| IMG_4984.jpeg | 0.5221 | 22 |
| IMG_5012.jpeg | 0.1665 | 32 |
| caliphotovideo - Group picture - DROC Track Day March 10 202500001.jpg | 0.0706 | 34 |
| caliphotovideo - Group picture - DROC Track Day March 10 202500002.jpg | 0.0712 | 33 |

## Ranked Quality Details (All Images)

_Each section below is from timed run 1 for that variant._

### gemma4:e4b | yolo=off | prompt=claude+system

| Rank | Image | Final | Tech | Vision | Subject | Light | Color | Emotion | Scroll | Crop | Failed | One line |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 | IG naramedia_official - DSC-5365-NaraMedia.jpeg | 0.8284 | 0.7780 | 51 | 9 | 8 | 8 | 9 | 9 | 8 | no | Strong, low-angle shot with excellent color contrast and dynamic posing, making it highly engaging for a Ducati enthusiast feed. |
| 2 | AB Group - Session 1 (Turn 16 Entry) - CVR_0230_Mar1025_817AM_CaliPhoto.jpg | 0.8092 | 0.7919 | 49 | 8 | 7 | 8 | 9 | 8 | 9 | no | Strong action shot with excellent subject focus and dynamic composition, making it highly engaging for an enthusiast feed. |
| 3 | IG naramedia_official - DSC-5897-NaraMedia.jpeg | 0.7901 | 0.6504 | 51 | 9 | 8 | 8 | 9 | 9 | 8 | no | Excellent dynamic shot with strong motion blur and vibrant subject contrast against the track, perfect for an enthusiast feed. |
| 4 | C Group - Session 2 Back Straight Speed Pans - CVR_2850_Mar1025_946AM_CaliPhoto.jpg | 0.7898 | 0.7661 | 48 | 8 | 8 | 8 | 8 | 8 | 8 | no | Vision scoring summary |
| 5 | IG naramedia_official - DSC-5992-NaraMedia.jpeg | 0.7871 | 0.7571 | 48 | 8 | 8 | 8 | 8 | 8 | 8 | no | Vision scoring summary |
| 6 | IG - renatobo - lineup 1st session - March 10 DROC Track Day-12 HDR.jpeg | 0.7837 | 0.5123 | 54 | 9 | 9 | 9 | 9 | 9 | 9 | no | Vision scoring summary |
| 7 | AB Group - Session 4 (Turn 16) - CVR_6545_Mar1025_1122AM_CaliPhoto.jpg | 0.7775 | 0.7249 | 48 | 8 | 8 | 8 | 8 | 8 | 8 | no | Vision scoring summary |
| 8 | IG cali_carnivores - DSC00013.jpg | 0.7704 | 0.7401 | 47 | 8 | 6 | 7 | 8 | 8 | 8 | no | Strong subject presence and dynamic angle, though the flat background limits overall impact; the Ducati bonus helps elevate the perceived power. |
| 9 | IG luckie.moto - 20250310-075123-luckie.moto.jpg | 0.7512 | 0.6374 | 48 | 8 | 8 | 8 | 8 | 8 | 8 | no | Vision scoring summary |
| 10 | 1-Around the Pits - CVR_0792_Mar1025_832AM_CaliPhoto.jpg | 0.7490 | 0.6688 | 47 | 8 | 7 | 8 | 8 | 8 | 8 | no | Strong, vibrant subject against a muted background, making it highly clickable for a Ducati enthusiast account. |
| 11 | AB Group - Session 1 (Turn 16 Entry) - CVR_0148_Mar1025_815AM_CaliPhoto.jpg | 0.7457 | 0.6190 | 48 | 8 | 6 | 7 | 9 | 8 | 8 | no | Strong dynamic pose and low angle convey power, though the background is slightly distracting for a perfect thumbnail. |
| 12 | AB Group - Session 2 (Turn 11) - CVR_1047_Mar1025_908AM_CaliPhoto.jpg | 0.7391 | 0.5583 | 49 | 8 | 7 | 8 | 9 | 9 | 8 | no | Excellent action shot with strong leading lines and vibrant color contrast, perfect for an enthusiast feed. |
| 13 | IG cali_carnivores - DSC09857.jpg | 0.7376 | 0.5921 | 48 | 8 | 8 | 8 | 8 | 8 | 8 | no | Vision scoring summary |
| 14 | IG kamiumitv - IMG-3981-Bao.jpeg | 0.7335 | 0.6171 | 47 | 8 | 7 | 8 | 8 | 8 | 8 | no | Strong composition with excellent color contrast against the blue sky, making it highly engaging for a motorcycle enthusiast feed. |
| 15 | IG desmo.donna - IMG-1080-Donna.jpeg | 0.7328 | 0.5760 | 48 | 8 | 8 | 8 | 8 | 8 | 8 | no | Vision scoring summary |
| 16 | IG desmo.donna - IMG-1015-Donna.jpeg | 0.7320 | 0.5732 | 48 | 8 | 8 | 8 | 8 | 8 | 8 | no | Vision scoring summary |
| 17 | DSC-5436-NaraMedia.jpeg | 0.7253 | 0.6287 | 46 | 8 | 6 | 7 | 7 | 8 | 8 | no | Strong leading lines and good subject placement make this highly usable, though the midday light is slightly flat. |
| 18 | 1-Around the Pits - 1-Around the Pits - CVR_0025_Mar1025_728AM_CaliPhoto.jpg | 0.7245 | 0.5483 | 48 | 8 | 8 | 8 | 8 | 8 | 8 | no | Vision scoring summary |
| 19 | IG - renatobo - lineup 1st session - March 10 DROC Track Day-02 HDR.jpeg | 0.7072 | 0.4907 | 48 | 9 | 7 | 8 | 8 | 8 | 8 | no | Strong subject focus with good color contrast against the arid background, making it highly clickable for a Ducati enthusiast account. |
| 20 | IG - renatobo - March 10 DROC Track Day-26 HDR.jpeg | 0.7030 | 0.4768 | 48 | 8 | 8 | 8 | 8 | 8 | 8 | no | Vision scoring summary |
| 21 | IG - renatobo - lineup 1st session - March 10 DROC Track Day-01 HDR.jpeg | 0.6729 | 0.3764 | 48 | 8 | 8 | 8 | 8 | 8 | 8 | no | Vision scoring summary |
| 22 | IMG_4984.jpeg | 0.5221 | 0.5421 | 42 | 8 | 6 | 7 | 6 | 6 | 7 | no | The subject is clear, but the indoor, cluttered background detracts from the potential, making it feel more like a product shot than an aspirational lifestyle shot. |
| 23 | IG guardiansvoice - IMG-4129-Delenian.jpeg | 0.5219 | 0.5413 | 42 | 7 | 7 | 7 | 7 | 7 | 7 | no | Vision scoring summary |
| 24 | C Group - Session 3 (Turn 9 and 8) - CVR_4982_Mar1025_1040AM_CaliPhoto.jpg | 0.5115 | 0.4978 | 42 | 7 | 7 | 7 | 7 | 7 | 7 | no | Vision scoring summary |
| 25 | C Group - Session 2 Back Straight Speed Pans - CVR_2628_Mar1025_943AM_CaliPhoto.jpg | 0.3929 | 0.7827 | 36 | 6 | 6 | 6 | 6 | 6 | 6 | no | Vision scoring summary |
| 26 | IG cali_carnivores - DSC09850.jpg | 0.3890 | 0.7612 | 36 | 6 | 6 | 6 | 6 | 6 | 6 | no | Vision scoring summary |
| 27 | 1-Around the Pits - CVR_0098_Mar1025_802AM_CaliPhoto.jpg | 0.3730 | 0.6723 | 36 | 6 | 6 | 6 | 6 | 6 | 6 | no | Vision scoring summary |
| 28 | IG ocredevil - DROC-Track-Day-3-10-2-filippo pagan.jpeg | 0.3296 | 0.4312 | 36 | 6 | 6 | 6 | 6 | 6 | 6 | no | Vision scoring summary |
| 29 | IG renatobo - IMG_5014.jpeg | 0.3198 | 0.3765 | 36 | 6 | 6 | 6 | 6 | 6 | 6 | no | Vision scoring summary |
| 30 | IG renatobo - IMG_5013.jpeg | 0.1836 | 0.5817 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 31 | IG - renatobo - March 10 DROC Track Day-58 HDR.jpeg | 0.1785 | 0.5335 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 32 | IMG_5012.jpeg | 0.1665 | 0.4195 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 33 | caliphotovideo - Group picture - DROC Track Day March 10 202500002.jpg | 0.0712 | 0.6484 | 24 | 4 | 4 | 4 | 4 | 4 | 4 | no | Vision scoring summary |
| 34 | caliphotovideo - Group picture - DROC Track Day March 10 202500001.jpg | 0.0706 | 0.6360 | 24 | 4 | 4 | 4 | 4 | 4 | 4 | no | Vision scoring summary |
| 35 | 2-Group Photo - Raffle - CVR_7677_Mar1025_1249PM_CaliPhoto.jpg | 0.0683 | 0.5853 | 24 | 4 | 4 | 4 | 4 | 4 | 4 | no | Vision scoring summary |
| 36 | IG m92663m - IMG-6019-Mark Momot.jpeg | 0.0670 | 0.6338 | 22 | 4 | 4 | 4 | 4 | 4 | 2 | no | Vision scoring summary |
| 37 | IG guardiansvoice - IMG-4160-Delenian.jpeg | 0.0643 | 0.4960 | 24 | 4 | 4 | 4 | 4 | 4 | 4 | no | Vision scoring summary |
| 38 | 1-Around the Pits - 1-Around the Pits - CVR_0012_Mar1025_702AM_CaliPhoto.jpg | 0.0599 | 0.3975 | 24 | 4 | 4 | 4 | 4 | 4 | 4 | no | Vision scoring summary |
| 39 | IG martangelenos - PXL_20250310_132329430.jpg | 0.0568 | 0.4839 | 20 | 3 | 4 | 4 | 4 | 4 | 1 | no | Vision scoring summary |
| 40 | IG desmo.donna - IMG-1151-Donna.jpeg | 0.0505 | 0.4230 | 18 | 3 | 3 | 3 | 3 | 3 | 3 | no | Vision scoring summary |
| 41 | IG - renatobo - March 10 DROC Track Day-28.jpeg | 0.0428 | 0.4833 | 12 | 2 | 2 | 2 | 2 | 2 | 2 | no | Vision scoring summary |
| 42 | 2-Group Photo - Raffle - CVR_7682_Mar1025_1249PM_CaliPhoto.jpg | 0.0346 | 0.5357 | 6 | 1 | 1 | 1 | 1 | 1 | 1 | no | Vision scoring summary |
