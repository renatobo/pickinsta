# Model Benchmark Report (Speed + Quality)

- Generated: `2026-04-09T10:24:38`
- Input folder: `/Users/renatobo/development/pickinsta/input`
- Candidates scored per run: `42`
- Runs per variant: `1`
- Warmup enabled: `False` (1 image + 10s wait)
- Ollama base URL: `http://localhost:11434`
- Ollama concurrency: `2`
- Ollama max retries: `2`
- Ollama keep_alive: `15m`

## Speed Summary

| Variant | Scorer | Model | YOLO | Avg sec/img | Avg imgs/min | Avg duration (s) | Avg failures/run | SDI | Unique tuples | Speed vs fastest |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| claude-sonnet-4-6 \| scorer=claude | claude | claude-sonnet-4-6 | off | 0.83 | 72.12 | 34.94 | 0.00 | 1.05 | 39 | 1.00x |
| gemma4:e4b \| yolo=off | ollama | gemma4:e4b | off | 43.92 | 1.37 | 1844.63 | 0.00 | 0.00 | 6 | 52.79x |

## Per-run Timing

| Variant | Run | Duration (s) | Sec/img | Imgs/min | Failures |
|---|---:|---:|---:|---:|---:|
| gemma4:e4b \| yolo=off | 1 | 1844.63 | 43.92 | 1.37 | 0 |
| claude-sonnet-4-6 \| scorer=claude | 1 | 34.94 | 0.83 | 72.12 | 0 |

## Image-by-Image Score Comparison (Run 1)

| Image | gemma4:e4b \| yolo=off final | gemma4:e4b \| yolo=off rank | claude-sonnet-4-6 \| scorer=claude final | claude-sonnet-4-6 \| scorer=claude rank |
|---|---:|---:|---:|---:|
| 1-Around the Pits - 1-Around the Pits - CVR_0012_Mar1025_702AM_CaliPhoto.jpg | 0.1642 | 39 | 0.0564 | 41 |
| 1-Around the Pits - 1-Around the Pits - CVR_0025_Mar1025_728AM_CaliPhoto.jpg | 0.5236 | 5 | 0.3997 | 18 |
| 1-Around the Pits - CVR_0098_Mar1025_802AM_CaliPhoto.jpg | 0.1931 | 21 | 0.5627 | 11 |
| 1-Around the Pits - CVR_0792_Mar1025_832AM_CaliPhoto.jpg | 0.1927 | 22 | 0.6179 | 7 |
| 2-Group Photo - Raffle - CVR_7677_Mar1025_1249PM_CaliPhoto.jpg | 0.1840 | 28 | 0.3224 | 25 |
| 2-Group Photo - Raffle - CVR_7682_Mar1025_1249PM_CaliPhoto.jpg | 0.1787 | 31 | 0.1542 | 35 |
| AB Group - Session 1 (Turn 16 Entry) - CVR_0148_Mar1025_815AM_CaliPhoto.jpg | 0.2145 | 13 | 0.6677 | 3 |
| AB Group - Session 1 (Turn 16 Entry) - CVR_0230_Mar1025_817AM_CaliPhoto.jpg | 0.1840 | 27 | 0.5793 | 10 |
| AB Group - Session 2 (Turn 11) - CVR_1047_Mar1025_908AM_CaliPhoto.jpg | 0.1811 | 30 | 0.6193 | 6 |
| AB Group - Session 4 (Turn 16) - CVR_6545_Mar1025_1122AM_CaliPhoto.jpg | 0.1986 | 20 | 0.6126 | 8 |
| C Group - Session 2 Back Straight Speed Pans - CVR_2628_Mar1025_943AM_CaliPhoto.jpg | 0.2047 | 14 | 0.4419 | 15 |
| C Group - Session 2 Back Straight Speed Pans - CVR_2850_Mar1025_946AM_CaliPhoto.jpg | 0.2029 | 16 | 0.4669 | 13 |
| C Group - Session 3 (Turn 9 and 8) - CVR_4982_Mar1025_1040AM_CaliPhoto.jpg | 0.5115 | 7 | 0.0784 | 36 |
| DSC-5436-NaraMedia.jpeg | 0.5429 | 4 | 0.4142 | 17 |
| IG - renatobo - March 10 DROC Track Day-26 HDR.jpeg | 0.3378 | 12 | 0.3098 | 26 |
| IG - renatobo - March 10 DROC Track Day-28.jpeg | 0.3390 | 11 | 0.1855 | 33 |
| IG - renatobo - March 10 DROC Track Day-58 HDR.jpeg | 0.1785 | 32 | 0.0608 | 40 |
| IG - renatobo - lineup 1st session - March 10 DROC Track Day-01 HDR.jpeg | 0.1620 | 41 | 0.1865 | 32 |
| IG - renatobo - lineup 1st session - March 10 DROC Track Day-02 HDR.jpeg | 0.1740 | 35 | 0.5004 | 12 |
| IG - renatobo - lineup 1st session - March 10 DROC Track Day-12 HDR.jpeg | 0.1763 | 33 | 0.3862 | 20 |
| IG cali_carnivores - DSC00013.jpg | 0.2002 | 19 | 0.6350 | 5 |
| IG cali_carnivores - DSC09850.jpg | 0.2024 | 17 | 0.2392 | 29 |
| IG cali_carnivores - DSC09857.jpg | 0.1847 | 26 | 0.4216 | 16 |
| IG desmo.donna - IMG-1015-Donna.jpeg | 0.1827 | 29 | 0.2194 | 31 |
| IG desmo.donna - IMG-1080-Donna.jpeg | 0.8028 | 1 | 0.3837 | 21 |
| IG desmo.donna - IMG-1151-Donna.jpeg | 0.1669 | 37 | 0.2791 | 28 |
| IG guardiansvoice - IMG-4129-Delenian.jpeg | 0.7224 | 3 | 0.2324 | 30 |
| IG guardiansvoice - IMG-4160-Delenian.jpeg | 0.1746 | 34 | 0.2993 | 27 |
| IG kamiumitv - IMG-3981-Bao.jpeg | 0.7451 | 2 | 0.5961 | 9 |
| IG luckie.moto - 20250310-075123-luckie.moto.jpg | 0.1894 | 24 | 0.3737 | 22 |
| IG m92663m - IMG-6019-Mark Momot.jpeg | 0.1891 | 25 | 0.1646 | 34 |
| IG martangelenos - PXL_20250310_132329430.jpg | 0.0638 | 42 | 0.0725 | 38 |
| IG naramedia_official - DSC-5365-NaraMedia.jpeg | 0.2042 | 15 | 0.6720 | 2 |
| IG naramedia_official - DSC-5897-NaraMedia.jpeg | 0.1908 | 23 | 0.6508 | 4 |
| IG naramedia_official - DSC-5992-NaraMedia.jpeg | 0.2020 | 18 | 0.4513 | 14 |
| IG ocredevil - DROC-Track-Day-3-10-2-filippo pagan.jpeg | 0.1678 | 36 | 0.0527 | 42 |
| IG renatobo - IMG_5013.jpeg | 0.3567 | 10 | 0.3917 | 19 |
| IG renatobo - IMG_5014.jpeg | 0.1620 | 40 | 0.3548 | 23 |
| IMG_4984.jpeg | 0.5221 | 6 | 0.6760 | 1 |
| IMG_5012.jpeg | 0.1665 | 38 | 0.3415 | 24 |
| caliphotovideo - Group picture - DROC Track Day March 10 202500001.jpg | 0.3665 | 9 | 0.0689 | 39 |
| caliphotovideo - Group picture - DROC Track Day March 10 202500002.jpg | 0.3687 | 8 | 0.0782 | 37 |

## Ranked Quality Details (All Images)

_Each section below is from timed run 1 for that variant._

### gemma4:e4b | yolo=off

| Rank | Image | Final | Tech | Vision | Subject | Light | Color | Emotion | Scroll | Crop | Failed | One line |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 | IG desmo.donna - IMG-1080-Donna.jpeg | 0.8028 | 0.5760 | 54 | 9 | 9 | 9 | 9 | 9 | 9 | no | Vision scoring summary |
| 2 | IG kamiumitv - IMG-3981-Bao.jpeg | 0.7451 | 0.6171 | 48 | 8 | 8 | 8 | 8 | 8 | 8 | no | Vision scoring summary |
| 3 | IG guardiansvoice - IMG-4129-Delenian.jpeg | 0.7224 | 0.5413 | 48 | 8 | 8 | 8 | 8 | 8 | 8 | no | Vision scoring summary |
| 4 | DSC-5436-NaraMedia.jpeg | 0.5429 | 0.6287 | 42 | 7 | 7 | 7 | 7 | 7 | 7 | no | Vision scoring summary |
| 5 | 1-Around the Pits - 1-Around the Pits - CVR_0025_Mar1025_728AM_CaliPhoto.jpg | 0.5236 | 0.5483 | 42 | 7 | 7 | 7 | 7 | 7 | 7 | no | Vision scoring summary |
| 6 | IMG_4984.jpeg | 0.5221 | 0.5421 | 42 | 7 | 7 | 7 | 7 | 7 | 7 | no | Vision scoring summary |
| 7 | C Group - Session 3 (Turn 9 and 8) - CVR_4982_Mar1025_1040AM_CaliPhoto.jpg | 0.5115 | 0.4978 | 42 | 7 | 7 | 7 | 7 | 7 | 7 | no | Vision scoring summary |
| 8 | caliphotovideo - Group picture - DROC Track Day March 10 202500002.jpg | 0.3687 | 0.6484 | 36 | 6 | 6 | 6 | 6 | 6 | 6 | no | Vision scoring summary |
| 9 | caliphotovideo - Group picture - DROC Track Day March 10 202500001.jpg | 0.3665 | 0.6360 | 36 | 6 | 6 | 6 | 6 | 6 | 6 | no | Vision scoring summary |
| 10 | IG renatobo - IMG_5013.jpeg | 0.3567 | 0.5817 | 36 | 6 | 6 | 6 | 6 | 6 | 6 | no | Vision scoring summary |
| 11 | IG - renatobo - March 10 DROC Track Day-28.jpeg | 0.3390 | 0.4833 | 36 | 6 | 6 | 6 | 6 | 6 | 6 | no | Vision scoring summary |
| 12 | IG - renatobo - March 10 DROC Track Day-26 HDR.jpeg | 0.3378 | 0.4768 | 36 | 6 | 6 | 6 | 6 | 6 | 6 | no | Vision scoring summary |
| 13 | AB Group - Session 1 (Turn 16 Entry) - CVR_0148_Mar1025_815AM_CaliPhoto.jpg | 0.2145 | 0.8766 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 14 | C Group - Session 2 Back Straight Speed Pans - CVR_2628_Mar1025_943AM_CaliPhoto.jpg | 0.2047 | 0.7827 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 15 | IG naramedia_official - DSC-5365-NaraMedia.jpeg | 0.2042 | 0.7780 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 16 | C Group - Session 2 Back Straight Speed Pans - CVR_2850_Mar1025_946AM_CaliPhoto.jpg | 0.2029 | 0.7661 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 17 | IG cali_carnivores - DSC09850.jpg | 0.2024 | 0.7612 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 18 | IG naramedia_official - DSC-5992-NaraMedia.jpeg | 0.2020 | 0.7571 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 19 | IG cali_carnivores - DSC00013.jpg | 0.2002 | 0.7401 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 20 | AB Group - Session 4 (Turn 16) - CVR_6545_Mar1025_1122AM_CaliPhoto.jpg | 0.1986 | 0.7249 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 21 | 1-Around the Pits - CVR_0098_Mar1025_802AM_CaliPhoto.jpg | 0.1931 | 0.6723 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 22 | 1-Around the Pits - CVR_0792_Mar1025_832AM_CaliPhoto.jpg | 0.1927 | 0.6688 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 23 | IG naramedia_official - DSC-5897-NaraMedia.jpeg | 0.1908 | 0.6504 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 24 | IG luckie.moto - 20250310-075123-luckie.moto.jpg | 0.1894 | 0.6374 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 25 | IG m92663m - IMG-6019-Mark Momot.jpeg | 0.1891 | 0.6338 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 26 | IG cali_carnivores - DSC09857.jpg | 0.1847 | 0.5921 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 27 | AB Group - Session 1 (Turn 16 Entry) - CVR_0230_Mar1025_817AM_CaliPhoto.jpg | 0.1840 | 0.5860 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 28 | 2-Group Photo - Raffle - CVR_7677_Mar1025_1249PM_CaliPhoto.jpg | 0.1840 | 0.5853 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 29 | IG desmo.donna - IMG-1015-Donna.jpeg | 0.1827 | 0.5732 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 30 | AB Group - Session 2 (Turn 11) - CVR_1047_Mar1025_908AM_CaliPhoto.jpg | 0.1811 | 0.5583 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 31 | 2-Group Photo - Raffle - CVR_7682_Mar1025_1249PM_CaliPhoto.jpg | 0.1787 | 0.5357 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 32 | IG - renatobo - March 10 DROC Track Day-58 HDR.jpeg | 0.1785 | 0.5335 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 33 | IG - renatobo - lineup 1st session - March 10 DROC Track Day-12 HDR.jpeg | 0.1763 | 0.5123 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 34 | IG guardiansvoice - IMG-4160-Delenian.jpeg | 0.1746 | 0.4960 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 35 | IG - renatobo - lineup 1st session - March 10 DROC Track Day-02 HDR.jpeg | 0.1740 | 0.4907 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 36 | IG ocredevil - DROC-Track-Day-3-10-2-filippo pagan.jpeg | 0.1678 | 0.4312 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 37 | IG desmo.donna - IMG-1151-Donna.jpeg | 0.1669 | 0.4230 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 38 | IMG_5012.jpeg | 0.1665 | 0.4195 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 39 | 1-Around the Pits - 1-Around the Pits - CVR_0012_Mar1025_702AM_CaliPhoto.jpg | 0.1642 | 0.3975 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 40 | IG renatobo - IMG_5014.jpeg | 0.1620 | 0.3765 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 41 | IG - renatobo - lineup 1st session - March 10 DROC Track Day-01 HDR.jpeg | 0.1620 | 0.3764 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Vision scoring summary |
| 42 | IG martangelenos - PXL_20250310_132329430.jpg | 0.0638 | 0.4839 | 24 | 4 | 4 | 4 | 4 | 4 | 4 | no | Vision scoring summary |

### claude-sonnet-4-6 | scorer=claude

| Rank | Image | Final | Tech | Vision | Subject | Light | Color | Emotion | Scroll | Crop | Failed | One line |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 | IMG_4984.jpeg | 0.6760 | 0.5421 | 44 | 9 | 6 | 7 | 7 | 7 | 8 | no | Strong front-facing Ducati Panigale V4R identity with tire warmer detail adds authenticity, but flat garage lighting and static presentation limit the aspirational energy needed for a standout cover. |
| 2 | IG naramedia_official - DSC-5365-NaraMedia.jpeg | 0.6720 | 0.7780 | 52 | 10 | 8 | 9 | 9 | 9 | 7 | no | Ducati 848 Corse in iconic red-white-black tricolor dominates the frame with a dramatic low-angle 3/4 front shot against clean desert sky — powerful brand identity but centered placement and wide format hurt portrait crop potential |
| 3 | AB Group - Session 1 (Turn 16 Entry) - CVR_0148_Mar1025_815AM_CaliPhoto.jpg | 0.6677 | 0.8766 | 49 | 9 | 7 | 9 | 9 | 8 | 7 | no | Red Ducati Panigale carving hard against dramatic desert mountains delivers strong brand energy, but slightly centered framing and midday flat light limit scroll-stop ceiling |
| 4 | IG naramedia_official - DSC-5897-NaraMedia.jpeg | 0.6508 | 0.6504 | 53 | 10 | 8 | 9 | 10 | 9 | 7 | no | A aggressive corner-carving Ducati Panigale V4 in full Corse livery with dramatic lean angle and motion blur creates an instinctive scroll-stop, but the centered-landscape composition limits portrait crop flexibility. |
| 5 | IG cali_carnivores - DSC00013.jpg | 0.6350 | 0.7401 | 49 | 10 | 7 | 9 | 8 | 8 | 7 | no | Bold red Panigale V4 pops hard against the pale sky with strong front-quarter presence, but flat midday light and near-centered framing limit drama and portrait-crop flexibility. |
| 6 | AB Group - Session 2 (Turn 11) - CVR_1047_Mar1025_908AM_CaliPhoto.jpg | 0.6193 | 0.5583 | 52 | 10 | 7 | 9 | 10 | 9 | 7 | no | Aggressive cornering angle on a Ducati Panigale V4 in signature Ducati red delivers raw speed and emotion, but flat midday light and centered composition slightly limit crop flexibility and depth drama. |
| 7 | 1-Around the Pits - CVR_0792_Mar1025_832AM_CaliPhoto.jpg | 0.6179 | 0.6688 | 49 | 10 | 7 | 9 | 8 | 8 | 7 | no | Red Ducati Panigale V4 pops hard against the desert mountain backdrop but flat midday light and dead-center framing flatten the drama that a low angle or golden-hour shot would deliver |
| 8 | AB Group - Session 4 (Turn 16) - CVR_6545_Mar1025_1122AM_CaliPhoto.jpg | 0.6126 | 0.7249 | 47 | 9 | 6 | 8 | 9 | 8 | 7 | no | Aggressive lean angle on a Ducati Hypermotard in full Dainese livery creates visceral tension, but flat midday desert light robs the chrome and bodywork of the drama this action deserves. |
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
