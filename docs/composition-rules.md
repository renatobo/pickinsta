# Motorcycle photography composition rules for automated cropping and scoring

> Implementation note: this document is a broad reference. The current
> `pickinsta` pipeline outputs portrait crops at **1080 x 1440 (3:4)**.

**An automated system for cropping and evaluating motorcycle photographs needs three core capabilities: detecting subject placement against compositional grids, scoring image quality across seven measurable dimensions, and intelligently reframing source images for Instagram's format zoo.** This specification translates professional motorcycle photography principles into quantifiable rules a developer can implement using YOLO-based segmentation and a multimodal LLM. The rules below are drawn from BikeEXIF's professional guides, Adventure Bike Rider's rule-of-thirds tutorial, and PhotographyCourses.biz, cross-referenced with general photographic composition theory and Instagram's current platform requirements.

---

## Compositional grid systems and where the motorcycle belongs

Two overlapping grid systems govern subject placement in motorcycle photography: the **Rule of Thirds** and the **Phi Grid** (golden ratio). Both should be evaluated, with the closer match scoring higher.

**Rule of Thirds grid (normalized 0–1 coordinates):** Vertical lines at x = 0.333 and x = 0.667; horizontal lines at y = 0.333 and y = 0.667. This produces four "power points" at their intersections: (0.333, 0.333), (0.667, 0.333), (0.333, 0.667), (0.667, 0.667). The motorcycle's center of visual mass should fall within **±0.05 of a power point** or ±0.05 of a grid line to be scored as "on thirds." Adventure Bike Rider's guide is explicit: "By placing your photo's key elements at the intersections of the grid lines, your photo becomes better balanced." Their example places the motorcycle at the left-third intersection with the rider's head at the upper-right intersection, layering foreground hills in the lower third, mountains in the middle, and sky in the upper third.

**Phi Grid (golden ratio):** Lines at x, y ∈ {0.382, 0.618}, shifted ~5% inward from thirds lines. Four intersection points: (0.382, 0.382), (0.618, 0.382), (0.382, 0.618), (0.618, 0.618). The Phi Grid works especially well for scenic motorcycle-in-landscape compositions where the bike sits against an expansive environment. The **Fibonacci Spiral** variant is useful when curving roads follow the spiral's arc, with the motorcycle positioned at the spiral's tightest point.

**Dead-center placement (subject at 0.5, 0.5) is penalized** except for one case: perfectly symmetrical head-on motorcycle shots where bilateral symmetry is the compositional intent. BikeEXIF and Adventure Bike Rider both warn against centering: it creates static, "snapshot" compositions and, practically, loses the subject in magazine gutters or Instagram grid crops.

**Horizon line placement:** The horizon should align with y = 0.333 or y = 0.667, never y = 0.5. BikeEXIF is emphatic that the horizon must be level (deviation < 2°) unless a deliberate Dutch angle is intended: "Slightly crooked photos mess with people on a subconscious level."

---

## Subject size, lead room, and negative space by shot type

The percentage of frame occupied by the motorcycle determines both shot classification and required compositional treatment. These ranges are derived from professional motorcycle photography benchmarks:

| Shot type | Subject area (% of frame) | Use case | Lead room required |
|---|---|---|---|
| Extreme close-up | 80–100% | Engine detail, badges, controls | None |
| Close-up | 50–80% | Headlight, tank, exhaust feature | Minimal |
| Medium shot | 20–50% | Full motorcycle with some context | Moderate (55–65% ahead) |
| Environmental | 10–25% | Motorcycle on a road or in setting | Standard (60–70% ahead) |
| Scenic/wide | 5–15% | Motorcycle dwarfed by landscape | Maximum (65–75% ahead) |
| Extreme wide | 1–5% | Establishing shot, vast scenery | Direction-flexible |

**Lead room (facing space) is the single most important directional rule.** When a motorcycle faces or moves in a direction, **60–70% of horizontal frame space should be in front of it**, with 30–40% behind. Algorithmically: if the motorcycle faces right, place its center at x ≈ 0.30–0.40; if left, at x ≈ 0.60–0.70. For head-on approaching shots, centered placement (x ≈ 0.45–0.55) is acceptable, and lead room shifts to the vertical axis (road extending below). **If space behind the subject exceeds space in front, the composition scores a significant penalty** — the viewer perceives the motorcycle as "running into" the frame edge.

**Negative space** should be purposeful, not accidental. PhotographyCourses.biz describes effective negative space as placing "a colourful, shiny bike amongst grey workday surroundings — the motorcycle stands out and has an impact even though it's occupying a small area of the image." For negative-space-driven compositions, the subject should occupy as little as **5–15% of frame** with ≥85% clean, low-detail surroundings. For standard compositions, **30–60% subject area with 40–70% negative space** is the target range. BikeEXIF warns against tight cropping: "Don't be tempted to crop the shot too tightly around the bike. Leave ample space."

---

## Motorcycle-specific angles, backgrounds, and what to avoid

Professional motorcycle photographers converge on several domain-specific rules that general composition theory doesn't cover.

**Camera height is the single biggest differentiator** between amateur and professional motorcycle photos. BikeEXIF's José Gallina states: "Lower your eyes and camera to the level of the tank or headlight. It's the one trick that makes any bike look much better." Three height zones apply: **tank/headlight level** (the default for natural-looking shots), **axle height** (aggressive, menacing), and **ground level** (maximum power and dominance). PhotographyCourses.biz confirms that low angles "make the motorcycle look powerful and dominant" and can "help you tidy up a slightly messy background." For an automated system evaluating existing photos, images shot from standing eye-height (~1.6m) with the camera angled downward toward the bike should be scored lower than those at or below tank level.

**The ¾ view** (showing the front corner and one full side) is the most flattering standard angle. BikeEXIF notes it produces "a compressed perspective effect when shooting the bike at ¾ angles, which flatters most bikes." Head-on and direct side profiles work for specific purposes but are less versatile.

**Background rules are strict.** BikeEXIF recommends backgrounds that "contrast slightly in color with the bike" — industrial garage doors, brick walls that aren't too busy, or open fields with clean sky. Three specific violations are called out:

- **Competing lines:** "Lines all over the place, distracting and interfering with the lines of the bike" — rod iron fences, complex architecture, cluttered signage
- **Merging objects:** "Be wary of telephone poles and trees, which may appear to be growing out from the bike"
- **Color clashes:** Background colors that match or fight the bike's paint scheme (e.g., orange garage doors behind an orange-paneled bike)

**Focal length matters for automated quality assessment.** BikeEXIF specifies a **minimum of 50mm equivalent (full-frame)** to avoid barrel distortion that "makes wheels look slightly 'out of round' when shot from side-on." Images showing visible wide-angle distortion (converging verticals, bulging proportions) should receive a quality penalty. An aperture around **f/4** is the professional sweet spot — enough depth of field to keep the entire bike sharp while softening the background.

---

## Instagram format specifications and safe zones

Instagram supports aspect ratios from **1.91:1** (widest landscape) to **4:5** (tallest portrait) in feed posts, with **9:16** for Stories and Reels. The automated cropping system must handle all five primary formats from a single source image.

| Format | Aspect ratio | Pixel dimensions | Subject placement priority |
|---|---|---|---|
| Feed portrait | 4:5 | 1080 × 1350 | Upper-third power points; bike in lower-center for low-angle shots |
| Feed square | 1:1 | 1080 × 1080 | Center-weighted; symmetrical compositions work best |
| Feed landscape | 16:9 | 1080 × 608 | Left or right third with lead room; horizon on thirds line |
| Stories/Reels | 9:16 | 1080 × 1920 | Subject within central safe zone (see below) |
| New grid format | 3:4 | 1080 × 1440 | Center-biased; grid thumbnail crops 4:5 posts to this ratio |

**Safe zones are critical for Stories and Reels.** Instagram's UI overlays consume significant screen real estate:

For **Stories (1080 × 1920):** The top **250 px** (13%) is occupied by the profile icon, username, and close button. The bottom **250 px** (13%) holds the reply bar and send buttons. Side margins consume **65 px** (6%) per side. The effective safe content area is approximately **950 × 1420 px** centered in the frame — about **74% of vertical space and 88% of horizontal space**.

For **Reels (1080 × 1920):** The bottom buffer is larger at **320 px** (16.7%) due to caption text, hashtags, and music info. The right side loses **120 px** (11.1%) to the like/comment/share button stack. The top loses **108 px** (5.6%). The effective safe area is approximately **900 × 1492 px** — about **72% of total frame area**. The motorcycle's primary visual mass must sit entirely within these safe zones.

**Grid preview consideration:** Instagram's profile grid now displays thumbnails in **3:4** ratio. For 4:5 feed posts, the grid crops to the center **1012 × 1350 px**. The cropping algorithm should ensure the motorcycle remains well-composed in both the full 4:5 view and the 3:4 grid preview.

**Cropping from a master image:** A **3:2 landscape** source (standard DSLR output, e.g., 6000 × 4000 px) provides adequate crop flexibility for all formats except 9:16, which requires removing ~63% of the horizontal content. The system should compose for the primary target format (usually 4:5) while checking that the subject remains well-positioned after reframing to every other ratio. For 9:16 crops from landscape sources, a letterbox/background treatment is often preferable to aggressive cropping that cuts the motorcycle.

---

## Seven-metric scoring rubric for automated evaluation

Each image should be scored on seven independently measurable dimensions, weighted by their importance to motorcycle photography specifically. The composite score uses a 1–10 scale.

**Metric 1 — Sharpness (weight: 0.18).** Compute Laplacian variance on the segmented motorcycle region. A ratio above 500 (on a 1080px-wide image) indicates tack-sharp resolution; below 50 indicates unacceptable blur. Only the subject region matters — background softness is desirable, not penalized. The subject-region Laplacian variance divided by background-region variance yields the **background separation score** (Metric 2).

**Metric 2 — Background separation (weight: 0.12).** The foreground-to-background sharpness ratio should exceed **3:1** for strong separation and **5:1** for excellent isolation. This is computed as subject Laplacian variance divided by background Laplacian variance. Motorcycle photography specifically benefits from separation because bikes have complex silhouettes with many small components (mirrors, levers, spokes) that disappear against busy backgrounds.

**Metric 3 — Composition adherence (weight: 0.20).** The highest-weighted metric. Compute the distance from the motorcycle's detected center-of-mass to the nearest Rule of Thirds or Phi Grid power point. Normalize by image diagonal. Score = 1.0 − (distance / (diagonal × 0.33)). Additionally check: horizon tilt (penalize > 2° deviation), lead room ratio (penalize if space behind subject exceeds space ahead), and left-right visual weight balance (acceptable range: 40:60 to 60:40).

**Metric 4 — Lighting quality (weight: 0.18).** Analyze the luminance histogram. Well-exposed motorcycle photos have < 2% of pixels clipped at either extreme (pure black or pure white), a mean luminance between 90–170 (8-bit), and the subject region should be at least as bright as the image mean. Chrome and metallic surfaces on motorcycles create specular highlights that can blow out — the system should flag highlight clipping specifically on detected reflective surfaces. BikeEXIF notes that golden hour light (low, even sun) is ideal and that "the top of the tank and polished metal parts won't be too bright" during these conditions.

**Metric 5 — Color harmony (weight: 0.13).** Map dominant colors to a hue wheel and evaluate alignment with Itten's harmony templates (complementary, analogous, triadic, split-complementary). Compute the Hasler-Süsstrunk colorfulness metric: C = √(σ²_rg + σ²_yb) + 0.3 × √(μ²_rg + μ²_yb). Moderate, consistent saturation scores highest. Critically, evaluate **subject-background color contrast** — BikeEXIF explicitly requires backgrounds that "contrast slightly in color with the bike."

**Metric 6 — Visual clutter (weight: 0.12).** Compute edge density (Canny edges per pixel) in the background region. Fewer edges = cleaner background. Count detected objects using YOLO — fewer distinct objects generally means cleaner composition for motorcycle photography. Compute a **saliency focus ratio**: what fraction of total image saliency falls within the motorcycle's segmentation mask. This ratio should exceed **0.60** for good compositions and **0.75** for excellent ones. PhotographyCourses.biz warns: "You don't want the bike to be lost against [the background] or competing with it for the viewer's attention."

**Metric 7 — Overall aesthetic (weight: 0.07).** Run the image through a NIMA model (MobileNet backbone, trained on the AVA dataset). NIMA outputs a 1–10 distribution; use the mean. Scores above 6.0 place the image in the top ~20% of rated photographs. This serves as a holistic sanity check that captures qualities the other six metrics might miss.

**Composite formula:**

```
TOTAL = 0.18 × Sharpness + 0.12 × BackgroundSeparation + 0.20 × Composition +
        0.18 × Lighting + 0.13 × ColorHarmony + 0.12 × VisualClutter + 0.07 × NIMA
```

Quality tiers: **8.0–10.0** (excellent, feature-worthy), **6.5–7.9** (good, post with minor adjustments), **5.0–6.4** (mediocre, needs re-cropping or editing), **3.0–4.9** (poor, significant issues), **1.0–2.9** (reject).

---

## Automated cropping algorithm logic

The cropping system should follow this decision sequence for each target aspect ratio:

**Step 1 — Detect and segment.** Run YOLO to identify the motorcycle bounding box and segmentation mask. Determine the motorcycle's facing direction (left or right) from the mask's orientation or from handlebar/wheel geometry. Compute the bike's center-of-mass and area percentage.

**Step 2 — Classify shot type.** Based on the motorcycle's area relative to the source image, classify as close-up (>50%), medium (20–50%), environmental (10–25%), or scenic (<10%). Each type has different target placement rules.

**Step 3 — Determine ideal placement.** For the target aspect ratio, compute the ideal crop window that places the motorcycle's center-of-mass nearest to the optimal power point while respecting lead room. For rightward-facing bikes: target x ≈ 0.33; for leftward: x ≈ 0.67. Vertical placement should put the bike's visual center at y ≈ 0.60–0.67 (lower-third area) for low-angle compositions, or y ≈ 0.50 for eye-level shots.

**Step 4 — Validate constraints.** Ensure the crop window keeps the entire motorcycle within frame (no clipping of wheels, handlebars, or exhaust). Verify that lead room ratio falls within 55–75% ahead of the bike. Check that the horizon (if detected) aligns with a thirds line and is level within 2°. For Stories/Reels crops, verify the motorcycle falls entirely within the safe content zone.

**Step 5 — Score and select.** If multiple valid crop positions exist, score each against the seven-metric rubric and select the highest-scoring option. Prioritize composition adherence and lead room over perfect thirds alignment — a well-balanced crop with the bike at x = 0.35 is better than a forced placement at exactly x = 0.333 that clips a wheel.

**Format-specific cropping notes:** When cropping a landscape 3:2 source to 4:5 portrait, the system removes ~33% of horizontal content. The crop should center on the motorcycle while biasing toward the side with more interesting background content (determined by saliency analysis of the non-subject region). When cropping to 1:1 square, the system extracts the most compositionally balanced square region containing the full motorcycle. For 9:16 from landscape sources, if the horizontal crop would clip the motorcycle, the system should flag the image as unsuitable for that format rather than producing a bad crop.

---

## Conclusion: from rules to implementation

The specification above provides **43 quantifiable parameters** across composition grids, subject sizing, lead room ratios, safe zones, and quality metrics — enough to build a functional automated system. Three implementation priorities stand out. First, **lead room detection** (determining which direction the motorcycle faces and ensuring adequate space ahead) is the single highest-impact rule because violations are immediately obvious and universally penalized. Second, **the composition score should weight Rule of Thirds and Phi Grid equally** and take the better of the two scores, since professional photographers use both frameworks interchangeably and the grids differ by only ~5%. Third, the system should treat these rules as **soft constraints with diminishing penalties** rather than binary pass/fail checks — a motorcycle at x = 0.38 is not meaningfully worse than one at x = 0.333, and the scoring function should reflect that through smooth gradients rather than sharp thresholds. The goal is not to enforce rigid rules but to approximate the judgment of a professional motorcycle photographer who has internalized these principles through years of practice.

---

## References and source mapping

This section lists the source references cited throughout this document and
maps them to rule groups.

### Primary sources cited

- **BikeEXIF** (professional motorcycle photography guides)
- **Adventure Bike Rider** (rule-of-thirds tutorial)
- **PhotographyCourses.biz** (composition, angle, and negative-space guidance)
- **Instagram platform requirements** (aspect ratios, grid behavior, safe zones)
- **General photographic composition theory** (cross-reference basis)

### Additional technical references cited

- **Johannes Itten color harmony templates** (complementary, analogous, triadic, split-complementary)
- **NIMA** (Neural Image Assessment; MobileNet backbone)
- **AVA dataset** (used for NIMA training)
- **YOLO** (subject detection/segmentation for crop logic)

### Rule-to-source mapping

| Rule area | Sources cited in this document |
|---|---|
| Rule of Thirds placement and grid intersections | Adventure Bike Rider, general composition theory |
| Phi Grid / Fibonacci placement | General composition theory |
| Dead-center penalty and level horizon | BikeEXIF, Adventure Bike Rider |
| Lead room and shot-type framing ranges | Professional motorcycle photography benchmarks, BikeEXIF |
| Negative space usage and anti-tight-crop guidance | PhotographyCourses.biz, BikeEXIF |
| Camera height, 3/4 angle, background selection | BikeEXIF, PhotographyCourses.biz |
| Focal length and distortion cautions | BikeEXIF |
| Instagram aspect ratios, Stories/Reels safe zones, grid preview | Instagram platform requirements |
| Seven-metric scoring rubric details | BikeEXIF, PhotographyCourses.biz, general composition theory, Itten templates, NIMA/AVA |
| Automated crop logic with detection/constraints | YOLO + composition rules synthesis in this document |
