# Tutorial: From Event Photos to Instagram Selects

This tutorial walks you through your first complete pickinsta run. By the end you will have a ranked set of Instagram-ready images, an interactive gallery to browse them, and a clear mental model of the workflow so you can adapt it to your next shoot.

---

## 1. What you will build

After running pickinsta on a folder of event photos, you get:

- **A scored, ranked shortlist** — the top images by combined technical quality and visual appeal.
- **Three output variants per selected image**:
  - `XX_cropped_<name>.jpg` — 1080×1440 smart crop ready to upload to Instagram.
  - `XX_hd_<name>.jpg` — 1920px longest-edge version in the original aspect ratio, good for Stories or sharing at full resolution.
  - `XX_full_<name>.<ext>` — the original source file, untouched, so you always have a lossless copy.
- **An HTML gallery** (`index.html`) in the output folder — open it in any browser to preview crops, check scores, and compare candidates side by side.

---

## 2. Before you start

**Python version.** You need Python 3.10 or newer.

```bash
python3 --version
```

**A virtual environment.** Always run pickinsta in its own venv to keep dependencies isolated.

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

**API key (only if you plan to use the Claude scorer).** The CLIP scorer used in this tutorial is free and runs entirely on your machine. If you want to use Claude later, you will need an Anthropic API key — skip this for now and revisit it in the Next steps section.

---

## 3. Install pickinsta

With your venv active, install pickinsta along with the CLIP and YOLO extras:

```bash
make install-dev
```

Or, if you are not working from the repository directly:

```bash
pip install -e ".[dev,clip,yolo]"
```

The first time you run CLIP, it downloads a model from HuggingFace (about 1.7 GB). This is a one-time download — subsequent runs use the cached model. Make sure you have an internet connection and enough disk space before continuing.

Verify the install:

```bash
pickinsta --help
```

If you see the usage message, you are ready.

---

## 4. Prepare your photos

**Folder structure.** pickinsta expects a flat folder of photos from a single shoot — it does not recurse into subfolders. A typical input folder looks like this:

```
2024-05-event/
  DSC_0001.JPG
  DSC_0002.JPG
  DSC_0003.JPG
  ...
```

**Supported formats.** pickinsta handles JPG, JPEG, PNG, WEBP, HEIC, TIFF, and BMP. If you shoot RAW, export a JPEG or HEIC copy first — most camera apps and tools like Lightroom, Capture One, or the macOS Photos app can do this in one step.

**No preparation required.** You do not need to rename files, sort by rating, or cull the shoot first. That is exactly what pickinsta is for.

---

## 5. Your first run

This command scores all images with the free CLIP scorer and writes the top 10 selects to `./selected`:

```bash
pickinsta ./input --output ./selected --top 10 --scorer clip
```

Replace `./input` with the actual path to your photo folder.

**What happens while it runs:**

1. Images are resized to a working resolution (max 1920px) in a temporary folder next to your input.
2. Near-duplicate and burst images are grouped; only the sharpest from each group advances.
3. Each image is scored on technical quality — sharpness, exposure, composition, color.
4. CLIP scores each image for visual appeal and Instagram potential.
5. A final score combining 30% technical and 70% visual is computed for each image.
6. The top 10 are cropped to 1080×1440, and three output variants are written to `./selected`.
7. An HTML gallery is generated at `./selected/index.html`.

The run typically takes one to several minutes depending on the number of photos, your machine, and whether the CLIP model is already cached.

---

## 6. Review the gallery

Open the gallery in any browser:

```bash
open ./selected/index.html        # macOS
xdg-open ./selected/index.html    # Linux
start ./selected/index.html       # Windows
```

**Gallery layout.** The main page shows all selected images as cards, sorted by final score descending. Each card displays:

- A preview of the cropped version.
- Score bars for technical and vision scores.
- A one-line AI caption summarising why the image scored well.
- A burst badge if the image was the best pick from a burst sequence.

**Detail panel.** Click any card to open a detail panel with:

- Tab switcher: **Cropped** (1080×1440), **HD** (1920px), **Full** (original).
- YOLO detection overlay showing what the model identified as the subject.
- EXIF information (camera, lens, exposure).
- Per-criterion score breakdown.
- An uncertain-crop warning badge if the smart crop was difficult to place.

Use the gallery to decide whether the top 10 is the right cut, or whether you want to expand to more candidates by re-running with a higher `--top` value.

---

## 7. Export your selects

The output folder contains everything you need:

```
selected/
  01_cropped_DSC_0042.jpg    ← upload directly to Instagram
  01_hd_DSC_0042.jpg         ← Stories, sharing, or print
  01_full_DSC_0042.JPG       ← original, lossless
  02_cropped_DSC_0107.jpg
  02_hd_DSC_0107.jpg
  02_full_DSC_0107.JPG
  ...
  index.html
```

The two-digit prefix is the rank. `01_*` is the top-scoring image, `02_*` is second, and so on. The `cropped` variant is the Instagram-ready 1080×1440 portrait. The `hd` and `full` variants give you flexibility for other uses without re-exporting.

---

## 8. Next steps

### Switch to the Claude scorer for higher-quality ranking

CLIP is a good starting point, but Claude has richer understanding of composition, lighting mood, and visual storytelling. Claude scoring costs roughly $0.005 per image.

First, set up your API key:

```bash
cp .env.example .env
# Open .env and add your key:
# ANTHROPIC_API_KEY=sk-ant-...
```

Then run with the Claude scorer:

```bash
pickinsta ./input --output ./selected --scorer claude --top 10
```

pickinsta caches Claude's responses per image, so re-running after adjusting `--top` does not cost anything for already-scored images.

### Tune how many images are selected

Adjust `--top` to control the output size. To see your top 20 instead of 10:

```bash
pickinsta ./input --output ./selected --top 20 --scorer clip
```

### Use dedup-only mode to cull bursts without scoring

If you just want to reduce a large burst-heavy shoot to one best shot per sequence — without any scoring or ranking — use `--dedup-only`. This outputs one image per burst group as full, HD, and cropped variants, covering the entire shoot:

```bash
pickinsta ./input --output ./deduped --dedup-only
```

This is useful as a first pass before committing to a full scored run, or when you want a complete deduplicated archive rather than a ranked top-N shortlist.

### Use a separate work folder

By default pickinsta writes intermediate resized images next to your input folder. On large shoots you may prefer to keep things tidy with an explicit work folder:

```bash
pickinsta ./input --output ./selected --work ./work --scorer clip --top 10
```

### Score all images instead of a fraction

By default pickinsta sends only the top 50% of technically-filtered images to the vision scorer. To score everything:

```bash
pickinsta ./input --output ./selected --scorer claude --all
```

This is more thorough but costs more when using Claude.
