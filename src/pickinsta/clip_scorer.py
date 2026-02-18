"""CLIP-based vision scoring helpers."""

from pathlib import Path

from PIL import Image


def load_clip_model():
    """Load CLIP model (cached by transformers/hub after first call)."""
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    return model, processor


def _clip_setup_hint(error: Exception) -> str:
    """Return a practical setup hint for common CLIP initialization failures."""
    text = str(error).lower()
    if "no module named" in text or "import" in text:
        return (
            "Install CLIP dependencies in your active environment:\n"
            "  python -m pip install -e '.[clip]'"
        )
    if "certificate" in text or "https" in text or "connection" in text or "offline" in text:
        return (
            "CLIP model download failed (network/TLS). Ensure internet access for the first run,\n"
            "or use --scorer claude."
        )
    return (
        "CLIP initialization failed. Ensure compatible versions of torch/transformers are installed.\n"
        "Run: python -c \"import transformers, torch; print(transformers.__version__, torch.__version__)\".\n"
        "Only if that fails, use Python 3.10-3.12."
    )


def score_with_clip(image_path: Path, model=None, processor=None) -> dict:
    """Score a single image using CLIP zero-shot classification."""
    import torch

    if model is None or processor is None:
        model, processor = load_clip_model()

    image = Image.open(image_path).convert("RGB")

    # Positive prompts (what we want)
    good_prompts = [
        "a dramatic cinematic motorcycle photo with beautiful moody lighting",
        "a stunning Ducati motorcycle rider in action on a scenic canyon road at sunset",
        "a visually striking red motorcycle in low light with strong color contrast",
        "an exciting motorcycle racing or cornering action shot",
    ]
    # Negative prompts (what we don't want)
    bad_prompts = [
        "a blurry poorly lit casual snapshot of people standing around motorcycles",
        "a cluttered unfocused group photo with no clear subject in flat lighting",
    ]

    all_prompts = good_prompts + bad_prompts

    inputs = processor(text=all_prompts, images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits_per_image.softmax(dim=1).numpy()[0]

    good_score = logits[: len(good_prompts)].mean()
    bad_score = logits[len(good_prompts) :].mean()
    combined = float(good_score - bad_score)

    # Map to 0-60 scale to match Claude's total range
    total = max(0, min(60, int(combined * 120 + 30)))

    return {
        "subject_clarity": int(logits[0] * 30 + 3),
        "lighting": int(logits[0] * 20 + 3),
        "color_pop": int(logits[2] * 20 + 3),
        "emotion": int(logits[3] * 20 + 3),
        "scroll_stop": int(combined * 20 + 5),
        "crop_4x5": 5,  # CLIP can't evaluate crop potential
        "total": total,
        "one_line": (
            f"CLIP score: {combined:.3f} "
            f"(cinematic={logits[0]:.2f}, action={logits[1]:.2f}, striking={logits[2]:.2f})"
        ),
    }
