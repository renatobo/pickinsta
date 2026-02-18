#!/usr/bin/env python3
"""Test YOLO-enhanced Claude scoring on a specific image."""

from pathlib import Path
import cv2
import json
from pickinsta.ig_image_selector import yolo_detect_subject, VISION_PROMPT

# Test image
test_image = Path("input/AB Group - Session 4 (Turn 6) - CVR_6108_Mar1025_1111AM_CaliPhoto.jpg")

print(f"Testing YOLO detection and Claude prompt enhancement on:")
print(f"  {test_image}\n")

# Run YOLO detection
img = cv2.imread(str(test_image))
if img is None:
    print(f"Error: Cannot read image {test_image}")
    exit(1)

detection = yolo_detect_subject(img, debug=True)

if detection:
    x, y, w, h, class_name, conf = detection
    img_h, img_w = img.shape[:2]

    # Calculate relative position
    center_x = (x + w/2) / img_w
    center_y = (y + h/2) / img_h
    size_ratio = (w * h) / (img_w * img_h)

    # Describe position
    h_pos = "left" if center_x < 0.33 else "right" if center_x > 0.66 else "center"
    v_pos = "top" if center_y < 0.33 else "bottom" if center_y > 0.66 else "middle"
    position = f"{v_pos}-{h_pos}" if v_pos != "middle" or h_pos != "center" else "centered"

    # Describe size
    size_desc = "large" if size_ratio > 0.3 else "medium" if size_ratio > 0.1 else "small"

    yolo_context = f"\n\n**Detected Subject**: {class_name} ({position}, {size_desc}, confidence: {conf:.0%})"

    print("\n" + "="*70)
    print("ENHANCED CLAUDE PROMPT:")
    print("="*70)
    print(VISION_PROMPT + yolo_context)
    print("="*70)

    print("\n✅ YOLO detection provides Claude with:")
    print(f"   - Subject type: {class_name}")
    print(f"   - Position: {position}")
    print(f"   - Size: {size_desc} ({size_ratio:.1%} of image)")
    print(f"   - Confidence: {conf:.0%}")
    print("\nThis context helps Claude make more informed judgments about:")
    print("   • Subject clarity (knows what to look for)")
    print("   • Crop potential (knows where subject is positioned)")
    print("   • Composition quality (can evaluate subject placement)")
else:
    print("\n⚠️  No subject detected by YOLO")
    print("Claude will receive the standard prompt without subject context")
    print("\n" + "="*70)
    print("STANDARD CLAUDE PROMPT:")
    print("="*70)
    print(VISION_PROMPT)
    print("="*70)
