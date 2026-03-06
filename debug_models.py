"""
Diagnostic script — run this FIRST to see exactly what each model outputs.
This tells you the actual label strings, probabilities, and whether the
label mapping is correct for each model.

Usage:
    python debug_models.py real_face.jpg
    python debug_models.py fake_face.jpg
"""

import sys
import torch
import torch.nn.functional as F
from PIL import Image

if len(sys.argv) < 2:
    print("Usage: python debug_models.py <image_path>")
    sys.exit(1)

image_path = sys.argv[1]
image = Image.open(image_path).convert('RGB')
print(f"\nImage: {image_path}  size={image.size}")
print("=" * 60)

# ── Model 1: vit_v2 ─────────────────────────────────────────────
print("\n[1] prithivMLmods/Deep-Fake-Detector-v2-Model (ViT)")
try:
    from transformers import ViTForImageClassification, ViTImageProcessor
    model = ViTForImageClassification.from_pretrained("prithivMLmods/Deep-Fake-Detector-v2-Model")
    processor = ViTImageProcessor.from_pretrained("prithivMLmods/Deep-Fake-Detector-v2-Model")
    model.eval()

    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1).squeeze(0)

    print(f"  id2label from config: {model.config.id2label}")
    for idx, label in model.config.id2label.items():
        print(f"  Index {idx}: '{label}' → probability = {probs[int(idx)].item():.4f}")

    argmax_idx = probs.argmax().item()
    argmax_label = model.config.id2label[argmax_idx]
    print(f"  >> Predicted (argmax): index={argmax_idx} label='{argmax_label}'")

except Exception as e:
    print(f"  ERROR: {e}")

# ── Model 2: siglip_v1 ───────────────────────────────────────────
print("\n[2] prithivMLmods/deepfake-detector-model-v1 (SigLIP)")
try:
    from transformers import AutoImageProcessor, SiglipForImageClassification
    model = SiglipForImageClassification.from_pretrained("prithivMLmods/deepfake-detector-model-v1")
    processor = AutoImageProcessor.from_pretrained("prithivMLmods/deepfake-detector-model-v1")
    model.eval()

    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1).squeeze(0)

    print(f"  id2label from config: {model.config.id2label}")
    for idx, label in model.config.id2label.items():
        print(f"  Index {idx}: '{label}' → probability = {probs[int(idx)].item():.4f}")

    argmax_idx = probs.argmax().item()
    argmax_label = model.config.id2label[argmax_idx]
    print(f"  >> Predicted (argmax): index={argmax_idx} label='{argmax_label}'")

except Exception as e:
    print(f"  ERROR: {e}")

# ── Model 3: vit_wvolf ───────────────────────────────────────────
print("\n[3] Wvolf/ViT_Deepfake_Detection (ViT)")
try:
    from transformers import ViTForImageClassification, ViTImageProcessor
    model = ViTForImageClassification.from_pretrained("Wvolf/ViT_Deepfake_Detection")
    processor = ViTImageProcessor.from_pretrained("Wvolf/ViT_Deepfake_Detection")
    model.eval()

    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1).squeeze(0)

    print(f"  id2label from config: {model.config.id2label}")
    for idx, label in model.config.id2label.items():
        print(f"  Index {idx}: '{label}' → probability = {probs[int(idx)].item():.4f}")

    argmax_idx = probs.argmax().item()
    argmax_label = model.config.id2label[argmax_idx]
    print(f"  >> Predicted (argmax): index={argmax_idx} label='{argmax_label}'")

except Exception as e:
    print(f"  ERROR: {e}")

print("\n" + "=" * 60)
print("Share this output so we can verify label mappings are correct.")
