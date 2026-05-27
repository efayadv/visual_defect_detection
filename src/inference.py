"""
inference.py — Run PatchCore on a single capsule image.

Outputs a verdict (GOOD / DEFECTIVE), anomaly score, and a heatmap overlay.

Usage
-----
    python src/inference.py --image path/to/capsule.jpg
    python src/inference.py --image path/to/capsule.jpg --threshold 11.568211
    python src/inference.py --image path/to/capsule.jpg --bank memory_bank.npz --out results/
"""

import argparse
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from patchcore_work import PatchCore, PatchCoreFeatureExtractor, build_transforms


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLE IMAGE SCORER
# ═══════════════════════════════════════════════════════════════════════════════

def score_image(image_path: str, model: PatchCore):
    """
    Score a single image.

    Returns
    -------
    score     : float  — raw anomaly score (higher = more anomalous)
    amap      : (256, 256) np.float32 — normalised anomaly heatmap [0, 1]
    image_np  : (256, 256, 3) np.float32 — the resized input image [0, 1]
    """
    t = build_transforms()
    image_pil = Image.open(image_path).convert("RGB")
    image_t = t(image_pil).unsqueeze(0)  # (1, 3, 256, 256)

    with torch.no_grad():
        patches, (H, W) = model.extractor(image_t)   # (1, H*W, D)
        patch_np = patches.reshape(-1, patches.shape[-1]).cpu().numpy()

    patch_scores = model._l2_knn_scores(patch_np, model.memory_bank,
                                        k=model.k_neighbours)
    patch_scores = patch_scores.reshape(H, W)

    # Image-level score = max patch score
    score = float(patch_scores.max())

    # Upsample anomaly map → 256×256
    amap_t = torch.tensor(patch_scores[None, None]).float()
    amap_t = F.interpolate(amap_t, size=(256, 256),
                           mode="bilinear", align_corners=False)
    amap = amap_t.squeeze().numpy()

    # Normalise to [0, 1] for visualisation
    amap_norm = (amap - amap.min()) / (amap.max() - amap.min() + 1e-8)

    # Original image as numpy for plotting (undo ImageNet normalisation)
    from torchvision import transforms
    inv = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
    )
    image_np = inv(image_t.squeeze()).permute(1, 2, 0).clamp(0, 1).numpy()

    return score, amap_norm, image_np


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALISATION
# ═══════════════════════════════════════════════════════════════════════════════

def save_result(image_path, image_np, amap_norm, score, threshold,
                is_defective, out_dir):
    """Save a 3-panel result image: original | heatmap | overlay."""
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    verdict = "DEFECTIVE" if is_defective else "GOOD"
    color   = "red"       if is_defective else "green"

    fig.suptitle(
        f"Verdict: {verdict}  |  Score: {score:.4f}  |  Threshold: {threshold:.4f}",
        fontsize=14, fontweight="bold", color=color
    )

    # Panel 1 — original image
    axes[0].imshow(image_np)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    # Panel 2 — raw heatmap
    im = axes[1].imshow(amap_norm, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Anomaly Heatmap")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # Panel 3 — overlay
    axes[2].imshow(image_np)
    axes[2].imshow(amap_norm, cmap="jet", alpha=0.45, vmin=0, vmax=1)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.tight_layout()

    # Save with the same stem as the input image
    stem = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(out_dir, f"{stem}_result.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    return out_path


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="PatchCore — Single Image Inference")
    parser.add_argument("--image",     required=True,
                        help="Path to the capsule image to inspect")
    parser.add_argument("--bank",      default="memory_bank.npz",
                        help="Path to the saved memory bank (default: memory_bank.npz)")
    parser.add_argument("--threshold", type=float, default=11.568211,
                        help="Anomaly score threshold (default: 11.568211)")
    parser.add_argument("--out",       default="results",
                        help="Directory to save the result image (default: results/)")
    args = parser.parse_args()

    # ── Validate input ───────────────────────────────────────────────────────
    if not os.path.isfile(args.image):
        print(f"ERROR: image not found → {args.image}")
        sys.exit(1)

    if not os.path.isfile(args.bank):
        print(f"ERROR: memory bank not found → {args.bank}")
        print("       Run patchcore_work.py --mode train first.")
        sys.exit(1)

    # ── Load model ───────────────────────────────────────────────────────────
    device = "cpu"
    model = PatchCore(device=device)
    model.load(args.bank)

    # ── Score ────────────────────────────────────────────────────────────────
    print(f"\nInspecting: {args.image}")
    score, amap_norm, image_np = score_image(args.image, model)
    is_defective = score > args.threshold

    # ── Print verdict ────────────────────────────────────────────────────────
    verdict = "DEFECTIVE ❌" if is_defective else "GOOD ✅"
    print("\n" + "=" * 45)
    print(f"  Verdict    : {verdict}")
    print(f"  Score      : {score:.6f}")
    print(f"  Threshold  : {args.threshold:.6f}")
    print("=" * 45)

    # ── Save result image ────────────────────────────────────────────────────
    out_path = save_result(args.image, image_np, amap_norm, score,
                           args.threshold, is_defective, args.out)
    print(f"\n  Result saved → {out_path}")


if __name__ == "__main__":
    main()