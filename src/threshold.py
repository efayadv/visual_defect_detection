import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from dataset_build import CapsuleDataset
from patchcore_work import PatchCore, build_transforms


def find_optimal_threshold(labels, scores, save_dir="results"):
    """
    Compute ROC curve and find the optimal threshold via Youden's J statistic:
        J = Sensitivity + Specificity - 1
    The threshold that maximises J gives the best tradeoff between
    catching defects (TPR) and avoiding false alarms (1 - FPR).
    """
    fpr, tpr, thresholds = roc_curve(labels, scores)
    auc = roc_auc_score(labels, scores)

    # Youden's J
    J = tpr - fpr
    best_idx = int(np.argmax(J))
    best_threshold = float(thresholds[best_idx])
    best_tpr = float(tpr[best_idx])
    best_fpr = float(fpr[best_idx])

    # ── Print results ────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print(f"  AUC                : {auc:.4f}")
    print(f"  Optimal threshold  : {best_threshold:.6f}")
    print(f"  Sensitivity (TPR)  : {best_tpr:.4f}  (defects correctly caught)")
    print(f"  False alarm (FPR)  : {best_fpr:.4f}  (good capsules wrongly flagged)")
    print(f"  Youden's J         : {J[best_idx]:.4f}")
    print("=" * 50)
    print(f"\n  → Use --threshold {best_threshold:.6f} in inference.py")
    print("=" * 50)

    # ── Plot ROC curve ───────────────────────────────────────────────────────
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ROC curve
    axes[0].plot(fpr, tpr, color="steelblue", lw=2,
                 label=f"ROC curve (AUC = {auc:.4f})")
    axes[0].scatter(best_fpr, best_tpr, color="red", zorder=5, s=100,
                    label=f"Optimal threshold = {best_threshold:.4f}")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve — Image-level")
    axes[0].legend(loc="lower right")
    axes[0].grid(alpha=0.3)

    # Score distribution
    good_scores = scores[labels == 0]
    defect_scores = scores[labels == 1]
    axes[1].hist(good_scores,   bins=30, alpha=0.6, color="green",
                 label="Good capsules")
    axes[1].hist(defect_scores, bins=30, alpha=0.6, color="red",
                 label="Defective capsules")
    axes[1].axvline(best_threshold, color="black", lw=2, linestyle="--",
                    label=f"Threshold = {best_threshold:.4f}")
    axes[1].set_xlabel("Anomaly Score")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Score Distribution")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(save_dir, "threshold_analysis.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Plot saved → {out_path}")

    return best_threshold


def main():
    parser = argparse.ArgumentParser(description="PatchCore — Threshold Finder")
    parser.add_argument("--data",    default="./capsule",
                        help="Path to the capsule dataset root")
    parser.add_argument("--bank",    default="memory_bank.npz",
                        help="Path to the saved memory bank")
    parser.add_argument("--batch",   type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--vis",     default="results",
                        help="Directory to save the ROC plot")
    args = parser.parse_args()

    device = "cpu"  # threshold finding is lightweight
    print(f"Using device: {device}")

    t = build_transforms()
    model = PatchCore(device=device, batch_size=args.batch,
                      num_workers=args.workers)
    model.load(args.bank)

    print("\nScoring test set …")
    test_ds = CapsuleDataset(args.data, split="test", image_transform=t)
    scores, _, labels, _ = model.predict(test_ds)

    find_optimal_threshold(labels, scores, save_dir=args.vis)


if __name__ == "__main__":
    main()