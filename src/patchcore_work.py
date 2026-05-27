"""
PatchCore — Towards Total Recall in Industrial Anomaly Detection
Reference: Roth et al., 2022 (https://arxiv.org/abs/2106.08265)

Usage
-----
Train:
    python patchcore.py --mode train --data ./capsule

Evaluate:
    python patchcore.py --mode eval --data ./capsule

Train + Evaluate:
    python patchcore.py --mode both --data ./capsule
"""

import os
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ─── Local import ────────────────────────────────────────────────────────────
from dataset_build import CapsuleDataset  # your existing file


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  FEATURE EXTRACTOR
# ═══════════════════════════════════════════════════════════════════════════════

class PatchCoreFeatureExtractor(torch.nn.Module):
    """
    Wide-ResNet50 backbone frozen at inference.
    Extracts intermediate feature maps from layer2 and layer3,
    then average-pools each spatial patch to a fixed dimension.

    Output shape per forward pass:
        (B, H*W, feature_dim)
    where H, W depend on the input resolution (256→32 for layer2, 256→16 for layer3).
    """

    def __init__(self, device="cpu"):
        super().__init__()
        backbone = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.DEFAULT)
        # Keep everything up to (and including) layer3
        self.layer0 = torch.nn.Sequential(backbone.conv1, backbone.bn1,
                                          backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2   # stride-8 feature map
        self.layer3 = backbone.layer3   # stride-16 feature map

        for p in self.parameters():
            p.requires_grad = False

        self.to(device)
        self.device = device

    # ── neighbourhood aggregation ───────────────────────────────────────────
    @staticmethod
    def _adaptive_avg_pool_patch(x, patch_size=3):
        """
        For each spatial location (h, w) average-pool over a (patch_size × patch_size)
        neighbourhood — equivalent to a depthwise avg-pool with same padding.
        """
        return F.avg_pool2d(x, kernel_size=patch_size, stride=1,
                            padding=patch_size // 2)

    def forward(self, x):
        x = x.to(self.device)
        x = self.layer0(x)
        x = self.layer1(x)
        f2 = self.layer2(x)           # (B, 512, H/8,  W/8)
        f3 = self.layer3(f2)          # (B, 1024, H/16, W/16)

        # neighbourhood-aggregated patches
        f2 = self._adaptive_avg_pool_patch(f2)
        f3 = self._adaptive_avg_pool_patch(f3)

        # upsample f3 → same spatial size as f2
        f3_up = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear",
                              align_corners=False)

        # concat along channel dim → (B, 1536, H/8, W/8)
        combined = torch.cat([f2, f3_up], dim=1)

        B, C, H, W = combined.shape
        # flatten spatial → (B, H*W, C)
        patches = combined.permute(0, 2, 3, 1).reshape(B, H * W, C)
        return patches, (H, W)


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  CORESET SUBSAMPLING  (greedy k-centre / random fallback)
# ═══════════════════════════════════════════════════════════════════════════════

def coreset_subsampling(features: np.ndarray, ratio: float = 0.1,
                        seed: int = 42) -> np.ndarray:
    """
    Greedy k-centre coreset selection as in the paper (Algorithm 1).
    Falls back to random subsampling when the desired k is very large
    (>50 k samples) to keep memory manageable.

    Parameters
    ----------
    features : (N, D) float32 array
    ratio    : fraction of N to keep
    seed     : random seed for reproducibility

    Returns
    -------
    (k, D) float32 array — the selected coreset
    """
    np.random.seed(seed)
    N = len(features)
    k = max(1, int(N * ratio))

    if k >= N:
        return features

    # For very large sets use random sampling (fast) to stay within RAM/time
    if k > 50_000:
        idx = np.random.choice(N, k, replace=False)
        return features[idx]

    # Greedy k-centre
    rng = np.random.default_rng(seed)
    first = rng.integers(0, N)
    selected = [first]
    # distance of every point to its nearest selected centre
    min_dists = np.full(N, np.inf, dtype=np.float32)

    for _ in tqdm(range(k - 1), desc="Coreset subsampling", leave=False):
        new = features[selected[-1]]                     # (D,)
        d = np.linalg.norm(features - new, axis=1)      # (N,)
        min_dists = np.minimum(min_dists, d)
        selected.append(int(np.argmax(min_dists)))

    return features[selected]


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  PATCHCORE MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class PatchCore:
    """
    Full PatchCore pipeline:
      - fit()   : build the memory bank from train images
      - predict(): score test images + produce anomaly maps
      - evaluate(): compute image-level AUROC and pixel-level AUROC
      - save() / load(): persist / restore the memory bank
    """

    def __init__(self, device=None, coreset_ratio=0.1, k_neighbours=9,
                 batch_size=8, num_workers=4):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.coreset_ratio = coreset_ratio
        self.k_neighbours = k_neighbours
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.extractor = PatchCoreFeatureExtractor(device=device)
        self.extractor.eval()
        self.memory_bank: np.ndarray | None = None   # (M, D)
        self.spatial_shape: tuple | None = None       # (H, W) of feature grid

    # ── helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _l2_knn_scores(query: np.ndarray, bank: np.ndarray,
                       k: int = 9) -> np.ndarray:
        """
        Batched approximate k-NN search (pure numpy, no faiss required).
        Returns the *mean* distance to the k nearest neighbours for each query.

        query : (Q, D)
        bank  : (M, D)
        → (Q,)
        """
        # Process in sub-batches of 1024 to avoid OOM
        Q = len(query)
        scores = np.empty(Q, dtype=np.float32)
        step = 1024

        for start in range(0, Q, step):
            q = query[start: start + step]            # (s, D)
            # squared L2: ||q - b||^2 = ||q||^2 + ||b||^2 - 2 q·b
            q2 = (q ** 2).sum(axis=1, keepdims=True)  # (s, 1)
            b2 = (bank ** 2).sum(axis=1)              # (M,)
            dist2 = q2 + b2 - 2.0 * (q @ bank.T)     # (s, M)
            dist2 = np.maximum(dist2, 0.0)            # numerical safety
            knn_idx = np.argpartition(dist2, k, axis=1)[:, :k]
            knn_d = np.take_along_axis(dist2, knn_idx, axis=1) ** 0.5
            scores[start: start + step] = knn_d.mean(axis=1)

        return scores

    # ── fit ─────────────────────────────────────────────────────────────────

    def fit(self, dataset: CapsuleDataset):
        """Extract all training patches and build the coreset memory bank."""
        loader = DataLoader(dataset, batch_size=self.batch_size,
                            shuffle=False, num_workers=self.num_workers,
                            pin_memory=self.device != "cpu")
        all_patches = []

        with torch.no_grad():
            for images, _, _, _ in tqdm(loader, desc="Extracting train features"):
                patches, spatial = self.extractor(images)   # (B, H*W, D)
                self.spatial_shape = spatial
                B, HW, D = patches.shape
                all_patches.append(patches.reshape(-1, D).cpu().numpy())

        all_patches = np.concatenate(all_patches, axis=0)   # (N, D)
        print(f"  Total patches before subsampling : {len(all_patches):,}")

        self.memory_bank = coreset_subsampling(all_patches,
                                               ratio=self.coreset_ratio)
        print(f"  Memory bank size after coreset   : {len(self.memory_bank):,}")

    # ── predict ─────────────────────────────────────────────────────────────

    def predict(self, dataset: CapsuleDataset):
        """
        Score every image in `dataset`.

        Returns
        -------
        image_scores : (N,)  — per-image anomaly score (max patch score)
        anomaly_maps : list of (H_orig, W_orig) np.float32 arrays
        labels       : (N,)  ground-truth binary labels
        gt_masks     : list of (H_orig, W_orig) np.float32 arrays
        """
        assert self.memory_bank is not None, "Call fit() first."
        loader = DataLoader(dataset, batch_size=self.batch_size,
                            shuffle=False, num_workers=self.num_workers,
                            pin_memory=self.device != "cpu")

        image_scores, anomaly_maps, labels, gt_masks = [], [], [], []

        with torch.no_grad():
            for images, lbls, masks, _ in tqdm(loader, desc="Scoring test images"):
                patches, (H, W) = self.extractor(images)    # (B, H*W, D)
                B, HW, D = patches.shape

                patch_np = patches.reshape(-1, D).cpu().numpy()
                patch_scores = self._l2_knn_scores(patch_np,
                                                   self.memory_bank,
                                                   k=self.k_neighbours)
                patch_scores = patch_scores.reshape(B, H, W)

                for i in range(B):
                    score_map = patch_scores[i]                 # (H, W)
                    image_scores.append(float(score_map.max()))

                    # upsample anomaly map → 256×256
                    amap = torch.tensor(score_map[None, None]).float()
                    amap = F.interpolate(amap, size=(256, 256),
                                         mode="bilinear", align_corners=False)
                    anomaly_maps.append(amap.squeeze().numpy())

                    labels.append(int(lbls[i]))
                    gt_masks.append(masks[i].squeeze().numpy())

        return (np.array(image_scores), anomaly_maps,
                np.array(labels),        gt_masks)

    # ── evaluate ─────────────────────────────────────────────────────────────

    def evaluate(self, dataset: CapsuleDataset, save_dir: str | None = None):
        """
        Compute image-level and pixel-level AUROC, optionally save vis.

        Returns
        -------
        img_auroc   : float
        pixel_auroc : float
        """
        scores, amaps, labels, gt_masks = self.predict(dataset)

        # ── image-level AUROC ───────────────────────────────────────────────
        img_auroc = roc_auc_score(labels, scores)
        print(f"\n  Image-level AUROC : {img_auroc:.4f}")

        # ── pixel-level AUROC ───────────────────────────────────────────────
        all_pred  = np.concatenate([a.ravel()  for a in amaps])
        all_gt    = np.concatenate([m.ravel()  for m in gt_masks])
        # Only compute pixel AUROC if there are positive pixels in GT
        if all_gt.max() > 0:
            pixel_auroc = roc_auc_score(all_gt.astype(int), all_pred)
            print(f"  Pixel-level AUROC : {pixel_auroc:.4f}")
        else:
            pixel_auroc = float("nan")
            print("  Pixel-level AUROC : N/A (no anomaly pixels in GT)")

        # ── optional visualisations ──────────────────────────────────────────
        if save_dir is not None:
            self._save_visualisations(dataset, scores, amaps, labels, gt_masks,
                                      save_dir)

        return img_auroc, pixel_auroc

    # ── visualisation ────────────────────────────────────────────────────────

    @staticmethod
    def _save_visualisations(dataset, scores, amaps, labels, gt_masks,
                             save_dir, n_samples=16):
        os.makedirs(save_dir, exist_ok=True)
        inv = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                                   std=[1/0.229, 1/0.224, 1/0.225])
        # Sort by score descending so we visualise worst cases first
        order = np.argsort(scores)[::-1]
        n = min(n_samples, len(order))

        fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
        if n == 1:
            axes = axes[None]

        for row, idx in enumerate(order[:n]):
            img_t, _, _, _ = dataset[idx]
            img_np = inv(img_t).permute(1, 2, 0).clamp(0, 1).numpy()

            gt = gt_masks[idx]
            amap = amaps[idx]
            amap_norm = (amap - amap.min()) / (amap.max() - amap.min() + 1e-8)

            axes[row, 0].imshow(img_np)
            axes[row, 0].set_title(
                f"{'DEFECT' if labels[idx] else 'GOOD'} | score={scores[idx]:.3f}")
            axes[row, 0].axis("off")

            axes[row, 1].imshow(gt, cmap="gray")
            axes[row, 1].set_title("Ground Truth")
            axes[row, 1].axis("off")

            axes[row, 2].imshow(img_np)
            axes[row, 2].imshow(amap_norm, cmap="jet", alpha=0.5)
            axes[row, 2].set_title("Anomaly Map (overlay)")
            axes[row, 2].axis("off")

        plt.tight_layout()
        out_path = os.path.join(save_dir, "anomaly_vis.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Visualisations saved → {out_path}")

    # ── persistence ──────────────────────────────────────────────────────────

    def save(self, path: str):
        """Save the memory bank (and spatial shape) to a .npz file."""
        np.savez_compressed(path,
                            memory_bank=self.memory_bank,
                            spatial_h=np.array(self.spatial_shape[0]),
                            spatial_w=np.array(self.spatial_shape[1]))
        print(f"  Memory bank saved → {path}")

    def load(self, path: str):
        """Load a previously saved memory bank."""
        data = np.load(path)
        self.memory_bank = data["memory_bank"]
        self.spatial_shape = (int(data["spatial_h"]), int(data["spatial_w"]))
        print(f"  Memory bank loaded ← {path}  ({len(self.memory_bank):,} vectors)")


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def build_transforms():
    """ImageNet-normalised transforms that match WideResNet50 pre-training."""
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def main():
    parser = argparse.ArgumentParser(description="PatchCore — Capsule Defect Detection")
    parser.add_argument("--mode",    choices=["train", "eval", "both"],
                        default="both",
                        help="train = build memory bank only; "
                             "eval = score test set; both = train then eval")
    parser.add_argument("--data",    default="./capsule",
                        help="Path to the capsule dataset root")
    parser.add_argument("--bank",    default="memory_bank.npz",
                        help="Path to save/load the memory bank")
    parser.add_argument("--ratio",   type=float, default=0.1,
                        help="Coreset subsampling ratio (default 0.10 = 10%%)")
    parser.add_argument("--k",       type=int,   default=9,
                        help="Number of k-NN neighbours (default 9)")
    parser.add_argument("--batch",   type=int,   default=8,
                        help="Batch size (default 8)")
    parser.add_argument("--workers", type=int,   default=4,
                        help="DataLoader worker count (default 4)")
    parser.add_argument("--vis",     default="results",
                        help="Directory to save anomaly visualisations "
                             "(set to '' to skip)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    t = build_transforms()
    model = PatchCore(device=device, coreset_ratio=args.ratio,
                      k_neighbours=args.k, batch_size=args.batch,
                      num_workers=args.workers)

    # ── TRAIN ──────────────────────────────────────────────────────────────
    if args.mode in ("train", "both"):
        print("\n[1/2] Building memory bank …")
        train_ds = CapsuleDataset(args.data, split="train",
                                  image_transform=t)
        model.fit(train_ds)
        model.save(args.bank)

    # ── EVAL ───────────────────────────────────────────────────────────────
    if args.mode in ("eval", "both"):
        print("\n[2/2] Evaluating on test set …")
        if args.mode == "eval":
            model.load(args.bank)

        test_ds = CapsuleDataset(args.data, split="test",
                                 image_transform=t)
        vis_dir = args.vis if args.vis else None
        img_auroc, pixel_auroc = model.evaluate(test_ds, save_dir=vis_dir)

        print("\n" + "=" * 45)
        print(f"  Image AUROC : {img_auroc:.4f}")
        print(f"  Pixel AUROC : {pixel_auroc:.4f}")
        print("=" * 45)


if __name__ == "__main__":
    main()






    






    

