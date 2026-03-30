"""
pipeline/train_direct.py
=========================
Trains ResNet-50 to predict carbohydrate range directly from food images.

Architecture:
  Input image -> ResNet-50 (ImageNet pretrained) -> 5-class carb range
  Range 0: 0-20g   Range 1: 21-40g   Range 2: 41-60g
  Range 3: 61-80g  Range 4: 81g+

Why ResNet-50:
  ResNet-50 is the most widely used backbone in food image classification
  literature (He et al., 2016). Its residual connections allow deep feature
  learning without vanishing gradients, making it well-suited to fine-grained
  visual tasks such as distinguishing between similar food categories.

Loss function - Ordinal Focal CE:
  Three components combined:
  1. Weighted CE     - corrects class imbalance across carb ranges
  2. Ordinal MSE     - penalises large ordinal errors clinically
                       (predicting range 4 for range 0 risks 4x insulin overdose)
  3. Focal term      - down-weights easy examples, focuses training on the
                       hardest class (41-60g, 25% accuracy). Lin et al. (2017).

Training strategy:
  Phase 1 (epochs 1 to PHASE2):  Head only - backbone frozen
  Phase 2 (epoch PHASE2 onward): layer4 + head unfrozen (fine-tuning)
"""

import json
import os
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    DATASET_DIR, DEVICE,
    BATCH_SIZE, EPOCHS_S1, LR, NUM_WORKERS,
    NUM_CARB_RANGES, CARB_RANGE_LABELS,
)

MODEL_DIR = os.path.join("models", "direct")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

TRAIN_TF = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(25),
    transforms.ColorJitter(brightness=0.5, contrast=0.5,
                           saturation=0.4, hue=0.15),
    transforms.RandomGrayscale(p=0.1),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

VAL_TF = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# ==================================================================
# DATASET
# ==================================================================

class CarbRangeDataset(Dataset):
    """
    Loads food images with carb range ground truth labels (0-4).
    Reads from data/carb_dataset/train_eval/ (Food-101 TRAIN split)
    which is built by pipeline/dataset.py using USDA-sourced carb values.
    """
    def __init__(self, root: str, transform, split: str = None):
        self.samples   = []
        self.transform = transform

        for range_dir in sorted(Path(root).iterdir()):
            if not range_dir.is_dir():
                continue
            name = range_dir.name
            if not name.startswith("range_"):
                continue
            label  = int(name.split("_")[1])
            images = list(range_dir.glob("*.jpg"))

            if split == "train":
                images = [i for i in images if hash(i.name) % 5 != 0]
            elif split == "val":
                images = [i for i in images if hash(i.name) % 5 == 0]

            for img in images:
                self.samples.append((str(img), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def load_data():
    train_dir = str(Path(DATASET_DIR) / "train_eval")
    train_ds  = CarbRangeDataset(train_dir, TRAIN_TF, split="train")
    val_ds    = CarbRangeDataset(train_dir, VAL_TF,   split="val")

    if len(train_ds) == 0:
        raise FileNotFoundError(
            f"No images found in {train_dir}. "
            "Run 'python run.py --stage data' first."
        )

    class_counts = Counter(label for _, label in train_ds.samples)
    print(f"\n  Dataset - Train: {len(train_ds):,}  Val: {len(val_ds):,}")
    print(f"  Class distribution (train):")
    for cls in range(NUM_CARB_RANGES):
        count = class_counts.get(cls, 0)
        label = CARB_RANGE_LABELS[cls]
        bar   = "=" * (count // 50)
        print(f"    Range {cls} ({label:<8}): {count:5d}  {bar}")

    # Weighted sampler - balances class distribution each batch
    total         = sum(class_counts.values())
    class_weights = {c: total / max(class_counts.get(c, 1), 1)
                     for c in range(NUM_CARB_RANGES)}
    sample_weights = [class_weights[label] for _, label in train_ds.samples]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights),
                                    replacement=True)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, sampler=sampler,
        num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS,
    )

    return train_loader, val_loader, class_counts


# ==================================================================
# MODEL
# ==================================================================

def build_model() -> nn.Module:
    """
    ResNet-50 with ImageNet pretrained weights (He et al., 2016).
    Classification head: Linear(2048->512) -> BN -> ReLU
                         -> Dropout(0.4) -> Linear(512->5)
    Backbone frozen initially; layer4 + head unfrozen in Phase 2.
    """
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False

    in_features = model.fc.in_features  # 2048
    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.4),
        nn.Linear(512, NUM_CARB_RANGES),
    )
    return model


def unfreeze_top_layers(model):
    """Unfreeze ResNet-50 layer4 + classification head for Phase 2."""
    for param in model.layer4.parameters():
        param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True


# ==================================================================
# ORDINAL LOSS
# ==================================================================

def ordinal_focal_loss(logits, labels, class_weights=None, gamma=2.0):
    """
    Combined weighted CE + focal term + ordinal MSE penalty.

    Components:
      1. Weighted CE        - corrects class imbalance
      2. Focal term         - (1-pt)^gamma down-weights easy examples so
                              training focuses on the hard 41-60g range.
                              gamma=2 is the value used in Lin et al. (2017).
      3. Ordinal MSE        - penalises predictions proportional to their
                              ordinal distance from the true class, encoding
                              clinical safety (Ozkaya et al., 2026).
    """
    # Focal CE: scale standard CE by (1 - p_true)^gamma
    log_probs  = F.log_softmax(logits, dim=1)
    probs      = log_probs.exp()
    ce_per     = F.nll_loss(log_probs, labels, weight=class_weights,
                            reduction="none")
    pt         = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
    focal_ce   = ((1 - pt) ** gamma * ce_per).mean()

    # Ordinal MSE
    class_vals = torch.arange(NUM_CARB_RANGES, dtype=torch.float32,
                               device=logits.device)
    expected   = (probs * class_vals).sum(dim=1)
    ordinal    = F.mse_loss(expected, labels.float())

    return focal_ce + 0.5 * ordinal


def make_weights(class_counts: dict) -> torch.Tensor:
    """
    Per-class loss weights capped at 5x mean.
    Raised from 3x to give the hardest class (41-60g, 25% accuracy)
    significantly more gradient budget without completely dominating.
    """
    total  = sum(class_counts.values())
    raw    = [total / max(class_counts.get(i, 1) * NUM_CARB_RANGES, 1)
              for i in range(NUM_CARB_RANGES)]
    mean_w = sum(raw) / len(raw)
    cap    = mean_w * 5.0
    capped = [min(w, cap) for w in raw]
    weights = torch.tensor(capped, dtype=torch.float32).to(DEVICE)
    print(f"  Loss weights: {[round(w.item(), 2) for w in weights]}")
    return weights


# ==================================================================
# TRAINING LOOP
# ==================================================================

def mixup_batch(images, labels, alpha=0.3):
    """
    Mixup augmentation (Zhang et al., 2018): blends pairs of training
    images and their labels. Softens decision boundaries between adjacent
    carb ranges, reducing the 41-60g / 21-40g confusion observed in results.
    alpha=0.3 gives moderate blending - high enough to help, low enough
    to keep individual class signal intact.
    """
    lam    = torch.distributions.Beta(alpha, alpha).sample().item()
    idx    = torch.randperm(images.size(0), device=images.device)
    images = lam * images + (1 - lam) * images[idx]
    return images, labels, labels[idx], lam


def train_epoch(model, loader, class_weights, optimizer, use_mixup=False):
    """
    use_mixup only enabled in Phase 2 when backbone is unfrozen.
    Mixup with a frozen backbone is counterproductive - blended features
    from fixed representations don't correspond to either class cleanly.
    """
    model.train()
    loss_sum = correct = total = 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        if use_mixup:
            images, labels_a, labels_b, lam = mixup_batch(images, labels)
            out  = model(images)
            loss = (lam       * ordinal_focal_loss(out, labels_a, class_weights) +
                    (1 - lam) * ordinal_focal_loss(out, labels_b, class_weights))
            correct += (out.argmax(1) == labels_a).sum().item()
        else:
            out  = model(images)
            loss = ordinal_focal_loss(out, labels, class_weights)
            correct += (out.argmax(1) == labels).sum().item()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        loss_sum += loss.item() * images.size(0)
        total    += images.size(0)
    return loss_sum / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, class_weights):
    model.eval()
    loss_sum = correct = clin = total = 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        out   = model(images)
        loss  = ordinal_focal_loss(out, labels, class_weights)
        preds = out.argmax(1)
        loss_sum += loss.item() * images.size(0)
        correct  += (preds == labels).sum().item()
        clin     += (torch.abs(preds - labels) <= 1).sum().item()
        total    += images.size(0)
    return loss_sum / total, correct / total, clin / total


def train(epochs: int = EPOCHS_S1):
    train_loader, val_loader, class_counts = load_data()

    save_path = os.path.join(MODEL_DIR, "resnet50_best.pth")
    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)

    model         = build_model().to(DEVICE)
    class_weights = make_weights(class_counts)
    optimizer     = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=1e-4,
    )
    PHASE2    = max(1, epochs // 3)
    best_acc  = 0.0
    best_clin = 0.0
    history   = []

    print(f"\n{'='*60}")
    print(f"  ResNet-50 - Direct Carb Range Classifier")
    print(f"  Loss: Ordinal Focal CE (weighted + focal + ordinal MSE)")
    print(f"  Epochs: {epochs}  |  Device: {DEVICE}  |  LR: {LR}")
    print(f"  Phase 1: epochs 1-{PHASE2-1}  |  Phase 2 (Mixup): epochs {PHASE2}-{epochs}")
    print(f"{'='*60}")

    # Phase 1 scheduler cycles fully over the head-warmup period
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=PHASE2)

    for epoch in range(1, epochs + 1):

        if epoch == PHASE2:
            print(f"\n  -> Epoch {epoch}: Unfreezing layer4 + head")
            unfreeze_top_layers(model)
            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=LR * 0.1, weight_decay=1e-4,
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs - PHASE2 + 1
            )

        in_phase2 = epoch >= PHASE2
        t0 = time.time()
        tr_loss, tr_acc          = train_epoch(model, train_loader,
                                               class_weights, optimizer,
                                               use_mixup=in_phase2)
        vl_loss, vl_acc, vl_clin = eval_epoch(model, val_loader, class_weights)
        scheduler.step()

        history.append({
            "epoch":     epoch,
            "train_acc": round(tr_acc,  4),
            "val_acc":   round(vl_acc,  4),
            "val_clin":  round(vl_clin, 4),
            "val_loss":  round(vl_loss, 4),
        })

        flag = ""
        if vl_acc > best_acc:
            best_acc  = vl_acc
            best_clin = vl_clin
            torch.save(model.state_dict(), save_path)
            flag = "  <- saved OK"

        print(f"  Ep {epoch:02d}/{epochs}  "
              f"tr={tr_acc:.3f}  vl={vl_acc:.3f}  "
              f"clin={vl_clin:.3f}  loss={vl_loss:.4f}  "
              f"({time.time()-t0:.1f}s){flag}")

    with open(os.path.join(MODEL_DIR, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n  OK Best val accuracy:      {best_acc:.3f} ({best_acc*100:.1f}%)")
    print(f"  OK Best clinical accuracy: {best_clin:.3f} ({best_clin*100:.1f}%)")
    print(f"  OK Saved: {save_path}")
    print(f"\n  OK Run: python run.py --stage evaluate")
