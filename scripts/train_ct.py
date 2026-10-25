from pathlib import Path
import sys
import random

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import (
    Dataset,
    DataLoader,
    WeightedRandomSampler
)

from src.medifederate.unet import UNet


# ============================================================
# Configuration
# ============================================================

DATA_ROOT = Path(
    "data/processed/ct_patches"
)

MODEL_DIR = Path("models")

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

BEST_MODEL = (
    MODEL_DIR /
    "unet_ct_patch256_best.pt"
)

INPUT_SIZE = 192

BATCH_SIZE = 4

EPOCHS = 10

LEARNING_RATE = 5e-4

WEIGHT_DECAY = 1e-4

POS_WEIGHT = 15.0

SEED = 42


# ============================================================
# Reproducibility
# ============================================================

random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)


# ============================================================
# Device
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("=" * 60)

print(
    "MediFederate AI - Improved CT U-Net"
)

print("=" * 60)

print(f"Device: {DEVICE}")

print(
    f"Input size: "
    f"{INPUT_SIZE}x{INPUT_SIZE}"
)

print(
    f"Batch size: {BATCH_SIZE}"
)

print(
    f"Epochs: {EPOCHS}"
)

print(
    f"Learning rate: {LEARNING_RATE}"
)

print(
    f"Positive weight: {POS_WEIGHT}"
)

print("=" * 60)


# ============================================================
# Dataset
# ============================================================

class CTPatchDataset(Dataset):

    def __init__(
        self,
        split,
        augment=False
    ):

        self.image_dir = (
            DATA_ROOT /
            split /
            "images"
        )

        self.mask_dir = (
            DATA_ROOT /
            split /
            "masks"
        )

        self.augment = augment

        self.files = sorted(
            self.image_dir.glob(
                "*.npy"
            )
        )

        if not self.files:

            raise RuntimeError(
                f"No CT patches found in "
                f"{self.image_dir}"
            )

    def __len__(self):

        return len(self.files)

    def __getitem__(
        self,
        index
    ):

        image_path = self.files[index]

        mask_path = (
            self.mask_dir /
            image_path.name
        )

        image = np.load(
            image_path
        ).astype(
            np.float32
        )

        mask = np.load(
            mask_path
        ).astype(
            np.float32
        )

        # ----------------------------------------------------
        # CT normalization
        # ----------------------------------------------------

        image = np.clip(
            image,
            -1000,
            400
        )

        image = (
            image + 1000.0
        ) / 1400.0

        # ----------------------------------------------------
        # Augmentation
        # ----------------------------------------------------

        if self.augment:

            if random.random() < 0.5:

                image = np.fliplr(
                    image
                ).copy()

                mask = np.fliplr(
                    mask
                ).copy()

            if random.random() < 0.5:

                image = np.flipud(
                    image
                ).copy()

                mask = np.flipud(
                    mask
                ).copy()

            if random.random() < 0.5:

                k = random.randint(
                    1,
                    3
                )

                image = np.rot90(
                    image,
                    k
                ).copy()

                mask = np.rot90(
                    mask,
                    k
                ).copy()

        # ----------------------------------------------------
        # Convert to tensors
        # ----------------------------------------------------

        image = torch.from_numpy(
            image
        ).float()

        mask = torch.from_numpy(
            mask
        ).float()

        # Add channel dimension
        image = image.unsqueeze(0)

        mask = mask.unsqueeze(0)

        # ----------------------------------------------------
        # Resize 256 -> 192
        # ----------------------------------------------------

        image = F.interpolate(
            image.unsqueeze(0),
            size=(
                INPUT_SIZE,
                INPUT_SIZE
            ),
            mode="bilinear",
            align_corners=False
        ).squeeze(0)

        mask = F.interpolate(
            mask.unsqueeze(0),
            size=(
                INPUT_SIZE,
                INPUT_SIZE
            ),
            mode="nearest"
        ).squeeze(0)

        mask = (
            mask > 0.5
        ).float()

        return image, mask


# ============================================================
# Dice Loss
# ============================================================

class DiceLoss(nn.Module):

    def __init__(
        self,
        smooth=1.0
    ):

        super().__init__()

        self.smooth = smooth

    def forward(
        self,
        logits,
        targets
    ):

        probabilities = torch.sigmoid(
            logits
        )

        probabilities = probabilities.reshape(
            probabilities.size(0),
            -1
        )

        targets = targets.reshape(
            targets.size(0),
            -1
        )

        intersection = (
            probabilities *
            targets
        ).sum(dim=1)

        dice = (
            2.0 * intersection +
            self.smooth
        ) / (
            probabilities.sum(dim=1) +
            targets.sum(dim=1) +
            self.smooth
        )

        return (
            1.0 - dice.mean()
        )


# ============================================================
# Combined Loss
# ============================================================

class CombinedLoss(nn.Module):

    def __init__(
        self,
        pos_weight=15.0
    ):

        super().__init__()

        self.pos_weight = pos_weight

        self.dice = DiceLoss()

    def forward(
        self,
        logits,
        targets
    ):

        weight = torch.tensor(
            [self.pos_weight],
            device=logits.device
        )

        bce = nn.functional.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=weight
        )

        dice = self.dice(
            logits,
            targets
        )

        return (
            0.5 * bce +
            0.5 * dice
        )


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(
    logits,
    targets
):

    probabilities = torch.sigmoid(
        logits
    )

    predictions = (
        probabilities >= 0.5
    ).float()

    predictions = predictions.reshape(
        -1
    )

    targets = targets.reshape(
        -1
    )

    intersection = (
        predictions *
        targets
    ).sum().item()

    pred_sum = (
        predictions.sum().item()
    )

    target_sum = (
        targets.sum().item()
    )

    dice = (
        2.0 * intersection +
        1e-6
    ) / (
        pred_sum +
        target_sum +
        1e-6
    )

    union = (
        pred_sum +
        target_sum -
        intersection
    )

    iou = (
        intersection +
        1e-6
    ) / (
        union +
        1e-6
    )

    return dice, iou


# ============================================================
# Datasets
# ============================================================

train_dataset = CTPatchDataset(
    "train",
    augment=True
)

val_dataset = CTPatchDataset(
    "val",
    augment=False
)

test_dataset = CTPatchDataset(
    "test",
    augment=False
)


print()

print(
    f"Train patches: "
    f"{len(train_dataset)}"
)

print(
    f"Validation patches: "
    f"{len(val_dataset)}"
)

print(
    f"Test patches: "
    f"{len(test_dataset)}"
)


# ============================================================
# Weighted sampling
# ============================================================

sample_weights = []

for path in train_dataset.files:

    mask = np.load(
        train_dataset.mask_dir /
        path.name
    )

    foreground = np.count_nonzero(
        mask
    )

    weight = 1.0 + min(
        foreground / 1000.0,
        3.0
    )

    sample_weights.append(
        weight
    )


sampler = WeightedRandomSampler(
    torch.DoubleTensor(
        sample_weights
    ),
    num_samples=len(
        sample_weights
    ),
    replacement=True
)


# ============================================================
# DataLoaders
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# Model
# ============================================================

model = UNet(
    in_channels=1,
    out_channels=1
)

model = model.to(
    DEVICE
)


print()

print(
    f"Model parameters: "
    f"{sum(p.numel() for p in model.parameters()):,}"
)


# ============================================================
# Optimizer
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2
)

criterion = CombinedLoss(
    POS_WEIGHT
)


# ============================================================
# Evaluation
# ============================================================

def evaluate(
    model,
    loader
):

    model.eval()

    total_loss = 0.0

    all_logits = []

    all_targets = []

    with torch.no_grad():

        for images, masks in loader:

            images = images.to(
                DEVICE
            )

            masks = masks.to(
                DEVICE
            )

            logits = model(
                images
            )

            loss = criterion(
                logits,
                masks
            )

            total_loss += (
                loss.item() *
                images.size(0)
            )

            all_logits.append(
                logits
            )

            all_targets.append(
                masks
            )

    logits = torch.cat(
        all_logits,
        dim=0
    )

    targets = torch.cat(
        all_targets,
        dim=0
    )

    dice, iou = calculate_metrics(
        logits,
        targets
    )

    average_loss = (
        total_loss /
        len(loader.dataset)
    )

    return (
        average_loss,
        dice,
        iou
    )


# ============================================================
# Training
# ============================================================

best_dice = -1.0


print()

print("=" * 60)

print(
    "STARTING IMPROVED TRAINING"
)

print("=" * 60)


for epoch in range(
    1,
    EPOCHS + 1
):

    model.train()

    running_loss = 0.0

    for images, masks in train_loader:

        images = images.to(
            DEVICE
        )

        masks = masks.to(
            DEVICE
        )

        optimizer.zero_grad()

        logits = model(
            images
        )

        loss = criterion(
            logits,
            masks
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )

        optimizer.step()

        running_loss += (
            loss.item() *
            images.size(0)
        )

    train_loss = (
        running_loss /
        len(train_loader.dataset)
    )

    val_loss, val_dice, val_iou = evaluate(
        model,
        val_loader
    )

    scheduler.step(
        val_dice
    )

    current_lr = (
        optimizer.param_groups[0]["lr"]
    )

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Dice: {val_dice:.4f} | "
        f"IoU: {val_iou:.4f} | "
        f"LR: {current_lr:.6f}"
    )

    if val_dice > best_dice:

        best_dice = val_dice

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "epoch":
                    epoch,

                "val_dice":
                    val_dice,

                "val_iou":
                    val_iou,

                "patch_size":
                    256,

                "input_size":
                    INPUT_SIZE
            },
            BEST_MODEL
        )

        print(
            f"  -> Best model saved "
            f"(Dice={val_dice:.4f})"
        )


# ============================================================
# Best Model
# ============================================================

print()

print("=" * 60)

print(
    "LOADING BEST MODEL"
)

print("=" * 60)


checkpoint = torch.load(
    BEST_MODEL,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint[
        "model_state_dict"
    ]
)


print(
    f"Best epoch: "
    f"{checkpoint['epoch']}"
)

print(
    f"Best validation Dice: "
    f"{checkpoint['val_dice']:.4f}"
)

print(
    f"Best validation IoU: "
    f"{checkpoint['val_iou']:.4f}"
)


# ============================================================
# Test
# ============================================================

test_loss, test_dice, test_iou = evaluate(
    model,
    test_loader
)


print()

print("=" * 60)

print(
    "FINAL TEST RESULTS"
)

print("=" * 60)

print(
    f"Test Loss: "
    f"{test_loss:.4f}"
)

print(
    f"Test Dice: "
    f"{test_dice:.4f}"
)

print(
    f"Test IoU: "
    f"{test_iou:.4f}"
)

print("=" * 60)

print()

print(
    "Best model:"
)

print(
    BEST_MODEL
)

print()

print(
    "Improved CT patch training completed."
)