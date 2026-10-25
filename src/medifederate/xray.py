
import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from .model import build_resnet18, save_checkpoint


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def loader(path, image_size=160, batch_size=8, train=False):
    """Load X-ray images from NORMAL/PNEUMONIA class folders."""

    ops = [
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((image_size, image_size)),
    ]

    if train:
        ops.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(7),
        ])

    ops.extend([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    ds = datasets.ImageFolder(
        path,
        transform=transforms.Compose(ops),
    )

    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=train,
        num_workers=0,
        pin_memory=(device_is_cuda := torch.cuda.is_available()),
    )

    return dl, ds


def evaluate(model, dl, device):
    """Calculate binary classification metrics."""

    ys = []
    ps = []
    pr = []
    loss_sum = 0.0
    n = 0

    criterion = nn.CrossEntropyLoss()
    model.eval()

    with torch.no_grad():
        for x, y in dl:
            x = x.to(device)
            y_device = y.to(device)

            out = model(x)
            loss = criterion(out, y_device)

            probabilities = torch.softmax(out, dim=1)[:, 1]

            ys.extend(y.tolist())
            ps.extend(out.argmax(dim=1).cpu().tolist())
            pr.extend(probabilities.cpu().tolist())

            loss_sum += loss.item() * len(y)
            n += len(y)

    if n == 0:
        raise ValueError("The evaluation dataset contains no images.")

    tn, fp, fn, tp = map(
        int,
        confusion_matrix(ys, ps, labels=[0, 1]).ravel(),
    )

    return {
        "loss": loss_sum / n,
        "accuracy": float(accuracy_score(ys, ps)),
        "precision": float(
            precision_score(ys, ps, zero_division=0)
        ),
        "recall": float(
            recall_score(ys, ps, zero_division=0)
        ),
        "f1": float(
            f1_score(ys, ps, zero_division=0)
        ),
        "roc_auc": (
            float(roc_auc_score(ys, pr))
            if len(set(ys)) == 2
            else None
        ),
        "sensitivity": (
            float(tp / (tp + fn))
            if tp + fn
            else 0.0
        ),
        "specificity": (
            float(tn / (tn + fp))
            if tn + fp
            else 0.0
        ),
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "class_names": dl.dataset.classes,
        "num_samples": n,
    }


def train_resnet(
    train_dir,
    val_dir,
    checkpoint,
    epochs=5,
    batch_size=8,
    lr=1e-3,
    device="cpu",
    patience=2,
):
    """Train a lightweight ResNet18 classifier using transfer learning."""

    tr, tr_ds = loader(
        train_dir,
        image_size=160,
        batch_size=batch_size,
        train=True,
    )

    va, va_ds = loader(
        val_dir,
        image_size=160,
        batch_size=batch_size,
        train=False,
    )

    if len(tr_ds) == 0:
        raise ValueError(f"No training images found in: {train_dir}")

    if len(va_ds) == 0:
        raise ValueError(f"No validation images found in: {val_dir}")

    if tr_ds.classes != va_ds.classes:
        raise ValueError(
            "Training and validation classes do not match. "
            f"Train: {tr_ds.classes}; Validation: {va_ds.classes}"
        )

    print(f"Training images: {len(tr_ds)}")
    print(f"Validation images: {len(va_ds)}")
    print(f"Classes: {tr_ds.classes}")
    print(f"Device: {device}")

    # Load pretrained ResNet18.
    model = build_resnet18(
        num_classes=2,
        pretrained=True,
    ).to(device)

    # Freeze the pretrained feature extractor.
    for param in model.parameters():
        param.requires_grad = False

    # Train only the final classification layer.
    for param in model.fc.parameters():
        param.requires_grad = True

    # Keep frozen backbone layers in evaluation mode so their
    # BatchNorm running statistics do not change during training.
    criterion_counts = torch.bincount(
        torch.tensor(tr_ds.targets),
        minlength=2,
    ).float()

    class_weights = (
        criterion_counts.sum()
        / (2 * criterion_counts.clamp_min(1))
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(
        filter(
            lambda param: param.requires_grad,
            model.parameters(),
        ),
        lr=lr,
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.3,
        patience=1,
        min_lr=1e-6,
    )

    best_f1 = -1.0
    bad_epochs = 0

    for epoch in range(epochs):
        model.train()

        # Keep the frozen backbone in eval mode.
        model.eval()
        model.fc.train()

        total = 0
        correct = 0
        loss_total = 0.0

        for images, labels in tr:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            batch_count = len(labels)
            loss_total += loss.item() * batch_count
            correct += (
                outputs.argmax(dim=1) == labels
            ).sum().item()
            total += batch_count

        train_loss = loss_total / max(total, 1)
        train_acc = correct / max(total, 1)

        val_metrics = evaluate(model, va, device)
        scheduler.step(val_metrics["f1"])

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_f1={val_metrics['f1']:.4f} "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            bad_epochs = 0

            save_checkpoint(
                model,
                checkpoint,
                tr_ds.classes,
            )

            print(f"Best checkpoint saved: {checkpoint}")

        else:
            bad_epochs += 1

        if bad_epochs >= patience:
            print("Early stopping.")
            break

    print(f"Training finished. Best validation F1: {best_f1:.4f}")
    return best_f1
