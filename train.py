"""
Fine-tuning Script for DenseNet121 Deepfake Detector.

Three-phase training strategy:
Phase 1: Frozen backbone, train head (10 epochs, lr=1e-3)
Phase 2: Unfreeze last 20 layers (20 epochs, lr=1e-4)
Phase 3: Full fine-tune (10 epochs, lr=1e-5)

Usage:
    python train.py --real-dir /path/to/real --fake-dir /path/to/fake

Dataset structure:
    data/
    ├── real/
    │   ├── image001.jpg
    │   └── ...
    └── fake/
        ├── image001.jpg
        └── ...
"""
import os
import sys
import argparse
import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DeepfakeDataset(Dataset):
    """Dataset for real/fake images."""

    EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}

    def __init__(self, real_dir: str, fake_dir: str, transform=None):
        self.transform = transform
        self.samples = []

        # Collect real images (label = 0)
        real_count = 0
        for path in Path(real_dir).rglob('*'):
            if path.suffix.lower() in self.EXTENSIONS:
                self.samples.append((str(path), 0))
                real_count += 1

        # Collect fake images (label = 1)
        fake_count = 0
        for path in Path(fake_dir).rglob('*'):
            if path.suffix.lower() in self.EXTENSIONS:
                self.samples.append((str(path), 1))
                fake_count += 1

        logger.info(f"Dataset: {real_count} real + {fake_count} fake = {len(self.samples)} total")

        if len(self.samples) == 0:
            raise ValueError(f"No images found in {real_dir} or {fake_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            image = Image.open(path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, torch.tensor(label, dtype=torch.float32)
        except Exception as e:
            logger.warning(f"Failed to load {path}: {e}")
            # Return a blank image with the same label
            blank = torch.zeros(3, 224, 224)
            return blank, torch.tensor(label, dtype=torch.float32)


def build_transforms(augment: bool = True):
    """Build data transforms."""
    if augment:
        train_transform = T.Compose([
            T.Resize(256),
            T.RandomCrop(224),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=20),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        train_transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    val_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_transform, val_transform


def build_model():
    """Build DenseNet121 with custom binary head."""
    model = models.densenet121(weights='IMAGENET1K_V1')
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, 1),
        nn.Sigmoid()
    )
    return model


def freeze_backbone(model):
    """Freeze all layers except the classifier."""
    for name, param in model.named_parameters():
        if 'classifier' not in name:
            param.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Frozen backbone: {trainable:,} trainable parameters (head only)")


def unfreeze_last_n_layers(model, n: int = 20):
    """Unfreeze the last N layers of DenseNet."""
    all_params = list(model.named_parameters())
    # Unfreeze last n + classifier
    threshold = max(0, len(all_params) - n)
    for i, (name, param) in enumerate(all_params):
        if 'classifier' in name or i >= threshold:
            param.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Unfroze last {n} layers: {trainable:,} trainable parameters")


def unfreeze_all(model):
    """Unfreeze all layers."""
    for param in model.parameters():
        param.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"All layers unfrozen: {trainable:,} trainable parameters")


def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch. Returns avg loss and accuracy."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = (outputs > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if (batch_idx + 1) % 20 == 0:
            logger.info(f"  Batch {batch_idx+1}/{len(loader)}: loss={loss.item():.4f}")

    return total_loss / len(loader), correct / total


def val_epoch(model, loader, criterion, device):
    """Validate for one epoch. Returns avg loss and accuracy."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = (outputs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(loader), correct / total


def save_checkpoint(model, path: str, metrics: dict):
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    logger.info(f"Saved checkpoint to {path} (acc={metrics.get('val_acc', 0):.4f})")


def train(args):
    """Main training function."""
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Training on: {device}")

    # Build datasets
    train_transform, val_transform = build_transforms(augment=True)
    full_dataset = DeepfakeDataset(args.real_dir, args.fake_dir)

    # Split 80/20
    val_size = max(1, int(0.2 * len(full_dataset)))
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    train_ds.dataset.transform = train_transform
    val_ds.dataset.transform = val_transform

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    logger.info(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    # Build model
    model = build_model().to(device)
    criterion = nn.BCELoss()
    best_val_acc = 0.0

    # ============================================================
    # Phase 1: Train head only (frozen backbone)
    # ============================================================
    logger.info("\n" + "="*50)
    logger.info("PHASE 1: Training head (frozen backbone)")
    logger.info("="*50)
    freeze_backbone(model)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.phase1_epochs)

    for epoch in range(args.phase1_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = val_epoch(model, val_loader, criterion, device)
        scheduler.step()
        logger.info(f"Epoch {epoch+1}/{args.phase1_epochs}: "
                    f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                    f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, args.output, {'val_acc': val_acc})

    # ============================================================
    # Phase 2: Fine-tune last 20 layers
    # ============================================================
    logger.info("\n" + "="*50)
    logger.info("PHASE 2: Fine-tuning last 20 layers")
    logger.info("="*50)
    unfreeze_last_n_layers(model, n=20)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.phase2_epochs)

    for epoch in range(args.phase2_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = val_epoch(model, val_loader, criterion, device)
        scheduler.step()
        logger.info(f"Epoch {epoch+1}/{args.phase2_epochs}: "
                    f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                    f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, args.output, {'val_acc': val_acc})

    # ============================================================
    # Phase 3: Full fine-tune
    # ============================================================
    logger.info("\n" + "="*50)
    logger.info("PHASE 3: Full model fine-tuning")
    logger.info("="*50)
    unfreeze_all(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.phase3_epochs)

    for epoch in range(args.phase3_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = val_epoch(model, val_loader, criterion, device)
        scheduler.step()
        logger.info(f"Epoch {epoch+1}/{args.phase3_epochs}: "
                    f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                    f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, args.output, {'val_acc': val_acc})

    logger.info(f"\n✅ Training complete! Best val accuracy: {best_val_acc:.4f}")
    logger.info(f"   Model saved to: {args.output}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fine-tune DenseNet121 for deepfake detection')
    parser.add_argument('--real-dir', required=True, help='Directory with real images')
    parser.add_argument('--fake-dir', required=True, help='Directory with fake images')
    parser.add_argument('--output', default='weights/densenet_deepfake.pth', help='Output weights path')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--phase1-epochs', type=int, default=10, help='Phase 1 epochs')
    parser.add_argument('--phase2-epochs', type=int, default=20, help='Phase 2 epochs')
    parser.add_argument('--phase3-epochs', type=int, default=10, help='Phase 3 epochs')
    args = parser.parse_args()
    train(args)
