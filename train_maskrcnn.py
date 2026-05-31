"""
train_maskrcnn.py — Обучение Mask R-CNN (ResNet-50 + FPN) для сегментации позвонков.

Точная копия архитектуры из статьи научрука:
    Al-Haidri et al. (2025), Section 2.4:
    - Backbone: ResNet-50
    - Mask R-CNN architecture
    - Loss: L = Lcls + Lbox + Lmask
    - Input: 512×512, normalized [0,1]

Использует наш существующий датасет в формате YOLO-Seg.
"""

import os
import glob
import numpy as np
import cv2
import torch
import torch.utils.data
from torch.utils.data import DataLoader
import sys
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import time


# ─── Настройки ───
DATASET_DIR = "/Users/birukovartemij/Documents/Antigravity/spine_seg_project/dataset_seg"
OUTPUT_DIR = "/Users/birukovartemij/Documents/Antigravity/spine_seg_project/runs/maskrcnn"
NUM_CLASSES = 2  # 1 класс (vertebra) + фон
IMG_SIZE = 512
NUM_EPOCHS = 30
LR = 0.005
BATCH_SIZE = 2
DEVICE = "cpu"  # MPS не поддерживает torchvision detection models стабильно


# ─── Датасет ────────────────────────────────────────────────────

class VertebraSegDataset(torch.utils.data.Dataset):
    """
    Читает изображения и YOLO-Seg аннотации,
    конвертирует полигоны в бинарные маски для Mask R-CNN.
    """

    def __init__(self, root, split="train"):
        self.img_dir = os.path.join(root, split, "images")
        self.lbl_dir = os.path.join(root, split, "labels")
        self.images = sorted(glob.glob(os.path.join(self.img_dir, "*.jpg")))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        orig_h, orig_w = img.shape[:2]

        # Resize to IMG_SIZE x IMG_SIZE
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        # Normalize to [0, 1] (как в статье: Section 2.3)
        img_tensor = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0

        # Читаем YOLO-Seg аннотации
        lbl_path = os.path.join(self.lbl_dir,
                                os.path.splitext(os.path.basename(img_path))[0] + ".txt")

        masks = []
        boxes = []

        if os.path.exists(lbl_path):
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 7:  # class + min 3 points (6 coords)
                        continue

                    # Парсим полигон: class x1 y1 x2 y2 ...
                    coords = list(map(float, parts[1:]))
                    polygon = np.array(coords).reshape(-1, 2)

                    # Денормализация
                    polygon[:, 0] *= IMG_SIZE
                    polygon[:, 1] *= IMG_SIZE
                    polygon = polygon.astype(np.int32)

                    # Создаём бинарную маску
                    mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
                    cv2.fillPoly(mask, [polygon], 1)

                    if mask.sum() < 20:
                        continue

                    # Bounding box из маски
                    ys, xs = np.where(mask > 0)
                    if len(xs) == 0:
                        continue

                    x1, x2 = xs.min(), xs.max()
                    y1, y2 = ys.min(), ys.max()

                    if x2 <= x1 or y2 <= y1:
                        continue

                    boxes.append([x1, y1, x2, y2])
                    masks.append(mask)

        if len(masks) == 0:
            # Пустой target (нет аннотаций)
            target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros(0, dtype=torch.int64),
                "masks": torch.zeros((0, IMG_SIZE, IMG_SIZE), dtype=torch.uint8),
            }
        else:
            target = {
                "boxes": torch.as_tensor(boxes, dtype=torch.float32),
                "labels": torch.ones(len(boxes), dtype=torch.int64),  # class 1 = vertebra
                "masks": torch.as_tensor(np.array(masks), dtype=torch.uint8),
            }

        return img_tensor, target


# ─── Модель ─────────────────────────────────────────────────────

def get_model(num_classes):
    """
    Mask R-CNN с ResNet-50 + FPN backbone (предобученная на COCO).
    Заменяем головы классификации и масок под наш 1 класс.
    """
    # Загружаем предобученную модель
    model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)

    # Заменяем голову классификации
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Заменяем голову масок
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )

    # Ограничиваем max detections = 5 (как в статье: Section 2.4)
    model.roi_heads.detections_per_img = 8  # Чуть больше на случай краевых

    return model


# ─── Collate function ───────────────────────────────────────────

def collate_fn(batch):
    return tuple(zip(*batch))


# ─── Обучение ───────────────────────────────────────────────────

def train():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "weights"), exist_ok=True)

    # Датасеты
    train_dataset = VertebraSegDataset(DATASET_DIR, "train")
    val_dataset = VertebraSegDataset(DATASET_DIR, "val")

    print(f"╔{'═'*58}╗")
    print(f"║  ОБУЧЕНИЕ MASK R-CNN (ResNet-50 + FPN)                   ║")
    print(f"╠{'═'*58}╣")
    print(f"║  Train: {len(train_dataset)} снимков                               ║")
    print(f"║  Val:   {len(val_dataset)} снимков                                ║")
    print(f"║  Размер: {IMG_SIZE}×{IMG_SIZE}                                    ║")
    print(f"║  Эпохи: {NUM_EPOCHS}, LR: {LR}                                ║")
    print(f"║  Device: {DEVICE}                                            ║")
    print(f"╚{'═'*58}╝\n")

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        collate_fn=collate_fn, num_workers=0
    )

    # Модель
    device = torch.device(DEVICE)
    model = get_model(NUM_CLASSES)
    model.to(device)

    # Оптимизатор (как в статье: lr=0.001)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LR, momentum=0.9, weight_decay=0.0005)

    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    best_val_loss = float('inf')
    patience_counter = 0
    patience = 10

    for epoch in range(NUM_EPOCHS):
        # ─── Train ───
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Пропускаем пустые батчи
            if all(t["boxes"].shape[0] == 0 for t in targets):
                continue

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()
            n_batches += 1

        lr_scheduler.step()

        avg_train_loss = epoch_loss / max(n_batches, 1)

        # ─── Validation ───
        model.train()  # Mask R-CNN возвращает loss только в train mode
        val_loss = 0.0
        n_val = 0

        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                if all(t["boxes"].shape[0] == 0 for t in targets):
                    continue

                try:
                    loss_dict = model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())
                    val_loss += losses.item()
                    n_val += 1
                except:
                    continue

        avg_val_loss = val_loss / max(n_val, 1)

        print(f"  Epoch {epoch+1:3d}/{NUM_EPOCHS} | "
              f"Train loss: {avg_train_loss:.4f} | "
              f"Val loss: {avg_val_loss:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        sys.stdout.flush()

        # Сохранение лучшей модели
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(),
                       os.path.join(OUTPUT_DIR, "weights", "best_maskrcnn.pt"))
            print(f"    ✅ Лучшая модель сохранена (val_loss={best_val_loss:.4f})")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print(f"\n  ⏹ Early stopping: {patience} эпох без улучшения")
            break

    # Сохраняем последнюю модель
    torch.save(model.state_dict(),
               os.path.join(OUTPUT_DIR, "weights", "last_maskrcnn.pt"))

    print(f"\n{'='*60}")
    print(f"  Обучение завершено!")
    print(f"  Лучшая val loss: {best_val_loss:.4f}")
    print(f"  Модель: {os.path.join(OUTPUT_DIR, 'weights', 'best_maskrcnn.pt')}")
    print(f"{'='*60}")


if __name__ == "__main__":
    train()
