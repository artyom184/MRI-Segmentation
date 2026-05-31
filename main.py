"""
main_maskrcnn.py — Запуск анализа деформаций через Mask R-CNN.

Загружает обученную Mask R-CNN модель, получает маски позвонков,
передаёт их в тот же самый алгоритм deformity_analysis.py.
"""

import os
import sys
import glob
import numpy as np
import cv2
import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

# Наш алгоритм деформаций (тот же самый)
from deformity_analysis import analyze_single_vertebra


# ─── Настройки ───
WEIGHTS_PATH = "/Users/birukovartemij/Documents/Antigravity/spine_seg_project/runs/maskrcnn/weights/best_maskrcnn.pt"
REPORTS_DIR = "/Users/birukovartemij/Documents/Antigravity/spine_seg_project/reports_maskrcnn"
IMG_SIZE = 512
NUM_CLASSES = 2
CONF_THRESHOLD = 0.5
DEVICE = "cpu"
PIXEL_SPACING_MM = 0.677


def load_model():
    """Загружает обученную Mask R-CNN."""
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    model = maskrcnn_resnet50_fpn(weights=None, weights_backbone=None)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, NUM_CLASSES)
    model.roi_heads.detections_per_img = 8

    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()

    return model


def analyze_with_maskrcnn(image_path, model, output_path=None):
    """
    Анализ одного МРТ-снимка через Mask R-CNN + наш алгоритм деформаций.
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"❌ Не удалось загрузить: {image_path}")
        return

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = img_rgb.shape[:2]

    # Resize и нормализация
    img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    img_tensor = torch.from_numpy(img_resized).float().permute(2, 0, 1) / 255.0
    img_tensor = img_tensor.to(DEVICE)

    # Инференс
    with torch.no_grad():
        predictions = model([img_tensor])[0]

    # Фильтруем по confidence
    scores = predictions["scores"].cpu().numpy()
    masks = predictions["masks"].cpu().numpy()
    boxes = predictions["boxes"].cpu().numpy()

    keep = scores >= CONF_THRESHOLD
    scores = scores[keep]
    masks = masks[keep]
    boxes = boxes[keep]

    if len(masks) == 0:
        print(f"⚠ Позвонки не обнаружены на {os.path.basename(image_path)}")
        return

    # Работаем в пространстве IMG_SIZE x IMG_SIZE
    vis_img = img_resized.copy()
    vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)

    # Сортировка сверху вниз
    centers_y = []
    for i, mask in enumerate(masks):
        binary = (mask[0] > 0.5).astype(np.uint8)
        ys = np.where(binary > 0)[0]
        cy = np.mean(ys) if len(ys) > 0 else 0
        centers_y.append((cy, i))
    centers_y.sort()
    sorted_indices = [idx for _, idx in centers_y]

    print(f"\n{'='*60}")
    print(f"  ОТЧЕТ О ДЕФОРМАЦИЯХ (Mask R-CNN)")
    print(f"  Снимок: {os.path.basename(image_path)}")
    print(f"  Обнаружено позвонков: {len(masks)}")
    print(f"{'='*60}\n")

    colors = [
        (50, 205, 50), (255, 191, 0), (0, 165, 255),
        (203, 92, 255), (0, 255, 255), (255, 105, 105), (100, 255, 100),
    ]

    for rank, orig_idx in enumerate(sorted_indices):
        mask = masks[orig_idx]
        binary = (mask[0] > 0.5).astype(np.uint8) * 255

        color = colors[rank % len(colors)]
        label = f"V{rank + 1}"

        landmarks, genant = analyze_single_vertebra(binary, PIXEL_SPACING_MM)

        if landmarks is None:
            print(f"  {label}: ⚠ Не удалось проанализировать")
            continue

        # Визуализация
        overlay = vis_img.copy()
        overlay[binary > 127] = color
        vis_img = cv2.addWeighted(vis_img, 0.7, overlay, 0.3, 0)

        for name, (x, y) in landmarks.items():
            cv2.circle(vis_img, (x, y), 4, (0, 0, 255), -1)
            cv2.circle(vis_img, (x, y), 4, (255, 255, 255), 1)

        cv2.line(vis_img, landmarks['Au'], landmarks['Al'], (0, 255, 255), 2)
        cv2.line(vis_img, landmarks['Pu'], landmarks['Pl'], (255, 0, 255), 2)
        cv2.line(vis_img, landmarks['Mu'], landmarks['Ml'], (0, 255, 0), 2)

        print(f"  {label} (conf={scores[orig_idx]:.2f}):")
        print(f"    Ah={genant['Ah']:.1f}mm  Mh={genant['Mh']:.1f}mm  Ph={genant['Ph']:.1f}mm")
        print(f"    Wedge:     {genant['wedge_pct']:.1f}% → {genant['wedge_grade_name']}")
        print(f"    Biconcave: {genant['biconcave_pct']:.1f}% → {genant['biconcave_grade_name']}")
        print()

    if output_path is None:
        os.makedirs(REPORTS_DIR, exist_ok=True)
        base = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(REPORTS_DIR, f"{base}_report.jpg")

    cv2.imwrite(output_path, vis_img)
    print(f"📊 Отчет сохранен: {output_path}")


def main():
    model = load_model()

    if len(sys.argv) > 1:
        path = sys.argv[1]
        if os.path.isdir(path):
            images = sorted(glob.glob(os.path.join(path, "*.jpg")))
        else:
            images = [path]
    else:
        # По умолчанию — валидационный набор
        val_dir = "/Users/birukovartemij/Documents/Antigravity/spine_seg_project/dataset_seg/val/images"
        images = sorted(glob.glob(os.path.join(val_dir, "*.jpg")))

    print(f"Анализ {len(images)} снимков...\n")

    for i, img_path in enumerate(images):
        print(f"── Снимок {i+1}/{len(images)}: {os.path.basename(img_path)} ──")
        analyze_with_maskrcnn(img_path, model)
        print()

    print(f"\n{'='*60}")
    print(f"✅ Все отчёты сохранены в: {REPORTS_DIR}")


if __name__ == "__main__":
    main()
