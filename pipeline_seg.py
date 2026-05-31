"""
pipeline_seg.py — Подготовка датасета для YOLOv8-Seg.

Читает DICOM-серии и .seg.nrrd маски из папки курсовой работы,
извлекает 2D-срезы с разметкой и конвертирует маски позвонков 
в полигоны формата YOLO-Seg.

Структура входных данных:
    курсовая/
    ├── 1-10db/
    │   ├── ARIFULINA/
    │   │   ├── IMG-0001/DICOM/    # Проекция 1 (корональная)
    │   │   ├── IMG-0002/DICOM/    # Проекция 2 (сагиттальная)
    │   │   ├── Segmentation.seg.nrrd    # Маска для одной проекции
    │   │   └── Segmentation_1.seg.nrrd  # Маска для другой проекции
    │   └── ...
    └── ...

Связка маска↔DICOM определяется по совпадению Spacing.
"""

import os
import sys
import glob
import shutil
import random
import numpy as np
import SimpleITK as sitk
import cv2


# ─── Настройки ──────────────────────────────────────────────────
DATA_ROOT = "/Users/birukovartemij/Desktop/Уник/курсовая"
OUTPUT_DIR = "/Users/birukovartemij/Documents/Antigravity/spine_seg_project/dataset_seg"
TRAIN_RATIO = 0.8
RANDOM_SEED = 42
# ────────────────────────────────────────────────────────────────


def find_dicom_series(dicom_dir):
    """Читает DICOM-серию из папки. Возвращает (sitk_image, spacing_tuple)."""
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(dicom_dir)
    if not series_ids:
        return None, None
    files = reader.GetGDCMSeriesFileNames(dicom_dir, series_ids[0])
    reader.SetFileNames(files)
    vol = reader.Execute()
    spacing = tuple(round(s, 4) for s in vol.GetSpacing())
    return vol, spacing


def match_mask_to_dicom(patient_dir):
    """
    Связывает каждую .seg.nrrd маску с соответствующей DICOM-серией
    по совпадению Spacing.
    
    Returns: list of (dicom_volume, mask_array, patient_name, projection_id)
    """
    patient_name = os.path.basename(patient_dir)
    
    # Собираем все DICOM-серии
    dicom_map = {}  # spacing -> (volume, img_dir_name)
    for img_dir in sorted(os.listdir(patient_dir)):
        if not img_dir.startswith("IMG-"):
            continue
        dicom_path = os.path.join(patient_dir, img_dir, "DICOM")
        if not os.path.isdir(dicom_path):
            continue
        vol, spacing = find_dicom_series(dicom_path)
        if vol is not None:
            dicom_map[spacing] = (vol, img_dir)
    
    # Собираем все маски
    pairs = []
    for f in sorted(os.listdir(patient_dir)):
        if not f.endswith(".seg.nrrd"):
            continue
        mask_path = os.path.join(patient_dir, f)
        mask_img = sitk.ReadImage(mask_path)
        mask_spacing = tuple(round(s, 4) for s in mask_img.GetSpacing())
        mask_arr = sitk.GetArrayFromImage(mask_img)  # (slices, H, W)
        
        # Ищем подходящий DICOM по spacing
        matched_vol = None
        matched_dir = f
        for sp, (vol, dirname) in dicom_map.items():
            if sp == mask_spacing:
                matched_vol = vol
                matched_dir = dirname
                break
        
        if matched_vol is None:
            # Fallback: совпадение по размеру
            for sp, (vol, dirname) in dicom_map.items():
                if vol.GetSize() == mask_img.GetSize():
                    matched_vol = vol
                    matched_dir = dirname
                    break
        
        if matched_vol is not None:
            pairs.append((matched_vol, mask_arr, patient_name, matched_dir))
        else:
            print(f"  ⚠ Не найден DICOM для маски {f} пациента {patient_name}")
    
    return pairs


def mask_to_yolo_polygons(single_vertebra_mask, img_h, img_w):
    """
    Конвертирует бинарную маску одного позвонка в нормализованный 
    YOLO-Seg полигон.
    """
    binary = (single_vertebra_mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Берем самый большой контур
    contour = max(contours, key=cv2.contourArea)
    
    if cv2.contourArea(contour) < 50:  # Слишком маленький
        return None
    
    # Упрощаем контур
    epsilon = 0.01 * cv2.arcLength(contour, True)
    contour = cv2.approxPolyDP(contour, epsilon, True)
    
    if len(contour) < 3:
        return None
    
    # Нормализуем координаты [0, 1]
    points = contour.reshape(-1, 2).astype(float)
    points[:, 0] /= img_w
    points[:, 1] /= img_h
    
    # Clamp
    points = np.clip(points, 0.0, 1.0)
    
    return points


def normalize_slice(dicom_slice):
    """Нормализует 2D-срез DICOM в uint8 изображение."""
    arr = dicom_slice.astype(float)
    
    # Обрезаем выбросы по перцентилям
    p1, p99 = np.percentile(arr, [1, 99])
    arr = np.clip(arr, p1, p99)
    
    # В [0, 255]
    if arr.max() > arr.min():
        arr = (arr - arr.min()) / (arr.max() - arr.min()) * 255
    
    return arr.astype(np.uint8)


def process_all_patients():
    """Главная функция: обходит все папки, извлекает срезы, сохраняет датасет."""
    
    # Очистка выходной папки
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    
    for split in ["train", "val"]:
        os.makedirs(os.path.join(OUTPUT_DIR, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, split, "labels"), exist_ok=True)
    
    # Собираем все папки пациентов
    all_patient_dirs = []
    for batch_dir in sorted(glob.glob(os.path.join(DATA_ROOT, "*db"))):
        if not os.path.isdir(batch_dir):
            continue
        for patient in sorted(os.listdir(batch_dir)):
            patient_path = os.path.join(batch_dir, patient)
            if os.path.isdir(patient_path) and not patient.startswith("."):
                all_patient_dirs.append(patient_path)
    
    print(f"╔{'═'*58}╗")
    print(f"║  ПОДГОТОВКА ДАТАСЕТА YOLO-SEG                            ║")
    print(f"╠{'═'*58}╣")
    print(f"║  Найдено пациентов: {len(all_patient_dirs):<38}║")
    print(f"╚{'═'*58}╝\n")
    
    # Перемешиваем и делим на train/val
    random.seed(RANDOM_SEED)
    random.shuffle(all_patient_dirs)
    split_idx = int(len(all_patient_dirs) * TRAIN_RATIO)
    train_patients = all_patient_dirs[:split_idx]
    val_patients = all_patient_dirs[split_idx:]
    
    print(f"  Train: {len(train_patients)} пациентов")
    print(f"  Val:   {len(val_patients)} пациентов\n")
    
    total_images = 0
    total_vertebrae = 0
    errors = []
    
    for split_name, patients in [("train", train_patients), ("val", val_patients)]:
        print(f"── {split_name.upper()} ──")
        for patient_dir in patients:
            patient_name = os.path.basename(patient_dir)
            
            try:
                pairs = match_mask_to_dicom(patient_dir)
            except Exception as e:
                err_msg = f"  ❌ {patient_name}: {e}"
                print(err_msg)
                errors.append(err_msg)
                continue
            
            if not pairs:
                print(f"  ⚠ {patient_name}: нет пар DICOM-маска")
                continue
            
            for dicom_vol, mask_arr, pname, proj_id in pairs:
                dicom_arr = sitk.GetArrayFromImage(dicom_vol)  # (slices, H, W)
                
                # Ищем срезы с разметкой
                for s_idx in range(mask_arr.shape[0]):
                    mask_slice = mask_arr[s_idx]
                    if not np.any(mask_slice > 0):
                        continue
                    
                    # Получаем соответствующий DICOM-срез
                    if s_idx >= dicom_arr.shape[0]:
                        continue
                    
                    dicom_slice = dicom_arr[s_idx]
                    img_h, img_w = dicom_slice.shape
                    
                    # Нормализуем
                    img_uint8 = normalize_slice(dicom_slice)
                    
                    # Сохраняем изображение (BGR для OpenCV, но grayscale → 3ch)
                    img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2BGR)
                    
                    # Формируем имя
                    safe_name = pname.replace(" ", "_").replace(".", "")
                    file_id = f"{safe_name}_{proj_id}_s{s_idx:02d}"
                    
                    img_path = os.path.join(OUTPUT_DIR, split_name, "images", f"{file_id}.jpg")
                    lbl_path = os.path.join(OUTPUT_DIR, split_name, "labels", f"{file_id}.txt")
                    
                    cv2.imwrite(img_path, img_bgr)
                    
                    # Извлекаем полигоны для каждого позвонка
                    label_lines = []
                    unique_labels = np.unique(mask_slice)
                    unique_labels = unique_labels[unique_labels > 0]
                    
                    for label_val in unique_labels:
                        single_mask = (mask_slice == label_val).astype(np.uint8)
                        polygon = mask_to_yolo_polygons(single_mask, img_h, img_w)
                        
                        if polygon is not None:
                            # class_id = 0 (один класс — vertebra)
                            coords_str = " ".join(f"{x:.6f} {y:.6f}" for x, y in polygon)
                            label_lines.append(f"0 {coords_str}")
                            total_vertebrae += 1
                    
                    if label_lines:
                        with open(lbl_path, "w") as f:
                            f.write("\n".join(label_lines))
                        total_images += 1
            
            print(f"  ✅ {patient_name}")
    
    print(f"\n{'═'*60}")
    print(f"  ИТОГО:")
    print(f"  Снимков: {total_images}")
    print(f"  Позвонков: {total_vertebrae}")
    if errors:
        print(f"  Ошибки ({len(errors)}):")
        for e in errors:
            print(f"    {e}")
    print(f"{'═'*60}")
    
    # Подсчитываем файлы по сплитам
    for split in ["train", "val"]:
        n_imgs = len(glob.glob(os.path.join(OUTPUT_DIR, split, "images", "*.jpg")))
        n_lbls = len(glob.glob(os.path.join(OUTPUT_DIR, split, "labels", "*.txt")))
        print(f"  {split}: {n_imgs} images, {n_lbls} labels")


if __name__ == "__main__":
    process_all_patients()
