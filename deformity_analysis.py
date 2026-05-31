"""
Deformity Analysis v2: Автоматическое определение деформаций позвонков по Genant.

Алгоритм воспроизводит методологию из статьи:
Al-Haidri et al. "Deep learning-assisted framework for automation of lumbar vertebral body 
segmentation, measurement, and deformity detection in MR images"
(Biomedical Signal Processing and Control, 2025)

Изменения в v2:
    - Поворот через cv2.minAreaRect (стабильнее PCA на маленьких масках)
    - Сглаживание гистограмм (Gaussian) для устранения шума
    - Корректная сортировка позвонков сверху вниз
    - Улучшенный поиск центральных точек (Mu, Ml)
    - Увеличен радиус точек и толщина линий для наглядности
"""

import numpy as np
import cv2
from ultralytics import YOLO
import os
import math
from scipy.ndimage import gaussian_filter1d


# ─────────────────────────────────────────────────────────────────
# Шаг 2: Постобработка маски (Section 2.5.1)
# ─────────────────────────────────────────────────────────────────

def postprocess_mask(binary_mask):
    """
    Удаление изолированных пикселей и заполнение пустот.
    + Оставляем только самый большой контур (убираем мусор).
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # Удаление мелкого мусора
    cleaned = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Заполнение дыр
    closed = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Оставляем только самый большой связный компонент
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return closed

    biggest = max(contours, key=cv2.contourArea)
    result = np.zeros_like(closed)
    cv2.drawContours(result, [biggest], -1, 255, cv2.FILLED)

    return result


# ─────────────────────────────────────────────────────────────────
# Шаг 3: Центр масс и поворот (Section 2.5.1)
# ─────────────────────────────────────────────────────────────────

def get_rotation_angle(binary_mask):
    """
    Определяет угол наклона позвонка через minAreaRect.
    Более стабильно, чем PCA, на маленьких и краевых масках.
    """
    points = np.column_stack(np.where(binary_mask > 0))
    if len(points) < 10:
        return 0.0, None

    # Центр масс
    cy = int(np.mean(points[:, 0]))
    cx = int(np.mean(points[:, 1]))

    # minAreaRect для определения угла наклона
    # Формат точек для OpenCV: (x, y)
    pts_cv = points[:, ::-1].astype(np.float32)  # (y,x) -> (x,y)
    rect = cv2.minAreaRect(pts_cv)
    angle = rect[2]
    rect_w, rect_h = rect[1]

    # minAreaRect возвращает угол от -90 до 0
    # Позвонок должен быть шире, чем высок (лежит горизонтально)
    if rect_w < rect_h:
        angle = angle + 90

    return angle, (cx, cy)


def rotate_mask(binary_mask, angle, center):
    """Поворачивает маску вокруг центра масс."""
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    h, w = binary_mask.shape
    rotated = cv2.warpAffine(binary_mask, M, (w, h), flags=cv2.INTER_NEAREST)
    return rotated, M


# ─────────────────────────────────────────────────────────────────
# Шаг 4–5: Поиск 6 ключевых точек (Section 2.5.2)
# ─────────────────────────────────────────────────────────────────

def find_six_landmarks(binary_mask):
    """
    Находит 6 ключевых точек позвонка (Au, Al, Pu, Pl, Mu, Ml)
    по алгоритму распределения пикселей маски.

    Улучшения v2:
        - Гауссово сглаживание гистограммы (убирает шум пикселей)
        - Поиск центральных точек через настоящий локальный минимум
        - Защита от выхода за границы маски
    """
    points = np.column_stack(np.where(binary_mask > 0))
    if len(points) < 50:
        return None

    # Центр масс
    cy_mass = int(np.mean(points[:, 0]))
    cx_mass = int(np.mean(points[:, 1]))

    h, w = binary_mask.shape

    # Разделяем маску на верхнюю и нижнюю части по центру масс
    upper_mask = binary_mask[:cy_mass, :]
    lower_mask = binary_mask[cy_mass:, :]

    if upper_mask.size == 0 or lower_mask.size == 0:
        return None

    # Распределение ненулевых пикселей по оси X (гистограмма)
    upper_dist_raw = np.sum(upper_mask > 0, axis=0).astype(float)
    lower_dist_raw = np.sum(lower_mask > 0, axis=0).astype(float)

    # Находим диапазон ненулевых столбцов
    upper_nonzero = np.where(upper_dist_raw > 0)[0]
    lower_nonzero = np.where(lower_dist_raw > 0)[0]

    if len(upper_nonzero) < 5 or len(lower_nonzero) < 5:
        return None

    u_start, u_end = upper_nonzero[0], upper_nonzero[-1]
    l_start, l_end = lower_nonzero[0], lower_nonzero[-1]
    u_len = u_end - u_start + 1
    l_len = l_end - l_start + 1

    if u_len < 5 or l_len < 5:
        return None

    # Сглаживание гистограмм (ключевое улучшение!)
    sigma = max(1.0, u_len * 0.03)  # Адаптивная ширина ядра
    upper_dist = gaussian_filter1d(upper_dist_raw, sigma=sigma)
    lower_dist = gaussian_filter1d(lower_dist_raw, sigma=sigma)

    # ─── Передние точки: максимум в первых 10% ───
    u_front_end = u_start + max(2, int(0.10 * u_len))
    l_front_end = l_start + max(2, int(0.10 * l_len))

    xAu = u_start + np.argmax(upper_dist[u_start:u_front_end + 1])
    xAl = l_start + np.argmax(lower_dist[l_start:l_front_end + 1])

    # ─── Задние точки: максимум в последних 10% ───
    u_back_start = u_end - max(2, int(0.10 * u_len))
    l_back_start = l_end - max(2, int(0.10 * l_len))

    xPu = u_back_start + np.argmax(upper_dist[u_back_start:u_end + 1])
    xPl = l_back_start + np.argmax(lower_dist[l_back_start:l_end + 1])

    # ─── Центральные точки: первый минимум вокруг центра масс ±10% ───
    # (Section 2.5.2: xMu и xMl находятся НЕЗАВИСИМО в верхней/нижней половинах)
    delta = max(5, int(0.10 * u_len))

    def find_center_minimum(dist, start, end, center_x):
        """Первый минимум сглаженной гистограммы вокруг центра масс (Section 2.5.2)."""
        search_lo = max(start + 1, center_x - delta)
        search_hi = min(end - 1, center_x + delta)

        if search_lo >= search_hi:
            return center_x

        segment = dist[search_lo:search_hi + 1].copy()

        # Маскируем нулевые значения (пустые столбцы)
        segment[segment == 0] = np.max(segment) + 1

        min_idx = np.argmin(segment)
        return search_lo + min_idx

    xMu = find_center_minimum(upper_dist, u_start, u_end, cx_mass)
    xMl = find_center_minimum(lower_dist, l_start, l_end, cx_mass)

    # ─── Y-координаты ───
    def first_nonzero_y(col_idx, mask_part):
        """Первый ненулевой пиксель сверху в столбце."""
        if col_idx < 0 or col_idx >= mask_part.shape[1]:
            return 0
        col = mask_part[:, col_idx]
        nz = np.where(col > 0)[0]
        return nz[0] if len(nz) > 0 else 0

    def last_nonzero_y(col_idx, mask_part, y_offset):
        """Последний ненулевой пиксель снизу в столбце."""
        if col_idx < 0 or col_idx >= mask_part.shape[1]:
            return y_offset
        col = mask_part[:, col_idx]
        nz = np.where(col > 0)[0]
        return (nz[-1] + y_offset) if len(nz) > 0 else y_offset

    yAu = first_nonzero_y(xAu, upper_mask)
    yPu = first_nonzero_y(xPu, upper_mask)
    yMu = first_nonzero_y(xMu, upper_mask)

    yAl = last_nonzero_y(xAl, lower_mask, cy_mass)
    yPl = last_nonzero_y(xPl, lower_mask, cy_mass)
    yMl = last_nonzero_y(xMl, lower_mask, cy_mass)

    return {
        'Au': (int(xAu), int(yAu)),
        'Al': (int(xAl), int(yAl)),
        'Pu': (int(xPu), int(yPu)),
        'Pl': (int(xPl), int(yPl)),
        'Mu': (int(xMu), int(yMu)),
        'Ml': (int(xMl), int(yMl)),
    }


# ─────────────────────────────────────────────────────────────────
# Шаг 6: Расчет высот и классификация Genant
# (Section 2.5.3 формулы 4–6, Section 2.5.4 формулы 7–10)
# ─────────────────────────────────────────────────────────────────

def calc_distance(p1, p2):
    """Евклидово расстояние между двумя точками (формулы 4–6)."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def classify_genant(Ah, Ph, Mh, pixel_spacing_mm=None):
    """
    Классификация деформации по Genant (формулы 7–10).
    """
    if pixel_spacing_mm:
        Ah_mm = Ah * pixel_spacing_mm
        Ph_mm = Ph * pixel_spacing_mm
        Mh_mm = Mh * pixel_spacing_mm
    else:
        Ah_mm = Ah
        Ph_mm = Ph
        Mh_mm = Mh

    if Ah <= 0 or Ph <= 0 or Mh <= 0:
        return {
            'Ah': Ah_mm, 'Ph': Ph_mm, 'Mh': Mh_mm,
            'wedge_pct': 0, 'biconcave_pct': 0,
            'wedge_grade': 0, 'biconcave_grade': 0,
            'wedge_type': 'N/A', 'biconcave_type': 'N/A'
        }

    if Ah < Ph:
        wedge_pct = max(0, 100 - (Ah / Ph) * 100)
        biconcave_pct = max(0, 100 - (Mh / Ph) * 100)
        wedge_type = "Wedge (Ah < Ph)"
    else:
        wedge_pct = max(0, 100 - (Ph / Ah) * 100)
        biconcave_pct = max(0, 100 - (Mh / Ah) * 100)
        wedge_type = "Wedge (Ph < Ah)"

    def grade(pct):
        if pct < 20:
            return 0
        elif pct < 25:
            return 1
        elif pct < 40:
            return 2
        else:
            return 3

    grade_names = {
        0: "Normal",
        1: "Mild (Grade 1)",
        2: "Moderate (Grade 2)",
        3: "Severe (Grade 3)"
    }

    w_grade = grade(wedge_pct)
    b_grade = grade(biconcave_pct)

    return {
        'Ah': round(Ah_mm, 2),
        'Ph': round(Ph_mm, 2),
        'Mh': round(Mh_mm, 2),
        'wedge_pct': round(wedge_pct, 1),
        'biconcave_pct': round(biconcave_pct, 1),
        'wedge_grade': w_grade,
        'biconcave_grade': b_grade,
        'wedge_grade_name': grade_names[w_grade],
        'biconcave_grade_name': grade_names[b_grade],
        'wedge_type': wedge_type,
    }


# ─────────────────────────────────────────────────────────────────
# Обработка одного позвонка: маска → 6 точек → Genant
# ─────────────────────────────────────────────────────────────────

def analyze_single_vertebra(binary_mask, pixel_spacing_mm=None):
    """
    Полный пайплайн от бинарной маски одного позвонка до диагноза Genant.
    """
    # Постобработка
    clean_mask = postprocess_mask(binary_mask)

    # Поворот для компенсации наклона
    angle, center = get_rotation_angle(clean_mask)
    if center is None:
        return None, None

    rotated_mask, rot_matrix = rotate_mask(clean_mask, angle, center)

    # Поиск 6 точек на повёрнутой маске
    landmarks_rotated = find_six_landmarks(rotated_mask)
    if landmarks_rotated is None:
        return None, None

    # Обратный поворот точек в оригинальную систему координат
    inv_matrix = cv2.invertAffineTransform(rot_matrix)
    landmarks_original = {}
    for name, (x, y) in landmarks_rotated.items():
        pt = np.array([x, y, 1.0])
        ox, oy = inv_matrix @ pt
        landmarks_original[name] = (int(round(ox)), int(round(oy)))

    # Расчёт высот (формулы 4–6)
    Ah = calc_distance(landmarks_original['Au'], landmarks_original['Al'])
    Ph = calc_distance(landmarks_original['Pu'], landmarks_original['Pl'])
    Mh = calc_distance(landmarks_original['Mu'], landmarks_original['Ml'])

    # Классификация
    result = classify_genant(Ah, Ph, Mh, pixel_spacing_mm)

    return landmarks_original, result


# ─────────────────────────────────────────────────────────────────
# Главная функция: снимок → полный отчёт
# ─────────────────────────────────────────────────────────────────

def analyze_image(image_path, model_path=None, output_path=None, pixel_spacing_mm=None):
    """
    Полный анализ МРТ-снимка:
        1. YOLOv8-Seg → маски позвонков
        2. Для каждой маски: 6 точек → 3 высоты → Genant
        3. Визуализация + текстовый отчёт
    """
    if model_path is None:
        model_path = os.path.join(os.path.dirname(__file__), "weights", "best.pt")

    if not os.path.exists(model_path):
        print(f"❌ Модель не найдена: {model_path}")
        return

    model = YOLO(model_path)
    results = model(image_path, conf=0.25)

    if not results or results[0].masks is None:
        print("⚠ Позвонки не обнаружены.")
        return

    result = results[0]
    img = result.orig_img.copy()
    h, w = img.shape[:2]

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    masks = result.masks.data.cpu().numpy()
    boxes = result.boxes

    # ─── Сортировка позвонков сверху вниз ───
    # Вычисляем центр Y каждой маски для сортировки
    centers_y = []
    for i, mask in enumerate(masks):
        mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        ys = np.where(mask_resized > 0.5)[0]
        cy = np.mean(ys) if len(ys) > 0 else 0
        centers_y.append((cy, i))

    # Сортируем по Y (сверху вниз)
    centers_y.sort(key=lambda x: x[0])
    sorted_indices = [idx for _, idx in centers_y]

    print(f"\n{'='*60}")
    print(f"  ОТЧЕТ О ДЕФОРМАЦИЯХ ПОЗВОНКОВ")
    print(f"  Снимок: {os.path.basename(image_path)}")
    print(f"  Обнаружено позвонков: {len(masks)}")
    print(f"{'='*60}\n")

    all_results = []

    colors = [
        (50, 205, 50),    # Зелёный
        (255, 191, 0),    # Голубой
        (0, 165, 255),    # Оранжевый
        (203, 92, 255),   # Фиолетовый
        (0, 255, 255),    # Жёлтый
        (255, 105, 105),  # Светло-синий
        (100, 255, 100),  # Салатовый
    ]

    for rank, orig_idx in enumerate(sorted_indices):
        mask = masks[orig_idx]
        mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        binary = (mask_resized > 0.5).astype(np.uint8) * 255

        color = colors[rank % len(colors)]
        label = f"V{rank + 1}"

        landmarks, genant = analyze_single_vertebra(binary, pixel_spacing_mm)

        if landmarks is None:
            print(f"  {label}: ⚠ Не удалось проанализировать")
            continue

        all_results.append((label, landmarks, genant))

        # ─── Визуализация ───

        # Полупрозрачная маска
        overlay = img.copy()
        overlay[mask_resized > 0.5] = color
        img = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)

        # 6 красных точек (крупнее)
        for name, (x, y) in landmarks.items():
            cv2.circle(img, (x, y), 4, (0, 0, 255), -1)
            cv2.circle(img, (x, y), 4, (255, 255, 255), 1)  # Белая обводка

        # Линии высот (толще)
        cv2.line(img, landmarks['Au'], landmarks['Al'], (0, 255, 255), 2)    # Ah — жёлтая
        cv2.line(img, landmarks['Pu'], landmarks['Pl'], (255, 0, 255), 2)    # Ph — фиолетовая
        cv2.line(img, landmarks['Mu'], landmarks['Ml'], (0, 255, 0), 2)      # Mh — зелёная

        # Печатаем результат в терминал
        unit = "mm" if pixel_spacing_mm else "px"
        print(f"  {label}:")
        print(f"    Ah={genant['Ah']:.1f}{unit}  Mh={genant['Mh']:.1f}{unit}  Ph={genant['Ph']:.1f}{unit}")
        print(f"    Wedge:     {genant['wedge_pct']:.1f}% → {genant['wedge_grade_name']}")
        print(f"    Biconcave: {genant['biconcave_pct']:.1f}% → {genant['biconcave_grade_name']}")
        print()

    # Сохраняем
    if output_path is None:
        base = os.path.splitext(image_path)[0]
        output_path = f"{base}_deformity_report.jpg"

    cv2.imwrite(output_path, img)
    print(f"📊 Отчет сохранен: {output_path}")

    return all_results


# ─────────────────────────────────────────────────────────────────
# Запуск
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        img_path = os.path.join(os.path.dirname(__file__),
                                "dataset_seg", "val", "images")
        import glob
        imgs = sorted(glob.glob(os.path.join(img_path, "*.jpg")))
        if imgs:
            img_path = imgs[0]

    analyze_image(img_path, pixel_spacing_mm=0.677)
