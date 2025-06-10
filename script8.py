import cv2
import numpy as np
import math
import mediapipe as mp
from tqdm import tqdm

def create_string_art_face(image_path, nails=400, max_lines=10000, output_size=800):
    # 1. Загрузка и подготовка изображения
    img = cv2.imread(image_path)
    if img is None:
        print("Ошибка загрузки изображения")
        return None
    
    img = cv2.resize(img, (output_size, output_size))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Детекция лица и ключевых точек
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:
        
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_img)
        
        if not results.multi_face_landmarks:
            print("Лицо не обнаружено")
            return None

        # 3. Создание карты важности на основе лица
        mask = np.zeros((output_size, output_size), dtype=np.float32)
        landmarks = results.multi_face_landmarks[0].landmark
        
        # Создаем тепловую карту лица
        for landmark in landmarks:
            x = int(landmark.x * output_size)
            y = int(landmark.y * output_size)
            if 0 <= x < output_size and 0 <= y < output_size:
                mask[y, x] = 1.0
        
        # Размываем для создания плавных градиентов
        mask = cv2.GaussianBlur(mask, (51,51), 0)
    
    # 4. Создание границ изображения
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blurred, 50, 150).astype(np.float32) / 255.0
    
    # 5. Комбинированная карта важности
    importance_map = cv2.addWeighted(mask, 0.8, edges, 0.2, 0, dtype=cv2.CV_32F)
    
    # 6. Распределение гвоздей по эллипсу
    center = (output_size//2, output_size//2)
    radius_x = int(output_size * 0.4)
    radius_y = int(output_size * 0.5)
    nails_pos = []
    
    for i in range(nails):
        angle = 2 * math.pi * i / nails
        x = int(center[0] + radius_x * math.cos(angle))
        y = int(center[1] + radius_y * math.sin(angle))
        nails_pos.append((x, y))
    
    # 7. Основной алгоритм string art
    result = np.ones((output_size, output_size, 3), dtype=np.uint8) * 255
    density_map = np.zeros((output_size, output_size), dtype=np.float32)
    
    # Создаем список всех возможных пар гвоздей
    all_pairs = []
    for i in range(nails):
        for j in range(i+1, min(i+100, nails)):
            all_pairs.append((i, j))
    
    # Сортируем пары по важности (предварительный расчет)
    def calculate_importance(i, j):
        pt1 = nails_pos[i]
        pt2 = nails_pos[j]
        line_mask = np.zeros_like(importance_map)
        cv2.line(line_mask, pt1, pt2, 1.0, 1)
        return cv2.mean(importance_map, mask=line_mask.astype(np.uint8))[0]
    
    all_pairs.sort(key=lambda x: -calculate_importance(x[0], x[1]))
    
    # Рисуем наиболее важные линии
    for i, j in tqdm(all_pairs[:max_lines], desc="Рисование линий"):
        pt1 = nails_pos[i]
        pt2 = nails_pos[j]
        
        # Рисуем линию
        cv2.line(result, pt1, pt2, (0,0,0), 1, cv2.LINE_AA)
        
        # Обновляем карту плотности
        line_mask = np.zeros_like(density_map)
        cv2.line(line_mask, pt1, pt2, 1.0, 1)
        density_map += line_mask * 0.1
        
        # Уменьшаем важность в этой области
        cv2.line(importance_map, pt1, pt2, 0.0, 1)
    
    # 8. Правильное преобразование для постобработки
    density_map_normalized = cv2.normalize(density_map, None, 0, 255, cv2.NORM_MINMAX)
    density_map_uint8 = density_map_normalized.astype(np.uint8)
    density_map_bgr = cv2.cvtColor(density_map_uint8, cv2.COLOR_GRAY2BGR)
    
    # Смешиваем с результатом
    result = cv2.addWeighted(
        result, 0.9,
        density_map_bgr, 0.1,
        0
    )
    
    # 9. Рисуем гвозди (опционально)
    for pt in nails_pos:
        cv2.circle(result, pt, 2, (0,0,255), -1)
    
    return result

# Использование
input_img = "face.jpg"
output_img = "string_art_result.jpg"

art = create_string_art_face(input_img, nails=400, max_lines=10000)
if art is not None:
    cv2.imwrite(output_img, art)
    cv2.imshow("String Art Face", art)
    cv2.waitKey(0)
    cv2.destroyAllWindows()