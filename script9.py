import cv2
import numpy as np
import math
import mediapipe as mp
from tqdm import tqdm

def create_face_silhouette(image_path, nails=500, max_lines=12000, output_size=800):
    # 1. Загрузка изображения
    img = cv2.imread(image_path)
    if img is None:
        print("Ошибка загрузки изображения")
        return None
    
    img = cv2.resize(img, (output_size, output_size))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Детекция лица с акцентом на черты
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.7
    ) as face_mesh:
        
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_img)
        
        if not results.multi_face_landmarks:
            print("Лицо не обнаружено")
            return None

        # 3. Создание точной маски лица
        mask = np.zeros((output_size, output_size), dtype=np.float32)
        landmarks = results.multi_face_landmarks[0].landmark
        
        
        face_points = []
        for landmark in landmarks:
            x = int(landmark.x * output_size)
            y = int(landmark.y * output_size)
            face_points.append([x, y])
        
        hull = cv2.convexHull(np.array(face_points))
        cv2.fillConvexPoly(mask, hull, 1.0)
        
      
        lips = list(mp_face_mesh.FACEMESH_LIPS)
        eyes = list(mp_face_mesh.FACEMESH_LEFT_EYE) + list(mp_face_mesh.FACEMESH_RIGHT_EYE)
        brows = list(mp_face_mesh.FACEMESH_LEFT_EYEBROW) + list(mp_face_mesh.FACEMESH_RIGHT_EYEBROW)
        
        for indices, weight in [(lips, 2.0), (eyes, 2.5), (brows, 1.5)]:
            for idx_pair in indices:
                pt1 = landmarks[idx_pair[0]]
                pt2 = landmarks[idx_pair[1]]
                x1, y1 = int(pt1.x * output_size), int(pt1.y * output_size)
                x2, y2 = int(pt2.x * output_size), int(pt2.y * output_size)
                cv2.line(mask, (x1,y1), (x2,y2), weight, 2)
    
    # 4. Создание карты важности
    mask = cv2.GaussianBlur(mask, (25,25), 0)
    edges = cv2.Canny(gray, 50, 150).astype(np.float32) / 255.0
    importance_map = cv2.addWeighted(mask, 0.9, edges, 0.1, 0, dtype=cv2.CV_32F)
    
    # 5. Распределение гвоздей с уплотнением в области лица
    center = (output_size//2, output_size//2)
    radius = int(output_size * 0.45)
    nails_pos = []
    
    for i in range(nails):
        angle = 2 * math.pi * i / nails
        # 30% гвоздей смещаем ближе к лицу
        r_factor = 0.8 if i % 3 == 0 else 1.0
        x = int(center[0] + radius * math.cos(angle) * r_factor)
        y = int(center[1] + radius * math.sin(angle) * r_factor)
        nails_pos.append((x, y))
    
    # 6. Основной алгоритм с приоритетом на лице
    result = np.ones((output_size, output_size, 3), dtype=np.uint8) * 255
    density_map = np.zeros((output_size, output_size), dtype=np.float32)
    
    # Предварительный расчет всех возможных пар
    all_pairs = []
    for i in range(nails):
        for j in range(i+1, min(i+150, nails)):
            all_pairs.append((i, j))
    
    # Сортируем по важности (линии через лицо имеют приоритет)
    def line_importance(i, j):
        pt1, pt2 = nails_pos[i], nails_pos[j]
        line_mask = np.zeros_like(importance_map)
        cv2.line(line_mask, pt1, pt2, 1.0, 1)
        return cv2.mean(importance_map, mask=line_mask.astype(np.uint8))[0]
    
    all_pairs.sort(key=lambda x: -line_importance(x[0], x[1]))
    
    # Рисуем линии
    for i, j in tqdm(all_pairs[:max_lines], desc="Создание силуэта"):
        pt1, pt2 = nails_pos[i], nails_pos[j]
        
        # Проверяем, проходит ли линия через важную область
        line_mask = np.zeros_like(importance_map)
        cv2.line(line_mask, pt1, pt2, 1.0, 1)
        importance = cv2.mean(importance_map, mask=line_mask.astype(np.uint8))[0]
        
        if importance > 0.2:  # Рисуем только значимые линии
            thickness = max(1, min(3, int(4 * importance)))
            cv2.line(result, pt1, pt2, (0,0,0), thickness, cv2.LINE_AA)
            
            # Обновляем карты
            density_map += line_mask * importance
            importance_map[line_mask > 0] *= 0.9  # Уменьшаем важность
    
    # 7. Финальная обработка
    density_map = cv2.GaussianBlur(density_map, (5,5), 0)
    density_map = cv2.normalize(density_map, None, 0, 255, cv2.NORM_MINMAX)
    result = cv2.addWeighted(
        result, 0.9,
        cv2.cvtColor(density_map.astype(np.uint8), cv2.COLOR_GRAY2BGR),
        0.1,
        0
    )
    
    return result

# Использование
input_img = "face.jpg"
output_img = "face_silhouette_art.jpg"

art = create_face_silhouette(input_img, nails=500, max_lines=12000)
if art is not None:
    cv2.imwrite(output_img, art)
    cv2.imshow("Face Silhouette Art", art)
    cv2.waitKey(0)
    cv2.destroyAllWindows()