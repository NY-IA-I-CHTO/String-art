import cv2
import numpy as np
import math

def create_face_string_art(image_path, nails=500, max_lines=30000):
    # 1. Загрузка и улучшение изображения
    img = cv2.imread(image_path)
    if img is None:
        print("Ошибка загрузки изображения")
        return None
    
    # 2. Детекция лица и ключевых точек
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # 3. Улучшенная обработка изображения
    equalized = cv2.equalizeHist(gray)
    blurred = cv2.GaussianBlur(equalized, (5,5), 0)
    
    # 4. Вариант 1: Детекция границ (для портретов без детектора лиц)
    edges = cv2.Canny(blurred, 30, 100)
    edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
    
    # 5. Создаем маску круга
    mask = np.zeros((h,w), dtype=np.uint8)
    center = (w//2, h//2)
    radius = int(min(h,w)*0.9/2)
    cv2.circle(mask, center, radius, 255, -1)
    edges = cv2.bitwise_and(edges, edges, mask=mask)
    
    # 6. Создаем "гвоздики" по окружности
    nail_points = []
    for i in range(nails):
        angle = 2 * math.pi * i / nails
        x = int(center[0] + radius * math.cos(angle))
        y = int(center[1] + radius * math.sin(angle))
        nail_points.append((x,y))
    
    # 7. Создаем карту приоритетов
    priority_map = cv2.GaussianBlur(edges, (15,15), 0)
    priority_map = priority_map.astype(np.float32)
    
    # 8. Основной алгоритм построения
    result = np.ones((h,w,3), dtype=np.uint8) * 255
    line_density = np.zeros((h,w), dtype=np.float32)
    
    # 9. Сначала рисуем основные контуры
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        for i in range(0, len(cnt), 5):
            pt = tuple(cnt[i][0])
            # Находим ближайшие гвоздики
            distances = [math.sqrt((pt[0]-n[0])**2 + (pt[1]-n[1])**2) for n in nail_points]
            idx1 = np.argmin(distances)
            distances[idx1] = float('inf')
            idx2 = np.argmin(distances)
            
            # Рисуем линию между ними
            pt1, pt2 = nail_points[idx1], nail_points[idx2]
            cv2.line(result, pt1, pt2, (0,0,0), 2, lineType=cv2.LINE_AA)
            cv2.line(line_density, pt1, pt2, 255, 1, lineType=cv2.LINE_AA)
    
    # 10. Затем заполняем остальные области
    for _ in range(max_lines):
        # Выбираем точки с учетом приоритета
        i, j = np.random.choice(nails, 2, replace=False)
        pt1, pt2 = nail_points[i], nail_points[j]
        
        # Проверяем важность линии
        line_mask = np.zeros((h,w), dtype=np.uint8)
        cv2.line(line_mask, pt1, pt2, 255, 1)
        line_value = cv2.mean(priority_map, mask=line_mask)[0]
        
        if line_value > 20:  # Порог можно регулировать
            thickness = max(1, int(line_value/40))
            cv2.line(result, pt1, pt2, (0,0,0), thickness, lineType=cv2.LINE_AA)
            cv2.line(line_density, pt1, pt2, line_value, 1, lineType=cv2.LINE_AA)
    
    # 11. Добавляем гвоздики
    for pt in nail_points:
        cv2.circle(result, pt, 3, (0,0,0), -1)
    cv2.circle(result, center, radius, (0,0,0), 2)
    
    return result

# Использование
input_img = "face.jpg"
output_img = "face_string_art.jpg"

art = create_face_string_art(input_img, nails=600, max_lines=40000)
if art is not None:
    cv2.imwrite(output_img, art)
    cv2.imshow("Face String Art", art)
    cv2.waitKey(0)
    cv2.destroyAllWindows()