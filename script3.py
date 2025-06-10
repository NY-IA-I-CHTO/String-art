import cv2
import numpy as np
import math

def create_string_art(image_path, nails=500, max_lines=30000):
    # 1. Загрузка и подготовка изображения
    img = cv2.imread(image_path)
    if img is None:
        print("Ошибка загрузки изображения")
        return None
    
    # 2. Конвертация в grayscale и улучшение контраста
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    h, w = gray.shape
    
    # 3. Создаем маску круга
    mask = np.zeros((h, w), dtype=np.uint8)
    center = (w//2, h//2)
    radius = int(min(h, w) * 0.45)  # Уменьшаем радиус для плотности
    cv2.circle(mask, center, radius, 255, -1)
    
    # 4. Применяем маску и инвертируем яркость
    masked = cv2.bitwise_and(gray, gray, mask=mask)
    inverted = 255 - masked
    
    # 5. Создаем точки крепления по окружности
    nails_pos = []
    for i in range(nails):
        angle = 2 * math.pi * i / nails
        x = int(center[0] + radius * math.cos(angle))
        y = int(center[1] + radius * math.sin(angle))
        nails_pos.append((x, y))
    
    # 6. Основной алгоритм построения
    result = np.ones((h, w, 3), dtype=np.uint8) * 255
    density_map = np.zeros((h, w), dtype=np.float32)
    
    for _ in range(max_lines):
        # Выбираем случайные точки
        i, j = np.random.choice(nails, 2, replace=False)
        pt1, pt2 = nails_pos[i], nails_pos[j]
        
        # Создаем маску линии
        line_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.line(line_mask, pt1, pt2, 255, 1)
        
        # Вычисляем важность линии
        brightness = cv2.mean(inverted, mask=line_mask)[0]
        
        # Рисуем только значимые линии
        if brightness > 40:  # Порог можно регулировать
            thickness = max(1, int(brightness / 60))
            cv2.line(result, pt1, pt2, (0, 0, 0), thickness, lineType=cv2.LINE_AA)
            cv2.line(density_map, pt1, pt2, brightness, 1, lineType=cv2.LINE_AA)
    
    # 7. Усиливаем контраст
    density_map = cv2.normalize(density_map, None, 0, 255, cv2.NORM_MINMAX)
    density_map = density_map.astype(np.uint8)
    
    # 8. Финальное смешивание
    result = cv2.addWeighted(result, 0.85, 
                           cv2.cvtColor(density_map, cv2.COLOR_GRAY2BGR), 
                           0.15, 0)
    
    # 9. Рисуем гвоздики и круг
    for pt in nails_pos:
        cv2.circle(result, pt, 2, (0, 0, 0), -1)
    cv2.circle(result, center, radius, (0, 0, 0), 1)
    
    return result

# Использование
input_img = "face1.jpg"
output_img = "string_art_result.jpg"

art = create_string_art(input_img, nails=800, max_lines=40000)
if art is not None:
    cv2.imwrite(output_img, art)
    cv2.imshow("String Art Portrait", art)
    cv2.waitKey(0)
    cv2.destroyAllWindows()