import cv2
import numpy as np
import math

def create_face_silhouette(image_path, nails=300, max_lines=3000, output_size=800):
    # 1. Загрузка и подготовка изображения
    img = cv2.imread(image_path)
    if img is None:
        print("Ошибка загрузки изображения")
        return None
    
    # Обрезка и ресайз до квадрата
    h, w = img.shape[:2]
    size = min(h, w)
    img = img[(h-size)//2:(h-size)//2+size, (w-size)//2:(w-size)//2+size]
    img = cv2.resize(img, (output_size, output_size))
    
    # 2. Детектирование лица и ключевых точек
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    
    if len(faces) == 0:
        print("Лицо не обнаружено")
        return None
    
    x, y, w, h = faces[0]
    face_roi = gray[y:y+h, x:x+w]
    
    # 3. Создание маски силуэта лица
    blurred = cv2.GaussianBlur(face_roi, (5,5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Улучшаем маску
    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Переносим маску на оригинальное изображение
    mask = np.zeros_like(gray)
    mask[y:y+h, x:x+w] = thresh
    masked_face = cv2.bitwise_and(gray, gray, mask=mask)
    
    # 4. Создание точек крепления по окружности
    center = (output_size//2, output_size//2)
    radius = int(output_size * 0.45)
    nails_pos = []
    for i in range(nails):
        angle = 2 * math.pi * i / nails
        x_nail = int(center[0] + radius * math.cos(angle))
        y_nail = int(center[1] + radius * math.sin(angle))
        nails_pos.append((x_nail, y_nail))
    
    # 5. Алгоритм построения силуэта
    result = np.ones((output_size, output_size, 3), dtype=np.uint8) * 255
    line_count = 0
    
    # Сначала рисуем контур лица
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(result, contours, -1, (0,0,0), 2)
    
    # Затем заполняем область лица линиями
    for i in range(nails):
        # Находим лучшую пару гвоздей для соединения
        best_j = -1
        best_score = 0
        
        for j in range(i+1, min(i+100, nails)):  # Ограничиваем диапазон поиска
            pt1 = nails_pos[i]
            pt2 = nails_pos[j]
            
            # Создаем маску линии
            line_mask = np.zeros_like(gray)
            cv2.line(line_mask, pt1, pt2, 255, 1)
            
            # Оцениваем важность линии
            overlap = cv2.bitwise_and(line_mask, mask)
            score = cv2.countNonZero(overlap)
            
            # Учитываем длину линии (предпочитаем более короткие)
            length = math.sqrt((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)
            score *= math.exp(-length/(radius*0.5))
            
            if score > best_score:
                best_score = score
                best_j = j
        
        if best_j != -1 and line_count < max_lines:
            pt1 = nails_pos[i]
            pt2 = nails_pos[best_j]
            
            # Определяем толщину линии на основе важности
            thickness = max(1, min(3, int(best_score / 500)))
            cv2.line(result, pt1, pt2, (0,0,0), thickness, lineType=cv2.LINE_AA)
            line_count += 1
    
    # 6. Рисуем гвоздики
    for pt in nails_pos:
        cv2.circle(result, pt, 2, (0,0,255), -1)
    
    return result

# Использование
input_img = "face1.jpg"
output_img = "face_silhouette.jpg"

art = create_face_silhouette(input_img, nails=350, max_lines=2500)
if art is not None:
    cv2.imwrite(output_img, art)
    cv2.imshow("Face Silhouette", art)
    cv2.waitKey(0)
    cv2.destroyAllWindows()