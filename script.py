import cv2
import numpy as np
import random
import math

def generate_string_art(image, nails=300, threads=5000, debug=False):
    # 1. Подготовка изображения
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    
    # 2. Создание контурного рисунка для гвоздиков
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # 3. Выбор точек-гвоздиков по контуру
    nail_points = []
    for cnt in contours:
        step = max(1, len(cnt)//nails)
        for i in range(0, len(cnt), step):
            nail_points.append(tuple(cnt[i][0]))
    
    # Если точек мало, добавляем случайные
    while len(nail_points) < nails:
        x = random.randint(0, gray.shape[1]-1)
        y = random.randint(0, gray.shape[0]-1)
        nail_points.append((x,y))
    
    nail_points = nail_points[:nails]
    
    # 4. Создание доски для стринг-арта
    board = np.ones_like(image) * 255
    brightness_map = np.zeros(gray.shape, dtype=np.float32)
    
    # 5. Имитация натяжения нитей (без tqdm)
    print("Создание стринг-арта...")
    for iteration in range(threads):
        # Выбираем две случайные точки
        i, j = random.sample(range(len(nail_points)), 2)
        pt1, pt2 = nail_points[i], nail_points[j]
        
        # Создаем маску линии
        line_mask = np.zeros_like(gray)
        cv2.line(line_mask, pt1, pt2, 255, 1)
        
        # Средняя яркость под линией
        line_brightness = cv2.mean(gray, mask=line_mask)[0]
        
        # Обновляем карту яркости
        cv2.line(brightness_map, pt1, pt2, line_brightness, 1, lineType=cv2.LINE_AA)
        
        # Промежуточная визуализация
        if debug and iteration % 500 == 0:
            display = board.copy()
            norm_brightness = cv2.normalize(brightness_map, None, 0, 255, cv2.NORM_MINMAX)
            display = cv2.addWeighted(display, 0.7, 
                                    cv2.cvtColor(norm_brightness.astype(np.uint8), cv2.COLOR_GRAY2BGR), 
                                    0.3, 0)
            cv2.imshow("Progress", display)
            cv2.waitKey(1)
            
        # Простой прогресс-бар в консоли
        if iteration % (threads//10) == 0:
            print(f"Прогресс: {iteration*100//threads}%")
    
    # 6. Финальная визуализация
    norm_brightness = cv2.normalize(brightness_map, None, 0, 255, cv2.NORM_MINMAX)
    result = cv2.addWeighted(board, 0.7, 
                           cv2.cvtColor(norm_brightness.astype(np.uint8), cv2.COLOR_GRAY2BGR), 
                           0.3, 0)
    
    # 7. Добавляем точки-гвоздики
    for pt in nail_points:
        cv2.circle(result, pt, 2, (50,50,50), -1)
    
    print("Готово!")
    return result

# Загрузка и обработка
image = cv2.imread('face1.jpg')
if image is not None:
    art = generate_string_art(image, nails=10000, threads=20000, debug=True)
    cv2.imwrite("string_art_result.jpg", art)
    cv2.imshow("String Art Result", art)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Ошибка загрузки изображения")