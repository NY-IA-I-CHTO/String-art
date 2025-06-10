import cv2
import numpy as np
import math
from tqdm import tqdm

class StringArtGenerator:
    def __init__(self, image_path):
        self.img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if self.img is None:
            raise ValueError("Ошибка загрузки изображения")
        
        self.height, self.width = self.img.shape
        self.center = (self.width//2, self.height//2)
        self.radius = min(self.width, self.height) // 2 - 20
        
    def preprocess_image(self, brightness=0, contrast=1, invert=False):
        """Подготовка изображения с настройками"""
        processed = cv2.convertScaleAbs(self.img, alpha=contrast, beta=brightness)
        if invert:
            processed = 255 - processed
        self.processed = cv2.equalizeHist(processed)
        
    def generate_nails(self, count):
        """Создание точек крепления по окружности"""
        self.nails = []
        for i in range(count):
            angle = 2 * math.pi * i / count
            x = int(self.center[0] + self.radius * math.cos(angle))
            y = int(self.center[1] + self.radius * math.sin(angle))
            self.nails.append((x, y))
    
    def create_string_art(self, lines_count=20000, opacity=10, line_color=(0,0,0), bg_color=(255,255,255)):
        """Основной алгоритм генерации String Art"""
        # Инициализация
        result = np.full((self.height, self.width, 3), bg_color, dtype=np.uint8)
        current_img = self.processed.copy()
        current_nail = 0
        opacity_value = opacity * 2.55  # Переводим % в 0-255
        
        # Гистограмма для визуализации прогресса
        progress = tqdm(total=lines_count, desc="Генерация String Art")
        
        for _ in range(lines_count):
            best_line = None
            best_score = float('inf')
            
            # Поиск оптимальной следующей линии
            for nail_idx in range(len(self.nails)):
                if nail_idx == current_nail:
                    continue
                    
                # Создаем маску линии
                line_mask = np.zeros_like(current_img)
                cv2.line(line_mask, self.nails[current_nail], self.nails[nail_idx], 255, 1)
                
                # Вычисляем среднюю яркость
                mean_brightness = cv2.mean(current_img, mask=line_mask)[0]
                
                if mean_brightness < best_score:
                    best_score = mean_brightness
                    best_line = nail_idx
            
            if best_line is None:
                break
            
            # Рисуем линию на результате
            cv2.line(result, self.nails[current_nail], self.nails[best_line], 
                    line_color, 1, lineType=cv2.LINE_AA)
            
            # "Удаляем" линию из исходного изображения
            line_mask = np.zeros_like(current_img)
            cv2.line(line_mask, self.nails[current_nail], self.nails[best_line], 
                    opacity_value, 1)
            current_img = cv2.add(current_img, line_mask)
            
            current_nail = best_line
            progress.update(1)
        
        progress.close()
        return result
    
    def save_result(self, result, output_path, svg=False):
        """Сохранение результата"""
        if svg:
            # Генерация SVG (упрощенная версия)
            with open(output_path, 'w') as f:
                f.write(f'<svg width="{self.width}" height="{self.height}" xmlns="http://www.w3.org/2000/svg">\n')
                f.write(f'<rect width="100%" height="100%" fill="rgb{self.bg_color}"/>\n')
                # Здесь должна быть логика экспорта линий в SVG
                f.write('</svg>')
        else:
            cv2.imwrite(output_path, result)

# Пример использования
if __name__ == "__main__":
    try:
        # Инициализация генератора
        generator = StringArtGenerator("face.jpg")
        
        # Настройки обработки изображения
        generator.preprocess_image(brightness=10, contrast=1.2, invert=True)
        
        # Создание точек крепления
        generator.generate_nails(count=500)
        
        # Генерация String Art
        result = generator.create_string_art(
            lines_count=30000,
            opacity=15,  # 15% непрозрачность
            line_color=(0, 0, 0),
            bg_color=(255, 255, 255)
        )
        
        # Сохранение результата
        generator.save_result(result, "string_art_result.png")
        generator.save_result(result, "string_art_result.svg", svg=True)
        
        print("String Art успешно сгенерирован!")
    except Exception as e:
        print(f"Ошибка: {str(e)}")