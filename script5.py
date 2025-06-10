import cv2
import numpy as np
import math
import json
from tqdm import tqdm
import random

class StringArtGenerator:
    def __init__(self, image_path):
        # Загрузка и инициализация изображения
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError("Не удалось загрузить изображение")
        
        self.height, self.width = self.image.shape[:2]
        self.center = (self.width // 2, self.height // 2)
        self.radius = min(self.width, self.height) // 2 - 20
        self.nails = []
        self.sequence = []
        
        # Параметры по умолчанию
        self.contrast = 1.0
        self.brightness = 0
        self.invert = False
        self.line_weight = 25  # 25%
        self.line_color = (0, 0, 0)
        self.background_color = (255, 255, 255)
        self.nails_count = 300
        self.lines_count = 10000
        
        # Инициализация обработки изображения
        self.processed_img = None
        self.pixels = None
        self.mask = None
    
    def preprocess_image(self):
        """Подготовка изображения с учетом настроек"""
        # Конвертация в grayscale
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        # Применение яркости/контраста
        processed = cv2.convertScaleAbs(gray, alpha=self.contrast, beta=self.brightness)
        
        # Инверсия при необходимости
        if self.invert:
            processed = 255 - processed
        
        # Выравнивание гистограммы
        self.processed_img = cv2.equalizeHist(processed)
        
        # Создание круглой маски
        self.mask = np.zeros((self.height, self.width), dtype=np.uint8)
        cv2.circle(self.mask, self.center, self.radius, 255, -1)
        self.processed_img = cv2.bitwise_and(self.processed_img, self.processed_img, mask=self.mask)
    
    def generate_nails(self):
        """Генерация точек крепления по окружности"""
        self.nails = []
        for i in range(self.nails_count):
            angle = 2 * math.pi * i / self.nails_count
            x = int(self.center[0] + self.radius * math.cos(angle))
            y = int(self.center[1] + self.radius * math.sin(angle))
            self.nails.append((x, y))
    
    def line_rasterization(self, x1, y1, x2, y2):
        """Алгоритм Брезенхема для растеризации линии"""
        points = set()
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        x, y = x1, y1
        sx = -1 if x1 > x2 else 1
        sy = -1 if y1 > y2 else 1
        
        if dx > dy:
            err = dx / 2.0
            while x != x2:
                if 0 <= x < self.width and 0 <= y < self.height:
                    points.add(y * self.width + x)
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != y2:
                if 0 <= x < self.width and 0 <= y < self.height:
                    points.add(y * self.width + x)
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy
        
        if 0 <= x < self.width and 0 <= y < self.height:
            points.add(y * self.width + x)
            
        return points
    
    def get_line_lightness(self, line_points):
        """Вычисление средней яркости линии"""
        total = 0
        count = 0
        for idx in line_points:
            y = idx // self.width
            x = idx % self.width
            if 0 <= x < self.width and 0 <= y < self.height:
                total += self.pixels[y, x]
                count += 1
        return total / count if count > 0 else 255
    
    def remove_line(self, line_points):
        """Обновление пикселей после добавления линии"""
        line_weight = int(255 * (self.line_weight / 100))
        for idx in line_points:
            y = idx // self.width
            x = idx % self.width
            if 0 <= x < self.width and 0 <= y < self.height:
                self.pixels[y, x] = min(255, self.pixels[y, x] + line_weight)
    
    def find_best_line(self, current_nail):
        """Поиск оптимальной следующей линии"""
        best_nail = None
        best_score = float('inf')
        best_line = None
        
        # Проверяем 50 случайных гвоздей для скорости
        for _ in range(50):
            j = random.randint(0, len(self.nails) - 1)
            if j == current_nail:
                continue
            
            x1, y1 = self.nails[current_nail]
            x2, y2 = self.nails[j]
            line_points = self.line_rasterization(x1, y1, x2, y2)
            lightness = self.get_line_lightness(line_points)
            
            if lightness < best_score:
                best_score = lightness
                best_nail = j
                best_line = line_points
        
        return best_nail, best_line
    
    def generate(self):
        """Основной алгоритм генерации String Art"""
        # Подготовка изображения
        self.preprocess_image()
        self.generate_nails()
        
        # Инициализация
        result = np.full((self.height, self.width, 3), self.background_color, dtype=np.uint8)
        self.pixels = self.processed_img.copy()
        current_nail = 0
        self.sequence = [current_nail]
        
        # Рисуем гвозди
        for (x, y) in self.nails:
            cv2.circle(result, (x, y), 2, (0, 0, 0), -1)
        
        # Основной цикл генерации
        for _ in tqdm(range(self.lines_count), desc="Генерация String Art"):
            next_nail, line_points = self.find_best_line(current_nail)
            
            if next_nail is None:
                break
            
            # Рисуем линию
            cv2.line(result, self.nails[current_nail], self.nails[next_nail], 
                    self.line_color, 1, lineType=cv2.LINE_AA)
            
            # Обновляем пиксели
            self.remove_line(line_points)
            
            current_nail = next_nail
            self.sequence.append(current_nail)
        
        return result
    
    def save_as_png(self, image, path):
        """Сохранение результата в PNG"""
        cv2.imwrite(path, image)
    
    def save_as_svg(self, path):
        """Сохранение результата в SVG"""
        svg = f'<svg viewBox="0 0 {self.width} {self.height}" xmlns="http://www.w3.org/2000/svg">\n'
        svg += f'<rect width="100%" height="100%" fill="rgb{self.background_color}"/>\n'
        
        # Круг для области String Art
        svg += f'<circle cx="{self.center[0]}" cy="{self.center[1]}" r="{self.radius}" fill="rgb{self.background_color}"/>\n'
        
        # Гвозди
        for (x, y) in self.nails:
            svg += f'<circle cx="{x}" cy="{y}" r="2" fill="black"/>\n'
        
        # Линии
        for i in range(1, len(self.sequence)):
            x1, y1 = self.nails[self.sequence[i-1]]
            x2, y2 = self.nails[self.sequence[i]]
            svg += f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="rgb{self.line_color}" stroke-width="1"/>\n'
        
        svg += '</svg>'
        
        with open(path, 'w') as f:
            f.write(svg)
    
    def save_as_stringart(self, path):
        """Сохранение в специальном формате String Art"""
        data = {
            'nails': self.nails,
            'color': self.line_color,
            'background': self.background_color,
            'sequence': self.sequence
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)

# Пример использования
if __name__ == "__main__":
    try:
        # Инициализация генератора
        generator = StringArtGenerator("face1.jpg")
        
        # Настройки (можно менять)
        generator.contrast = 1.0      # Контрастность (1.0 = оригинал)
        generator.brightness = 0     # Яркость (0 = оригинал)
        generator.invert = False       # Инверсия яркости
        generator.line_weight = 10    # Непрозрачность линий (15%)
        generator.nails_count = 250   # Количество гвоздей
        generator.lines_count = 1000 # Количество линий
        
        # Генерация String Art
        result = generator.generate()
        
        # Сохранение результатов
        generator.save_as_png(result, "string_art.png")
        generator.save_as_svg("string_art.svg")
        generator.save_as_stringart("string_art.json")
        
        print("String Art успешно сгенерирован и сохранен!")
        
    except Exception as e:
        print(f"Ошибка: {str(e)}")