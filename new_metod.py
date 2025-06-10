import cv2
import numpy as np
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
import urllib.request
from skimage.metrics import structural_similarity as ssim

class StringArtGenerator:
    def __init__(self, image_path, nails=200, iterations=2000, output_size=500):
        self.nails = nails
        self.iterations = iterations
        self.output_size = output_size
        
        # Загрузка и подготовка изображения
        if isinstance(image_path, str) and image_path.startswith(('http://', 'https://')):
            self.image = self._download_image(image_path)
        else:
            self.image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        if self.image is None:
            raise ValueError("Не удалось загрузить изображение")
        
        self.image = cv2.resize(self.image, (output_size, output_size))
        self.target = 255 - self.image  # Инверсия для работы с темными линиями
        self.target_float = self.target.astype(np.float32) / 255.0
        
        # Инициализация холста (белый)
        self.canvas = np.ones((output_size, output_size), dtype=np.float32)
        self.nail_positions = self._calculate_nail_positions()
        self.error_history = []
        self.ssim_history = []
    
    def _download_image(self, url):
        resp = urllib.request.urlopen(url)
        image = np.asarray(bytearray(resp.read()), dtype=np.uint8)
        return cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
    
    def _calculate_nail_positions(self):
        center = self.output_size // 2
        radius = self.output_size // 2 - 10
        angles = np.linspace(0, 2*np.pi, self.nails, endpoint=False)
        x = (center + radius * np.cos(angles)).astype(int)
        y = (center + radius * np.sin(angles)).astype(int)
        return np.column_stack((x, y))
    
    def _draw_line(self, start_idx, end_idx, intensity=0.15):
        """Рисует линию с учетом накопленной интенсивности"""
        start = self.nail_positions[start_idx]
        end = self.nail_positions[end_idx]
        line_mask = np.zeros_like(self.canvas)
        cv2.line(line_mask, tuple(start), tuple(end), intensity, 1)
        self.canvas = np.clip(self.canvas - line_mask, 0, 1)
    
    def _calculate_error(self):
        """Вычисляет MSE и SSIM между холстом и целевым изображением"""
        mse = np.mean((self.target_float - self.canvas)**2)
        
        # Нормализация для SSIM
        canvas_uint8 = (self.canvas * 255).astype(np.uint8)
        target_uint8 = (self.target_float * 255).astype(np.uint8)
        ssim_val = ssim(target_uint8, canvas_uint8, data_range=255)
        
        return mse, ssim_val
    
    def _find_best_next_nail(self, current_nail):
        best_error = float('inf')
        best_nail = current_nail
        
        for candidate in np.random.choice(self.nails, size=min(50, self.nails), replace=False):
            if candidate == current_nail:
                continue
                
            temp_canvas = self.canvas.copy()
            self._draw_line(current_nail, candidate)
            error = np.mean((self.target_float - self.canvas)**2)
            self.canvas = temp_canvas
            
            if error < best_error:
                best_error = error
                best_nail = candidate
                
        return best_nail
    
    def generate(self):
        current_nail = 0
        
        for i in tqdm(range(self.iterations)):
            next_nail = self._find_best_next_nail(current_nail)
            self._draw_line(current_nail, next_nail)
            current_nail = next_nail
            
            if i % 100 == 0:
                mse, ssim_val = self._calculate_error()
                self.error_history.append(mse)
                self.ssim_history.append(ssim_val)
        
        result = (1 - self.canvas) * 255
        return result.astype(np.uint8)

# Тестовое изображение (простая геометрическая фигура для отладки)
TEST_URL = "edik.jpg"

# Параметры
params = {
    'nails': 500,
    'iterations': 10000, 
    # 10000
    'output_size': 800
    # 600
}

# Генерация
generator = StringArtGenerator(TEST_URL, **params)
result = generator.generate()

# Визуализация
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(255 - generator.target, cmap='gray')
plt.title("Original (inverted)")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(result, cmap='gray')
plt.title(f"String Art\nNails: {params['nails']}, Iterations: {params['iterations']}")
plt.axis('off')

plt.tight_layout()
plt.show()

# График ошибок
plt.figure(figsize=(10, 4))
plt.plot(generator.error_history, label='MSE')
plt.plot(generator.ssim_history, label='SSIM')
plt.xlabel('Iteration (x100)')
plt.ylabel('Metric Value')
plt.title('Convergence Metrics')
plt.legend()
plt.grid()
plt.show()