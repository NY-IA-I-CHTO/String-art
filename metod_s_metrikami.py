import cv2
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import structural_similarity as ssim

class StringArtGenerator:
    def __init__(self, image_path, nails=200, iterations=2000, output_size=500):
        self.nails = nails
        self.iterations = iterations
        self.output_size = output_size
        
        # Загрузка изображения
        if isinstance(image_path, str) and image_path.startswith(('http://', 'https://')):
            self.image = self._download_image(image_path)
        else:
            self.image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        if self.image is None:
            raise ValueError("Не удалось загрузить изображение")
        
        # Предварительная обработка
        self.image = cv2.resize(self.image, (output_size, output_size))
        self.target = 255 - self.image  # Инверсия для работы с темными линиями
        self.target_float = self.target.astype(np.float32) / 255.0
        
        # Инициализация холста
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
        start = self.nail_positions[start_idx]
        end = self.nail_positions[end_idx]
        line_mask = np.zeros_like(self.canvas)
        cv2.line(line_mask, tuple(start), tuple(end), intensity, 1)
        self.canvas = np.clip(self.canvas - line_mask, 0, 1)
    
    def _calculate_metrics(self):
        canvas_uint8 = (self.canvas * 255).astype(np.uint8)
        target_uint8 = (self.target_float * 255).astype(np.uint8)
        return {
            'PSNR': psnr(target_uint8, canvas_uint8),
            'MSE': mse(target_uint8, canvas_uint8),
            'SSIM': ssim(target_uint8, canvas_uint8, data_range=255),
            'IQ5': self._calculate_iq5(target_uint8, canvas_uint8)
        }
    
    def _calculate_iq5(self, original, processed):
        diff = np.abs(original.astype(float) - processed.astype(float))
        return np.percentile(diff, 5)
    
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
        
        for i in tqdm(range(self.iterations), desc="Generating String Art"):
            next_nail = self._find_best_next_nail(current_nail)
            self._draw_line(current_nail, next_nail)
            current_nail = next_nail
            
            if i % 100 == 0:
                metrics = self._calculate_metrics()
                self.error_history.append(metrics['MSE'])
                self.ssim_history.append(metrics['SSIM'])
        
        result = (1 - self.canvas) * 255
        return result.astype(np.uint8)
    
    def analyze_results(self, result):
        original = self.target
        if len(original.shape) == 3:
            original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        if len(result.shape) == 3:
            result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        
        metrics = self._calculate_metrics()
        
        plt.figure(figsize=(18, 12))
        
        # 1. Исходное vs Результат
        plt.subplot(2, 3, 1)
        plt.imshow(original, cmap='gray')
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(2, 3, 2)
        plt.imshow(result, cmap='gray')
        plt.title('String Art Result')
        plt.axis('off')
        
        # 2. Разница изображений
        plt.subplot(2, 3, 3)
        diff = cv2.absdiff(original, result)
        plt.imshow(diff, cmap='hot')
        plt.title(f'Difference Map\nMSE: {metrics["MSE"]:.2f}')
        plt.colorbar()
        plt.axis('off')
        
        # 3. Гистограммы
        plt.subplot(2, 3, 4)
        plt.hist(original.ravel(), 256, [0, 256], color='blue', alpha=0.5)
        plt.hist(result.ravel(), 256, [0, 256], color='red', alpha=0.5)
        plt.title('Intensity Histograms\nBlue=Original, Red=Result')
        
        # 4. Метрики
        plt.subplot(2, 3, 5)
        text = "\n".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        plt.text(0.1, 0.5, text, fontsize=14)
        plt.axis('off')
        plt.title('Quality Metrics')
        
        # 5. График сходимости
        plt.subplot(2, 3, 6)
        plt.plot(self.error_history, label='MSE')
        plt.plot(self.ssim_history, label='SSIM')
        plt.xlabel('Iteration (x100)')
        plt.ylabel('Value')
        plt.title('Convergence Metrics')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        return metrics

# Параметры
params = {
    'image_path': "edik.jpg",
    'nails': 500,
    'iterations': 10000,
    'output_size': 1000
}

# Генерация и анализ
generator = StringArtGenerator(**params)
result = generator.generate()
metrics = generator.analyze_results(result)

# Сохранение результатов
cv2.imwrite("original.png", generator.target)
cv2.imwrite("string_art_result.png", result)
print("Final Metrics:", metrics)