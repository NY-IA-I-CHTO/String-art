import cv2
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

class StringArtGenerator:
    def __init__(self, image_path, nails=200, iterations=2000, output_size=500, gaussian_blur=True, blur_kernel=(5,5)):
        self.nails = nails
        self.iterations = iterations
        self.output_size = output_size
        self.gaussian_blur = gaussian_blur
        self.blur_kernel = blur_kernel
        
        # Загрузка изображения
        if isinstance(image_path, str) and image_path.startswith(('http://', 'https://')):
            self.original_image = self._download_image(image_path)
        else:
            self.original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        if self.original_image is None:
            raise ValueError("Не удалось загрузить изображение")
        
        # Предварительная обработка
        self.original_image = cv2.resize(self.original_image, (output_size, output_size))
        self.target = 255 - self.original_image
        self.target_float = self.target.astype(np.float32) / 255.0
        
        if self.gaussian_blur:
            self.target_blurred = cv2.GaussianBlur(self.target, self.blur_kernel, 0)
            self.target_float_blurred = self.target_blurred.astype(np.float32) / 255.0
        else:
            self.target_blurred = self.target
            self.target_float_blurred = self.target_float
        
        # Инициализация холста
        self.canvas = np.ones((output_size, output_size), dtype=np.float32)
        self.nail_positions = self._calculate_nail_positions()
        self.metric_history = {'PSNR': [], 'SSIM': [], 'IQ5': []}
        self.metric_history_blurred = {'PSNR': [], 'SSIM': [], 'IQ5': []}
    
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
    
    def _calculate_iq5(self, original, processed):
        diff = np.abs(original.astype(float) - processed.astype(float))
        return np.percentile(diff, 5)
    
    def _calculate_metrics(self, blurred=False):
        canvas_uint8 = (self.canvas * 255).astype(np.uint8)
        target_uint8 = (self.target_float * 255).astype(np.uint8)
        
        if blurred:
            canvas_uint8 = cv2.GaussianBlur(canvas_uint8, self.blur_kernel, 0)
        
        return {
            'PSNR': psnr(target_uint8, canvas_uint8),
            'SSIM': ssim(target_uint8, canvas_uint8, data_range=255),
            'IQ5': self._calculate_iq5(target_uint8, canvas_uint8)
        }
    
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
        
        for i in tqdm(range(self.iterations), desc="Генерация String Art"):
            next_nail = self._find_best_next_nail(current_nail)
            self._draw_line(current_nail, next_nail)
            current_nail = next_nail
            
            if i % 100 == 0:
                metrics = self._calculate_metrics(blurred=False)
                for k in metrics:
                    self.metric_history[k].append(metrics[k])
                
                if self.gaussian_blur:
                    metrics_blurred = self._calculate_metrics(blurred=True)
                    for k in metrics_blurred:
                        self.metric_history_blurred[k].append(metrics_blurred[k])
        
        result = (1 - self.canvas) * 255
        result_blurred = cv2.GaussianBlur(result, self.blur_kernel, 0) if self.gaussian_blur else None
        
        return result.astype(np.uint8), result_blurred.astype(np.uint8) if result_blurred is not None else None
    
    def _plot_results(self, result, history, title_suffix=""):
        metrics = self._calculate_metrics(blurred=("blurred" in title_suffix.lower()))
        
        plt.figure(figsize=(18, 10))
        
        # 1. Оригинальное изображение
        plt.subplot(2, 3, 1)
        plt.imshow(self.original_image, cmap='gray')
        plt.title('Оригинальное изображение')
        plt.axis('off')
        
        # 2. Инвертированное изображение
        plt.subplot(2, 3, 2)
        plt.imshow(self.target, cmap='gray')
        plt.title('Инвертированное изображение')
        plt.axis('off')
        
        # 3. Результат String Art
        plt.subplot(2, 3, 3)
        plt.imshow(result, cmap='gray')
        plt.title(f'Результат String Art {title_suffix}'.strip())
        plt.axis('off')
        
        # 4-6. Графики метрик с значениями
        metrics_data = [
            ('PSNR', 'PSNR (пиковое отношение сигнал/шум)', 'blue', ' дБ', f"{metrics['PSNR']:.2f}"),
            ('SSIM', 'SSIM (структурное сходство)', 'green', '', f"{metrics['SSIM']:.4f}"),
            ('IQ5', 'IQ5 (5-й процентиль ошибок)', 'red', '', f"{metrics['IQ5']:.4f}")
        ]
        
        for i, (metric_key, metric_name, color, unit, value) in enumerate(metrics_data, start=4):
            plt.subplot(2, 3, i)
            plt.plot(history[metric_key], color=color)
            plt.title(f'{metric_name} = {value}{unit}')
            plt.xlabel('Итерации (x100)')
            plt.ylabel(metric_name.split(' ')[0] + unit)
            plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def analyze_results(self, result, result_blurred):
        # Вывод обычного результата
        print("\nРезультаты для обычного String Art:")
        self._plot_results(result, self.metric_history)
        
        # Вывод сглаженного результата (если есть)
        if result_blurred is not None:
            print("\nРезультаты для сглаженного String Art:")
            self._plot_results(result_blurred, self.metric_history_blurred, "(сглаженный)")

# Параметры
params = {
    'image_path': "edik.jpg",
    'nails': 250,
    'iterations': 200,
    'output_size': 400,
    'gaussian_blur': True,
    'blur_kernel': (5,5)
}

# Генерация и анализ
print("Начало обработки...")
generator = StringArtGenerator(**params)
result, result_blurred = generator.generate()
generator.analyze_results(result, result_blurred)

# Сохранение результатов
cv2.imwrite("original.png", generator.original_image)
cv2.imwrite("string_art_result.png", result)
if result_blurred is not None:
    cv2.imwrite("string_art_result_blurred.png", result_blurred)
print("Обработка завершена. Результаты сохранены.")