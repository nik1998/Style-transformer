import os
import cv2
import numpy as np
from PIL import Image
import random
from scipy.ndimage import gaussian_filter


class RainGenerator:
    """
    Класс для генерации реалистичного эффекта дождя на изображениях
    Вдохновлен библиотекой Albumentations, но реализован без её использования
    """
    
    def __init__(self, seed=None):
        """
        Инициализация генератора дождя
        
        Args:
            seed (int, optional): Seed для генератора случайных чисел. По умолчанию None.
        """
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            self.random_state = np.random.RandomState(seed)
        else:
            self.random_state = np.random.RandomState()
    
    def _create_raindrop_mask(self, shape, length, width):
        """
        Создание маски одной капли дождя с градиентом прозрачности
        
        Args:
            shape (tuple): Размер маски (высота, ширина)
            length (int): Длина капли
            width (int): Ширина капли
            
        Returns:
            numpy.ndarray: Маска капли с градиентом прозрачности
        """
        # Создаем пустую маску
        h, w = shape
        mask = np.zeros((h, w), dtype=np.float32)
        
        # Центр капли
        center_y, center_x = h // 2, w // 2
        
        # Создаем эллипс для капли
        cv2.ellipse(mask, 
                   (center_x, center_y),  # центр
                   (width // 2, length // 2),  # полуоси
                   0,  # угол поворота
                   0, 360,  # начальный и конечный угол
                   1.0,  # цвет (белый)
                   -1)  # заполненный
        
        # Создаем градиент прозрачности от центра к краям
        y, x = np.ogrid[:h, :w]
        # Нормализованное расстояние от центра (0 в центре, 1 на границе эллипса)
        dist_x = (x - center_x) / (width / 2)
        dist_y = (y - center_y) / (length / 2)
        dist = np.sqrt(dist_x**2 + dist_y**2)
        
        # Применяем градиент только внутри эллипса
        gradient = np.clip(1.0 - dist, 0, 1) * mask
        
        return gradient
    
    def _create_rain_layer_algorithm1(self, shape, slant, drop_length, drop_width, drop_color, density):
        """
        Алгоритм 1: Улучшенная форма капель с градиентом прозрачности
        
        Args:
            shape (tuple): Размер изображения (высота, ширина)
            slant (int): Угол наклона дождя (в градусах от вертикали)
            drop_length (int): Длина капель дождя
            drop_width (int): Ширина капель дождя
            drop_color (float): Яркость капель (0-1)
            density (float): Плотность дождя (0.0-1.0)
            
        Returns:
            numpy.ndarray: Слой с каплями дождя
        """
        # Создаем пустой слой
        h, w = shape
        rain_layer = np.zeros(shape, dtype=np.float32)
        
        # Вычисляем количество капель на основе плотности и размера изображения
        num_drops = int(density * w * h / 100)
        
        # Преобразуем угол наклона в радианы (90 - вертикальный дождь)
        slant_rad = np.deg2rad(slant - 90)
        
        # Создаем маску капли с градиентом
        drop_mask_size = (drop_length * 2, drop_width * 2)
        drop_mask = self._create_raindrop_mask(drop_mask_size, drop_length, drop_width)
        
        # Поворачиваем маску капли в соответствии с углом наклона
        rotation_matrix = cv2.getRotationMatrix2D(
            (drop_mask_size[1] // 2, drop_mask_size[0] // 2),
            90 - slant,  # угол поворота
            1.0  # масштаб
        )
        rotated_drop = cv2.warpAffine(drop_mask, rotation_matrix, 
                                     (drop_mask_size[1], drop_mask_size[0]))
        
        # Генерируем случайные позиции для капель
        x_positions = self.random_state.randint(0, w, num_drops)
        y_positions = self.random_state.randint(0, h, num_drops)
        
        # Размещаем капли на слое
        for i in range(num_drops):
            x = x_positions[i]
            y = y_positions[i]
            
            # Определяем область для размещения капли
            x1 = max(0, x - drop_mask_size[1] // 2)
            y1 = max(0, y - drop_mask_size[0] // 2)
            x2 = min(w, x + drop_mask_size[1] // 2)
            y2 = min(h, y + drop_mask_size[0] // 2)
            
            # Определяем соответствующую область в маске капли
            mx1 = max(0, drop_mask_size[1] // 2 - x)
            my1 = max(0, drop_mask_size[0] // 2 - y)
            mx2 = drop_mask_size[1] - max(0, x + drop_mask_size[1] // 2 - w)
            my2 = drop_mask_size[0] - max(0, y + drop_mask_size[0] // 2 - h)
            
            # Проверяем, что области имеют положительный размер
            if x2 > x1 and y2 > y1 and mx2 > mx1 and my2 > my1:
                # Размещаем каплю на слое с учетом градиента прозрачности
                drop_region = rotated_drop[my1:my2, mx1:mx2]
                rain_layer[y1:y2, x1:x2] = np.maximum(
                    rain_layer[y1:y2, x1:x2],
                    drop_region * drop_color
                )
        
        return rain_layer
    
    def _create_rain_layer_algorithm2(self, shape, slant, drop_length, drop_width, drop_color, density, layer_index=0):
        """
        Алгоритм 2: Многослойный дождь с разной скоростью и размерами капель
        
        Args:
            shape (tuple): Размер изображения (высота, ширина)
            slant (int): Угол наклона дождя (в градусах от вертикали)
            drop_length (int): Длина капель дождя
            drop_width (int): Ширина капель дождя
            drop_color (float): Яркость капель (0-1)
            density (float): Плотность дождя (0.0-1.0)
            layer_index (int): Индекс слоя (0 - задний план, 2 - передний план)
            
        Returns:
            numpy.ndarray: Слой с каплями дождя
        """
        # Создаем пустой слой
        h, w = shape
        rain_layer = np.zeros(shape, dtype=np.float32)
        
        # Настраиваем параметры в зависимости от слоя
        if layer_index == 0:  # Задний план
            # Меньшие, более медленные капли
            drop_length = int(drop_length * 0.7)
            drop_width = max(1, int(drop_width * 0.7))
            blur_sigma = 0.3
        elif layer_index == 1:  # Средний план
            # Средние капли
            blur_sigma = 0.5
        else:  # Передний план
            # Более крупные, быстрые капли
            drop_length = int(drop_length * 1.3)
            drop_width = int(drop_width * 1.3)
            blur_sigma = 0.7
        
        # Вычисляем количество капель на основе плотности и размера изображения
        num_drops = int(density * w * h / 100)
        
        # Преобразуем угол наклона в радианы (90 - вертикальный дождь)
        slant_rad = np.deg2rad(slant - 90)
        
        # Вычисляем смещение по X на основе угла наклона и длины капли
        dx = int(drop_length * np.sin(slant_rad))
        dy = int(drop_length * np.cos(slant_rad))
        
        # Генерируем случайные позиции для капель
        x_positions = self.random_state.randint(0, w, num_drops)
        y_positions = self.random_state.randint(0, h, num_drops)
        
        # Рисуем каждую каплю
        for i in range(num_drops):
            x = x_positions[i]
            y = y_positions[i]
            
            # Вычисляем конечные координаты капли
            x2 = x + dx
            y2 = y + dy
            
            # Проверяем, что капля находится в пределах изображения
            if 0 <= x < w and 0 <= y < h and 0 <= x2 < w and 0 <= y2 < h:
                # Для переднего плана добавляем motion blur
                if layer_index == 2:
                    # Создаем маску для капли с motion blur
                    line_mask = np.zeros(shape, dtype=np.float32)
                    cv2.line(line_mask, (x, y), (x2, y2), 1.0, drop_width)
                    # Применяем размытие в направлении движения
                    kernel_size = max(3, drop_width * 2 + 1)
                    kernel = np.zeros((kernel_size, kernel_size))
                    angle_deg = 90 - slant
                    angle_rad = np.deg2rad(angle_deg)
                    x_dir = np.cos(angle_rad)
                    y_dir = np.sin(angle_rad)
                    center = kernel_size // 2
                    for i in range(kernel_size):
                        for j in range(kernel_size):
                            dist = abs((i - center) * y_dir - (j - center) * x_dir)
                            if dist < 1.0:
                                kernel[i, j] = 1.0
                    kernel = kernel / kernel.sum()
                    line_mask = cv2.filter2D(line_mask, -1, kernel)
                    rain_layer = np.maximum(rain_layer, line_mask * drop_color)
                else:
                    # Для среднего и заднего плана используем обычные линии
                    cv2.line(rain_layer, (x, y), (x2, y2), drop_color, drop_width)
        
        # Применяем размытие в зависимости от слоя
        rain_layer = gaussian_filter(rain_layer, sigma=blur_sigma)
        
        return rain_layer
    
    def _create_rain_layer_algorithm3(self, shape, slant, drop_length, drop_width, drop_color, density):
        """
        Алгоритм 3: Физически более реалистичное моделирование капель
        
        Args:
            shape (tuple): Размер изображения (высота, ширина)
            slant (int): Угол наклона дождя (в градусах от вертикали)
            drop_length (int): Длина капель дождя
            drop_width (int): Ширина капель дождя
            drop_color (float): Яркость капель (0-1)
            density (float): Плотность дождя (0.0-1.0)
            
        Returns:
            numpy.ndarray: Слой с каплями дождя
        """
        # Создаем пустой слой
        h, w = shape
        rain_layer = np.zeros(shape, dtype=np.float32)
        highlight_layer = np.zeros(shape, dtype=np.float32)
        
        # Вычисляем количество капель на основе плотности и размера изображения
        num_drops = int(density * w * h / 100)
        
        # Преобразуем угол наклона в радианы (90 - вертикальный дождь)
        slant_rad = np.deg2rad(slant - 90)
        
        # Генерируем случайные позиции для капель
        x_positions = self.random_state.randint(0, w, num_drops)
        y_positions = self.random_state.randint(0, h, num_drops)
        
        # Генерируем случайные длины и ширины для капель
        if isinstance(drop_length, tuple):
            lengths = self.random_state.randint(drop_length[0], drop_length[1] + 1, num_drops)
        else:
            lengths = np.full(num_drops, drop_length)
            
        if isinstance(drop_width, tuple):
            widths = self.random_state.randint(drop_width[0], drop_width[1] + 1, num_drops)
        else:
            widths = np.full(num_drops, drop_width)
        
        # Рисуем каждую каплю
        for i in range(num_drops):
            x = x_positions[i]
            y = y_positions[i]
            length = lengths[i]
            width = widths[i]
            
            # Вычисляем конечные координаты капли
            dx = int(length * np.sin(slant_rad))
            dy = int(length * np.cos(slant_rad))
            x2 = x + dx
            y2 = y + dy
            
            # Проверяем, что капля находится в пределах изображения
            if 0 <= x < w and 0 <= y < h and 0 <= x2 < w and 0 <= y2 < h:
                # Создаем маску для капли с переменной шириной (тоньше на конце)
                drop_mask = np.zeros(shape, dtype=np.float32)
                
                # Рисуем каплю с переменной шириной
                # Разделяем каплю на несколько сегментов
                num_segments = 8
                for s in range(num_segments):
                    # Вычисляем координаты сегмента
                    t1 = s / num_segments
                    t2 = (s + 1) / num_segments
                    x1 = int(x + dx * t1)
                    y1 = int(y + dy * t1)
                    x2 = int(x + dx * t2)
                    y2 = int(y + dy * t2)
                    
                    # Ширина сегмента уменьшается к концу капли
                    segment_width = max(1, int(width * (1 - 0.7 * t1)))
                    
                    # Рисуем сегмент
                    cv2.line(drop_mask, (x1, y1), (x2, y2), 1.0, segment_width)
                
                # Добавляем блик в верхней части капли (имитация преломления света)
                highlight_x = x
                highlight_y = y
                highlight_size = max(1, width // 2)
                cv2.circle(highlight_layer, (highlight_x, highlight_y), 
                          highlight_size, drop_color * 1.5, -1)
                
                # Добавляем каплю на слой
                rain_layer = np.maximum(rain_layer, drop_mask * drop_color)
        
        # Применяем размытие для более естественного вида
        rain_layer = gaussian_filter(rain_layer, sigma=0.5)
        highlight_layer = gaussian_filter(highlight_layer, sigma=0.3)
        
        # Объединяем слой капель и бликов
        combined_layer = np.maximum(rain_layer, highlight_layer)
        
        return combined_layer
    
    def _generate_rain_streaks(self, shape, algorithm=1, angle=90, length=(40, 80), width=(1, 3), 
                              intensity=1.0, density=0.08, transparency=0.7):
        """
        Генерация полос дождя с одинаковым углом наклона
        
        Args:
            shape (tuple): Размер выходного изображения (высота, ширина)
            algorithm (int): Алгоритм генерации дождя (1, 2 или 3)
            angle (float): Угол наклона капель дождя в градусах (90 - вертикальный)
            length (tuple or int): Диапазон длин капель или фиксированная длина
            width (tuple or int): Диапазон ширин капель или фиксированная ширина
            intensity (float): Интенсивность (яркость) капель дождя (0.0-1.0)
            density (float): Плотность дождя (0.0-1.0)
            transparency (float): Прозрачность дождя (0.0-1.0)
            
        Returns:
            numpy.ndarray: Маска дождя с значениями в диапазоне [0, 1]
        """
        h, w = shape
        
        # Определяем длину и ширину капель
        if isinstance(length, tuple):
            drop_length = self.random_state.randint(length[0], length[1] + 1)
        else:
            drop_length = length
            
        if isinstance(width, tuple):
            drop_width = self.random_state.randint(width[0], width[1] + 1)
        else:
            drop_width = width
        
        # Создаем маску дождя в зависимости от выбранного алгоритма
        rain_mask = np.zeros(shape, dtype=np.float32)
        
        if algorithm == 1:
            # Алгоритм 1: Улучшенная форма капель с градиентом прозрачности
            rain_layer = self._create_rain_layer_algorithm1(
                shape, angle, drop_length, drop_width, intensity, density)
            rain_mask = np.maximum(rain_mask, rain_layer)
            
            # Добавляем слой с более короткими каплями для разнообразия
            short_drop_length = max(10, drop_length // 2)
            short_drop_width = max(1, drop_width)
            short_rain_layer = self._create_rain_layer_algorithm1(
                shape, angle, short_drop_length, short_drop_width, 
                intensity * 0.9, density * 0.4)
            rain_mask = np.maximum(rain_mask, short_rain_layer)
            
        elif algorithm == 2:
            # Алгоритм 2: Многослойный дождь с разной скоростью и размерами капель
            # Задний план (маленькие капли)
            back_layer = self._create_rain_layer_algorithm2(
                shape, angle, drop_length, drop_width, 
                intensity * 0.7, density * 0.4, layer_index=0)
            rain_mask = np.maximum(rain_mask, back_layer)
            
            # Средний план
            mid_layer = self._create_rain_layer_algorithm2(
                shape, angle, drop_length, drop_width, 
                intensity * 0.9, density * 0.3, layer_index=1)
            rain_mask = np.maximum(rain_mask, mid_layer)
            
            # Передний план (крупные капли)
            front_layer = self._create_rain_layer_algorithm2(
                shape, angle, drop_length, drop_width, 
                intensity, density * 0.3, layer_index=2)
            rain_mask = np.maximum(rain_mask, front_layer)
            
        else:  # algorithm == 3
            # Алгоритм 3: Физически более реалистичное моделирование капель
            rain_layer = self._create_rain_layer_algorithm3(
                shape, angle, drop_length, drop_width, intensity, density)
            rain_mask = np.maximum(rain_mask, rain_layer)
        
        # Применяем небольшое размытие для более естественного вида
        rain_mask = gaussian_filter(rain_mask, sigma=0.5)
        
        # Нормализуем значения до диапазона [0, 1]
        if rain_mask.max() > 0:
            rain_mask = rain_mask / rain_mask.max()
        
        # Применяем прозрачность
        rain_mask = rain_mask * (1.0 - transparency)
        
        return rain_mask
    
    def apply_rain(self, image, algorithm=1, angle=90, length=(40, 80), width=(1, 3), 
                  transparency=0.7, density=0.08, intensity=1.0, seed=None):
        """
        Применение эффекта дождя к изображению
        
        Args:
            image (numpy.ndarray): Исходное изображение (RGB)
            algorithm (int): Алгоритм генерации дождя (1, 2 или 3)
            angle (float): Угол наклона капель дождя в градусах (90 - вертикальный)
            length (tuple or int): Диапазон длин капель или фиксированная длина
            width (tuple or int): Диапазон ширин капель или фиксированная ширина
            transparency (float): Прозрачность дождя (0.0-1.0), где 0 - полностью прозрачный
            density (float): Плотность дождя (0.0-1.0)
            intensity (float): Интенсивность (яркость) капель дождя (0.0-1.0)
            seed (int, optional): Seed для генератора случайных чисел
            
        Returns:
            numpy.ndarray: Изображение с эффектом дождя
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            random_state = np.random.RandomState(seed)
        else:
            random_state = self.random_state
        
        # Преобразование в float32 для обработки
        image_float = image.astype(np.float32) / 255.0
        
        # Генерация маски дождя с заданными параметрами
        rain_mask = self._generate_rain_streaks(
            image.shape[:2],
            algorithm=algorithm,
            angle=angle,
            length=length,
            width=width,
            intensity=intensity,
            density=density,
            transparency=transparency
        )
        
        # Расширение маски до 3 каналов для RGB изображения
        rain_mask_3d = np.stack([rain_mask] * 3, axis=-1)
        
        # Создаем эффект затемнения и синеватого оттенка для дождливой атмосферы
        # Чем сильнее дождь, тем сильнее затемнение
        darkness_factor = 1.0 - (density * intensity * 0.3)
        darkness_factor = max(0.7, darkness_factor)  # Не затемнять слишком сильно
        
        # Синеватый оттенок для дождливой погоды
        blue_tint = np.array([
            darkness_factor * 0.9,  # R - уменьшение красного
            darkness_factor * 0.95, # G - небольшое уменьшение зеленого
            darkness_factor         # B - сохранение синего
        ])
        
        # Применение затемнения и синего оттенка
        darkened_image = image_float * blue_tint.reshape(1, 1, 3)
        
        # Добавление легкого размытия для имитации мокрого стекла/линзы
        blur_sigma = random_state.uniform(0.3, 0.7)
        darkened_image = gaussian_filter(darkened_image, sigma=[blur_sigma * 0.5, blur_sigma * 0.5, 0])
        
        # Цвет дождя (слегка голубоватый)
        rain_color = np.array([0.9, 0.9, 1.0])
        
        # Применение эффекта дождя
        rainy_image = darkened_image * (1.0 - rain_mask_3d) + rain_mask_3d * rain_color.reshape(1, 1, 3)
        
        # Добавление шума для имитации капель на линзе/стекле
        noise_level = min(0.02, density * 0.01)
        noise = random_state.normal(0, noise_level, rainy_image.shape)
        rainy_image = np.clip(rainy_image + noise, 0, 1)
        
        # Преобразование обратно в uint8
        return (rainy_image * 255).astype(np.uint8)
    
    def apply_rain_to_image(self, image_path, output_path=None, algorithm=1, angle=90, length=(40, 80), 
                           width=(1, 3), transparency=0.7, density=0.08, intensity=1.0, seed=None):
        """
        Применение эффекта дождя к изображению из файла и сохранение результата
        
        Args:
            image_path (str): Путь к исходному изображению
            output_path (str, optional): Путь для сохранения результата. По умолчанию None (не сохранять).
            algorithm (int): Алгоритм генерации дождя (1, 2 или 3)
            angle (float): Угол наклона капель дождя в градусах (90 - вертикальный)
            length (tuple or int): Диапазон длин капель или фиксированная длина
            width (tuple or int): Диапазон ширин капель или фиксированная ширина
            transparency (float): Прозрачность дождя (0.0-1.0), где 0 - полностью прозрачный
            density (float): Плотность дождя (0.0-1.0)
            intensity (float): Интенсивность (яркость) капель дождя (0.0-1.0)
            seed (int, optional): Seed для генератора случайных чисел
            
        Returns:
            numpy.ndarray: Изображение с эффектом дождя
        """
        # Чтение изображения
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Не удалось прочитать изображение: {image_path}")
        
        # Преобразование BGR в RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Применение эффекта дождя с заданными параметрами
        rainy_image = self.apply_rain(
            image_rgb, 
            algorithm=algorithm,
            angle=angle,
            length=length,
            width=width,
            transparency=transparency,
            density=density,
            intensity=intensity,
            seed=seed
        )
        
        # Сохранение результата, если указан путь
        if output_path:
            # Создание директории, если она не существует
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Преобразование RGB в BGR для сохранения
            rainy_image_bgr = cv2.cvtColor(rainy_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, rainy_image_bgr)
            print(f"Изображение с дождем сохранено в {output_path}")
        
        return rainy_image


if __name__ == "__main__":
    # Демонстрация работы генератора дождя
    import matplotlib.pyplot as plt
    import os
    
    # Использование изображения из датасета
    input_image_path = "../dataset/input/patch_0_0.png"
    output_dir = "./"
    seed = 42  # Фиксированный seed для воспроизводимости
    
    # Создание генератора дождя
    rain_generator = RainGenerator(seed=seed)
    
    # Чтение изображения
    image = cv2.imread(input_image_path)
    if image is None:
        print(f"Не удалось прочитать изображение: {input_image_path}")
        exit(1)
    
    # Преобразование BGR в RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Применение эффекта дождя с разными алгоритмами
    rainy_image1 = rain_generator.apply_rain(
        image_rgb.copy(), 
        algorithm=1,         # Алгоритм 1: Улучшенная форма капель с градиентом прозрачности
        angle=90,            # Вертикальный дождь
        length=(40, 80),     # Длинные капли
        width=(1, 3),        # Тонкие капли
        transparency=0.8,    # Высокая прозрачность
        density=0.08,        # Средняя плотность
        intensity=1.0,       # Высокая интенсивность
        seed=seed
    )
    
    rainy_image2 = rain_generator.apply_rain(
        image_rgb.copy(), 
        algorithm=2,         # Алгоритм 2: Многослойный дождь с разной скоростью и размерами капель
        angle=90,            # Вертикальный дождь
        length=(40, 80),     # Длинные капли
        width=(1, 3),        # Тонкие капли
        transparency=0.8,    # Высокая прозрачность
        density=0.08,        # Средняя плотность
        intensity=1.0,       # Высокая интенсивность
        seed=seed
    )
    
    rainy_image3 = rain_generator.apply_rain(
        image_rgb.copy(), 
        algorithm=3,         # Алгоритм 3: Физически более реалистичное моделирование капель
        angle=90,            # Вертикальный дождь
        length=(40, 80),     # Длинные капли
        width=(1, 3),        # Тонкие капли
        transparency=0.8,    # Высокая прозрачность
        density=0.08,        # Средняя плотность
        intensity=1.0,       # Высокая интенсивность
        seed=seed
    )
    
    # Отображение результатов
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.title('Исходное изображение')
    plt.imshow(image_rgb)
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    plt.title('Алгоритм 1: Улучшенная форма капель')
    plt.imshow(rainy_image1)
    plt.axis('off')
    
    plt.subplot(2, 2, 3)
    plt.title('Алгоритм 2: Многослойный дождь')
    plt.imshow(rainy_image2)
    plt.axis('off')
    
    plt.subplot(2, 2, 4)
    plt.title('Алгоритм 3: Физически реалистичные капли')
    plt.imshow(rainy_image3)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("rain_comparison.png")
    print(f"Сравнение сохранено в rain_comparison.png")
    
    # Сохранение результатов
    cv2.imwrite(os.path.join(output_dir, "rainy_output_alg1.png"), 
               cv2.cvtColor(rainy_image1, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(output_dir, "rainy_output_alg2.png"), 
               cv2.cvtColor(rainy_image2, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(output_dir, "rainy_output_alg3.png"), 
               cv2.cvtColor(rainy_image3, cv2.COLOR_RGB2BGR))
    
    print("\nПример использования класса RainGenerator с тремя алгоритмами:")
    print("```python")
    print("from utils.rain_generator import RainGenerator")
    print("import cv2")
    print("")
    print("# Создание генератора дождя")
    print("rain_generator = RainGenerator(seed=42)")
    print("")
    print("# Применение эффекта дождя с алгоритмом 1 (улучшенная форма капель)")
    print("rainy_image1 = rain_generator.apply_rain_to_image(")
    print("    'input.png',")
    print("    'output_alg1.png',")
    print("    algorithm=1,         # Алгоритм 1: Улучшенная форма капель с градиентом прозрачности")
    print("    angle=90,            # Угол наклона капель (90 - вертикальный)")
    print("    length=(40, 80),     # Диапазон длин капель")
    print("    width=(1, 3),        # Диапазон ширин капель")
    print("    transparency=0.8,    # Прозрачность дождя (0.0-1.0)")
    print("    density=0.08,        # Плотность дождя")
    print("    intensity=1.0        # Интенсивность (яркость) капель (0.0-1.0)")
    print(")")
    print("")
    print("# Применение эффекта дождя с алгоритмом 2 (многослойный дождь)")
    print("rainy_image2 = rain_generator.apply_rain_to_image(")
    print("    'input.png',")
    print("    'output_alg2.png',")
    print("    algorithm=2,         # Алгоритм 2: Многослойный дождь с разной скоростью и размерами капель")
    print("    angle=90,            # Угол наклона капель (90 - вертикальный)")
    print("    length=(40, 80),     # Диапазон длин капель")
    print("    width=(1, 3),        # Диапазон ширин капель")
    print("    transparency=0.8,    # Прозрачность дождя (0.0-1.0)")
    print("    density=0.08,        # Плотность дождя")
    print("    intensity=1.0        # Интенсивность (яркость) капель (0.0-1.0)")
    print(")")
    print("")
    print("# Применение эффекта дождя с алгоритмом 3 (физически реалистичные капли)")
    print("rainy_image3 = rain_generator.apply_rain_to_image(")
    print("    'input.png',")
    print("    'output_alg3.png',")
    print("    algorithm=3,         # Алгоритм 3: Физически более реалистичное моделирование капель")
    print("    angle=90,            # Угол наклона капель (90 - вертикальный)")
    print("    length=(40, 80),     # Диапазон длин капель")
    print("    width=(1, 3),        # Диапазон ширин капель")
    print("    transparency=0.8,    # Прозрачность дождя (0.0-1.0)")
    print("    density=0.08,        # Плотность дождя")
    print("    intensity=1.0        # Интенсивность (яркость) капель (0.0-1.0)")
    print(")")
    print("```")
