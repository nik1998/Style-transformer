import os
import cv2
import numpy as np
from PIL import Image
import random
from scipy.ndimage import gaussian_filter


class SnowGenerator:
    """
    Класс для генерации эффекта снега на изображениях
    """
    
    def __init__(self, seed=None):
        """
        Инициализация генератора снега
        
        Args:
            seed (int, optional): Seed для генератора случайных чисел. По умолчанию None.
        """
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
            self.random_state = np.random.RandomState(seed)
        else:
            self.random_state = np.random.RandomState()
        
        # Кэш для шаблонов снежинок
        self.snowflake_templates = None
    
    def generate_snowflake_shapes(self, num_shapes=10, size_range=(0.0005, 0.005)):
        """
        Генерация различных форм снежинок с более выраженной структурой
        
        Args:
            num_shapes (int, optional): Количество различных форм. По умолчанию 10.
            size_range (tuple, optional): Диапазон размеров снежинок. По умолчанию (0.0005, 0.005) - маленькие снежинки.
            
        Returns:
            list: Список шаблонов снежинок
        """
        shapes = []
        for _ in range(num_shapes):
            size = int(max(1, self.random_state.uniform(size_range[0], size_range[1]) * 1000))
            # Создание базовой формы снежинки с 6 лучами
            shape = np.zeros((size*2+1, size*2+1), dtype=np.float32)
            center = size
            
            # Основные лучи (6 направлений)
            angles = np.linspace(0, 2*np.pi, 6, endpoint=False)
            for angle in angles:
                x_dir, y_dir = np.cos(angle), np.sin(angle)
                for i in range(1, size+1):
                    x = int(center + x_dir * i)
                    y = int(center + y_dir * i)
                    if 0 <= x < shape.shape[1] and 0 <= y < shape.shape[0]:
                        shape[y, x] = 1.0
                        
                        # Добавление более выраженных ответвлений для создания структуры снежинки
                        if i > size//4 and self.random_state.random() < 0.3:  # Уменьшена вероятность ответвлений
                            # Создаем ответвления под более выраженными углами
                            branch_angle = angle + self.random_state.uniform(-np.pi/4, np.pi/4)
                            branch_x_dir, branch_y_dir = np.cos(branch_angle), np.sin(branch_angle)
                            # Уменьшаем длину ответвлений
                            max_branch_length = max(1, size//3)
                            branch_length = self.random_state.randint(1, max_branch_length + 1)
                            for j in range(1, branch_length+1):
                                bx = int(x + branch_x_dir * j)
                                by = int(y + branch_y_dir * j)
                                if 0 <= bx < shape.shape[1] and 0 <= by < shape.shape[0]:
                                    shape[by, bx] = 0.9  # Более яркие ответвления
            
            # Легкое размытие для более естественного вида, но сохраняем четкость структуры
            shape = gaussian_filter(shape, sigma=0.2)
            shapes.append(shape)
        return shapes
    
    def initialize_snowflake_cache(self, num_templates=15, size_range=(0.0005, 0.005)):
        """
        Предварительная генерация и кэширование шаблонов снежинок
        
        Args:
            num_templates (int, optional): Количество шаблонов. По умолчанию 15.
            size_range (tuple, optional): Диапазон размеров снежинок. По умолчанию (0.0005, 0.005) - маленькие снежинки.
            
        Returns:
            list: Список шаблонов снежинок
        """
        self.snowflake_templates = self.generate_snowflake_shapes(
            num_shapes=num_templates,
            size_range=size_range
        )
        return self.snowflake_templates
    
    def generate_snow_particles(self, shape, density=0.01, size_range=(0.0001, 0.001), seed=None):
        """
        Генерация частиц снега разного размера с более выраженной структурой снежинок
        
        Args:
            shape (tuple): Размер выходного изображения (высота, ширина)
            density (float, optional): Плотность снега (доля пикселей). По умолчанию 0.01.
            size_range (tuple, optional): Диапазон размеров снежинок. По умолчанию (0.0001, 0.001) - маленькие снежинки.
            seed (int, optional): Seed для генератора случайных чисел. По умолчанию None.
            
        Returns:
            numpy.ndarray: Маска снега с значениями в диапазоне [0, 1]
        """
        if seed is not None:
            random_state = np.random.RandomState(seed)
        else:
            random_state = self.random_state
            
        # Создание пустой маски
        snow_mask = np.zeros(shape, dtype=np.float32)
        
        # Определение количества снежинок
        num_particles = int(shape[0] * shape[1] * density)
        
        # Генерация случайных позиций снежинок
        positions_y = random_state.randint(0, shape[0], num_particles)
        positions_x = random_state.randint(0, shape[1], num_particles)
        
        # Генерация случайных размеров снежинок
        # Преобразуем диапазон размеров в пиксели
        min_size_px = max(1, int(size_range[0] * min(shape)))
        max_size_px = max(2, int(size_range[1] * min(shape)))
        sizes = random_state.randint(min_size_px, max_size_px + 1, num_particles)
        
        # Генерация случайных интенсивностей снежинок (белые снежинки)
        intensities = random_state.uniform(0.8, 1.0, num_particles)
        
        # Генерация случайных типов снежинок (для разнообразия форм)
        if self.snowflake_templates is None:
            self.initialize_snowflake_cache(num_templates=15, size_range=size_range)
        
        snowflake_types = random_state.randint(0, len(self.snowflake_templates), num_particles)
        
        # Размещение снежинок на маске
        for i in range(num_particles):
            y, x = positions_y[i], positions_x[i]
            size = sizes[i]
            intensity = intensities[i]
            
            # Для маленьких снежинок используем простые формы (точки)
            if size <= 1 or random_state.random() < 0.7:  # 70% снежинок - простые формы
                # Создание снежинки как маленького круга
                y_min, y_max = max(0, y - size), min(shape[0], y + size + 1)
                x_min, x_max = max(0, x - size), min(shape[1], x + size + 1)
                
                for yy in range(y_min, y_max):
                    for xx in range(x_min, x_max):
                        # Расстояние от центра снежинки
                        dist = np.sqrt(((yy - y) / 1.0) ** 2 + ((xx - x) / 1.0) ** 2)
                        if dist <= size:
                            # Гауссово распределение интенсивности от центра к краям
                            falloff = np.exp(-0.5 * (dist / size) ** 2)
                            snow_mask[yy, xx] = max(snow_mask[yy, xx], intensity * falloff)
            else:
                # Использование сложной формы снежинки из кэша
                snowflake_type = snowflake_types[i]
                template = self.snowflake_templates[snowflake_type]
                
                # Масштабирование шаблона до нужного размера
                target_size = (int(size*2+1), int(size*2+1))
                if target_size[0] > 0 and target_size[1] > 0:
                    try:
                        scaled_template = cv2.resize(template, target_size, interpolation=cv2.INTER_LINEAR)
                        
                        # Размещение шаблона на маске
                        y_min, y_max = max(0, y - size), min(shape[0], y + size + 1)
                        x_min, x_max = max(0, x - size), min(shape[1], x + size + 1)
                        
                        # Вычисление координат для шаблона
                        template_y_min = max(0, size - y) if y < size else 0
                        template_y_max = min(size*2+1, size + (shape[0] - y)) if y + size >= shape[0] else size*2+1
                        template_x_min = max(0, size - x) if x < size else 0
                        template_x_max = min(size*2+1, size + (shape[1] - x)) if x + size >= shape[1] else size*2+1
                        
                        # Копирование части шаблона на маску
                        template_slice = scaled_template[template_y_min:template_y_max, template_x_min:template_x_max]
                        mask_slice = snow_mask[y_min:y_max, x_min:x_max]
                        
                        # Проверяем, что размеры совпадают
                        if template_slice.shape == mask_slice.shape:
                            snow_mask[y_min:y_max, x_min:x_max] = np.maximum(
                                mask_slice,
                                template_slice * intensity
                            )
                    except Exception:
                        # В случае ошибки масштабирования, используем простую форму
                        y_min, y_max = max(0, y - 1), min(shape[0], y + 2)
                        x_min, x_max = max(0, x - 1), min(shape[1], x + 2)
                        if y_min < y_max and x_min < x_max:
                            snow_mask[y_min:y_max, x_min:x_max] = intensity
        
        return snow_mask
    
    def apply_wind_effect(self, snow_mask, wind_direction=180, wind_strength=5):
        """
        Применение эффекта ветра к снежинкам для создания эффекта падения сбоку
        
        Args:
            snow_mask (numpy.ndarray): Маска снега
            wind_direction (float, optional): Направление ветра в градусах. По умолчанию 180 (слева направо).
            wind_strength (float, optional): Сила ветра. По умолчанию 5.
            
        Returns:
            numpy.ndarray: Маска снега с примененным эффектом ветра
        """
        # Преобразование направления ветра в радианы
        wind_rad = np.deg2rad(wind_direction)
        
        # Вычисление смещения по x и y
        dx = int(np.cos(wind_rad) * wind_strength)
        dy = int(np.sin(wind_rad) * wind_strength)
        
        # Применение смещения
        rows, cols = snow_mask.shape
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        shifted_mask = cv2.warpAffine(snow_mask, M, (cols, rows))
        
        # Добавление размытия в направлении ветра для создания эффекта движения
        kernel_size = max(3, min(wind_strength // 2, 5))
        kernel = np.zeros((kernel_size, kernel_size))
        center = kernel_size // 2
        
        # Создание направленного ядра размытия
        for i in range(kernel_size):
            for j in range(kernel_size):
                # Вычисление расстояния от центра
                y_dist = i - center
                x_dist = j - center
                
                # Проверка, находится ли точка в направлении ветра
                angle = np.arctan2(y_dist, x_dist)
                if abs(angle - wind_rad) < np.pi/4 or abs(angle - wind_rad) > 7*np.pi/4:
                    dist = np.sqrt(y_dist**2 + x_dist**2)
                    kernel[i, j] = max(0, 1 - dist/center)
        
        # Нормализация ядра
        if kernel.sum() > 0:
            kernel = kernel / kernel.sum()
            
            # Применение направленного размытия
            shifted_mask = cv2.filter2D(shifted_mask, -1, kernel)
        
        return shifted_mask
    
    def generate_falling_snow(self, shape, num_layers=3, density=0.01, size_range=(0.0001, 0.001)):
        """
        Генерация падающего снега с эффектом глубины
        
        Args:
            shape (tuple): Размер выходного изображения (высота, ширина)
            num_layers (int, optional): Количество слоев снега. По умолчанию 3.
            density (float, optional): Базовая плотность снега. По умолчанию 0.01.
            size_range (tuple, optional): Базовый диапазон размеров снежинок. По умолчанию (0.0001, 0.001).
            
        Returns:
            numpy.ndarray: Маска снега с эффектом глубины
        """
        combined_mask = np.zeros(shape, dtype=np.float32)
        
        for layer in range(num_layers):
            # Меньшая плотность для дальних слоев, большая для ближних
            layer_density = density * (0.5 + layer * 0.5)
            
            # Меньший размер для дальних слоев, больший для ближних
            min_size = size_range[0] * (0.8 + layer * 0.2)
            max_size = size_range[1] * (0.8 + layer * 0.2)
            
            # Генерация слоя снега
            layer_mask = self.generate_snow_particles(
                shape, 
                density=layer_density,
                size_range=(min_size, max_size)
            )
            
            # Размытие дальних слоев сильнее для эффекта глубины
            blur_sigma = 0.5 - (layer * 0.1)
            if blur_sigma > 0:
                layer_mask = gaussian_filter(layer_mask, sigma=blur_sigma)
            
            # Добавление слоя к общей маске с учетом прозрачности
            layer_opacity = 0.7 + (layer * 0.1)
            combined_mask = np.maximum(combined_mask, layer_mask * layer_opacity)
        
        return combined_mask
    
    def generate_perlin_noise(self, shape, scale=100, octaves=6, persistence=0.5, lacunarity=2.0, seed=None):
        """
        Генерация шума Перлина для создания естественных паттернов снега
        
        Args:
            shape (tuple): Размер выходного изображения (высота, ширина)
            scale (float, optional): Масштаб шума. По умолчанию 100.
            octaves (int, optional): Количество октав шума. По умолчанию 6.
            persistence (float, optional): Персистентность шума. По умолчанию 0.5.
            lacunarity (float, optional): Лакунарность шума. По умолчанию 2.0.
            seed (int, optional): Seed для генератора случайных чисел. По умолчанию None.
            
        Returns:
            numpy.ndarray: Шум Перлина размером shape с значениями в диапазоне [0, 1]
        """
        if seed is not None:
            np.random.seed(seed)
            
        def generate_octave(shape, frequency):
            noise = np.random.rand(*shape)
            noise = cv2.resize(noise, None, fx=frequency, fy=frequency, interpolation=cv2.INTER_LINEAR)
            noise = cv2.resize(noise, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
            return noise
        
        noise = np.zeros(shape)
        amplitude = 1.0
        frequency = 1.0 / scale
        
        for _ in range(octaves):
            noise += amplitude * generate_octave(shape, frequency)
            amplitude *= persistence
            frequency *= lacunarity
            
        # Нормализация к диапазону [0, 1]
        noise = (noise - noise.min()) / (noise.max() - noise.min())
        return noise
    
    def apply_snow_algorithm1(self, image, seed=None, intensity=0.5, snow_size=(0.0001, 0.001), wind_direction=180, wind_strength=5):
        """
        Алгоритм 1: Физическое моделирование падения снежинок
        
        Args:
            image (numpy.ndarray): Исходное изображение
            seed (int, optional): Seed для генератора случайных чисел. По умолчанию None.
            intensity (float, optional): Интенсивность снегопада (0.0-1.0). По умолчанию 0.5.
            snow_size (tuple, optional): Диапазон размеров снежинок (0.0-1.0). По умолчанию (0.0001, 0.001).
            wind_direction (float, optional): Направление ветра в градусах. По умолчанию 180 (слева направо).
            wind_strength (float, optional): Сила ветра. По умолчанию 5.
            
        Returns:
            numpy.ndarray: Изображение с реалистичным эффектом снега
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Преобразование в float32
        image_float = image.astype(np.float32) / 255.0
        
        # Использование seed для инициализации случайного состояния
        if seed is not None:
            random_state = np.random.RandomState(seed)
        else:
            random_state = self.random_state
        
        # Расчет плотности снега на основе интенсивности
        snow_density = 0.005 + (intensity * 0.015)  # От 0.005 до 0.02
        
        # Генерация многослойного снега с эффектом глубины
        num_layers = 2 + int(intensity * 3)  # От 2 до 5 слоев
        falling_snow = self.generate_falling_snow(
            image.shape[:2],
            num_layers=num_layers,
            density=snow_density,
            size_range=snow_size
        )
        
        # Применение эффекта ветра к падающему снегу
        falling_snow_with_wind = self.apply_wind_effect(
            falling_snow,
            wind_direction=wind_direction,
            wind_strength=wind_strength
        )
        
        # Добавление эффекта глубины резкости для дальних снежинок
        # Создаем карту глубины (простая линейная градация сверху вниз)
        height, width = image.shape[:2]
        depth_map = np.linspace(0, 1, height).reshape(-1, 1)
        depth_map = np.tile(depth_map, (1, width))
        
        # Размытие снежинок на основе глубины
        blur_factor = 0.5 + (intensity * 0.5)  # От 0.5 до 1.0
        for y in range(height):
            depth = depth_map[y, 0]
            blur_sigma = blur_factor * (1.0 - depth)
            if blur_sigma > 0.1:  # Применяем размытие только если оно заметно
                falling_snow_with_wind[y, :] = gaussian_filter(falling_snow_with_wind[y, :], sigma=blur_sigma)
        
        # Расширение до 3 каналов
        snow_mask_3d = np.stack([falling_snow_with_wind] * 3, axis=-1)
        
        # Цвет снега (чисто белый с легким голубоватым оттенком для реалистичности)
        snow_color = np.array([0.98, 0.99, 1.0])
        
        # Добавление снега на изображение
        snow_intensity = 0.8 + (intensity * 0.2)  # От 0.8 до 1.0
        snowy_image = image_float * (1 - snow_mask_3d * snow_intensity) + snow_mask_3d * snow_color.reshape(1, 1, 3) * snow_intensity
        
        # Добавление легкого эффекта атмосферного рассеивания
        # Чем дальше объект, тем больше он подвержен атмосферному рассеиванию
        fog_factor = 0.05 + (intensity * 0.05)  # От 0.05 до 0.1
        fog_color = np.array([0.95, 0.97, 1.0])
        
        for y in range(height):
            depth = depth_map[y, 0]
            fog_amount = fog_factor * depth
            snowy_image[y, :] = snowy_image[y, :] * (1 - fog_amount) + fog_color * fog_amount
        
        # Ограничение значений до допустимого диапазона
        snowy_image = np.clip(snowy_image, 0, 1)
        
        # Преобразование обратно в uint8
        return (snowy_image * 255).astype(np.uint8)
    
    def apply_snow_algorithm2(self, image, seed=None, intensity=0.5, snow_size=(0.0001, 0.001), wind_direction=180, wind_strength=5):
        """
        Алгоритм 2: Использование текстур и наложений для создания эффекта объемности
        
        Args:
            image (numpy.ndarray): Исходное изображение
            seed (int, optional): Seed для генератора случайных чисел. По умолчанию None.
            intensity (float, optional): Интенсивность снегопада (0.0-1.0). По умолчанию 0.5.
            snow_size (tuple, optional): Диапазон размеров снежинок (0.0-1.0). По умолчанию (0.0001, 0.001).
            wind_direction (float, optional): Направление ветра в градусах. По умолчанию 180 (слева направо).
            wind_strength (float, optional): Сила ветра. По умолчанию 5.
            
        Returns:
            numpy.ndarray: Изображение с реалистичным эффектом снега
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Преобразование в float32
        image_float = image.astype(np.float32) / 255.0
        
        # Использование seed для инициализации случайного состояния
        if seed is not None:
            random_state = np.random.RandomState(seed)
        else:
            random_state = self.random_state
        
        height, width = image.shape[:2]
        
        # 1. Создание базовой текстуры снега с помощью шума Перлина
        snow_texture = self.generate_perlin_noise(
            (height, width),
            scale=50 + random_state.randint(-20, 20),
            octaves=4,
            persistence=0.5,
            lacunarity=2.0,
            seed=seed
        )
        
        # Применение порога для создания более выраженных снежинок
        snow_threshold = 0.7 - (intensity * 0.2)  # От 0.7 до 0.5
        snow_texture = np.where(snow_texture > snow_threshold, 
                               (snow_texture - snow_threshold) / (1 - snow_threshold), 
                               0)
        
        # 2. Создание нескольких слоев снежинок разного размера
        snow_layers = []
        num_layers = 3 + int(intensity * 2)  # От 3 до 5 слоев
        
        for layer in range(num_layers):
            # Для каждого слоя используем разный размер снежинок
            layer_min_size = snow_size[0] * (0.8 + layer * 0.2)
            layer_max_size = snow_size[1] * (0.8 + layer * 0.2)
            
            # Генерация слоя снежинок
            layer_density = 0.005 + (intensity * 0.005) * (layer + 1)  # Увеличиваем плотность для ближних слоев
            snow_layer = self.generate_snow_particles(
                (height, width),
                density=layer_density,
                size_range=(layer_min_size, layer_max_size)
            )
            
            # Применение эффекта ветра
            snow_layer = self.apply_wind_effect(
                snow_layer,
                wind_direction=wind_direction + random_state.uniform(-10, 10),  # Небольшая вариация направления
                wind_strength=wind_strength * (0.8 + layer * 0.2)  # Увеличиваем силу ветра для ближних слоев
            )
            
            # Размытие дальних слоев для эффекта глубины
            if layer < num_layers - 1:  # Не размываем самый ближний слой
                blur_sigma = 0.5 - (layer * 0.1)
                if blur_sigma > 0:
                    snow_layer = gaussian_filter(snow_layer, sigma=blur_sigma)
            
            snow_layers.append(snow_layer)
        
        # 3. Объединение всех слоев снега
        combined_snow = np.zeros((height, width), dtype=np.float32)
        for i, layer in enumerate(snow_layers):
            layer_weight = 0.7 + (i * 0.1)  # Увеличиваем вес для ближних слоев
            combined_snow = np.maximum(combined_snow, layer * layer_weight)
        
        # 4. Добавление текстуры снега
        combined_snow = np.maximum(combined_snow, snow_texture * 0.3)
        
        # 5. Расширение до 3 каналов
        snow_mask_3d = np.stack([combined_snow] * 3, axis=-1)
        
        # 6. Создание эффекта объемности с помощью освещения
        # Имитация направленного света, создающего тени и блики на снежинках
        light_direction = np.array([0.5, 0.5, 1.0])  # Направление света (x, y, z)
        light_direction = light_direction / np.linalg.norm(light_direction)
        
        # Создание карты нормалей из текстуры снега
        dx, dy = np.gradient(snow_texture)
        normal_map = np.zeros((height, width, 3), dtype=np.float32)
        normal_map[:, :, 0] = -dx
        normal_map[:, :, 1] = -dy
        normal_map[:, :, 2] = 1.0
        
        # Нормализация векторов нормалей
        norm = np.sqrt(np.sum(normal_map**2, axis=2))
        norm = np.maximum(norm, 1e-10)  # Избегаем деления на ноль
        normal_map[:, :, 0] /= norm
        normal_map[:, :, 1] /= norm
        normal_map[:, :, 2] /= norm
        
        # Расчет освещения (скалярное произведение нормали и направления света)
        lighting = np.sum(normal_map * light_direction, axis=2)
        lighting = np.clip(lighting, 0, 1)
        lighting = lighting.reshape(height, width, 1)
        
        # Применение освещения к снежинкам
        snow_color = np.array([0.98, 0.99, 1.0])  # Белый с легким голубым оттенком
        snow_color_3d = snow_color.reshape(1, 1, 3)
        
        # Добавление бликов и теней
        snow_with_lighting = snow_mask_3d * snow_color_3d * (0.7 + 0.3 * lighting)
        
        # 7. Добавление снега на изображение
        snow_intensity = 0.8 + (intensity * 0.2)  # От 0.8 до 1.0
        snowy_image = image_float * (1 - snow_mask_3d * snow_intensity) + snow_with_lighting * snow_intensity
        
        # 8. Добавление эффекта атмосферного рассеивания
        # Создание карты глубины (простая линейная градация сверху вниз)
        depth_map = np.linspace(0, 1, height).reshape(-1, 1)
        depth_map = np.tile(depth_map, (1, width))
        
        # Атмосферное рассеивание
        fog_factor = 0.05 + (intensity * 0.05)  # От 0.05 до 0.1
        fog_color = np.array([0.95, 0.97, 1.0])
        
        for y in range(height):
            depth = depth_map[y, 0]
            fog_amount = fog_factor * depth
            snowy_image[y, :] = snowy_image[y, :] * (1 - fog_amount) + fog_color * fog_amount
        
        # Ограничение значений до допустимого диапазона
        snowy_image = np.clip(snowy_image, 0, 1)
        
        # Преобразование обратно в uint8
        return (snowy_image * 255).astype(np.uint8)
    
    def apply_snow(self, image, algorithm=1, seed=None, intensity=0.5, snow_size=(0.0001, 0.001), wind_direction=180, wind_strength=5):
        """
        Применение реалистичного эффекта снега к изображению с выбором алгоритма
        
        Args:
            image (numpy.ndarray): Исходное изображение
            algorithm (int, optional): Алгоритм генерации снега (1 или 2). По умолчанию 1.
            seed (int, optional): Seed для генератора случайных чисел. По умолчанию None.
            intensity (float, optional): Интенсивность снегопада (0.0-1.0). По умолчанию 0.5.
            snow_size (tuple, optional): Диапазон размеров снежинок (0.0-1.0). По умолчанию (0.0001, 0.001).
            wind_direction (float, optional): Направление ветра в градусах. По умолчанию 180 (слева направо).
            wind_strength (float, optional): Сила ветра. По умолчанию 5.
            
        Returns:
            numpy.ndarray: Изображение с реалистичным эффектом снега
        """
        if algorithm == 1:
            return self.apply_snow_algorithm1(
                image, 
                seed=seed,
                intensity=intensity,
                snow_size=snow_size,
                wind_direction=wind_direction,
                wind_strength=wind_strength
            )
        else:
            return self.apply_snow_algorithm2(
                image, 
                seed=seed,
                intensity=intensity,
                snow_size=snow_size,
                wind_direction=wind_direction,
                wind_strength=wind_strength
            )
    
    def apply_snow_to_image(self, image_path, output_path=None, algorithm=1, seed=None, intensity=0.5, snow_size=(0.0001, 0.001), wind_direction=180, wind_strength=5):
        """
        Применение эффекта снега к изображению и сохранение результата
        
        Args:
            image_path (str): Путь к исходному изображению
            output_path (str, optional): Путь для сохранения результата. 
                                        По умолчанию None (не сохранять).
            algorithm (int, optional): Алгоритм генерации снега (1 или 2). По умолчанию 1.
            seed (int, optional): Seed для генератора случайных чисел. По умолчанию None.
            intensity (float, optional): Интенсивность снегопада (0.0-1.0). По умолчанию 0.5.
            snow_size (tuple, optional): Диапазон размеров снежинок (0.0-1.0). По умолчанию (0.0001, 0.001).
            wind_direction (float, optional): Направление ветра в градусах. По умолчанию 180 (слева направо).
            wind_strength (float, optional): Сила ветра. По умолчанию 5.
            
        Returns:
            numpy.ndarray: Изображение с эффектом снега
        """
        # Чтение изображения
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Не удалось прочитать изображение: {image_path}")
        
        # Преобразование BGR в RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Применение эффекта снега
        snowy_image = self.apply_snow(
            image_rgb, 
            algorithm=algorithm,
            seed=seed,
            intensity=intensity,
            snow_size=snow_size,
            wind_direction=wind_direction,
            wind_strength=wind_strength
        )
        
        # Сохранение результата, если указан путь
        if output_path:
            # Создание директории, если она не существует
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Преобразование RGB в BGR для сохранения
            snowy_image_bgr = cv2.cvtColor(snowy_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, snowy_image_bgr)
            print(f"Изображение со снегом сохранено в {output_path}")
        
        return snowy_image


if __name__ == "__main__":
    # Демонстрация работы генератора снега
    import matplotlib.pyplot as plt
    import os
    
    # Использование изображения из датасета
    input_image_path = "../dataset/input/patch_0_0.png"
    output_dir = "../examples/generated/snow"
    os.makedirs(output_dir, exist_ok=True)
    
    seed = 42  # Фиксированный seed для воспроизводимости
    
    # Создание генератора снега
    snow_generator = SnowGenerator(seed=seed)
    
    # Чтение изображения
    image = cv2.imread(input_image_path)
    if image is None:
        print(f"Не удалось прочитать изображение: {input_image_path}")
        exit(1)
    
    # Преобразование BGR в RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Применение эффекта снега с разными параметрами и алгоритмами
    
    # Алгоритм 1: Физическое моделирование падения снежинок
    # Очень маленькие снежинки, очень низкая интенсивность
    snowy_image_alg1_light = snow_generator.apply_snow(
        image_rgb, 
        algorithm=1,
        intensity=0.2, 
        snow_size=(0.00005, 0.0005),
        wind_strength=3
    )
    
    # Маленькие снежинки, низкая интенсивность
    snowy_image_alg1_medium = snow_generator.apply_snow(
        image_rgb, 
        algorithm=1,
        intensity=0.4, 
        snow_size=(0.0001, 0.001),
        wind_strength=5
    )
    
    # Средние снежинки, средняя интенсивность
    snowy_image_alg1_heavy = snow_generator.apply_snow(
        image_rgb, 
        algorithm=1,
        intensity=0.6, 
        snow_size=(0.0002, 0.002),
        wind_strength=7
    )
    
    # Алгоритм 2: Использование текстур и наложений
    # Очень маленькие снежинки, очень низкая интенсивность
    snowy_image_alg2_light = snow_generator.apply_snow(
        image_rgb, 
        algorithm=2,
        intensity=0.2, 
        snow_size=(0.00005, 0.0005),
        wind_strength=3
    )
    
    # Маленькие снежинки, низкая интенсивность
    snowy_image_alg2_medium = snow_generator.apply_snow(
        image_rgb, 
        algorithm=2,
        intensity=0.4, 
        snow_size=(0.0001, 0.001),
        wind_strength=5
    )
    
    # Средние снежинки, средняя интенсивность
    snowy_image_alg2_heavy = snow_generator.apply_snow(
        image_rgb, 
        algorithm=2,
        intensity=0.6, 
        snow_size=(0.0002, 0.002),
        wind_strength=7
    )
    
    # Отображение результатов
    plt.figure(figsize=(15, 10))
    
    plt.subplot(3, 3, 1)
    plt.title('Исходное изображение')
    plt.imshow(image_rgb)
    plt.axis('off')
    
    plt.subplot(3, 3, 2)
    plt.title('Алгоритм 1: Очень легкий снегопад (очень маленькие снежинки)')
    plt.imshow(snowy_image_alg1_light)
    plt.axis('off')
    
    plt.subplot(3, 3, 3)
    plt.title('Алгоритм 2: Очень легкий снегопад (очень маленькие снежинки)')
    plt.imshow(snowy_image_alg2_light)
    plt.axis('off')
    
    plt.subplot(3, 3, 5)
    plt.title('Алгоритм 1: Легкий снегопад (маленькие снежинки)')
    plt.imshow(snowy_image_alg1_medium)
    plt.axis('off')
    
    plt.subplot(3, 3, 6)
    plt.title('Алгоритм 2: Легкий снегопад (маленькие снежинки)')
    plt.imshow(snowy_image_alg2_medium)
    plt.axis('off')
    
    plt.subplot(3, 3, 8)
    plt.title('Алгоритм 1: Средний снегопад (средние снежинки)')
    plt.imshow(snowy_image_alg1_heavy)
    plt.axis('off')
    
    plt.subplot(3, 3, 9)
    plt.title('Алгоритм 2: Средний снегопад (средние снежинки)')
    plt.imshow(snowy_image_alg2_heavy)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "snow_comparison.png"))
    print(f"Сравнение сохранено в {os.path.join(output_dir, 'snow_comparison.png')}")
    
    # Сохранение результатов
    cv2.imwrite(os.path.join(output_dir, "snow_alg1_light.png"), cv2.cvtColor(snowy_image_alg1_light, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(output_dir, "snow_alg1_medium.png"), cv2.cvtColor(snowy_image_alg1_medium, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(output_dir, "snow_alg1_heavy.png"), cv2.cvtColor(snowy_image_alg1_heavy, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(output_dir, "snow_alg2_light.png"), cv2.cvtColor(snowy_image_alg2_light, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(output_dir, "snow_alg2_medium.png"), cv2.cvtColor(snowy_image_alg2_medium, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(output_dir, "snow_alg2_heavy.png"), cv2.cvtColor(snowy_image_alg2_heavy, cv2.COLOR_RGB2BGR))
    
    print("\nПример использования класса SnowGenerator:")
    print("```python")
    print("from utils.snow_generator import SnowGenerator")
    print("import cv2")
    print("")
    print("# Создание генератора снега")
    print("snow_generator = SnowGenerator(seed=42)")
    print("")
    print("# Применение эффекта снега к изображению с разными параметрами")
    print("# algorithm - алгоритм генерации снега (1 или 2)")
    print("# intensity - интенсивность снегопада (0.0-1.0)")
    print("# snow_size - диапазон размеров снежинок (0.0-1.0)")
    print("# wind_direction - направление ветра в градусах")
    print("# wind_strength - сила ветра")
    print("snowy_image = snow_generator.apply_snow_to_image(")
    print("    'input.png', ")
    print("    'output.png',")
    print("    algorithm=1,")
    print("    intensity=0.4,")
    print("    snow_size=(0.0001, 0.001),")
    print("    wind_direction=180,")
    print("    wind_strength=5")
    print(")")
    print("```")
