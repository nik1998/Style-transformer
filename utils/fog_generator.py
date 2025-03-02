import os
import cv2
import numpy as np
from PIL import Image
import random
from scipy.ndimage import gaussian_filter


class FogGenerator:
    """
    Класс для генерации эффекта тумана на изображениях
    """
    
    def __init__(self, seed=None):
        """
        Инициализация генератора тумана
        
        Args:
            seed (int, optional): Seed для генератора случайных чисел. По умолчанию None.
        """
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
            self.random_state = np.random.RandomState(seed)
        else:
            self.random_state = np.random.RandomState()
    
    def generate_perlin_noise(self, shape, scale=100, octaves=6, persistence=0.5, lacunarity=2.0, seed=None):
        """
        Генерация шума Перлина для создания естественных паттернов тумана
        
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
    
    def simulate_atmospheric_scattering(self, image, fog_density, depth_factor=1.0):
        """
        Симуляция эффекта атмосферного рассеивания в тумане
        
        Args:
            image (numpy.ndarray): Исходное изображение
            fog_density (float): Плотность тумана
            depth_factor (float, optional): Фактор глубины. По умолчанию 1.0.
            
        Returns:
            numpy.ndarray: Изображение с эффектом атмосферного рассеивания
        """
        # Преобразование в float
        image_float = image.astype(np.float32) / 255.0
        
        # Создание карты глубины (простой градиент сверху вниз)
        height = image.shape[0]
        depth_map = np.linspace(0, 1, height)
        depth_map = np.tile(depth_map.reshape(-1, 1), (1, image.shape[1]))
        depth_map = np.dstack([depth_map] * 3)
        
        # Применение рассеивания, зависящего от глубины
        transmission = np.exp(-fog_density * depth_map * depth_factor)
        airlight = np.array([1.0, 1.0, 1.0])  # Белый туман
        
        scattered = image_float * transmission + airlight.reshape(1, 1, 3) * (1 - transmission)
        return np.clip(scattered, 0, 1)
    
    def apply_noise(self, image, seed=None):
        """
        Применение реалистичного эффекта тумана к изображению
        
        Args:
            image (numpy.ndarray): Исходное изображение
            seed (int, optional): Seed для генератора случайных чисел. По умолчанию None.
            
        Returns:
            numpy.ndarray: Изображение с эффектом тумана
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Преобразование в float32
        image_float = image.astype(np.float32) / 255.0
        
        # Использование seed для инициализации случайного состояния для согласованных параметров
        if seed is not None:
            random_state = np.random.RandomState(seed)
        else:
            random_state = self.random_state
            
        # Генерация базового шума Перлина для паттерна тумана с согласованными параметрами
        base_scale = random_state.uniform(50, 150)
        base_octaves = random_state.randint(4, 9)
        base_persistence = random_state.uniform(0.4, 0.6)
        base_lacunarity = random_state.uniform(1.8, 2.2)
        
        fog_pattern = self.generate_perlin_noise(
            image.shape[:2], 
            scale=base_scale,
            octaves=base_octaves,
            persistence=base_persistence,
            lacunarity=base_lacunarity,
            seed=seed if seed is not None else random_state.randint(0, 1000000)
        )
        
        # Добавление многомасштабных деталей с тем же seed
        detail_scale = random_state.uniform(20, 50)
        detail_pattern = self.generate_perlin_noise(
            image.shape[:2],
            scale=detail_scale,
            octaves=3,
            persistence=0.3,
            seed=seed if seed is not None else random_state.randint(0, 1000000)
        )
        
        fog_pattern = 0.8 * fog_pattern + 0.2 * detail_pattern
        
        # Использование того же random_state для размытия
        blur_sigma = random_state.uniform(3, 5)  # Увеличенное размытие для более густого тумана
        fog_pattern = gaussian_filter(fog_pattern, sigma=blur_sigma)
        
        # Расширение до 3 каналов
        fog_pattern = np.stack([fog_pattern] * 3, axis=-1)
        
        # Использование того же random_state для вариации цвета
        fog_color = np.array([
            random_state.uniform(0.97, 1.0),  # R - увеличенная яркость
            random_state.uniform(0.97, 1.0),  # G
            random_state.uniform(0.97, 1.0)   # B
        ])
        fog = fog_pattern * fog_color.reshape(1, 1, 3)
        
        # Использование того же random_state для параметров тумана
        fog_intensity = random_state.uniform(0.7, 0.95)
        fog_density = random_state.uniform(0.7, 0.9)
        
        # Применение атмосферного рассеивания с согласованной плотностью
        scattered = self.simulate_atmospheric_scattering(
            image, 
            fog_density=fog_density
        )
        
        # Комбинирование эффектов с более сильным влиянием тумана
        noisy_image = scattered * (1 - fog_intensity * fog_pattern) + fog * fog_intensity
        
        # Добавление тонкого гауссова шума с согласованными параметрами
        noise_std = random_state.uniform(0.01, 0.03)
        noise = random_state.normal(0, noise_std, noisy_image.shape)  # Уменьшенный шум для более чистого тумана
        noisy_image = noisy_image + noise
        
        # Ограничение значений до допустимого диапазона
        noisy_image = np.clip(noisy_image, 0, 1)
        
        # Преобразование обратно в uint8
        return (noisy_image * 255).astype(np.uint8)
    
    def apply_fog_to_image(self, image_path, output_path=None, seed=None):
        """
        Применение эффекта тумана к изображению и сохранение результата
        
        Args:
            image_path (str): Путь к исходному изображению
            output_path (str, optional): Путь для сохранения результата. 
                                        По умолчанию None (не сохранять).
            seed (int, optional): Seed для генератора случайных чисел. По умолчанию None.
            
        Returns:
            numpy.ndarray: Изображение с эффектом тумана
        """
        # Чтение изображения
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Не удалось прочитать изображение: {image_path}")
        
        # Преобразование BGR в RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Применение эффекта тумана
        foggy_image = self.apply_noise(image_rgb, seed=seed)
        
        # Сохранение результата, если указан путь
        if output_path:
            # Создание директории, если она не существует
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Преобразование RGB в BGR для сохранения
            foggy_image_bgr = cv2.cvtColor(foggy_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, foggy_image_bgr)
            print(f"Изображение с туманом сохранено в {output_path}")
        
        return foggy_image


if __name__ == "__main__":
    # Демонстрация работы генератора тумана
    import matplotlib.pyplot as plt
    import os
    
    # Использование изображения из датасета
    input_image_path = "../dataset/input/patch_0_0.png"
    output_image_path = "foggy_output.png"
    seed = 42  # Фиксированный seed для воспроизводимости
    
    # Создание генератора тумана
    fog_generator = FogGenerator(seed=seed)
    
    # Чтение изображения
    image = cv2.imread(input_image_path)
    if image is None:
        print(f"Не удалось прочитать изображение: {input_image_path}")
        exit(1)
    
    # Преобразование BGR в RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Применение эффекта тумана
    foggy_image = fog_generator.apply_noise(image_rgb)
    
    # Отображение результатов
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.title('Исходное изображение')
    plt.imshow(image_rgb)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title('Изображение с туманом')
    plt.imshow(foggy_image)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("comparison.png")
    print(f"Сравнение сохранено в comparison.png")
    
    # Сохранение результата
    foggy_image_bgr = cv2.cvtColor(foggy_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_image_path, foggy_image_bgr)
    print(f"Изображение с туманом сохранено в {output_image_path}")
    
    print("\nПример использования класса FogGenerator:")
    print("```python")
    print("from utils.fog_generator import FogGenerator")
    print("import cv2")
    print("")
    print("# Создание генератора тумана")
    print("fog_generator = FogGenerator(seed=42)")
    print("")
    print("# Применение эффекта тумана к изображению")
    print("foggy_image = fog_generator.apply_fog_to_image('input.png', 'output.png')")
    print("```")
