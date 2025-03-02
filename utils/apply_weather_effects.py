#!/usr/bin/env python3
import os
import sys
import cv2
import numpy as np
import random
import glob
from tqdm import tqdm

# Import the weather generators
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.rain_generator import RainGenerator
from utils.fog_generator import FogGenerator
from utils.snow_generator import SnowGenerator

def main():
    # Set up paths
    examples_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'examples')
    patches_dir = os.path.join(examples_dir, 'patches')
    generated_dir = os.path.join(examples_dir, 'generated')
    
    # Create output directories if they don't exist
    rain_output_dir = os.path.join(generated_dir, 'rain')
    fog_output_dir = os.path.join(generated_dir, 'fog')
    snow_output_dir = os.path.join(generated_dir, 'snow')
    
    os.makedirs(rain_output_dir, exist_ok=True)
    os.makedirs(fog_output_dir, exist_ok=True)
    os.makedirs(snow_output_dir, exist_ok=True)
    
    # Check if patches directory exists and contains files
    if not os.path.exists(patches_dir):
        print(f"Error: Patches directory {patches_dir} does not exist.")
        return
    
    # Get list of patch files
    patch_files = glob.glob(os.path.join(patches_dir, '*.png'))
    if not patch_files:
        print(f"Error: No patch files found in {patches_dir}")
        return
    
    print(f"Found {len(patch_files)} patch files.")
    
    # Initialize weather generators
    seed = 42  # Fixed seed for reproducibility
    rain_generator = RainGenerator(seed=seed)
    fog_generator = FogGenerator(seed=seed)
    snow_generator = SnowGenerator(seed=seed)
    
    # Process each patch
    for patch_file in tqdm(patch_files, desc="Applying weather effects"):
        # Get the base filename without extension
        base_name = os.path.basename(patch_file)
        
        # Генерируем случайный угол наклона для каждого изображения
        # Угол в диапазоне от 85 до 95 градусов (небольшой наклон влево или вправо)
        random_angle = random.uniform(85, 95)
        
        # Увеличиваем плотность дождя в три раза (с 0.08 до 0.24)
        increased_density = 0.24  # 0.08 * 3
        
        # Apply rain effect with three different algorithms
        # Algorithm 1: Улучшенная форма капель с градиентом прозрачности
        rain_output_path1 = os.path.join(rain_output_dir, f"rain_alg1_{base_name}")
        try:
            rain_generator.apply_rain_to_image(
                patch_file, 
                rain_output_path1,
                algorithm=1,         # Алгоритм 1: Улучшенная форма капель с градиентом прозрачности
                angle=random_angle,  # Случайный угол наклона для каждого изображения
                length=(40, 80),     # Увеличенный диапазон длин капель
                width=(1, 3),        # Диапазон ширин капель (вытянутость)
                transparency=0.8,    # Высокая прозрачность (0.0-1.0)
                density=increased_density,  # Увеличенная в 3 раза плотность дождя
                intensity=1.0        # Максимальная интенсивность капель
            )
        except Exception as e:
            print(f"Error applying rain algorithm 1 to {patch_file}: {e}")
        
        # Генерируем новый случайный угол для разнообразия
        random_angle = random.uniform(85, 95)
        
        # Algorithm 2: Многослойный дождь с разной скоростью и размерами капель
        rain_output_path2 = os.path.join(rain_output_dir, f"rain_alg2_{base_name}")
        try:
            rain_generator.apply_rain_to_image(
                patch_file, 
                rain_output_path2,
                algorithm=2,         # Алгоритм 2: Многослойный дождь с разной скоростью и размерами капель
                angle=random_angle,  # Случайный угол наклона для каждого изображения
                length=(40, 80),     # Увеличенный диапазон длин капель
                width=(1, 3),        # Диапазон ширин капель (вытянутость)
                transparency=0.8,    # Высокая прозрачность (0.0-1.0)
                density=increased_density,  # Увеличенная в 3 раза плотность дождя
                intensity=1.0        # Максимальная интенсивность капель
            )
        except Exception as e:
            print(f"Error applying rain algorithm 2 to {patch_file}: {e}")
        
        # Генерируем новый случайный угол для разнообразия
        random_angle = random.uniform(85, 95)
        
        # Algorithm 3: Физически более реалистичное моделирование капель
        rain_output_path3 = os.path.join(rain_output_dir, f"rain_alg3_{base_name}")
        try:
            rain_generator.apply_rain_to_image(
                patch_file, 
                rain_output_path3,
                algorithm=3,         # Алгоритм 3: Физически более реалистичное моделирование капель
                angle=random_angle,  # Случайный угол наклона для каждого изображения
                length=(40, 80),     # Увеличенный диапазон длин капель
                width=(1, 3),        # Диапазон ширин капель (вытянутость)
                transparency=0.8,    # Высокая прозрачность (0.0-1.0)
                density=increased_density,  # Увеличенная в 3 раза плотность дождя
                intensity=1.0        # Максимальная интенсивность капель
            )
        except Exception as e:
            print(f"Error applying rain to {patch_file}: {e}")
        
        # Apply fog effect
        fog_output_path = os.path.join(fog_output_dir, f"fog_{base_name}")
        try:
            fog_generator.apply_fog_to_image(patch_file, fog_output_path)
        except Exception as e:
            print(f"Error applying fog to {patch_file}: {e}")
        
        # Применяем два разных алгоритма генерации снега
        # Алгоритм 1: Физическое моделирование падения снежинок
        snow_output_path1 = os.path.join(snow_output_dir, f"snow_alg1_{base_name}")
        try:
            # Используем уменьшенные параметры для алгоритма 1
            snow_generator.apply_snow_to_image(
                patch_file, 
                snow_output_path1, 
                algorithm=1,          # Алгоритм 1: Физическое моделирование
                intensity=0.4,        # Уменьшенная интенсивность (0.0-1.0)
                snow_size=(0.0001, 0.001),  # Уменьшенные в два раза размеры снежинок
                wind_direction=180,   # Горизонтальное направление (слева направо)
                wind_strength=5       # Умеренный эффект ветра
            )
        except Exception as e:
            print(f"Error applying snow algorithm 1 to {patch_file}: {e}")
            
        # Алгоритм 2: Использование текстур и наложений
        snow_output_path2 = os.path.join(snow_output_dir, f"snow_alg2_{base_name}")
        try:
            # Используем уменьшенные параметры для алгоритма 2
            snow_generator.apply_snow_to_image(
                patch_file, 
                snow_output_path2, 
                algorithm=2,          # Алгоритм 2: Текстуры и наложения
                intensity=0.4,        # Уменьшенная интенсивность (0.0-1.0)
                snow_size=(0.0001, 0.001),  # Уменьшенные в два раза размеры снежинок
                wind_direction=180,   # Горизонтальное направление (слева направо)
                wind_strength=5       # Умеренный эффект ветра
            )
        except Exception as e:
            print(f"Error applying snow to {patch_file}: {e}")
    
    # Count the number of generated images
    rain_files = len(glob.glob(os.path.join(rain_output_dir, '*.png')))
    fog_files = len(glob.glob(os.path.join(fog_output_dir, '*.png')))
    snow_files = len(glob.glob(os.path.join(snow_output_dir, '*.png')))
    
    print(f"\nGenerated weather effect images:")
    print(f"Rain: {rain_files} images")
    print(f"Fog: {fog_files} images")
    print(f"Snow: {snow_files} images")
    print(f"Total: {rain_files + fog_files + snow_files} images")

if __name__ == "__main__":
    main()
