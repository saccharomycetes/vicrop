from PIL import Image
import numpy as np

def visualize(image_path, map):

    image = Image.open(image_path).convert('RGB')

    image_width, image_height = image.size

    patch_width = image_width // 16

    patch_height = image_height // 16

    brightened_image_data = np.zeros((image_height, image_width, 3), dtype=np.uint8)

    top_30_percentile = np.percentile(map, 95)

    map = np.clip(map, 0, top_30_percentile)

    map = (map - map.min())/ (map.max() - map.min())

    for i in range(16):
        for j in range(16):
            start_x = j * patch_width
            end_x = (j + 1) * patch_width
            start_y = i * patch_height
            end_y = (i + 1) * patch_height
            patch = np.array(image.crop((start_x, start_y, end_x, end_y)))
            brightness_factor = map[i, j]
            brightened_patch = np.clip(patch * brightness_factor, 0, 255).astype(np.uint8)
            brightened_image_data[start_y:end_y, start_x:end_x] = brightened_patch
    
    brightened_image = Image.fromarray(brightened_image_data)
    return brightened_image