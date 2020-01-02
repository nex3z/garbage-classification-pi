import numpy as np
from PIL import Image


def preprocess_image_file(image_file, target_size):
    image = Image.open(image_file).convert('RGB')
    image_data = preprocess(image, target_size)
    return image_data


def preprocess(image, target_size):
    image = image.resize(target_size, Image.ANTIALIAS)
    image = np.array(image, dtype=np.float32)
    image /= 127.5
    image -= 1.
    return image


def center_crop(image):
    width, height = image.size
    if width / height < 1:
        vertical_offset = int((height - width) / 2)
        padded = image.crop((0, vertical_offset, width, height - vertical_offset))
    else:
        horizontal_offset = int((width - height) / 2)
        padded = image.crop((horizontal_offset, 0, width - horizontal_offset, height))
    return padded
    