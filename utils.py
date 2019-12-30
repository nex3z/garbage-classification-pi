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
