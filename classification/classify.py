import argparse
import logging
import time

import numpy as np
from PIL import Image
from tflite_runtime.interpreter import Interpreter

LABEL_PATH = 'labels.txt'
MODEL_PATH = "./model/model.tflite"
TOP_NUM = 5


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    labels = load_labels(LABEL_PATH)

    interpreter, meta_info = load_model(MODEL_PATH)

    img = Image.open(args.image_file).convert('RGB')
    img = preprocess(img, meta_info['input_size'])

    input_data = np.expand_dims(img, axis=0)
    start_time = time.time()
    interpreter.set_tensor(meta_info['input_tensor'], input_data)
    interpreter.invoke()

    predictions = interpreter.get_tensor(meta_info['output_tensor'])[0]
    elapsed = time.time() - start_time
    logger.info("Elapsed {:.6f}s".format(elapsed))

    top_k_idx = np.argsort(predictions)[::-1][:TOP_NUM]

    for idx in top_k_idx:
        print(labels[idx], predictions[idx])


def load_labels(label_path):
    with open(label_path, 'r') as f:
        labels = list(map(str.strip, f.readlines()))
    return labels


def load_model(model_path):
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    logger.debug("input_details = {}, output_details = {}".format(input_details, output_details))

    input_shape = input_details[0]['shape']
    input_size = input_shape[:2] if len(input_shape) == 3 else input_shape[1:3]

    meta_info = {
        'input_size': input_size,
        'input_tensor': input_details[0]['index'],
        'output_tensor': output_details[0]['index'],
    }
    return interpreter, meta_info


def preprocess(img, target_size):
    img = img.resize(target_size)
    img = np.array(img, dtype=np.float32)
    img /= 127.5
    img -= 1.
    return img


def build_arg_parser():
    parser = argparse.ArgumentParser(description='Image Classification')
    parser.add_argument('-f', dest='image_file', type=str, help='Image file', required=True)
    return parser


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logger = logging.getLogger(__name__)
    main()
