import logging
import time
from collections import OrderedDict

import numpy as np
from tflite_runtime.interpreter import Interpreter

logger = logging.getLogger(__name__)


class Classifier(object):
    def __init__(self, model_path, label_path):
        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.labels = load_labels(label_path)

        input_details = self.interpreter.get_input_details()
        self.input_tensor = input_details[0]['index']
        input_shape = input_details[0]['shape']
        self.input_size = input_shape[:2] if len(input_shape) == 3 else input_shape[1:3]

        output_details = self.interpreter.get_output_details()
        self.output_tensor = output_details[0]['index']

    def classify(self, input_data, top=5):
        if len(input_data.shape) < 4:
            input_data = np.expand_dims(input_data, axis=0)

        start_time = time.time()
        self.interpreter.set_tensor(self.input_tensor, input_data)
        self.interpreter.invoke()
        predictions = self.interpreter.get_tensor(self.output_tensor)[0]
        elapsed = time.time() - start_time
        logger.info("Elapsed {:.6f}s".format(elapsed))

        top_k_idx = np.argsort(predictions)[::-1][:top]
        result = OrderedDict()
        for idx in top_k_idx:
            result[self.labels[idx]] = predictions[idx]
        return result


def load_labels(label_path):
    with open(label_path, 'r') as f:
        labels = list(map(str.strip, f.readlines()))
    return labels
