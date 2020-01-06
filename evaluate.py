import argparse

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm


# noinspection PyUnresolvedReferences
def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    val_generator = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

    val_iter = val_generator.flow_from_directory(
        args.data_dir,
        target_size=(args.input_size, args.input_size),
        batch_size=1,
        shuffle=False,
        color_mode='rgb',
        class_mode='categorical',
    )

    interpreter = interpreter = tf.lite.Interpreter(model_path=args.tflite_model_file)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    count, acc = 0, 0
    loop = tqdm(val_iter, ascii=True)
    for i, (data, label) in enumerate(loop):
        interpreter.set_tensor(input_details[0]['index'], data)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])
        count += 1
        if np.argmax(predictions) == np.argmax(label):
            acc += 1
        loop.set_postfix(acc="{:.4f}".format(acc / count))
        if i == len(val_iter):
            break
    loop.close()
    print("accuracy: {:.4f}".format(acc / count))


def build_arg_parser():
    parser = argparse.ArgumentParser(description='Convert TensorFlow models to T.')
    parser.add_argument('--tflite_model_file', type=str, dest='tflite_model_file',
                        help='Path to TensorFlow Lite model file.')
    parser.add_argument('--data_dir', type=str, dest='data_dir',
                        help='Path to evaluate data directory.')
    parser.add_argument('--input_size', type=int, dest='input_size', default=224,
                        help='Model input size.')
    return parser


if __name__ == '__main__':
    main()
