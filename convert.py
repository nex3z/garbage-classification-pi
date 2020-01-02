import argparse

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    print(args)

    if args.keras_model_file is not None:
        model = load_model(args.keras_model_file)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
    elif args.keras_model_file is not None:
        converter = tf.lite.TFLiteConverter.from_saved_model(args.keras_model_file)
    else:
        return

    if args.quant == 'default':
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    elif args.quant == 'size':
        converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    elif args.quant == 'latency':
        converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]
    elif args.quant == 'float16':
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.lite.constants.FLOAT16]
    elif args.quant == 'int':
        data_generator = ImageDataGenerator(
            preprocessing_function=preprocess_input
        )
        data_iterator = data_generator.flow_from_directory(
            args.data_dir,
            target_size=(args.input_size, args.input_size),
            batch_size=1,
            color_mode='rgb',
            class_mode=None
        )

        def representative_dataset_gen():
            for i in range(len(data_iterator)):
                yield [next(data_iterator)]

        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset_gen
        # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        # converter.inference_input_type = tf.uint8
        # converter.inference_output_type = tf.uint8

    tflite_model = converter.convert()
    open(args.output_file, 'wb').write(tflite_model)


def build_arg_parser():
    parser = argparse.ArgumentParser(description='Convert TensorFlow models to T.')
    parser.add_argument('--saved_model_dir', type=str, dest='saved_model_dir',
                        help='Path to saved model directory.')
    parser.add_argument('--keras_model_file', type=str, dest='keras_model_file',
                        help='Path to kears HDF5 model file.')
    parser.add_argument('--output_file', type=str, dest='output_file', help='Path of the output file.')
    parser.add_argument('--quant', type=str, dest='quant',
                        help='Quantization options, one of default, size, latency, float16, int. '
                             'Default no quantization.')
    parser.add_argument('--data_dir', type=str, dest='data_dir',
                        help='Path to representative data directory. Mandatory when use int quantization.')
    parser.add_argument('--input_size', type=int, dest='input_size', default=224,
                        help='Model input size. Mandatory when use int quantization.')
    return parser


if __name__ == '__main__':
    main()
