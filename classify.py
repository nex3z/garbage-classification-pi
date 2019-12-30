import argparse
import logging

import utils as utils
from classifier import Classifier

LABEL_PATH = 'labels.txt'
MODEL_PATH = "./model/tf_mobilenet_v2_01_val_acc_8600_optimized.tflite"
TOP_NUM = 5


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    classifier = Classifier(MODEL_PATH, LABEL_PATH)
    image_data = utils.preprocess_image_file(args.image_file, classifier.input_size)
    result = classifier.classify(image_data, TOP_NUM)

    for label, prob in result.items():
        print(label, prob)


def build_arg_parser():
    parser = argparse.ArgumentParser(description='Image Classification')
    parser.add_argument('-f', dest='image_file', type=str, help='Image file', required=True)
    return parser


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logger = logging.getLogger(__name__)
    main()
