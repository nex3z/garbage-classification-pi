import io
import logging
import sys
import picamera
from PIL import Image

import utils as utils
from classifier import Classifier

LABEL_PATH = 'labels.txt'
MODEL_PATH = "./model/tf_mobilenet_v2_01_val_acc_8600_optimized.tflite"
TOP_NUM = 1
CAMERA_WIDTH = 480
CAMERA_HEIGHT = 480


def main():
    classifier = Classifier(MODEL_PATH, LABEL_PATH)

    with picamera.PiCamera(resolution=(CAMERA_WIDTH, CAMERA_HEIGHT), framerate=30) as camera:
        camera.start_preview()
        try:
            stream = io.BytesIO()
            for _ in camera.capture_continuous(stream, format='jpeg', use_video_port=True):
                stream.seek(0)
                image = Image.open(stream).convert('RGB')
                image_data = utils.preprocess(image, classifier.input_size)
                result = classifier.classify(image_data, TOP_NUM)

                output = ""
                for label, prob in result.items():
                    output += "{}: {}".format(label, prob)
                
                print(output, end='\r')

                stream.seek(0)
                stream.truncate()

        finally:
            camera.stop_preview()


if __name__ == '__main__':
    logging.basicConfig(level=logging.WARN, format='%(levelname)s: %(message)s')
    logger = logging.getLogger(__name__)
    main()
