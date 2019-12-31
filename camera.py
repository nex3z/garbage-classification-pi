import io
import logging
import sys
import picamera
from PIL import Image
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

import utils as utils
from classifier import Classifier

LABEL_PATH = 'labels.txt'
MODEL_PATH = "./model/tf_mobilenet_v2_01_val_acc_8600_optimized.tflite"
TOP_NUM = 1
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 360
CAMERA_RESOLUTION = (CAMERA_WIDTH, CAMERA_HEIGHT)


def main():
    classifier = Classifier(MODEL_PATH, LABEL_PATH)
    font = ImageFont.truetype('heiti.ttf', 48)
    overlay = None
    with picamera.PiCamera(resolution=CAMERA_RESOLUTION, framerate=30) as camera:
        camera.start_preview()
        try:
            stream = io.BytesIO()
            for _ in camera.capture_continuous(stream, format='jpeg', use_video_port=True):
                stream.seek(0)
                image = Image.open(stream).convert('RGB')
                image = utils.center_crop(image)
                image_data = utils.preprocess(image, classifier.input_size)
                result = classifier.classify(image_data, TOP_NUM)

                output = ""
                for label, prob in result.items():
                    output += "{}({:.4f})".format(label, prob)
                print(output, end='\r')

                img = Image.new('RGBA', CAMERA_RESOLUTION, (255, 0, 0, 0))
                draw = ImageDraw.Draw(img)
                draw.text((0, CAMERA_HEIGHT-50), output, fill='white', font=font)
                pad = Image.new('RGBA', (
                    ((img.size[0] + 31) // 32) * 32,
                    ((img.size[1] + 15) // 16) * 16,
                ))
                pad.paste(img, (0, 0))

                o = camera.add_overlay(pad.tobytes(), size=img.size)
                o.alpha = 128
                o.layer = 3

                if overlay is not None:
                    camera.remove_overlay(overlay)
                overlay = o

                stream.seek(0)
                stream.truncate()

        finally:
            camera.stop_preview()


if __name__ == '__main__':
    logging.basicConfig(level=logging.WARN, format='%(levelname)s: %(message)s')
    logger = logging.getLogger(__name__)
    main()
