"""
Detections script

"""

import os
import sys
import argparse
import logging
import json

from PIL import Image
import pandas as pd
import numpy as np

sys.path += [os.path.abspath('.'), os.path.abspath('..')]
from yolo3.yolo import YOLO
from yolo3.utils import update_path


def arg_params_yolo():
    # class YOLO defines the default value, so suppress any default HERE
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    # Command line options
    parser.add_argument('--path_model', type=str,
                        help='path to model weight file',
                        default=YOLO.get_defaults("model_path"))
    parser.add_argument('--path_anchors', type=str,
                        help='path to anchor definitions',
                        default=YOLO.get_defaults("anchors_path"))
    parser.add_argument('--path_classes', type=str,
                        help='path to class definitions',
                        default=YOLO.get_defaults("classes_path"))
    parser.add_argument('--gpu_num', type=int, help='Number of GPU to use',
                        default=str(YOLO.get_defaults("gpu_num")))
    parser.add_argument("--path_output", nargs='?', type=str, default='.',
                        help='path to the output directory')
    return parser


def parse_params():
    # class YOLO defines the default value, so suppress any default HERE
    parser = arg_params_yolo()
    parser.add_argument('--path_image', nargs='*', type=str, required=False,
                        help='images to be processed')
    parser.add_argument("--path_video", nargs='*', type=str, required=False,
                        help='Video to be processed')
    params = parser.parse_args()
    # if there is only single path still make it as a list
    if hasattr(params, 'path_image') and not isinstance(params.path_image, list):
        params.path_image = [params.path_image]
    # if there is only single path still make it as a list
    if hasattr(params, 'path_video') and not isinstance(params.path_video, list):
        params.path_video = [params.path_video]
    return params


def predict_image(yolo, path_image, path_output=None):
    path_img = update_path(path_image)
    try:
        image = Image.open(path_img)
    except Exception:
        logging.error('Fail open image "%s"', path_img)
        return
    else:
        image_pred, pred_items = yolo.detect_image(image)
        if path_output is None or not os.path.isdir(path_output):
            image_pred.show()
        else:
            name = os.path.splitext(os.path.basename(path_image))[0]
            path_img = os.path.join(path_output, name + '_visual.jpg')
            path_csv = os.path.join(path_output, name + '.csv')
            logging.info('exporting "%s" and "%s"', path_img, path_csv)
            image_pred.save(path_img)
            pd.DataFrame(pred_items).to_csv(path_csv)


def predict_video(yolo, path_video, path_output=None):
    path_video = update_path(path_video)
    assert os.path.isfile(path_video), 'missing: %s' % path_video

    import cv2
    # Create a video capture object to read videos
    vid = cv2.VideoCapture(path_video)

    b_output = path_output is not None and os.path.isdir(path_output)

    if b_output:
        video_format = cv2.VideoWriter_fourcc('F', 'M', 'P', '4')
        video_fps = vid.get(cv2.CAP_PROP_FPS)
        video_size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                      int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        name = os.path.splitext(os.path.basename(path_video))[0]
        path_out = os.path.join(path_output, name + '_visual.avi')
        logging.debug('export viode: %a', path_out)
        out = cv2.VideoWriter(path_out, video_format, video_fps, video_size)
        frame_preds = []

    while vid.isOpened():
        success, frame = vid.read()
        if not success:
            break
        image = Image.fromarray(frame)
        image_pred, pred_items = yolo.detect_image(image)
        frame = np.asarray(image_pred)

        if b_output:
            out.write(frame)
            frame_preds.append(pred_items)
        else:
            # show frame
            cv2.imshow('YOLO', frame)

    if b_output:
        out.release()
        path_json = os.path.join(path_output, name + '.json')
        logging.info('export predictions: %s', path_json)
        with open(path_json, 'w') as fp:
            json.dump(frame_preds, fp)


def _main(path_model, path_anchors, path_classes, path_output, gpu_num=0, **kwargs):
    path_model = update_path(path_model)
    assert os.path.isfile(path_model), 'missing "%s"' % path_model
    path_anchors = update_path(path_anchors)
    assert os.path.isfile(path_anchors), 'missing "%s"' % path_anchors
    path_classes = update_path(path_classes)
    assert os.path.isfile(path_classes), 'missing "%s"' % path_classes
    path_output = update_path(path_output)
    assert os.path.isdir(path_output), 'missing "%s"' % path_output

    yolo = YOLO(model_path=path_model, anchors_path=path_anchors, classes_path=path_classes)

    if 'path_image' in kwargs:
        for path_img in kwargs['path_image']:
            logging.info('processing: "%s"', path_img)
            predict_image(yolo, path_img, path_output)
    if 'path_video' in kwargs:
        for path_vid in kwargs['path_video']:
            logging.info('processing: "%s"', path_img)
            predict_video(yolo, path_vid, path_output)

    yolo.close_session()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    # class YOLO defines the default value, so suppress any default HERE
    arg_params = parse_params()

    _main(**vars(arg_params))

    logging.info('Done')
