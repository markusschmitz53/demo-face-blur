"""
Copyright (c) 2023, Markus Schmitz

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import os
import cv2
import time
import logging
import argparse
import numpy as np
from pathlib import Path
from imutils.video import FileVideoStream, FPS
from VideoShow import VideoShow
from util import log_timing_info, anonymize_roi
from alive_progress import alive_bar

show_video = True
fvs_queue_size = 64
kernel_size = (20, 20)
device = 'cuda:0'
use_yolo = False
resize_video = False
parser = argparse.ArgumentParser()
parser.add_argument("path", nargs="?", type=Path)
args = parser.parse_args()

logging.basicConfig(
    format='%(asctime)s %(message)s',
    datefmt='%d.%m.%y %H:%M',
    stream=os.sys.stdout,
)
logging.getLogger().setLevel(logging.INFO)


# @profile
def process_file(input_video):
    datetime = time.strftime("%Y-%m-%d_%H%M")
    input_video_path = str(input_video.resolve())

    logging.info(f'Processing {input_video} start time {datetime}')

    fvs = FileVideoStream(input_video_path, None, fvs_queue_size).start()
    time.sleep(1.0)
    fps = FPS().start()
    frame = fvs.read()

    assert len(frame.shape) == 3, logging.critical('Frame has invalid shape')
    frame_height, frame_width, _ = frame.shape
    assert frame_height > 0, logging.critical('Frame height too low')
    assert frame_width > 0, logging.critical('Frame width too low')
    aspect_ratio = frame_width / frame_height
    frame_count = int(fvs.stream.get(cv2.CAP_PROP_FRAME_COUNT))

    if show_video:
        video_shower = VideoShow(frame).start()

    target_frame_width = frame_width
    target_frame_height = frame_height

    if resize_video:
        target_frame_width = 1280
        target_frame_height = int(target_frame_width / aspect_ratio)

    logging.info('Original video dimensions %dx%d' % (frame_width,
                                                      frame_height))

    if resize_video:
        logging.info('Applying video dimensions %dx%d' % (target_frame_width,
                                                          target_frame_height))

    yunet_model = cv2.FaceDetectorYN_create(
        model='models/face_detection_yunet_2022mar.onnx',
        config='',
        input_size=[target_frame_width, target_frame_height],
        score_threshold=0.7,
        nms_threshold=0.8,
        top_k=40)

    frames_processed = 0
    try:
        with alive_bar(frame_count) as bar:
            while fvs.running():
                frame = fvs.read()
                if frame is None:
                    fps.update()
                    bar()
                    continue

                _, faces = yunet_model.detect(frame)
                faces = faces if faces is not None else []
                faces = np.asarray(faces, dtype=np.int16)

                for face in faces:
                    x1, y1, w, h = face[:4]
                    x2 = x1 + w
                    y2 = y1 + h

                    frame = anonymize_roi(frame, x1, y1, x2, y2)

                # write frame to output video file here

                if show_video:
                    video_shower.frame = frame

                fps.update()
                frames_processed += 1
                bar()
        return True
    except KeyboardInterrupt:
        return False
    finally:
        fps.stop()
        fvs.stop()
        if show_video:
            video_shower.stop()
        if frames_processed > 0:
            log_timing_info(fps, frames_processed)


# @profile
def main():
    if args.path is None:
        logging.critical("Missing input video path")
        raise SystemExit(1)
    else:
        input_video = Path(args.path)
        if not input_video.exists():
            logging.critical("Input file doesn't exist")
            raise SystemExit(1)
    process_file(input_video)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()