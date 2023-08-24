import cv2
import math
import logging


def anonymize_roi(frame, x1, y1, x2, y2):
    roi = frame[y1:y2, x1:x2]
    if not roi.size:
        return frame

    return cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), -1)


def hms(seconds):
    h = int(seconds // 3600)
    m = int(seconds % 3600 // 60)
    s = int(seconds % 3600 % 60)
    return '{:02d}:{:02d}:{:02d}'.format(h, m, s)


def log_timing_info(fps, frames_processed):
    assert frames_processed > 0, logging.error('No frames processed')
    time_per_frame = fps.elapsed() / frames_processed
    logging.info('########## INFO ##########')
    logging.info("elapsed time:\t%s" % hms(fps.elapsed()))
    logging.info("approx. FPS:\t%s" % math.floor(fps.fps()))
    logging.info("time/frame:\t{:.2f}s".format(time_per_frame))
    logging.info('##########################')