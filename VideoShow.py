"""
Copyright (c) 2018 Najam Syed

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

@see https://github.com/nrsyed/computer-vision
"""

from threading import Thread
import cv2


class VideoShow:
    """
    Class that continuously shows a frame using a dedicated thread.
    """

    def __init__(self, frame=None):
        self.frame = frame
        self.stopped = False

    def start(self):
        Thread(target=self.show, args=()).start()
        return self

    def show(self):
        while not self.stopped:
            cv2.imshow("demo-face-blur", self.frame)
            if cv2.waitKey(1) == ord("q"):
                self.stopped = True

    def stop(self):
        self.stopped = True
