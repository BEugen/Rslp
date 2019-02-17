import numpy as np
import cv2
import math


class MotionDetect:
    def __init__(self, blur=11, evlc=10.0):
        self.blur = blur
        self.evlc = evlc
        self.older_image = None

    def evc_detect(self, image):
        blur = cv2.GaussianBlur(image, (self.blur, self.blur), 0)
        ret, thresh = cv2.threshold(blur, 25, 255, cv2.THRESH_BINARY)
        if self.older_image is None:
            self.older_image = thresh
            return 0.0
        evl = np.linalg.norm(thresh - self.older_image)/100
        if evl >= self.evlc:
            self.older_image = blur
        return round(evl, 1)

