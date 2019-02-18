import numpy as np
import cv2
import math


class MotionDetect:
    def __init__(self, blur=15, evlc=7.0, scadr=3):
        self.blur = blur
        self.evlc = evlc
        self.older_image = None
        self.scadr = scadr
        self.sc_count = 0

    def evc_detect(self, image):
        image = cv2.resize(image, (64, 64))
        blur = cv2.GaussianBlur(image, (self.blur, self.blur), 0)
        ret, thresh = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY)
        if self.older_image is None:
            self.older_image = thresh
            return 0.0
        cv2.imshow('Older', self.older_image)
        cv2.imshow('New', thresh)
        evl = np.linalg.norm(thresh - self.older_image)/15
        if self.sc_count > self.scadr and evl >= self.evlc:
            self.sc_count = 0
            self.older_image = thresh
        self.sc_count += 1
        return round(evl, 1)

