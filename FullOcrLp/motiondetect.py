import numpy as np
import cv2
import math


class MotionDetect:
    def __init__(self, blur=3, evlc=2.0, fon_count=200):
        self.blur = blur
        self.evlc = evlc
        self.fon_cadr_count = fon_count
        self.older_image = None
        self.fc_count = 0
        self.fc_end = fon_count
        self.old_evl = 0.00001

    def evc_detect(self, image):
        image = cv2.resize(image, (64, 64))
        blur1 = cv2.GaussianBlur(image, (3, 3), 0)
        blur2 = cv2.GaussianBlur(image, (25, 25), 12)
        imgsub = cv2.subtract(blur1, blur2)
        if self.older_image is None:
            self.older_image = imgsub
            return 0.0
        diff = cv2.absdiff(imgsub, self.older_image)
        ret, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        cv2.imshow('New', thresh)
        evl = np.linalg.norm(thresh - 0) / 1000.0
        delta = math.fabs(evl - self.old_evl) / evl if evl > 0.0 else 0.0
        if delta > 1.5:
            self.fc_count += 1
        if evl >= self.evlc or self.fc_count > self.fc_end:
            self.fc_count = 0
            self.older_image = imgsub
            self.old_evl = evl
        return round(evl, 1), round(delta, 2), self.fc_count
