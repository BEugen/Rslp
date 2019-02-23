import numpy as np
import cv2
import math


class MotionDetect:
    def __init__(self, blur=3, evlc=2.0, fon_count=100, scadr=2):
        self.blur = blur
        self.evlc = evlc
        self.fon_cadr_count = fon_count
        self.older_image = None
        self.scadr = scadr
        self.sc_count = 0
        self.fc_count = 0
        self.fc_end = fon_count
        self.old_evl = 0.00001

    def evc_detect(self, image):
        image = cv2.resize(image, (64, 64))
        blur = cv2.medianBlur(image, self.blur)
        ret, thresh = cv2.threshold(blur, 45, 255, cv2.THRESH_BINARY)
        if self.older_image is None:
            self.older_image = thresh
            return 0.0
        diff = cv2.absdiff(thresh, self.older_image)
        cv2.imshow('New', diff)
        evl = np.linalg.norm(diff - 0) / 1000.0
        delta = math.fabs(evl - self.old_evl) / evl if evl > 0.0 else 0.0
        if delta > 2.5:
            self.fc_count += 1
        if (self.sc_count < self.scadr and evl >= self.evlc) or self.fc_count > self.fc_end:
            self.sc_count = 0
            self.fc_count = 0
            self.older_image = thresh
            self.old_evl = evl
        self.sc_count += 1
        return round(evl, 1), round(delta, 2), self.fc_count
