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

    def bw_area_open(self, image, areapixel=5):
        img = image.copy()
        cntr, _ = cv2.findContours(img.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for idx in np.arange(len(cntr)):
            area = cv2.contourArea(cntr[idx])
            if 0 <= area <= areapixel:
                cv2.drawContours(img, cntr, idx, (0, 0, 0), -1)
        return img

    def evc_detect(self, image):
        image = cv2.resize(image, (64, 64))
        blur = cv2.medianBlur(image, self.blur)
        ret, thresh = cv2.threshold(blur, 45, 255, cv2.THRESH_BINARY)
        if self.older_image is None:
            self.older_image = thresh
            return 0.0
        diff = cv2.absdiff(thresh, self.older_image)
        diff = self.bw_area_open(diff, areapixel=3)
        cv2.imshow('New', diff)
        evl = np.linalg.norm(diff - 0) / 1000.0
        delta = math.fabs(evl - self.old_evl) / evl if evl > 0.0 else 0.0
        if delta > 1.5:
            self.fc_count += 1
        if evl >= self.evlc or self.fc_count > self.fc_end:
            self.fc_count = 0
            self.older_image = thresh
            self.old_evl = evl
        return round(evl, 1), round(delta, 2), self.fc_count
