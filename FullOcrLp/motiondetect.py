import numpy as np
import cv2
import math


class MotionDetect:
    def __init__(self, region=(0, 50, 0, 50), limit_height=10, limit_width=10,
                 blur=21, motion_obj_size=50, motoin_l_limit=0, motion_h_limit=11):
        self.limit_heigh = limit_height
        self.limit_width = limit_width
        self.motion_l_limit = motoin_l_limit
        self.motion_h_limit = motion_h_limit
        self.blur = blur
        self.motion_obj_size = motion_obj_size
        x1, x2, y1, y2 = region
        self.region_control = (x1 + limit_width, x2 - limit_width, y1 + limit_height, y2 - limit_height)
        self.older_image = None
        self.x = 0
        self.y = 0
        self.xo = 0
        self.yo = 0
        self.w = 0
        self.h = 0

    def evc_detect(self, image, blur=11, evlc=3.0):
        blur = cv2.GaussianBlur(image, (blur, blur), 0)
        if self.older_image is None:
            self.older_image = blur
            return 0.0
        ret, thresh1 = cv2.threshold(blur, 25, 255, cv2.THRESH_BINARY)
        ret, thresh2 = cv2.threshold(self.older_image, 25, 255, cv2.THRESH_BINARY)
        evl = np.linalg.norm(thresh1 - thresh2)/100
        if evl >= evlc:
            self.older_image = blur
        return round(evl, 1)


    def detect(self, image):
        result = False
        blur = cv2.GaussianBlur(image, (self.blur, self.blur), 0)
        if self.older_image is None:
            self.older_image = blur
            return False
        diff = cv2.absdiff(self.older_image, blur)
        self.older_image = blur
        ret, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=3)
        _, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        (x, y, w, h) = 0, 0, 0, 0
        if len(contours) != 0:
            for cont in contours:
                epsilon = 0.025 * cv2.arcLength(cont, True)
                approx = cv2.approxPolyDP(cont, epsilon, True)
                (x, y, w, h) = cv2.boundingRect(approx)
                if h < self.motion_obj_size and w < self.motion_obj_size:
                    (x, y, w, h) = 0, 0, 0, 0
                    continue
                else:
                    self.w = w
                    self.h = h
                    break

            # cnt = contours[0]
            # max_area = cv2.contourArea(cnt)
            # for cont in contours:
            #     if cv2.contourArea(cont) > max_area:
            #         cnt = cont
            #         max_area = cv2.contourArea(cont)
            # epsilon = 0.025 * cv2.arcLength(cnt, True)
            # approx = cv2.approxPolyDP(cnt, epsilon, True)
            # (x, y, w, h) = cv2.boundingRect(approx)
            # if h < self.motion_obj_size and w < self.motion_obj_size:
            #     return False
        xd, yd = (math.fabs(x - self.x), math.fabs(y - self.y))
        if self.motion_l_limit <= xd < self.motion_h_limit \
                and self.motion_l_limit <= yd < self.motion_h_limit \
                and (self.x < 1.5 and self.y < 1.5) and (self.xo != 0 or self.yo != 0):
                #and (self.w > 100 or self.h > 100):
            print('Recognise!!!')
            print(xd, self.x, self.xo, '||', yd, self.y, self.yo, '||', self.w, self.h)
            result = True
        self.xo = self.x
        self.yo = self.y
        self.x = x
        self.y = y
        return result

