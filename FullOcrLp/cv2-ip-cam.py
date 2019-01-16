import base64
import time
import urllib3

import cv2
import numpy as np


"""
Examples of objects for image frame aquisition from both IP and
physically connected cameras
Requires:
 - opencv (cv2 bindings)
 - numpy
"""

import numpy as np

def main():
    import cv2
    cap = cv2.VideoCapture('http:///mjpg/video.mjpg')
    im_prev = None
    while True:
        ret, frame = cap.read()
        im_p = cv2.cvtColor(frame, cv2.IMREAD_GRAYSCALE)
        x1, x2, y1, y2 = (125, 950, 280, 700)
        img = im_p[y1:y2, x1:x2]
        if im_prev is None:
            im_prev = img
            continue
        ld = np.sum((img.astype("float") - im_prev.astype("float")) ** 2)
        ld /= float(im_p.shape[0] * im_p.shape[1])
        ld = round(ld / 100.0, 2)
        x1, x2, y1, y2 = (125, 950, 280, 700)
        cv2.rectangle(im_p, (0, 0), (im_p.shape[1], 50), (255, 255, 255), cv2.FILLED)
        cv2.rectangle(im_p, (x1, y1), (x2, y2), (0, 0, 255), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(im_p, str(ld), (10, 50), font, 2, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow('Video', im_p)
        if cv2.waitKey(1) == 27:
            exit(0)

if __name__ == '__main__':
    main()