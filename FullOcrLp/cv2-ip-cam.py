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
        if im_prev is None:
            im_prev = im_p
            continue
        ld = np.sum((im_p.astype("float") - im_prev.astype("float"))**2)
        ld /= float(im_p.shape[0]*im_p.shape[1])
        ld = round(ld/100, 2)
        cv2.rectangle(im_p, (0, 0), (1000, 50), (255, 255, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(im_p, str(ld), (10, 50), font, 2, (0,0,0), 2, cv2.LINE_AA)
        cv2.imshow('Video', im_p)
        if cv2.waitKey(1) == 27:
         exit(0)

if __name__ == '__main__':
    main()