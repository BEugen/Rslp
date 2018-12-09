import cv2
import numpy as np
import math
import os




def main():
    img_path = '/mnt/misk/misk/lplate/data/data_rt/1'
    for file in os.listdir(img_path):
        img = cv2.imread(os.path.join(img_path, file))
        if img is None:
            continue
        img = cv2.resize(img, (152, 34))
        # img_bw = 255 * (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) > 5).astype('uint8')
        # #
        # se1 = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
        # se2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        # mask = cv2.morphologyEx(img_bw, cv2.MORPH_CLOSE, se1)
        # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se2)
        # #
        # mask = np.dstack([mask, mask, mask]) / 255
        # out = img * mask
        # canvas = np.zeros(img.shape, np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img[img.shape[0] - 4: img.shape[0], 0:img.shape[1]] = 255
        img[0: 4, 0:img.shape[1]] = 255
        ret, img = cv2.threshold(img, 100, 255, 0)
        im2, contours, hierarchy = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_KCOS)
        for cnt in contours:
            if cv2.contourArea(cnt) < 60:
                cv2.fillPoly(img, pts=[cnt], color=(255, 255, 255))
        im2, contours, hierarchy = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_KCOS)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for cnt in contours:
            epsilon = 0.05 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            approx = np.reshape(approx, (approx.shape[0], 2))
            min_x, min_y = np.min(approx, axis=0)
            max_x, max_y = np.max(approx, axis=0)
            if (max_x - min_x) > 0:
                koeff = math.fabs((max_y - min_y) / (max_x - min_x))
                if 0.5 < koeff < 2.2 and cv2.contourArea(cnt) > 80:
                    print(koeff, max_x - min_x)
                    cv2.rectangle(img, (min_x, min_y), (max_x, max_y), (0, 0, 255), 1)
            #cv2.drawContours(canvas, approx, -1, (0, 255, 0), 1)
        cv2.imshow("Contour", cv2.resize(img, (img.shape[1] * 3, img.shape[0] * 3)))
        cv2.waitKey(2000)

   # cv2.imshow('Output', out)

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

