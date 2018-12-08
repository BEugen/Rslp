import cv2
import numpy as np




def main():
    img = cv2.imread('X:/Books/FullPlate/data/ddenoise/test_in/16_654_C520YH35.bmp')
    # img_bw = 255 * (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) > 5).astype('uint8')
    #
    # se1 = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
    # se2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # mask = cv2.morphologyEx(img_bw, cv2.MORPH_CLOSE, se1)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se2)
    #
    # mask = np.dstack([mask, mask, mask]) / 255
    # out = img * mask
    canvas = np.zeros(img.shape, np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    im2, contours, hierarchy = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for cnt in contours:
        epsilon = 0.005 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        approx = np.reshape(approx, (approx.shape[0], 2))
        min_x, min_y = np.min(approx, axis=0)
        max_x, max_y = np.max(approx, axis=0)
        cv2.rectangle(img, (min_x, min_y), (max_x, max_y), (0, 0, 255), 1)
        #cv2.drawContours(canvas, approx, -1, (0, 255, 0), 1)
    cv2.imshow("Contour", cv2.resize(img, (img.shape[1] * 3, img.shape[0] * 3)))


   # cv2.imshow('Output', out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

