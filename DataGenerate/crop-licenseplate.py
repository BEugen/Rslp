import cv2
import numpy as np
import os
import math

MASK_PATH = ''
IMG_PATH = ''
IMG_LP = ''


def main():
    for file in os.listdir(MASK_PATH):
        mask = cv2.imread(os.path.join(MASK_PATH, file), cv2.IMREAD_GRAYSCALE)
        img = cv2.imread(os.path.join(IMG_PATH, file.replace('msk_', 'img_').replace('.jpg', '.bmp')),
                         cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (512, 512))
        mask = cv2.resize(mask, (512, 512))
        print(file)
        select_countur(mask, img, file)



def select_countur(mask, img, filename):
    _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    if len(contours) == 0:
        return
    cnt = contours[0]
    max_area = cv2.contourArea(cnt)

    for cont in contours:
        if cv2.contourArea(cont) > max_area:
            cnt = cont
            max_area = cv2.contourArea(cont)
    epsilon = 0.025 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    approx = np.reshape(approx, (approx.shape[0], 2))
    min_x, min_y = np.min(approx, axis=0)
    max_x, max_y = np.max(approx, axis=0)
    img_hh = (max_y - min_y)/2 + min_y
    img_hw = (max_x - min_x)/2 + min_x
    y_top_m = np.max(approx[approx[:, 1] < img_hh][:, 1])
    x_top_m = np.min(approx[approx[:, 1] == y_top_m][:, 0])
    #y_top_l = np.min(approx[approx[:, 0] > img_hl][:, 1])
    a = math.atan(math.fabs(min_y - y_top_m)/math.fabs(min_x-max_x))*180/math.pi
    a = a if x_top_m > img_hw else a * -1.0
    print(approx, a, x_top_m, y_top_m)

    ## cv2.drawContours(canvas, hull, -1, (0, 0, 255), 3) # only displays a few points as well.

    out = np.zeros_like(img)
    out[mask == 255] = img[mask == 255]
    #(x, y) = np.where(mask == 255)
    #(topx, topy) = (np.min(x), np.min(y))
    #(bottomx, bottomy) = (np.max(x), np.max(y))
    for i in range(-10, 10):
        out = img[min_y:max_y + 1, min_x:max_x + 1]
        out = rotateImage(out, i)
        out = cv2.GaussianBlur(out, (5, 5), 0)
        out = cv2.adaptiveThreshold(out, 200, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 9, 3)
        out = cv2.resize(out, (152, 34))
        cv2.imwrite(os.path.join(IMG_LP, str(i + 10) + '_' + filename), out)


def rotateImage(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

if __name__ == '__main__':
    main()
