import os
import cv2
from keras.models import *
from keras.layers import *
from keras.optimizers import *
import logging
import shutil


IMG_FOLDER = '/mnt/misk/misk/lplate/nchars'
IMG_CHARS = '/mnt/misk/misk/lplate/cchars'
LETTERS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A',
                'B', 'C', 'E', 'H', 'K', 'M', 'O', 'P', 'T', 'X', 'Y', ' ']

def main():
    model = get_model_ocr_lp('nn/', 'model-ocr-lp')
    for fl in os.listdir(IMG_FOLDER):
        file = os.path.join(IMG_FOLDER, fl)
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        pclass = image_ocr(model, img)
        if pclass is None:
            continue
        folder_store = os.path.join(IMG_CHARS, LETTERS[int(pclass)])
        if not os.path.exists(folder_store):
            os.makedirs(folder_store)
        shutil.copy(file, os.path.join(folder_store, fl))


def image_ocr(model, images):
    try:
        images = np.array(images) / 255
        images = np.reshape(images, (1, ) + images.shape + (1,))
        predict = model.predict_classes(images)
        return predict
    except:
        logging.exception('')
        return None

def get_model_ocr_lp(folder_nn, nn_ocr_lp):
    json_file = open(folder_nn + nn_ocr_lp + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(folder_nn + nn_ocr_lp + '.h5')
    loaded_model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy')
    return loaded_model


if __name__ == '__main__':
    main()
