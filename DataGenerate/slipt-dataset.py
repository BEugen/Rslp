import os
import shutil

IMG_PATH_TRAIN = '/mnt/misk/misk/lplate/data/data/train'
IMG_PATH_TEST = '/mnt/misk/misk/lplate/data/data/test'
IMG_PATH = '/mnt/misk/misk/lplate/data/jpg'
MSK_PATH = '/mnt/misk/misk/lplate/data/msk'
SPLIT_SIZE = 0.25


def main():
    list_file = os.listdir(IMG_PATH)

    split_size = len(list_file) - int(len(list_file) * SPLIT_SIZE)
    print(split_size)
    train = list_file[:split_size]
    test = list_file[split_size:]
    for file in train:
        shutil.move(IMG_PATH + '/' + file, IMG_PATH_TRAIN + '/img/' + file)
        file = file.replace('img_', 'msk_')
        shutil.move(MSK_PATH + '/' + file, IMG_PATH_TRAIN + '/msk/' + file)

    for file in test:
        shutil.move(IMG_PATH + '/' + file, IMG_PATH_TEST + '/img/' + file)
        file = file.replace('img_', 'msk_')
        shutil.move(MSK_PATH + '/' + file, IMG_PATH_TEST + '/msk/' + file)


if __name__ == '__main__':
    main()
