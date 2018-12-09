import os
import shutil


IMG_PATH = '/mnt/misk/misk/lplate/data/data_rt'
IMG_PATH_TMP = '/mnt/misk/misk/lplate/data/data_rt/tmp'
IMG_TRAIN = '/mnt/misk/misk/lplate/data/data_rt/train'
IMG_TEST = '/mnt/misk/misk/lplate/data/data_rt/test'
CLASS_0 = 0
CLASS_1 = 1
SPLIT_SIZE =0.25

def main():
    img_path = os.path.join(IMG_PATH, str(CLASS_0))
    for file in os.listdir(img_path):
        file_name = os.path.splitext(file)
        shutil.move(os.path.join(img_path, file), os.path.join(IMG_PATH_TMP, file_name[0] + '_' +
                                                               str(CLASS_0) + file_name[1]))
    img_path = os.path.join(IMG_PATH, str(CLASS_1))
    for file in os.listdir(img_path):
        file_name = os.path.splitext(file)
        shutil.move(os.path.join(img_path, file), os.path.join(IMG_PATH_TMP, file_name[0] + '_' +
                                                               str(CLASS_1) + file_name[1]))

    list_file = os.listdir(IMG_PATH_TMP)
    split_size = len(list_file) - int(len(list_file) * SPLIT_SIZE)
    print(split_size)
    train = list_file[:split_size]
    test = list_file[split_size:]
    for file in train:
        shutil.move(os.path.join(IMG_PATH_TMP, file), os.path.join(IMG_TRAIN, file))

    for file in test:
        shutil.move(os.path.join(IMG_PATH_TMP, file), os.path.join(IMG_TEST, file))


if __name__ == '__main__':
    main()

