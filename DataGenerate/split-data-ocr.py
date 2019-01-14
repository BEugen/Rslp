import config
import os
import shutil

IMG_SRC = '/mnt/misk/misk/lplate/augchar'
IMG_TEST = '/mnt/misk/misk/lplate/augchar/test'
IMG_TRAIN = '/mnt/misk/misk/lplate/augchar/train'
SPLIT_SIZE = 0.25

def main():
    letters = config.LETTERS
    for icl in range(0, len(letters)):
        folder = os.path.join(IMG_SRC, letters[icl])
        list_file = os.listdir(folder)

        split_size = len(list_file) - int(len(list_file) * SPLIT_SIZE)
        print(split_size)
        train = list_file[:split_size]
        test = list_file[split_size:]
        for file in train:
            img_path = os.path.join(folder, file)
            shutil.copy(img_path, os.path.join(IMG_TRAIN, file + '_' + str(icl) + '.jpg'))

        for file in test:
            img_path = os.path.join(folder, file)
            shutil.copy(img_path, os.path.join(IMG_TEST, file + '_' + str(icl) + '.jpg'))


if __name__ == '__main__':
    main()
