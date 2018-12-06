from PIL import Image, ImageDraw, ImageFont
import os


DIGI_8 = [(10, 11), (30, 11), (46, 11), (62, 11), (80, 11), (96, 11), (119, 8), (133, 8)]
DIGI_9 = [(10, 11), (27, 11), (42, 11), (59, 11), (75, 11), (92, 11), (112, 8), (121, 8), (134, 8)]
FONT_SIZE8 = (35, 35, 35, 35, 35, 35, 26, 26)
FONT_SIZE9 = (35, 35, 35, 35, 35, 35, 24, 24, 24)

IMG_DIR = 'X:/Books/FullPlate/data/ddenoise/test_in'
IMG_SAVE = 'X:/Books/FullPlate/data/ddenoise/test'
def main():
    for file in os.listdir(IMG_DIR):
        file = os.path.splitext(file)[0]
        file = file.split('_')
        lipl(file[len(file) - 1])


def lipl(text):
    img = Image.new('RGB', (152, 34), (255, 255, 255))
    d = ImageDraw.Draw(img)
    lt = len(text)
    for i in range(0, lt):
        fnt = ImageFont.truetype('RoadNumbers2.0.ttf', FONT_SIZE8[i] if lt == 8 else FONT_SIZE9[i])
        d.text(DIGI_8[i] if lt == 8 else DIGI_9[i], text[i], font=fnt, fill=(0, 0, 0))
    img.save(os.path.join(IMG_SAVE, text + '.png'))


if __name__ == '__main__':
    main()


