from PIL import Image, ImageDraw, ImageFont



def main():
    img = Image.new('RGB', (152, 34), (255, 255, 255))
    fnt = ImageFont.truetype('RoadNumbers2.0.ttf', 30)
    d = ImageDraw.Draw(img)
    d.text((11, 15), "M", font=fnt, fill=(0, 0, 0))
    img.save('test.png')


if __name__ == '__main__':
    main()

