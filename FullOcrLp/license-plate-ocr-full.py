import os
import nwirecognise
import cv2
import uuid
from multiprocessing import Process, Queue
from threading import Thread
import atexit
import time
import motiondetect
import socket
import argparse
import json
import logging

IMG_PATH = '/mnt/misk/misk/lplate/images'
IMG_FOR_OCR = 'E:/temp/chars'
MOTION_H_LEVEL = 15.0
MOTION_L_LEVEL = 0.0
MOTION_HW_OBJECT = 50


def ocr(qo, qi):
    rc = nwirecognise.RecognizeLp()
    while True:
        if qi.qsize() > 0:
            fn = os.path.join(IMG_FOR_OCR, str(uuid.uuid4()))
            if not os.path.exists(fn):
                os.makedirs(fn)
            image = qi.get()
            img_path = os.path.join(fn, str(uuid.uuid4()) + '.jpg')
            cv2.imwrite(img_path, image)
            rc.recognize(image, fn)
            qo.put([rc.ok_ocr, rc.date_ocr, rc.number_ocr, img_path])
            if rc.ok_ocr:
                while qi.qsize() > 0:
                    qi.get()
            print('End ocr')
        time.sleep(0.1)


async def tcp_echo_client(server, port, message):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((server, port))
    print('Send: %r' % message)
    s.send(message.encode())
    s.close()


def ocr_kill(ocr):
    ocr.terminate()


def video_capture(source, width, height):
    try:
        cap = cv2.VideoCapture(source)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        return cap
    except:
        logging.exception('')
        return None


def main(args):
    number = ''
    _img_count = 0
    dsf = os.path.dirname(os.path.realpath(__file__))
    js_path = os.path.join(dsf, args.config_file)
    json_file = open(js_path).read()
    js = json.loads(json_file)
    config = js[str(args.config)]
    x1, x2, y1, y2 = config['region']  # (125, 950, 100, 620)#src='http:///mjpg/video.mjpg',
    cap = video_capture(config['videosource'], config['width'], config['height'])
    if cap is None:
        return
    server = config['server']
    port = config['port']
    qo = Queue()
    qi = Queue()
    p = Process(target=ocr, args=(qo, qi))
    p.start()
    atexit.register(ocr_kill, p)
    md = motiondetect.MotionDetect()
    while True:
        try:
            ret, image = cap.read()
            if image is None:
                cap.release()
                cap = video_capture(config['videosource'], config['width'], config['height'])
                time.sleep(2)
                continue
            img = image[y1:y2, x1:x2]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cv2.rectangle(image, (0, 0), (image.shape[1], 50), (255, 255, 255), cv2.FILLED)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            evc_l = md.evc_detect(img.copy())
            cv2.putText(image, str(evc_l), (620, 200), font, 1, (255, 100, 0), 2, cv2.LINE_AA)
            if evc_l >= 5.0 and _img_count < 10:
                _img_count += 1
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                qi.put(img)
            if qo.qsize() > 0:
                ocr_data = qo.get()
                _img_count -= 1
                if ocr_data[0]:
                    number = ocr_data[2] + '&' + ocr_data[1].strftime('%d.%m.%Y %H:%M:%S')
                    cv2.rectangle(image, (0, 0), (image.shape[1], 50), (0, 255, 0), 2)
                    t = Thread(target=tcp_echo_client, args=(server, port, str(args.config) + '&' + number + '&' +
                                                             ocr_data[3] + '&'))
                    t.start()
                else:
                    cv2.rectangle(image, (0, 0), (image.shape[1], 50), (0, 0, 255), 2)
            cv2.putText(image, number, (10, 35), font, 1, (255, 100, 0), 2, cv2.LINE_AA)
            # #image[y1:y2, x1:x2] = imgc
            cv2.imshow('Video', image)
            if cv2.waitKey(1) == 27:
                cap.release()
                cv2.destroyAllWindows()
                exit(0)
            if not p.is_alive():
                p = Process(target=ocr, args=(qo, qi))
                p.start()
        except:
            logging.exception('')
            cap.release()
            cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=0, type=int,
                        help="Number config to recognise")
    parser.add_argument('--config_file', default='config.json',
                        help="Config file")
    args = parser.parse_args()
    main(args)
