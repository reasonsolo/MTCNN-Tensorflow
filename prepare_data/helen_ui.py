import cv2
import numpy as np
import os
import time

ANNOTATION_DIR = 'annotation'
IMAGE_DIRS = ['helen_1']


def read_annotation(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        striped = [x.strip() for x in lines]
        name, landmarks = striped[0], striped[1:]
        return name, [(int(float(x.strip())), int(float(y.strip()))) for x, y in [landmark.split(',') for landmark in landmarks]]

screen_res = 1280, 720

if __name__ == '__main__':
    cv2.namedWindow('helen', cv2.WINDOW_NORMAL)
    font = cv2.FONT_HERSHEY_SIMPLEX

    for annotation in [os.path.join(ANNOTATION_DIR, f) for f in os.listdir(ANNOTATION_DIR)]:
        name, landmarks = read_annotation(annotation)
        possible_image_files = [os.path.join(p, name + '.jpg') for p in IMAGE_DIRS]
        for pif in possible_image_files:
            if not os.path.isfile(pif):
                continue
            image = cv2.imread(pif)
            print('open image %s' % pif)
            scale_width = screen_res[0] / image.shape[1]
            scale_height = screen_res[1] / image.shape[0]
            scale = min(scale_width, scale_height)
            window_width = int(image.shape[1] * scale)
            window_height = int(image.shape[0] * scale)
            print('image open success %s, rescale windows to %dx%d' % (pif, window_width, window_height))
            if window_width > 0 and window_height > 0:
                cv2.resizeWindow('helen', window_width, window_height)
            else:
                continue
            cv2.imshow('helen', image)
            k = cv2.waitKey(0)
            for i, landmark in enumerate(landmarks):
                # cv2.circle(image, landmark, 1, (0, 255, 255))
                cv2.putText(image, str(i), landmark, font, 0.4, (0, 255, 255))
                cv2.imshow('helen', image)
                k = cv2.waitKey(1)
                if k == ord(' '):
                    break
            k = cv2.waitKey(0)
    cv2.destroyAllWindows()


