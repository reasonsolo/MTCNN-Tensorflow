import os
import numpy as np
from BBox_utils import BBox


def read_muct_annotation(path):
    with open(path, 'r') as f:
        iterlines = iter(f)
        next(iterlines)
        for line in iterlines:
            segs = line.strip().split(',')
            img_file, landmarks = segs[0], [float(i) for i in segs[1:]]
            yield img_file, zip(landmarks[::2], landmarks[1::2])

def load_muct_annotation(filepath):
    for img_file, landmarks in read_muct_annotation(filepath):
        max_x, max_y = 0, 0
        min_x, min_y = np.inf, np.inf
        for x, y in landmarks:
            max_x = max(int(x), max_x)
            max_y = max(int(y), max_y)
            min_x = min(int(x), min_x)
            min_y = min(int(y), min_y)
        yield img_file, [min_x, min_y, max_x, max_y], landmarks


def read_helen_annotation(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        striped = [x.strip() for x in lines]
        name, landmarks = striped[0], striped[1:]
        return name, [(float(x.strip()), float(y.strip())) for x, y in [landmark.split(',') for landmark in landmarks]]

def load_helen_annotation(filedir):
    for anno_file in os.listdir(filedir):
        file_path =  os.path.join(filedir, anno_file)
        img_name, landmarks = read_helen_annotation(file_path)
        # print('anno path %s img file %s' % (file_path, img_name))
        max_x, max_y = 0, 0
        min_x, min_y = np.inf, np.inf
        for x, y in landmarks:
            max_x = max(int(x), max_x)
            max_y = max(int(y), max_y)
            min_x = min(int(x), min_x)
            min_y = min(int(y), min_y)
        yield img_name, [min_x, min_y, max_x, max_y], landmarks


