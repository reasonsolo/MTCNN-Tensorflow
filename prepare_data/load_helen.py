import os
import numpy as np
from BBox_utils import BBox

def read_annotation(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        striped = [x.strip() for x in lines]
        name, landmarks = striped[0], striped[1:]
        return name, [(float(x.strip()), float(y.strip())) for x, y in [landmark.split(',') for landmark in landmarks]]

def load_annotation(filedir):
    for anno_file in os.listdir(filedir):
        file_path =  os.path.join(filedir, anno_file)
        img_name, landmarks = read_annotation(file_path)
        print('anno path %s img file %s' % (file_path, img_name))
        max_x, max_y = 0, 0
        min_x, min_y = np.inf, np.inf
        for x, y in landmarks:
            max_x = max(int(x), max_x)
            max_y = max(int(y), max_y)
            min_x = min(int(x), min_x)
            min_y = min(int(y), min_y)
        yield img_name, [min_x, min_y, max_x, max_y], landmarks


