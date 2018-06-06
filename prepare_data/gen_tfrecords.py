import os
import sys
import cv2
import tensorflow
import random

from tfrecord_utils import _process_image_withoutcoder, _convert_to_example_simple
from train_models.MTCNN_config import config

LANDMARK_LEN = config.LANDMARK_SIZE * 2
DATA_TYPES = ['pos', 'neg', 'part', 'landmark']

def _add_to_tfrecord(filename, image_record, tfrecord_writer):
    image_data, height, width = _process_image_withoutcoder(filename)
    example = _convert_to_example_simple(image_example, image_data)
    tfrecord_writer.write(example.SerializeToString())

def line_to_example(i, line):
    segs = line.strip().split(' ')
    data_example = {}
    data_example['filename'] = segs[0]
    data_example['label'] = int(segs[1])
    if len(segs) == 6:
        bbox = {}
        bbox['xmin'] = float(segs[2])
        bbox['ymin'] = float(segs[3])
        bbox['xmax'] = float(segs[4])
        bbox['ymax'] = float(segs[5])
        data_example['bbox'] = bbox
    elif len(segs) == LANDMARK_LEN + 2:
        data_example['landmark'] = segs[2:]
    else:
        print("unexpected line length of %d %s:%d" % (len(segs), dataset_path, i))
        return None
    return data_example

def load_dataset(filepath):
    dataset = []
    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            data_example = line_to_example(i, line)
            if data_example is not None:
                dataset.append(data_example)
    return dataset

def gen_tfrecord_file(dataset_path, tfrecord_path, shuffling=True):
    if tf.gfile.Exists(tfrecord_path):
        print('skip %s, already exists' % tfrecord_path)
        return
    if shuffling:
        tfrecord_filename += '_shuffle'
        random.shuffle(dataset)

    dataset = load_dataset(dataset_path)
    total_size = len(dataset)
    with tf.python_io.TFRecordWriter(tfrecord_filename) as tfrecord_writer:
        for i, data_example in enumerate(dataset):
            filename = data_example['filename']
            print('>> Converting image %d/%d %s' % (i+1, total_size, filename))
            _add_to_tfrecord(filename, data_example, tfrecord_writer)

def gen_pnet_tfrecord(dataset_dir, output_dir, net='PNet'):
    file_name = 'train_%s_landmark.txt' % net
    dataset_path = os.path.join(dataset_dir, file_name)
    tfrecord_path = '%s/train_%s_landmark.tfrecord' % (output_dir, net)
    gen_tfrecord_file(dataset_path, tfrecord_path)


def gen_rnet_tfrecord(dataset_dir, output_dir, net='RNet'):
    for data_type in DATA_TYPES:
        file_name = '%s_%s.txt' % net
        dataset_path = os.path.join(dataset_dir, file_name)
        tfrecord_path = '%s/%s_landmark.tfrecord' % (output_dir, data_type)
        gen_tfrecord_file(dataset_path, tfrecord_path)

def gen_onet_tfrecord(dataset_dir, output_dir, net='ONet'):
    for data_type in DATA_TYPES:
        file_name = '%s_%s.txt' % net
        dataset_path = os.path.join(dataset_dir, file_name)
        tfrecord_path = '%s/%s_landmark.tfrecord' % (output_dir, data_type)
        gen_tfrecord_file(dataset_path, tfrecord_path)

if __name__ == '__main__':
    net = sys.argv[1]
    output_dir = 'imglists/%s' % net
    if net == 'PNet':
        gen_pnet_tfrecord('.', output_dir, shuffling=True)
    elif net == 'RNet':
        gen_rnet_tfrecord('.', output_dir, shuffling=True)
    elif net == 'ONet':
        gen_onet_tfrecord('.', output_dir, shuffling=True)
    else:
        print('invalid net %s' % net)





