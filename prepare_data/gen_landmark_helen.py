from load_dataset import  load_helen_annotation, load_muct_annotation
from BBox_utils import BBox, IoU
from Landmark_utils import rotate
import sys
sys.path.append('../')
from train_models.MTCNN_config import config
import cv2
import os
import random
import numpy as np


LANDMARK_LEN = config.LANDMARK_SIZE * 2

NETS_IMG_SIZE = config.IMAGE_SIZES
RANDOM_SHIFT_TIMES = 50

IOU_POS = 0.65
IOU_NEG = 0.3

BASE_LANDMARK_DIR = 'train_%s_landmark'
BASE_LANDMARK_FILE = 'landmark_%s.txt'
BASE_IMG_DIR = 'train_%s_landmark'

def generate_data(anno_dir, image_dir, net, load_annotation):
    size = NETS_IMG_SIZE[net]

    f_imgs = []
    f_landmarks = []
    img_count= 0
    for image_name, box_corner, landmarks_gt in load_annotation(anno_dir):
        if len(landmarks_gt) != config.LANDMARK_SIZE:
            print('invalid landmakr size %d file %s' % (len(landmarks_gt), image_name))
            continue
        image_path = os.path.join(image_dir, "%s.jpg" % image_name)
        img = cv2.imread(image_path)
        if img is None:
            continue
        # print('transform image %s' % image_path + 'landmarks len %d' % len(landmarks_gt))
        img_h, img_w, img_c = img.shape
        bbox = BBox(box_corner)
        gt_box = np.array([bbox.left, bbox.top, bbox.right, bbox.bottom])
        face = img[bbox.top:bbox.bottom+1, bbox.left:bbox.right+1]
        try:
            face = cv2.resize(face, (size, size))
        except Exception as ex:
            print("canno resize file %s" % image_path)

        # normalized landmark in (0, 1)
        f_landmark = np.zeros((len(landmarks_gt), 2))
        for i, lm in enumerate(landmarks_gt):
            rv = ((lm[0] - gt_box[0]) / (gt_box[2] - gt_box[0]),
                  (lm[1] - gt_box[1]) / (gt_box[3] - gt_box[1]))
            f_landmark[i] = rv

        f_imgs.append(face)
        f_landmarks.append(f_landmark.reshape(np.prod(f_landmark.shape)))
        img_count += 1
        if img_count % 100 == 0:
            print("%d images done" % img_count)
        x1, y1, x2, y2 = gt_box
        gt_w = x2 - x1 + 1
        gt_h = y2 - y1 + 1

        if max(gt_w, gt_h) < 40 or x1 < 0 or y1 < 0:
            continue

        for i in range(RANDOM_SHIFT_TIMES):
            bbox_size = np.random.randint(int(min(gt_w, gt_h) * 0.8),
                                          np.ceil(1.25 * max(gt_w, gt_h)))
            delta_x = np.random.randint(-gt_w * 0.2, gt_w * 0.2)
            delta_y = np.random.randint(-gt_h * 0.2, gt_h * 0.2)

            nx1 = int(max(x1 + gt_w / 2 - bbox_size / 2 + delta_x, 0))
            ny1 = int(max(y1 + gt_h / 2 - bbox_size / 2 + delta_y, 0))

            nx2 = nx1 + bbox_size
            ny2 = ny1 + bbox_size
            if nx2 > img_w or ny2 > img_h:
                continue
            crop_box = np.array([nx1, ny1, nx2, ny2])
            # print([nx1, ny1, nx2, ny2])
            cropped_img = img[ny1:ny2+1, nx1:nx2+1, :]
            resized_img = cv2.resize(cropped_img, (size, size))
            #cal iou
            iou = IoU(crop_box, np.expand_dims(gt_box,0))

            if iou > IOU_POS:
                f_landmark = np.zeros((len(landmarks_gt), 2))
                for j, lm in enumerate(landmarks_gt):
                    rv = ((lm[0] - nx1) / bbox_size, (lm[1] - ny1) / bbox_size)
                    f_landmark[j] = rv

                shifted_landmark = f_landmark.copy()
                f_landmarks.append(f_landmark)
                f_imgs.append(resized_img)
                bbox = BBox([nx1, ny1, nx2, ny2])

                #print('shifted landmark shape %s' % str(shifted_landmark.shape))

                # rotate image and landmark
                rotate_alpha = random.choice([-1, 1]) * np.random.randint(5, 10)
                rotated_face, rotated_landmark = rotate(img, bbox,
                                                        bbox.reprojectLandmark(shifted_landmark),
                                                        rotate_alpha)
                rotated_landmark = bbox.projectLandmark(rotated_landmark)
                if np.sum(np.where(rotated_landmark < 0, 1, 0)) > 0:
                    continue
                if np.sum(np.where(rotated_landmark > 1, 1, 0)) > 0:
                    continue
                # print('rotated_landmark %s' % str(rotated_landmark))
                rotated_cropped_img = cv2.resize(rotated_face, (size, size))
                f_imgs.append(rotated_cropped_img)
                f_landmarks.append(rotated_landmark)

    np_imgs, np_landmarks = np.asarray(f_imgs), np.asarray(f_landmarks)
    print('np_imgs shape %s, np_landmarks shape %s' % (np_imgs.shape, np_landmarks.shape))
    # print(np_landmarks)

    output_dir = net
    landmark_dir = os.path.join(output_dir, BASE_LANDMARK_DIR % net)
    landmark_file = os.path.join(output_dir, BASE_LANDMARK_FILE % net)
    img_dir = os.path.join(output_dir, BASE_IMG_DIR % net)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if not os.path.exists(landmark_dir):
        os.mkdir(landmark_dir)
    if not os.path.exists(img_dir):
        os.mkdir(img_dir)
    print('writing to landmark %s' % landmark_file)
    with open(landmark_file, 'w') as f:
        img_count = 0
        for i, img in enumerate(np_imgs):
            if np.sum(np.where(np_landmarks[i] <= 0, 1, 0)) > 0:
                continue
            if np.sum(np.where(np_landmarks[i] >= 1, 1, 0)) > 0:
                continue
            img_count += 1
            img_file_path = os.path.join(img_dir, "%d.jpg" % (img_count))
            cv2.imwrite(img_file_path, img)
            flattened_landmark = map(str, list(np_landmarks[i].reshape(np.prod(np_landmarks[i].shape))))
            f.write(" ".join([img_file_path, "-2"] + flattened_landmark))
            f.write("\n")
        print('total img %d' % img_count)

if __name__ == '__main__':
    net = sys.argv[1]
    dataset = sys.argv[2]
    if dataset == 'helen':
        generate_data('helen/annotation','helen/image', net, load_helen_annotation)
    elif dataset == 'muct':
        generate_data('muct/muct76-opencv.csv', 'muct/image', net, load_muct_annotation)

