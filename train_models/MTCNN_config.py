#coding:utf-8

from easydict import EasyDict as edict

config = edict()

config.BATCH_SIZE = 192
config.CLS_OHEM = True
config.CLS_OHEM_RATIO = 0.7
config.BBOX_OHEM = False
config.BBOX_OHEM_RATIO = 0.7

config.EPS = 1e-14
config.LR_EPOCH = [6,14,20]

config.PNET_IMAGE_SIZE = 12
config.RNET_IMAGE_SIZE = 24
config.ONET_IMAGE_SIZE = 64
config.IMAGE_SIZES = {
    "PNet": 12,
    "RNet": 24,
    "ONet": 64
}

#config.LANDMARK_SIZE = 5
#config.LANDMARK_SIZE = 194  # helen
config.LANDMARK_SIZE = 76  # muct
