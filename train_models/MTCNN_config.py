#coding:utf-8

from easydict import EasyDict as edict

config = edict()

config.BATCH_SIZE = 384
config.CLS_OHEM = True
config.CLS_OHEM_RATIO = 0.7
config.BBOX_OHEM = False
config.BBOX_OHEM_RATIO = 0.7

config.EPS = 1e-14
config.LR_EPOCH = [6,14,20]

config.PNET_IMAGE_SIZE = 12
config.RNET_IMAGE_SIZE = 24
config.ONET_IMAGE_SIZE = 48

config.LANDMARK_SIZE = 5
