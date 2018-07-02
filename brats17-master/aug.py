from __future__ import absolute_import, print_function

import numpy as np
import random
from scipy import ndimage
import time
import os
import sys
import nibabel
import tensorflow as tf
import tensorlayer as tl
#from tensorflow.contrib.data import iterator
from tensorflow.contrib.layers.python.layers import regularizers
#from niftynet.layer.loss_segmentation import LossFunction
from util.data_loader import *
from util.train_test_func import *
from util.parse_config import parse_config
import cv2
#from util.MSNet import MSNet


def distort_imgs(data):
    """ data augumentation """
    x1, x2, x3, x4, y = data
    # x1, x2, x3, x4, y = tl.prepro.flip_axis_multi([x1, x2, x3, x4, y],  # previous without this, hard-dice=83.7
    #                         axis=0, is_random=True) # up down
    x1, x2, x3, x4, y = tl.prepro.flip_axis_multi([x1, x2, x3, x4, y], axis=1, is_random=False) # left right
    #x1, x2, x3, x4, y = tl.prepro.elastic_transform_multi([x1, x2, x3, x4, y],
                            #alpha=720, sigma=24, is_random=True)
    #x1, x2, x3, x4, y = tl.prepro.rotation_multi([x1, x2, x3, x4, y], rg=20,
                            #is_random=True, fill_mode='constant') # nearest, constant
    #x1, x2, x3, x4, y = tl.prepro.shift_multi([x1, x2, x3, x4, y], wrg=0.10,
                            #hrg=0.10, is_random=True, fill_mode='constant')
    #x1, x2, x3, x4, y = tl.prepro.shear_multi([x1, x2, x3, x4, y], 0.05,
                            #is_random=True, fill_mode='constant')
    #x1, x2, x3, x4, y = tl.prepro.zoom_multi([x1, x2, x3, x4, y],
                            #zoom_range=[0.9, 1.1], is_random=True,
                            #fill_mode='constant')
    return x1, x2, x3, x4, y

def vis_imgs(X, y, path):
    """ show one slice """
    if y.ndim == 2:
        y = y[:,:,np.newaxis]
    assert X.ndim == 3
    tl.vis.save_images(np.asarray([X[:,:,0,np.newaxis],
        X[:,:,1,np.newaxis], X[:,:,2,np.newaxis],
        X[:,:,3,np.newaxis], y]), size=(1, 5),
        image_path=path)

def vis_imgs2(X, y_, y, path):
    """ show one slice with target """
    if y.ndim == 2:
        y = y[:,:,np.newaxis]
    if y_.ndim == 2:
        y_ = y_[:,:,np.newaxis]
    assert X.ndim == 3
    tl.vis.save_images(np.asarray([X[:,:,0,np.newaxis],
        X[:,:,1,np.newaxis], X[:,:,2,np.newaxis],
        X[:,:,3,np.newaxis], y_, y]), size=(1, 6),
        image_path=path)

config_file = str(sys.argv[1])
assert(os.path.isfile(config_file))
print(os.path.isfile(config_file))
config = parse_config(config_file)
config_data  = config['data']
batch_size  = config_data.get('batch_size', 5)
dataloader = DataLoader(config_data)
print("DEBUG: Started Loading Data")
dataloader.load_data()
print("DEBUG: Finished Loading Data")
#X, _, Y = dataloader.get_subimage_batch()
aug_pair = dataloader.data
#print(aug_pair[0][0].shape)
#print("length",len(aug_pair[0])) #4 -> number of modalities
patient_names = dataloader.patient_names
volume,label = dataloader.g_load() #loaded one volume with its label
print("albels",label.shape)
x_flair_test = volume[0,70,:,:,np.newaxis]
print("X_Flair",x_flair_test.shape)
x_t1_test = volume[1,70,:,:,np.newaxis]
x_t1ce_test= volume[2,70,:,:,np.newaxis]
x_t2_test = volume[3,70,:,:,np.newaxis]
y = label[70,:,:,np.newaxis]
print("YYY",y.shape)
X_r = np.concatenate((x_flair_test, x_t1_test, x_t1ce_test, x_t2_test), axis=2)
vis_imgs(X_r,y, 'samples/all/test.png')
print(volume.shape)

x_flair_r,x_t1_r,x_t1ce_r,x_t2_r,la = distort_imgs([x_flair_test,x_t1_test,x_t1ce_test,x_t2_test,y])
x_dis = np.concatenate((x_flair_r, x_t1_r, x_t1ce_r, x_t2_r), axis=2)
vis_imgs(x_dis,la, 'samples/all/test_result.png')

patient_frames_flair = []
patient_frames_t1 = []
patient_frames_t1ce = []
patient_frames_t2 = []
for frame in range(155):
    x_flair, x_t1, x_t1ce, x_t2, l = distort_imgs([volume[0,frame,:,:,np.newaxis], volume[1,frame,:,:,np.newaxis], volume[2,frame,:,:,np.newaxis], volume[3,frame,:,:,np.newaxis], label[frame,:,:,np.newaxis]])
    patient_frames_flair.append(x_flair)
    patient_frames_t1.append(x_t1)
    patient_frames_t1ce.append(x_t1ce)
    patient_frames_t2.append(x_t2)

result_flair = np.dstack(patient_frames_flair)
result_t1 = np.dstack(patient_frames_t1)
print("Testing",result_t1.shape)
result = nibabel.Nifti1Image(result_t1, affine=np.eye(4))
nibabel.save(result, 'samples/all/_train_result_nii_test.nii.gz')
#X = aug_pair['images']
#Y = aug_pair['labels']
#print(X.shape)
#print (Y.shape)
#X_test = X[0,17,:,:,:]
#y = Y[0,:,:,:,0]
#print(X_test.shape)
#x_flair_test = X_test[:,:,0,np.newaxis]
#x_t1_test = X_test[:,:,1,np.newaxis]
#x_t1ce_test= X_test[:,:,2,np.newaxis]
#x_t2_test = X_test[:,:,3,np.newaxis]
#X_r = np.concatenate((x_flair_test, x_t1_test, x_t1ce_test, x_t2_test), axis=2)
#vis_imgs(X_r,y, 'samples/all/test.png')