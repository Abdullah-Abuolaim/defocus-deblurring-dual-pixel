"""
This is the configuration module has all the gobal variables and basic
libraries to be shared with other modules in the same project.

Copyright (c) 2020-present, Abdullah Abuolaim
This source code is licensed under the license found in the LICENSE file in
the root directory of this source tree.

Note: this code is the implementation of the "Defocus Deblurring Using Dual-
Pixel Data" paper accepted to ECCV 2020. Link to GitHub repository:
https://github.com/Abdullah-Abuolaim/defocus-deblurring-dual-pixel

Email: abuolaim@eecs.yorku.ca
"""

import numpy as np
import os
import math
import cv2
import random
from skimage import measure
from sklearn.metrics import mean_absolute_error

# results and model name
res_model_name='l5_s512_f0.7_d0.4'
op_phase='test'

# image mini-batch size
img_mini_b = 5

#########################################################################
# READ & WRITE DATA PATHS									            #
#########################################################################
# run on server or local machine
server=False

sub_folder=['source/','target/']

if op_phase=='train':
    dataset_name='_canon_patch'
    # resize flag to resize input and output images
    resize_flag=False
elif op_phase=='test':
    dataset_name='_canon'
    # resize flag to resize input and output images
    resize_flag=True
else:
    raise NotImplementedError

# path to save model
if server:
    path_save_model='/local/ssd/abuolaim/defocus_deblurring_dp_'+res_model_name+'.hdf5'
else:
    path_save_model='./ModelCheckpoints/defocus_deblurring_dp_'+res_model_name+'.hdf5'
    

# paths to read data
path_read_train = './dd_dp_dataset'+dataset_name+'/'
path_read_val_test = './dd_dp_dataset'+dataset_name+'/'

# path to write results
path_write='./results/res_'+res_model_name+'_dd_dp'+dataset_name+'/'

#########################################################################
# NUMBER OF IMAGES IN THE TRAINING, VALIDATION, AND TESTING SETS	    #
#########################################################################
if op_phase=='train':
    total_nb_train = len([path_read_train + 'train_c/' + sub_folder[0] + f for f
                    in os.listdir(path_read_train + 'train_c/' + sub_folder[0])
                    if f.endswith(('.jpg','.JPG', '.png', '.PNG', '.TIF'))])
    
    total_nb_val = len([path_read_val_test + 'val_c/' + sub_folder[0] + f for f
                    in os.listdir(path_read_val_test + 'val_c/' + sub_folder[0])
                    if f.endswith(('.jpg','.JPG', '.png', '.PNG', '.TIF'))])
    
    # number of training image batches
    nb_train = int(math.ceil(total_nb_train/img_mini_b))
    # number of validation image batches
    nb_val = int(math.ceil(total_nb_val/img_mini_b))
    
elif op_phase=='test':
    total_nb_test = len([path_read_val_test + 'test_c/' + sub_folder[0] + f for f
                    in os.listdir(path_read_val_test + 'test_c/' + sub_folder[0])
                    if f.endswith(('.jpg','.JPG', '.png', '.PNG', '.TIF'))])

#########################################################################
# MODEL PARAMETERS & TRAINING SETTINGS									#
#########################################################################

# input image size
img_w = 1680
img_h = 1120

# input patch size
patch_w=512
patch_h=512

# mean value pre-claculated
src_mean=0
trg_mean=0

# number of epochs
nb_epoch = 200

# number of input channels
nb_ch_all= 6
# number of output channels
nb_ch=3  # change conv9 in the model and the folowing variable

# color flag:"1" for 3-channel 8-bit image or "0" for 1-channel 8-bit grayscale
# or "-1" to read image as it including bit depth
color_flag=-1

bit_depth=16

norm_val=(2**bit_depth)-1

# after how many epochs you change learning rate
scheduling_rate=60

dropout_rate=0.4

# generate learning rate array
lr_=[]
lr_.append(2e-5) #initial learning rate
for i in range(int(nb_epoch/scheduling_rate)):
    lr_.append(lr_[i]*0.5)

train_set, val_set, test_set = [], [], []

size_set, portrait_orientation_set = [], []

mse_list, psnr_list, ssim_list, mae_list = [], [], [], []