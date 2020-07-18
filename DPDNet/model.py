"""
This module has the DNN architecture design

Copyright (c) 2020-present, Abdullah Abuolaim
This source code is licensed under the license found in the LICENSE file in
the root directory of this source tree.

Note: this code is the implementation of the "Defocus Deblurring Using Dual-
Pixel Data" paper accepted to ECCV 2020. Link to GitHub repository:
https://github.com/Abdullah-Abuolaim/defocus-deblurring-dual-pixel

Email: abuolaim@eecs.yorku.ca
"""

from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import *
from keras import backend
from config import *

def unet(_in_data):
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same',
                   kernel_initializer = 'he_normal')(_in_data)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same',
                   kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same',
                   kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same',
                   kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same',
                   kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same',
                   kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same',
                   kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same',
                   kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(dropout_rate)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same',
                   kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same',
                   kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(dropout_rate)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same',
            kernel_initializer = 'he_normal')(UpSampling2D(size= (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same',
                   kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same',
                   kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same',
            kernel_initializer = 'he_normal')(UpSampling2D(size= (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same',
                   kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same',
                   kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same',
            kernel_initializer = 'he_normal')(UpSampling2D(size= (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same',
                   kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same',
                   kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same',
            kernel_initializer = 'he_normal')(UpSampling2D(size= (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same',
                   kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same',
                   kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(3, 3, activation = 'relu', padding = 'same',
                   kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(nb_ch, 1, activation = 'sigmoid')(conv9)

    return conv10

# m = unet()
# m.summary()
