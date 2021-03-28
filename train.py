import numpy as np
#from tensorflow import keras
#from tensorflow.keras import layers
import os
from contextlib import suppress
import matplotlib.pyplot as plt

import numpy as np
import os
import warnings
from zipfile import ZipFile

from skimage.io import imread, imsave

from keras import applications
from keras import optimizers
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Flatten
from keras.utils import to_categorical, plot_model
from keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import accuracy_score, classification_report
import keras
from keras.layers import Dense, GlobalAveragePooling2D,Reshape,Permute,multiply,GlobalMaxPooling2D, Conv2D, BatchNormalization, add, Lambda
from keras import backend as K
K.set_image_data_format('channels_last')
from keras.models import load_model
from glob import glob
import cv2
import numpy as np
import keras
import model



os.chdir("./data/dataset")
!pwd


############################################ dataset
# Model / data parameters
# num_classes = 10
input_shape = (256, 1, 1)

# the data, split between train and test sets

x_train = np.load('x_train.npy')
x_train = x_train[...,np.newaxis, np.newaxis]

y_train_noise = np.load('y_train.npy')
y_train_noise = y_train_noise[...,np.newaxis, np.newaxis]

x_valid = np.load('x_valid.npy')
x_valid = x_valid[...,np.newaxis, np.newaxis]

y_valid = np.load('y_valid.npy')
y_valid = y_valid[...,np.newaxis, np.newaxis]

print('x_train shape: ', x_train.shape)
print('x_valid shape: ', y_valid.shape)


print("min: ", np.min(x_train))
print("max: ", np.max(x_train))

print("min: ", np.min(x_train))
print("max: ", np.max(x_train))

############################################ load model

model= model.load_model()

############################################ metrics

opt = keras.optimizers.Adam(
    learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
    name='Adam')

def metrics(y_true, y_pred):
    return 1 - K.mean(K.square(y_pred - y_true), axis=-1)

model.compile(loss="mean_squared_error", optimizer=opt, metrics= metrics)


############################################ checkpoint

callback_model= keras.callbacks.ModelCheckpoint("./checkpoint/model.h5", monitor= "val_loss", verbose= 1, save_best_only= True, mode= "min")
callback_CSV= keras.callbacks.CSVLogger("./checkpointmodel.log", append=True)

############################################ training

batch_size = 64
epochs = 100
history= model.fit(x_train, y_train_noise, batch_size= batch_size, epochs= epochs, validation_data=(x_valid, y_valid), callbacks=[callback_model, callback_CSV], shuffle= True)








