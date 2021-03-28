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


def squeeze_excite_block(input, ratio=16):
    ''' Create a squeeze-excite block
    Args:
        input: input tensor
        filters: number of output filters
        k: width factor
    Returns: a keras tensor
    '''
    init = input
    #channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    #filters = init._keras_shape[channel_axis]
    filters = init.shape[-1]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    if K.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)

    x = multiply([init, se])
    return x

def load_model ():
  #Create Model 17k Parameters
  input_model= keras.layers.Input((256,1,1))
  conv_skip= Conv2D(filters=64, kernel_size=(3,1), strides=1, padding= "same", activation= "relu")(input_model)


  #Resblock 1
  conv_1= Conv2D(filters=64, kernel_size=(3,1), strides=1, padding= "same", activation= "relu")(conv_skip)
  conv_1= BatchNormalization()(conv_1)
  conv_2= Conv2D(filters=64, kernel_size=(3,1), strides=1, padding= "same", activation= "relu")(conv_1)
  conv_2= BatchNormalization()(conv_2)
  conv_2= squeeze_excite_block(conv_2, ratio=16) # Channel Attention
  conv_3= Conv2D(filters=1, kernel_size=(5,1), strides=1, padding= "same", activation= "sigmoid")(conv_skip) # Spacial Attention
  conv_4= multiply([conv_2, conv_3])
  conv= add([conv_skip,conv_4])
  conv_skip= Conv2D(filters=16, kernel_size=(3,1), strides=1, padding= "same", activation= "relu")(conv)


  #Resblock 2
  conv_1= Conv2D(filters=16, kernel_size=(3,1), strides=1, padding= "same", activation= "relu")(conv_skip)
  conv_1= BatchNormalization()(conv_1)
  conv_2= Conv2D(filters=16, kernel_size=(3,1), strides=1, padding= "same", activation= "relu")(conv_1)
  conv_2= BatchNormalization()(conv_2)
  conv_2= squeeze_excite_block(conv_2, ratio=16) # Channel Attention
  conv_3= Conv2D(filters=1, kernel_size=(5,1), strides=1, padding= "same", activation= "sigmoid")(conv_skip) # Spacial Attention
  conv_4= multiply([conv_2, conv_3])
  conv_= add([conv_skip,conv_4])

  final= Conv2D(filters=1, kernel_size=(3,1), strides=1, padding= "same")(conv_)
  #final= BatchNormalization()(conv)
  #final= add([input_model,conv])

  #final = Lambda(lambda x: (x * sd) + mean, output_shape=(256, 1, 1))(final) 

  model= keras.models.Model(inputs= input_model, outputs= final)
  opt = keras.optimizers.Adam(
      learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
      name='Adam')

  def metrics(y_true, y_pred):
      return 1 - K.mean(K.square(y_pred - y_true), axis=-1)

  model.compile(loss="mean_squared_error", optimizer=opt, metrics= metrics)

  model.summary()
  return model
#plot_model(model, "/content/drive/MyDrive/Mostafa/Callback/CNN_30k_With_batch_Final_without_add3.pdf")



