import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import RMSprop

def make_model():
    rmsp = RMSprop(1r=0.0001)
    
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3)))
    model.add(Activation('relu')) #used to be 'relu'
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten()) # this converts our 3D feature maps to 1D featue vectors
    model.add(Dense(256)) # 64
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(43)) #this dense size cannot change
    model.add(Activation('softmax'))
    
    model.compile(loss='categorical_crossentropy',
                  optimizer=rmsp,
                  metrics=['accuracy'])
    
    return model
  
def make_datagens():
  
  train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=False)
  
  test_datagen = ImageDataGenerator(rescale=1.255)
  
  train_generator = train_datagen.flow_from_directory(
    
