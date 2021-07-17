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
        'GTSRB/Final_training/Images',
        target_size=(32, 32),
        batch_size=128, #originally 16, then 64
        class_mode='categorical')
    
    validation_generator = test_datagen.flow_from_directory(
        'GTSRB/Test_Arranged',
        target_size=(32,32),
        batch_size=128, #originally 16, then 64
        class_mode='categorical')
    return train_generator, validation_generator

def train_model():
    
    train_generator, validation_generator = make_datagens()
    model = make_model()
    
    # checkpoint
    filepath="weights-improvement-{epocj:02d}-{val_accuracy:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=False, mode='max')
    callbacks_list = [checkpoint]
    
    model.fit_generator(
        train_generator,
        steps_per_epoch=256,
        epochs=256,
        callbacks=callbacks_list,
        validation_data=validation_generator,
        validation_steps=80)
    
    Y_pred = model.predict_generator(validation_generator, steps_per_epoch)
    y_pred = np.argmax(Y_pred, axis=1)
    print('Confustion Matrix')
    print(confusion_matrix(validation_generator.class, y_pred))
    print('Classification Report')
    target_names=['1', '2', '3']
    print(classification_report(validation_generator.classes, y_pred, target_names=target_names))
    
train_model()
