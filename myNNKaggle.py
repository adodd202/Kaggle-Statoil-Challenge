# Andrew Dodd
# January 12th, 2018

# Summary: A simple neural network to assess the validity of this approach. 

############################################################
################  IMPORTS AND DATA CLEANING  ###############
############################################################

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import keras
from matplotlib import pyplot
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, Activation
from keras.layers import GlobalMaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from keras.models import Model
from keras import initializers
from keras.optimizers import Adam
from keras.optimizers import rmsprop
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from sklearn.model_selection import train_test_split


train = pd.read_json('../train.json') #online is '../train.json'
#train = pd.read_json('/Users/adodd202/Documents/GitHub/Statoil_Data/train.json')
y_train=train['is_iceberg']
test = pd.read_json('../test.json')
#test = pd.read_json('/Users/adodd202/Documents/GitHub/Statoil_Data/test.json') #online is '../test.json'


###### Deal with incident angle train and test data ################
train['inc_angle']=pd.to_numeric(train['inc_angle'], errors='coerce')
train['inc_angle']=train['inc_angle'].fillna(method='pad')  #We have only 133 NAs.
x_angle=train['inc_angle']

test['inc_angle']=pd.to_numeric(test['inc_angle'], errors='coerce')
x_test_angle=test['inc_angle']

############### Put image train data in bands ######################

#Generate the training data
x_band_1=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_1"]])
x_band_2=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_2"]])
x_band_3=(x_band_1+x_band_2)/2
#X_band_3=np.array([np.full((75, 75), angel).astype(np.float32) for angel in train["inc_angle"]])
x_train = np.concatenate([x_band_1[:, :, :, np.newaxis]
                          , x_band_2[:, :, :, np.newaxis]
                         , x_band_3[:, :, :, np.newaxis]], axis=-1)



x_band_test_1=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_1"]])
x_band_test_2=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_2"]])
x_band_test_3=(x_band_test_1+x_band_test_2)/2
#X_band_test_3=np.array([np.full((75, 75), angel).astype(np.float32) for angel in test["inc_angle"]])
x_test = np.concatenate([x_band_test_1[:, :, :, np.newaxis]
                          , x_band_test_2[:, :, :, np.newaxis]
                         , x_band_test_3[:, :, :, np.newaxis]], axis=-1)

# Train: x_train, y_train, Submission/Test: x_test
# Shape: 

x_train, x_val, y_train, y_val = train_test_split(
								   x_train, y_train, test_size=0.3, random_state=42)

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

print ("x_train", x_train.shape)
print ("y_train", y_train.shape)
print ("x_val", x_val.shape)
print ("y_val", y_val.shape)

print ("x_test", x_test.shape)




############################################################
################  DATA AUGMENTATION  #######################
############################################################




############################################################
################  DEFINE THE MODEL  ########################
############################################################

batch_size = 32
num_classes = 2
model = Sequential()
model.add(Conv2D(batch_size, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(75,75,3)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))


############################################################
################  TRAIN AND PREDICT  #######################
############################################################

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(), # ADAM or SGD
              metrics=['accuracy'])

epochs = 10
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_val, y_val))
score = model.evaluate(x_val, y_val, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])