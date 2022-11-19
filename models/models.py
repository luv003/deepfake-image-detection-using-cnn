from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    BatchNormalization, SeparableConv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense
)
def create_model():

     model = Sequential()

     input_shape = (224,224,3)
     activation = 'relu'
     padding = 'same'
     droprate = 0.1 
     epsilon=0.001

     model = Sequential()
     model.add(BatchNormalization(input_shape=input_shape))
     model.add(Conv2D(filters=16, kernel_size=3, activation=activation, padding=padding))
     model.add(MaxPooling2D(pool_size=2))
     model.add(BatchNormalization(epsilon=epsilon))


     model.add(Conv2D(filters=32, kernel_size=3, activation=activation, padding=padding))
     model.add(MaxPooling2D(pool_size=2))
     model.add(BatchNormalization(epsilon=epsilon))
     model.add(Dropout(droprate))

     model.add(Conv2D(filters=64, kernel_size=3, activation=activation, padding=padding))
     model.add(MaxPooling2D(pool_size=2))
     model.add(BatchNormalization(epsilon=epsilon))
     model.add(Dropout(droprate))

     model.add(Conv2D(filters=128, kernel_size=3, activation=activation, padding=padding))
     model.add(MaxPooling2D(pool_size=2))
     model.add(BatchNormalization(epsilon=epsilon))
     model.add(Dropout(droprate))

     model.add(Conv2D(filters=256, kernel_size=3, activation=activation, padding=padding))
     model.add(MaxPooling2D(pool_size=2))
     model.add(BatchNormalization(epsilon=epsilon))
     model.add(Dropout(droprate))

     model.add(GlobalAveragePooling2D())
     model.add(Flatten())
     model.add(Dropout(droprate))
     model.add(Dense(1, activation='sigmoid'))

     model.compile(loss='binary_crossentropy',optimizer=keras.optimizers.Adam(0.001), metrics=['accuracy'])
     
     return model