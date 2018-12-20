#SIZE 512*512

#Part 1 : Building a CNN


#import Keras packages
import keras as k
from keras.models import Sequential

from keras.layers import Convolution2D

from keras.layers import MaxPooling2D

from keras.layers import Flatten

from keras.layers import Dense

from keras.layers import Dropout

import numpy as np

import scipy.ndimage
#%matplotlib inline

#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg


from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


# Initializing the CNN


np.random.seed(1337)
classifier = Sequential()
classifier.add(Convolution2D(256, 3, 3, input_shape = (512, 512, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Convolution2D(128, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Convolution2D(64, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dropout(p = 0.5))
classifier.add(Flatten())

#output layer
classifier.add(Dense(output_dim = 2, activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier.summary()


#Part 2 - fitting the data set

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory('train1',target_size=(512, 512),batch_size=20,class_mode='categorical')
test_set = test_datagen.flow_from_directory('val1',target_size=(512, 512),batch_size=20,class_mode='categorical')


#START THE TRAINING
history= classifier.fit_generator(training_set,steps_per_epoch=50,epochs=50,validation_data=test_set,validation_steps=5,verbose=2)

#CALLBACK IF NO IMPROVMENT IN EPOCHS
k.callbacks.EarlyStopping(monitor='val_loss', min_delta=1, patience=0, verbose=0, mode='auto', baseline=None, restore_best_weights=False)

classifier.save_weights('keras_retina_trained_model_weights_1000.h5')

print('Saved trained model as %s ' % 'keras_retina_trained_model_weights_1000.h5')

## END OF MOHIT SIR SRC