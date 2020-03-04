"""TODO WRITE THIS"""
##In[]
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Activation, Dropout, Conv2D, MaxPooling2D
import paths
from os import listdir
from time import time

## In[]
start = time()

BATCH_SIZE = 64
IMG_HEIGHT = 63
IMG_WIDTH = 64
IMG_CHANNELS = 1

#TODO look into this. rescaling is supposed to be better but seems to yield worse results?
img_gen = ImageDataGenerator()
# img_gen = ImageDataGenerator(rescale=1./255)
data_path = paths.join(paths.getDBPath(), 'ETL1PNG')

#list of arguments to be extracted for use in data generator objects
args = {'batch_size':BATCH_SIZE, 
        'shuffle':True,
        'target_size':(IMG_HEIGHT, IMG_WIDTH),
        'color_mode':'grayscale',
        'class_mode':'categorical' #binary for binary encoded labels, categorical for hot-encoded and sparse for integer labels
        }
train_data_gen = img_gen.flow_from_directory(directory=paths.join(data_path, 'train'), **args)
test_data_gen = img_gen.flow_from_directory(directory=paths.join(data_path, 'test'),**args)


print('Took: ',time() - start, 'seconds')

model = Sequential()
model.add(Conv2D(32,(5,5), padding='same', activation='relu', 
                                 input_shape=(IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS)))
# model.add(MaxPooling2D((3, 3)))
model.add(MaxPooling2D((8, 8)))
model.add(Conv2D(64, (5, 5), padding='same', activation='softmax'))
model.add(Flatten())
model.add(Dense(97, activation='softmax')) #this is supposed to be the number of labels.

#sparse_categorical_crossentropy results in the model only allowing the number of labels input into Dense -1.
#one-hot encoding is required with categorical_crossentropy
#https://stackoverflow.com/a/59148543
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

time
start = time()

fit_history = model.fit(train_data_gen,
          epochs=10, #TODO change this back to higher numbers to try achieving higher accuracy
          verbose=1)
end=time()
print(f'Training took {(end-start)//60:.0}min {int(end-start)%60}seconds')

test_loss, test_acc = model.evaluate(test_data_gen, 
                verbose=1)

print(f'Test loss: {test_loss}, test accuracy: {test_acc}')
# model.fit(x=x_train, y=y_train, epochs=1, steps_per_epoch=1, verbose=1)
#TODO save weighted models to compare them more easily. possibly write wrapping functions/classes