##In[]
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Activation, Dropout, Conv2D, MaxPooling2D
import paths
from os import listdir
from time import time, asctime, struct_time

## In[]
start = time()

BATCH_SIZE = 32
IMG_HEIGHT = 63
IMG_WIDTH = 64
IMG_CHANNELS = 1
img_gen = ImageDataGenerator()
# img_gen = ImageDataGenerator(rescale=1./255)
data_path = paths.join(paths.getDBPath(), 'ETL1PNG')

args = {'batch_size':BATCH_SIZE, 
        'shuffle':True,
        'target_size':(IMG_HEIGHT, IMG_WIDTH),
        'color_mode':'grayscale',
        'class_mode':'categorical' #binary for binary encoded labels, categorical for hot-encoded and sparse for integer labels
        }
train_data_gen = img_gen.flow_from_directory(directory=paths.join(data_path, 'train'), **args)
test_data_gen = img_gen.flow_from_directory(directory=paths.join(data_path, 'test'),**args)
                                            # batch_size=BATCH_SIZE, 
                                            # shuffle=True,
                                            # target_size=(IMG_HEIGHT, IMG_WIGHT),
                                            # color_mode='grayscale',
                                            # class_mode='sparse')#change to binary for binary encoded labels


print('Took: ',time() - start, 'seconds')

# y_train = tf.convert_to_tensor(y_train)
# x_train = tf.reshape(x_train, [len(x_train), 63,64,1])
model = Sequential()
model.add(Conv2D(32,(5,5), padding='same', activation='relu', 
                                 input_shape=(IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS)))
#model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (5, 5), padding='same', activation='relu'))
#model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='softmax'))
model.add(Dense(1))


#sparse_categorical_crossentropy results in the model only allowing the number of labels input into Dense -1.
#one-hot encoding is required with categorical_crossentropy
#https://stackoverflow.com/a/59148543
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

time
start = asctime(time())

model.fit(train_data_gen,
          epochs=1, #TODO change this back to higher numbers to try achieving higher accuracy
          verbose=1)
end=asctime(time())
print(f'Start time: {ascitime(start)}, End time: {ascitime(end)}, took {asctime(end-start)}')
# model.fit(x=x_train, y=y_train, epochs=1, steps_per_epoch=1, verbose=1)