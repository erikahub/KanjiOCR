##In[]
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Activation, Dropout, Conv2D, MaxPooling2D
import paths
from os import listdir
import time

## In[]
start = time.time()

BATCH_SIZE = 32
IMG_HEIGHT = 63
IMG_WIGHT = 64
img_gen = ImageDataGenerator()
# img_gen = ImageDataGenerator(rescale=1./255)
data_path = paths.join(paths.getDBPath(), 'ETL1PNG')

args = {'batch_size':BATCH_SIZE, 
        'shuffle':True,
        'target_size':(IMG_HEIGHT, IMG_WIGHT),
        'color_mode':'grayscale',
        'class_mode':'sparse' #change to binary for binary encoded labels
        }
train_data_gen = img_gen.flow_from_directory(directory=paths.join(data_path, 'train'), **args)
test_data_gen = img_gen.flow_from_directory(directory=paths.join(data_path, 'test'),**args)
                                            # batch_size=BATCH_SIZE, 
                                            # shuffle=True,
                                            # target_size=(IMG_HEIGHT, IMG_WIGHT),
                                            # color_mode='grayscale',
                                            # class_mode='sparse')#change to binary for binary encoded labels

#TODO TODO TODO TODO TODO look into how to use the data_gen. Possible issue with the data shape (63,64,1)

y_train = []
x_train = []

print('Took: ',time.time() - start, 'seconds')

##In[]
y_train = tf.convert_to_tensor(y_train)
x_train = tf.reshape(x_train, [len(x_train), 63,64,1])

model = Sequential()
model.add(Conv2D(32,(5,5),activation='relu', 
                                 input_shape=(63,64,1)))
#model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu'))
#model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(300, activation='softmax'))


model.compile(loss='sparse_categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])


model.fit(x_train, y_train,
          batch_size=100,
          epochs=5,
          verbose=1)

# model.fit(x=x_train, y=y_train, epochs=1, steps_per_epoch=1, verbose=1)