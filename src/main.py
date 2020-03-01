import imageio
import paths
from os import listdir

import time
import dataconverter
dc = dataconverter.DataConverter()
start = time.time()
data_path = paths.join(paths.getDBPath(), 'ETL1PNG')
tmp = listdir(paths.join(data_path, 'train'))
data_paths = []
# data_
for folder in tmp:
    data_paths += listdir(paths.join(data_path, 'train', folder))
del tmp
y_train = []
x_train = []
for file in data_paths[:100]:
    label = file[5:-4]
    # label = file[len(dc.charset)+5:-4]
    x_train.append([x/255. for x in imageio.imread(paths.join(data_path, 'train', label, file))])
    y_train.append(int(label)) #didn't have any effect, maybe go back to strings

print('Took: ',time.time() - start, 'seconds')

import tensorflow as tf
y_train = tf.convert_to_tensor(y_train)
x_train = tf.reshape(x_train, [len(x_train), 63,64,1])
# x_train = tf.reshape(x_train, [len(x_train), 64*63])
# x = tf.convert_to_tensor(x_train, dtype=tf.float32)
# print(x_train.shape)
def printSome(dataset, number: int):
    c = 0
    for example in dataset:
        c+=1
        print('x:', example[0].numpy()[:10], ' y:', example[1].numpy())
        if c >= number:
            break

#
"""
dc.load()
#ds stands for dataset
labels = dc.labels
# features = [list(_) for _ in dc.features]
features = dc.features

ds_joint = tf.data.Dataset.from_tensor_slices((features, labels)) #x: features, y: labels

######################\"""
print('ds_join: ',type(ds_joint))
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
print('mnist train: ',type(x_train))
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])
predictions = model(x_train[:1]).numpy()
# print(predictions)
######################\"""

tf.random.set_seed(182)
ds = ds_joint.shuffle(buffer_size=1000, reshuffle_each_iteration=False)
# printSome(ds, 4)

train_size, ds_size = 0.7, len(labels)
take_size_train = int(train_size * ds_size)
take_size_test = ds_size - take_size_train
#named datasets can apparently be loaded with tf.Datasets.load() and split()
ds_train = ds.take(take_size_train), ds.skip(take_size_train).take(take_size_test)

# ds_train = ds_train.map(lambda x: (x['features'], x['label']))
# ds_test = ds_test.map(lambda x: (x['features'], x['label']))
"""

# ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
######################\
# """
# print(list(ds_train.as_numpy_iterator()))
# print('ds_train: ', type(ds_train))
print('x_train: ', type(x_train))
print('y_train: ', type(y_train))

######################\

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Activation, Dropout, Conv2D, MaxPooling2D
model = Sequential()
model.add(Conv2D(32,(5,5),activation='relu', 
                                 input_shape=(63,64,1)))
#model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu'))
#model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(200, activation='softmax'))


model.compile(loss='sparse_categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])


model.fit(x_train, y_train,
          batch_size=100,
          epochs=5,
          verbose=1)

"""
model = tf.keras.Sequential()
# model.add(tf.keras.layers.Conv2D(
#     filters=32, kernel_size=(5, 5),
#     strides=(1, 1), padding='same',
#     data_format='channels_last',
#     name='conv_1', activation='relu'))
model.add(tf.keras.layers.Dense(16, activation='sigmoid', input_shape=(1,)))
model.add(tf.keras.layers.Dense(3, activation='softmax'))
# model.add(tf.keras.layers.Flatten())
print(model.compute_output_shape(
    input_shape=(None, 1)
))

model.compile(optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

model.summary()

model.fit(ds_train, epochs=1, steps_per_epoch=1, verbose=1)
"""
model.fit(x=x_train, y=y_train, epochs=1, steps_per_epoch=1, verbose=1)