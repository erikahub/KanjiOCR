# import imageio
# import paths
# from os import listdir

# data_path = paths.join(paths.getDBPath(), 'ETL1PNG')
# tmp = listdir(paths.join(data_path, 'train'))
# data_paths = []
# data_
# for folder in tmp:
#     data_paths += listdir(paths.join(data_path, 'train', folder))
# del tmp
# img = imageio.imread(paths.join(data_path, 'train', '0', '0001_0.png'))

import tensorflow as tf
import dataconverter

def printSome(dataset, number: int):
    c = 0
    for example in dataset:
        c+=1
        print('x:', example[0].numpy()[:10], ' y:', example[1].numpy())
        if c >= number:
            break

dc = dataconverter.DataConverter()
dc.load()
#ds stands for dataset

# for i in range(len(dc.labels)):
    # dc.labels[i]+=f'_{i}'
# x = tf.Variable(dc.features(), tf.int8)
# , dc.labels
ds_joint = tf.data.Dataset.from_tensor_slices((dc.features(), dc.labels)) #x: features, y: labels

tf.random.set_seed(182)
ds = ds_joint.shuffle(buffer_size=1000, reshuffle_each_iteration=False)
# printSome(ds, 4)

train_size, ds_size = 0.7, len(dc.labels)
take_size_train = int(train_size * ds_size)
take_size_test = ds_size - take_size_train
#named datasets can apparently be loaded with tf.Datasets.load() and split()
ds_train = ds.take(take_size_train), ds.skip(take_size_train).take(take_size_test)

ds_train = ds_train.map(lambda x: (x['features'], x['label']))
# ds_test = ds_test.map(lambda x: (x['features'], x['label']))

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(16, activation='sigmoid', input_shape=(63,64)))
model.add(tf.keras.layers.Dense(3, activation='softmax'))

# model.summary()

model.compile(optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

model.fit(ds_train, epochs=1, steps_per_epoch=1, verbose=1)