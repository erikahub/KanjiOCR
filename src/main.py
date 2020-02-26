import matplotlib.pyplot as plt
# import struct
# from PIL import Image, ImageEnhance
# from os import listdir, mkdir, sep
from os.path import isfile, isdir, join, exists, dirname, abspath
# folder = join(dirname(abspath(__file__))[:-4],'Data','DatasetETLCDB')
# sourceFolder = join(folder, 'ETL1SPLIT')

# import time

# start = time.time()

# allfolders = [join(sourceFolder, f) for f in listdir(sourceFolder) if isdir(join(sourceFolder, f))]
# allfiles= list()
# for direc in allfolders:
#     allfiles += [join(direc,f) for f in listdir(direc) if f != '0' and isfile(join(direc, f))]
# testlist = [allfiles[4]]

# data, targets = [], []
# for filename in testlist:
# # for filename in allfiles:
#     with open(filename, 'rb') as f:
#         # f.seek(skip * 2052)
#         while f.readable():
#             #get start position of name of the file used from absolute path
#             s = f.read(2052)
#             if s == None or len(s) < 2052:
#                 #TODO print error message to see problems
#                 break
#             #19 records, omitted records (x option) with undefined data
#             r = struct.unpack('>H2sH6BI4H4B4x2016s4x', s)
#             iF = Image.frombytes('F', (64, 63), r[18], 'bit', 4)
#             #P maps to 8bit pixels in color, L maps to 8bit pixels black and white
#             iP = iF.convert('L')
#             # r[2]:=Sheet Index (characters A and B may have the same index, but no A has the same index as another one), 
#             # r[3]:=Character Code (JIS X0201)
#             enhancer = ImageEnhance.Brightness(iP)
#             iE = enhancer.enhance(8)
#             tmp = bytes()
#             tmp = struct.pack('>2016s', iE.tobytes())
#             data+=[struct.unpack('>2016B', tmp)]
#             targets+=['JIS X0201 '+str(r[3])]
#             # data+=[iE.tobytes()]


# end = time.time()
# print(end-start)


# import from __future__ import absolute_import, division, print_function, unicode_literals


import tensorflow as tf
from tensorflow.keras import layers, Model

# Our input feature map is 150x150x3: 150x150 for the image pixels, and 3 for
# the three color channels: R, G, and B
img_input = layers.Input(shape=(64, 64, 1))

# First convolution extracts 16 filters that are 3x3
# Convolution is followed by max-pooling layer with a 2x2 window
x = layers.Conv2D(16, 3, activation='relu')(img_input)
x = layers.MaxPooling2D(2)(x)

# Second convolution extracts 32 filters that are 3x3
# Convolution is followed by max-pooling layer with a 2x2 window
x = layers.Conv2D(32, 3, activation='relu')(x)
x = layers.MaxPooling2D(2)(x)

# Third convolution extracts 64 filters that are 3x3
# Convolution is followed by max-pooling layer with a 2x2 window
x = layers.Conv2D(64, 3, activation='relu')(x)
x = layers.MaxPooling2D(2)(x)

# Create a fully connected layer with ReLU activation and 512 hidden units
x = layers.Dense(512, activation='relu')(x)

# Create output layer with a single node and sigmoid activation
output = layers.Dense(1, activation='sigmoid')(x)

# Create model:
# input = input feature map
# output = input feature map + stacked convolution/maxpooling layers + fully 
# connected layer + sigmoid output layer
model = Model(img_input, output)


# print(model.summary())


from tensorflow.keras.optimizers import RMSprop

model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=0.001),
              metrics=['acc'])


import paths
sourceFolder = join(paths.getRootPath(), 'Data', 'DatasetETLCDB', 'ETL1PNG')
train_dir = join(sourceFolder, 'train')
validation_dir = join(sourceFolder, 'test')

# Directory with our training cat pictures
train_cats_dir = train_dir

# Directory with our training dog pictures
train_dogs_dir = train_dir

# Directory with our validation cat pictures
validation_cats_dir = validation_dir

# Directory with our validation dog pictures
validation_dogs_dir = validation_dir

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

# Flow training images in batches of 20 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
        train_dir,  # This is the source directory for training images
        target_size=(64, 64),  # All images will be resized to 150x150
        batch_size=20,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

# Flow validation images in batches of 20 using val_datagen generator
validation_generator = val_datagen.flow_from_directory(
        validation_dir,
        target_size=(64, 64),
        batch_size=20,
        class_mode='binary')



history = model.        (
      train_generator,
#       steps_per_epoch=100,  # 2000 images = batch_size * steps
      epochs=15,
      validation_data=validation_generator,
#       validation_steps=50,  # 1000 images = batch_size * steps
      verbose=2)



import numpy as np
import random
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Let's define a new Model that will take an image as input, and will output
# intermediate representations for all layers in the previous model after
# the first.
successive_outputs = [layer.output for layer in model.layers[1:]]
visualization_model = Model(img_input, successive_outputs)

# Let's prepare a random input image of a cat or dog from the training set.
cat_img_files = [join(train_cats_dir, f) for f in train_cat_fnames]
dog_img_files = [join(train_dogs_dir, f) for f in train_dog_fnames]
img_path = random.choice(cat_img_files + dog_img_files)

img = load_img(img_path, target_size=(64, 64))  # this is a PIL image
x = img_to_array(img)  # Numpy array with shape (150, 150, 3)
x = x.reshape((1,) + x.shape)  # Numpy array with shape (1, 150, 150, 3)

# Rescale by 1/255
x /= 255

# Let's run our image through our network, thus obtaining all
# intermediate representations for this image.
successive_feature_maps = visualization_model.predict(x)

# These are the names of the layers, so can have them as part of our plot
layer_names = [layer.name for layer in model.layers]

# Now let's display our representations
for layer_name, feature_map in zip(layer_names, successive_feature_maps):
  if len(feature_map.shape) == 4:
    # Just do this for the conv / maxpool layers, not the fully-connected layers
    n_features = feature_map.shape[-1]  # number of features in feature map
    # The feature map has shape (1, size, size, n_features)
    size = feature_map.shape[1]
    # We will tile our images in this matrix
    display_grid = np.zeros((size, size * n_features))
    for i in range(n_features):
      # Postprocess the feature to make it visually palatable
      x = feature_map[0, :, :, i]
      x -= x.mean()
      x /= x.std()
      x *= 64
      x += 128
      x = np.clip(x, 0, 255).astype('uint8')
      # We'll tile each filter into this big horizontal grid
      display_grid[:, i * size : (i + 1) * size] = x
    # Display the grid
    scale = 20. / n_features
    plt.figure(figsize=(scale * n_features, scale))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')