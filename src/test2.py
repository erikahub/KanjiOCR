import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

import dataconverter
import JISDictClass
import re
from PIL import Image
import math

print(tf.__version__)

dc = dataconverter.DataConverter()
dc.load()
dc.convertFeaturesToNumpyArray()
dc.convertLabels()

print(math.floor(len(dc.features) * 0.8))
train_images = dc.features[0:math.floor(len(dc.features) * 0.8)]
test_images = dc.features[math.floor(len(dc.features) * 0.8)+1:len(dc.features)]
train_labels = dc.labels[0:math.floor(len(dc.features) * 0.8)]
test_labels = dc.labels[math.floor(len(dc.features) * 0.8)+1:len(dc.features)]

JIS = JISDictClass.JISDictClass()
# int(re.split("_", train_labels[200])[-1]))

# print(JIS.getValues())

# print(train_labels[200])

# print(re.match("_\d*", train_labels[200]))


# print(train_labels[200])

# image = Image.frombytes('F',(64,63), train_images[200],'bit',4)

# pix = np.array(image)

# print(len(train_images[50]))
# print(train_images[50])


# fashion_mnist = keras.datasets.fashion_mnist
# (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


# class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
            #    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#formats test dataset to 1s
# train_images = train_images / 255.0
# test_images = test_images / 255.0

# print(train_labels[200])

#test output, if dataset in correct format
# plt.figure(figsize=(64,63))
# for i in range(0,50):
#     plt.subplot(10,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(JIS.lookUp(int(train_labels[i])))
#     plt.colorbar()
# plt.show()

print(np.unique(train_labels))

#creates model: 
#Flatten shapes the array to one-dimensional,
#Dense: fully connected Layers with 128 Nodes, 
#2nd Dense: softmax limit of nodes (equal amount to classlabels, so everything has the prob. of 1)
print("train_label:",len(set(train_labels)),"\nDictLabel:",len(JIS.getValues()),"\nRatio:", len(set(train_labels))/len(JIS.getValues()))

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(63,64)),
    keras.layers.Dense(128,activation='selu'),
    keras.layers.Dense(len(np.unique(train_labels)))
])

#compile model
#optmizer function: updates Model based on data and loss function,
#loss function: measures accuracy during training (minimization is key),
#metrics: monitor training and testing steps (here based on accuracy)
# tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam',
                loss=tf.keras.losses.Huber(),
                metrics=['accuracy'])

#train model with training_data and do it 10 times (improve accuracy)
model.fit(train_images,train_labels, epochs=10)

#test model on test_data
test_loss, test_acc = model.evaluate(test_images,test_labels, verbose=2)

print('\nTest accuracy:\t',test_acc)

#creates prediction model using trained model
probability_model = tf.keras.Sequential([
    model,
    tf.keras.layers.Softmax()
])

#model makes a prediction on test_data
#outputs a list of numpy arrays with probabilities of each class label
predictions = probability_model.predict(test_images)

#confidence array, max probability is the label
print(JIS.lookUp(np.argmax(predictions[0])))

def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array, true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(JIS.lookUp(predicted_label),
                                100*np.max(predictions_array),
                                JIS.lookUp(true_label)),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array, true_label[i]
  plt.grid(False)
  plt.xticks(range(len(JIS.getValues())))
  plt.yticks([])
  thisplot = plt.bar(range(len(JIS.getValues())), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

 
def predictAllTest_Images():
    num_rows = 5
    num_cols = 3
    num_images = num_rows*num_cols
    plt.figure(figsize=(2*2*num_cols, 2*num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, 2*num_cols, 2*i+1)
        plot_image(i, predictions[i], test_labels, test_images)
        plt.subplot(num_rows, 2*num_cols, 2*i+2)
        plot_value_array(i, predictions[i], test_labels)
        plt.tight_layout()
    plt.show()

# predictAllTest_Images()


def predictSingleImage(img):
    img = (np.expand_dims(img,0))
    predictions_single = probability_model.predict(img)
    plot_value_array(200,predictions_single[0], test_labels)
    _ = plt.xticks(range(len(JIS.getValues())),JIS.getValues(),rotation=45)
    plt.show()

img = test_images[200]

plt.figure()
plt.imshow(img)
plt.colorbar()
plt.grid(False)
plt.show()

predictSingleImage(img)