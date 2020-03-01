import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# plt.figure()
# plt.imshow(train_images[0])
# plt.colorbar()
# plt.grid(False)
# plt.show()

# print("Label 40:",train_labels[40])

#formats test dataset to 1s
train_images = train_images / 255.0
test_images = test_images / 255.0

#test output, if dataset in correct format
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.colorbar()
    plt.xlabel(class_names[train_labels[i]])
plt.show()

#creates model: 
#Flatten shapes the array to one-dimensional,
#Dense: fully connected Layers with 128 Nodes, 
#2nd Dense: softmax limit of nodes (equal amount to classlabels, so everything has the prob. of 1)
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128,activation='relu'),
    keras.layers.Dense(10)
])

#compile model
#optmizer function: updates Model based on data and loss function,
#loss function: measures accuracy during training (minimization is key),
#metrics: monitor training and testing steps (here based on accuracy)
model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])


print("Train_Imgs:\n",train_images)

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
print(class_names[np.argmax(predictions[0])])

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

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array, true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
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

predictAllTest_Images()


def predictSingleImage(img):
    img = (np.expand_dims(img,0))
    predictions_single = probability_model.predict(img)
    plot_value_array(1,predictions_single[0], test_labels)
    _ = plt.xticks(range(10),class_names,rotation=45)
    plt.show()

img = test_images[0]
predictSingleImage(img)