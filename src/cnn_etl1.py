"""TODO WRITE THIS"""
##In[]
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Activation, Dropout, Conv2D, MaxPooling2D
import paths
from os import listdir
from time import time

class CNN_ETL1():
    """This class trains a model based on the data of ETL1.
    For now only supports the ETL1PNG folder and only if split into train and test folders. See DataConverter.exportPNGOrganised"""
    def __init__(self):
        super().__init__()
        self.model = Sequential()
        self.fit_args = {'verbose':1}

        #TODO look into this. rescaling is supposed to be better but seems to yield worse results?
        # img_gen = ImageDataGenerator()
        img_gen = ImageDataGenerator(rescale=1./255)
        data_path = paths.join(paths.getDBPath(), 'ETL1PNG')

        self.BATCH_SIZE = 64
        self.IMG_HEIGHT = 63
        self.IMG_WIDTH = 64
        self.IMG_CHANNELS = 1

        start = time()
        #list of arguments to be extracted for use in data generator objects
        args = {'batch_size':BATCH_SIZE, 
                'shuffle':True,
                'target_size':(IMG_HEIGHT, IMG_WIDTH),
                'color_mode':'grayscale',
                'class_mode':'categorical' #binary for binary encoded labels, categorical for hot-encoded and sparse for integer labels
                }
        self.train_data_gen = img_gen.flow_from_directory(directory=paths.join(data_path, 'train'), **args)
        self.test_data_gen = img_gen.flow_from_directory(directory=paths.join(data_path, 'test'),**args)
        print('Took: ',time() - start, 'seconds')

        self.model.add(Conv2D(32,(5,5), padding='same', activation='relu', 
                                         input_shape=(IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS)))
        # model.add(MaxPooling2D((3, 3)))
        self.model.add(MaxPooling2D((8, 8)))
        self.model.add(Conv2D(64, (5, 5), padding='same', activation='softmax'))
        self.model.add(Flatten())
        self.model.add(Dense(97, activation='softmax')) #this is supposed to be the number of labels.
        
        #sparse_categorical_crossentropy results in the model only allowing the number of labels input into Dense -1.
        #one-hot encoding is required with categorical_crossentropy
        #https://stackoverflow.com/a/59148543
        self.model.compile(loss='categorical_crossentropy',
                      optimizer='sgd',
                      metrics=['accuracy'])
        

    def train(self, epochs=1, callbacks=None):
        start = time()
        fit_history = self.model.fit(self.train_data_gen,
          epochs=epochs, #TODO change this back to higher numbers to try achieving higher accuracy
          **self.fit_args)
        end=time()
        print(f'Training took {(end-start)//60:.0}min {int(end-start)%60}seconds')


    def test(self):
        test_loss, test_acc = self.model.evaluate(self.test_data_gen, 
                        verbose=1)
        print(f'Test loss: {test_loss}, test accuracy: {test_acc}')


    def saveModel(self):
        """Save a (compiled) model including its architecture and weights to a file."""
        fn = input('Save as: ')
        if any('\/*><"|?:') in fn:
            print('File could not be created, invalid file name.')
            return
        self.model.save(paths.join(paths.getModelsPath(), fn))


    def loadModel(self):
        """Load (compiled) model from file."""
        models = listdir(paths.getModelsPath())
        for i in range(len(models)):
            print(i, models[i])
        cm = input('Choose a number: ')
        try:
            cm = int(cm)
        except ValueError as e:
            print('Choice was not an integer.')
            return
        self.model.load_weights(paths.join(paths.getModelsPath, cm))