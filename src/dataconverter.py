"""This code is based off code found in the specification pages of http://etlcdb.db.aist.go.jp/ for the corresponding dataset.
See http://etlcdb.db.aist.go.jp/specification-of-etl-1
author: rohue"""

from colors import bcolors
import paths
from collections import defaultdict
import struct
import re
import numpy as np
from PIL import Image, ImageEnhance
from os import listdir, mkdir, sep
from os.path import isfile, isdir, join, exists
import sklearn.model_selection as sk
import tensorflow as tf
from keras.utils import to_categorical

#TODO this is only ETL1
#TODO maybe insert progress bars
class DataConverter():
    """Converts data from the ETL Databases to either images or simply splits the data from the Databases into separate files"""    
    def __init__(self):
        self.rootDir = paths.getDBPath()
        #TODO maybe change to dict containing data structure for struct
        self.ETL = ['ETL1']
        self.CHARSETS = {self.ETL[0] : 'JIS_X0201_'}
        self.db = self.ETL[0]
        self.charset = self.CHARSETS[self.db]
        # self.data = defaultdict(list)
        self.features = list()
        self.labels = list()


    def getDB(self):
        return self._db
    def setDB(self, db: str):
        if db not in self.ETL:
            raise NotImplementedError("{db} is not a supported ETLCDB")
        self._db = db

    db = property(getDB, setDB)

    def split(self):
        """Splits the databases into organised subfolders that each contain files of all their characters split into one each file.
        So 'A' has its own file, '0' has its own file etc."""
        ignore = ['ETL1INFO', 'ETL2INFO','ETL3INFO' ,'ETL4INFO' ,'ETL5INFO' ,'ETL6INFO' ,\
            'ETL7INFO' ,'ETL8INFO' ,'ETL9INFO' ,'ETL10INFO' ,'ETL11INFO' ,'ETL12INFO' ,'ETL13INFO']
        folder = join(self.rootDir, self.db)
        allfiles = [f for f in listdir(folder) if f not in ignore and isfile(join(folder, f))]
        targetFolder = folder + 'SPLIT'
        if not exists(targetFolder):
            mkdir(targetFolder)
        for file in allfiles:
            sourceFolder = join(folder, file)
            with open(sourceFolder, 'rb') as f:
                while f.readable():
                    target = join(targetFolder, file)
                    s = f.read(2052)
                    if s == None or len(s) < 2052:
                        break
                    if not exists(target):
                        mkdir(target)
                    #append data to the file, if it does not exist it will be created
                    target = join(target,'{:1d}'.format(s[6]))
                    with open(target, 'ab') as wf:
                        wf.write(s)


    def load(self, ignore: list = []):
        """Loads the data of a single full split(!) database into memory"""
        self.features.clear()
        self.labels.clear()
        self.currentDB = self.db
        sourceFolder = join(self.rootDir, self.db + 'SPLIT')
        #get all files from the ETLxSPLIT subfolders
        allfolders = [join(sourceFolder, f) for f in listdir(sourceFolder) if isdir(join(sourceFolder, f))]
        allfiles= list()
        for direc in allfolders:
            allfiles += [join(direc,f) for f in listdir(direc) if f not in ignore and isfile(join(direc, f))]

        i = 0
        for filename in allfiles:
            i+=1
            with open(filename, 'rb') as f:
                while f.readable():
                    s = f.read(2052)
                    if len(s) == 0: #len of 0 indicates eof
                        break
                    if s == None or len(s) < 2052: 
                        var = 'None' if s == None else f'{len(s)}B'
                        print(f"{bcolors.WARNING}Warning: There was a problem reading the data from file {filename}. Data equals to {var}{bcolors.ENDC}")
                        break
                    #19 records, omitted records (x option) with undefined data
                    r = struct.unpack('>H2sH6BI4H4B4x2016s4x', s)
                    self.features.append(r[18])
                    self.labels.append(self.charset + str(r[3]))

            #prompt user for every 10th file that is being read from
            # if i % 10 == 0:
                # answ = input('Continue? (y/n)')
                # if answ[0] != 'y':
                    # break

    def exportPNGOrganised(self, train_size = None, random_state: int = None):
        """Exports all loaded images currently in the features and labels lists as PNG in the following structure:
            Path/DBnamePNG/
            |_train
            |   |_FolderCharacter1 (contains character images of character 1 only)
            |   |_FolderCharacter2 (same as above for character 2)
            |   |_...
            |_test
                |_FolderCharacter1
                |_...
            """
        dbFolder = join(self.rootDir, self.currentDB + 'PNG')
        traindir = join(dbFolder, 'train')
        testdir = join(dbFolder, 'test')
        if not exists(dbFolder):
            mkdir(dbFolder)
        if not exists(traindir):
            mkdir(traindir)
        if not exists(testdir):
            mkdir(testdir)

        (x_train, x_test, y_train, y_test ) = \
          sk.train_test_split(self.features, self.labels, train_size=train_size, random_state=random_state)

        self._export(x_train, y_train, traindir)
        self._export(x_test, y_test, testdir)



    def _export(self, features, labels, path):
        count = defaultdict(int)
        if len(labels) != len(features):
            raise RuntimeError("labels and features do not match in size")
        for idx in range(len(labels)):
            label = labels[idx]
            ln = label[len(self.charset):]
            count[ln]+=1
            targetDir = join(path, ln)
            if not exists(targetDir):
                mkdir(targetDir)
            iF = Image.frombytes('F', (64, 63), features[idx], 'bit', 4)
            #P maps to 8bit pixels in color, L maps to 8bit pixels black and white
            iP = iF.convert('L')
            fn = f"{count[ln]:04d}_{ln}.png"
            # iP.save('Data/' + fn, 'PNG', bits=4)
            enhancer = ImageEnhance.Brightness(iP)
            iE = enhancer.enhance(16)
            iE.save(join(targetDir, fn), 'PNG')


    def exportPNGSplit(self, fileList: list, p: float):
        """Not implemented"""
        pass

    def setFeaturesToBinary(self):
        tmpFeatures = []
        for feature in self.features:
            img = feature / 10.0
            # print(feature,":\t",img)
            tmpFeatures.append(img)
        self.features = tmpFeatures

    def convertFeaturesToNumpyArray(self):
        tmpFeatures = []
        for feature in self.features:
            img = Image.frombytes('F',(64,63), feature,'bit',4)
            pix = np.array(img)
            tmpFeatures.append(pix)
        self.features = tmpFeatures
        self.setFeaturesToBinary()
        self.features = np.array(self.features)
        print(self.features.shape)
    

    def convertLabels(self):
        tmpLabels = []
        for label in self.labels:
            tmpLabels.append(np.array(int(re.split("_", label)[-1])))
        self.labels = tmpLabels
        self.labels = np.array(self.labels)


# dc = DataConverter()
# dc.load()
# dc.exportPNGOrganised(0.7, 182)
