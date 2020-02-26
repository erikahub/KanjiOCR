"""This code is based off code found in the specification pages of http://etlcdb.db.aist.go.jp/ for the corresponding dataset.
See http://etlcdb.db.aist.go.jp/specification-of-etl-1
author: rohue"""

from colors import bcolors
import paths
from collections import defaultdict
import struct
from PIL import Image, ImageEnhance
from os import listdir, mkdir, sep
from os.path import isfile, isdir, join, exists

#TODO this is only ETL1
#TODO maybe insert progress bars
class DataConverter():
    """Converts data from the ETL Databases to either images or simply splits the data from the Databases into separate files"""    
    def __init__(self):
        self.rootDir = join(paths.getRootPath(), 'Data','DatasetETLCDB')
        #TODO maybe change to dict containing data structure for struct
        self.ETL = ['ETL1']
        self.db = self.ETL[0]
        self.data = defaultdict(list)

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
        self.data.clear()
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
                    self.data['JIS X0201 '+str(r[3])] += [r[18]]

            #prompt user for every 10th file that is being read from
            # if i % 10 == 0:
                # answ = input('Continue? (y/n)')
                # if answ[0] != 'y':
                    # break

    def exportPNGOrganised(self):
        dbFolder = join(self.rootDir, self.currentDB + 'PNG')
        keys = self.data.keys()
        if len(keys) > 0:
            if not exists(dbFolder):
                mkdir(dbFolder)
        else:
            print(f"{bcolors.WARNING}Warning: No labels found{bcolors.ENDC}")
        lcount = 0 #labelcount
        for label in keys:
            features = self.data[label]
            count = 0 #feature number within the label
            ln = label[len('JIS X0201 '):]
            targetDir = join(dbFolder, ln)
            if not exists(targetDir):
                mkdir(targetDir)
            for f in features:
                count+=1
                iF = Image.frombytes('F', (64, 63), f, 'bit', 4)
                #P maps to 8bit pixels in color, L maps to 8bit pixels black and white
                iP = iF.convert('L')
                fn = f"{count:04d}_{ln}.png"
                # iP.save('Data/' + fn, 'PNG', bits=4)
                enhancer = ImageEnhance.Brightness(iP)
                iE = enhancer.enhance(16)

                if not exists(targetDir):
                    mkdir(targetDir)
                iE.save(join(targetDir, fn), 'PNG')


    def exportPNGSplit(self, fileList: list, p: float):
        pass

# dc = DataConverter()
# dc.split()
# dc.load()
# dc.exportPNGOrganised()
# print(dc.data.keys())
# print(dc.data['JIS X0201 193'][0])