"""This code is based off code found in the specification pages of http://etlcdb.db.aist.go.jp/ for the corresponding dataset.
See http://etlcdb.db.aist.go.jp/specification-of-etl-1
author: rohue"""

import struct
from PIL import Image, ImageEnhance
from os import listdir, mkdir, sep
from os.path import isfile, isdir, join, exists, dirname, abspath
folder = join(dirname(abspath(__file__)),'DatasetETLCDB')
sourceFolder = join(folder, 'ETL1SPLIT')
rootImageFolder = join(folder, 'ETL1PNG')
if not exists(rootImageFolder):
    mkdir(rootImageFolder)

#get all files from the ETL1SPLIT subfolders
allfolders = [join(sourceFolder, f) for f in listdir(sourceFolder) if isdir(join(sourceFolder, f))]
allfiles= list()
for direc in allfolders:
    allfiles += [join(direc,f) for f in listdir(direc) if isfile(join(direc, f))]
# print(allfiles)

# skip = 100
for filename in allfiles:
    with open(filename, 'rb') as f:
        # f.seek(skip * 2052)
        while f.readable():
            #get start position of name of the file used from absolute path
            fnpos=str.rfind(filename, sep)
            targetDir = join(rootImageFolder,filename[str.rfind(filename[:fnpos], sep)+1:fnpos])
            if not exists(targetDir):
                mkdir(targetDir)

            s = f.read(2052)
            if s == None or len(s) < 2052:
                #TODO print error message to see problems
                break
            #19 records, omitted records (x option) with undefined data
            r = struct.unpack('>H2sH6BI4H4B4x2016s4x', s)
            iF = Image.frombytes('F', (64, 63), r[18], 'bit', 4)
            #P maps to 8bit pixels in color, L maps to 8bit pixels black and white
            # iP = iF.convert('P')
            iP = iF.convert('L')
            # r[2]:=Sheet Index (characters A and B may have the same index, but no A has the same index as another one), 
            # r[3]:=Character Code (JIS X0201)
            fn = "{:4d}_{:2x}.png".format(r[2], r[3])
            # iP.save('Data/' + fn, 'PNG', bits=4)
            enhancer = ImageEnhance.Brightness(iP)
            iE = enhancer.enhance(16)

            targetDir = join(targetDir, filename[fnpos+1:])
            if not exists(targetDir):
                mkdir(targetDir)
            iE.save(join(targetDir, fn), 'PNG')
    
    #prompt user for each file that is being read from
    answ = input('Continue? (y/n)')
    if answ != 'y' or answ != 'yes':
        break