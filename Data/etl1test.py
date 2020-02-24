"""this script is a test script which converts all files with a given name ('0') into pictures in folder t
author: rohue"""

import struct
from collections import defaultdict
from PIL import Image, ImageEnhance
from os import listdir, mkdir
from os.path import isfile, join, exists, isdir, dirname
folder = join(dirname(abspath(__file__)),'DatasetETLCDB','ETL1SPLIT')
if not exists(join(folder, 't')):
    mkdir(join(folder, 't'))
# folder = join('Data', 'DatasetETLCDB','ETL1SPLIT')
allfolders = [join(folder, f) for f in listdir(folder) if isdir(join(folder, f))]
allfiles=[]
for direc in allfolders:
    allfiles += [join(direc,f) for f in listdir(direc) if isfile(join(direc, f)) and f == '0']
target = folder+'SPLIT'
print(allfiles)

for filename in allfiles:
    with open(filename, 'rb') as f:
        while f.readable():
            s = f.read(2052)
            if s == None or len(s) < 2052:
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
            iE.save(join(folder,'t',fn), 'PNG')