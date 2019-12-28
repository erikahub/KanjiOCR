"""This code is largely imported from the specification sites of http://etlcdb.db.aist.go.jp/ for the corresponding dataset"""

import struct
from PIL import Image, ImageEnhance
from os import listdir, mkdir
from os.path import isfile, join, exists
folder = 'DatasetETLCDB/ETL1'
allfiles = [f for f in listdir(folder) if isfile(join(folder, f))]
filename = folder + '/ETL1C_07'
# skip = 100
with open(filename, 'rb') as f:
    # f.seek(skip * 2052)
    fileFolder = folder + 'PNG/' + filename.split('/')[-1] + '/'
    if not exists(folder + 'PNG/'):
        mkdir(folder + 'PNG/')
        if not exists(fileFolder):
            mkdir(fileFolder)
    while f.readable():
        s = f.read(2052)
        if s == None or len(s) < 2052:
            break
        r = struct.unpack('>H2sH6BI4H4B4x2016s4x', s)
        iF = Image.frombytes('F', (64, 63), r[18], 'bit', 4)
        # iP = iF.convert('P')
        iP = iF.convert('L')
        fn = "{:1d}_{:4d}_{:2x}.png".format(r[0], r[2], r[3])
        # iP.save('Data/' + fn, 'PNG', bits=4)
        enhancer = ImageEnhance.Brightness(iP)
        iE = enhancer.enhance(16)
        iE.save(folder + 'PNG/' + filename.split('/')[-1] + '/' + fn, 'PNG')