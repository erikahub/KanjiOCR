"""This code is based off code found in the specification pages of http://etlcdb.db.aist.go.jp/ for the corresponding dataset.
See http://etlcdb.db.aist.go.jp/specification-of-etl-1"""

import struct
from PIL import Image, ImageEnhance
from os import listdir
from os.path import isfile, join
folder = join(dirname(__file__),'DatasetETLCDB','ETL1SPLIT')
allfiles = [f for f in listdir(folder) if isfile(join(folder, f))]
filename = join(folder,'ETL1C_07')
# skip = 100
with open(filename, 'rb') as f:
    # f.seek(skip * 2052)
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
        iE.save(join(folder + 'PNG', filename.split('/')[-1], fn, 'PNG')
