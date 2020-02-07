"""This code is largely imported from the specification sites of http://etlcdb.db.aist.go.jp/ for the corresponding dataset.
See http://etlcdb.db.aist.go.jp/specification-of-etl-1"""

import struct
from PIL import Image, ImageEnhance
from os import listdir
from os.path import isfile, join
folder = 'Data/DatasetETLCDB/ETL1'
allfiles = [f for f in listdir(folder) if isfile(join(folder, f))]
filename = folder + '/ETL1C_07'
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
        # iP = iF.convert('P')
        iP = iF.convert('L')
        #r[0]:=Data Index, r[2]:=Sheet Index, r[3]:=Character Code (JIS X0201)
        fn = "{:1d}_{:4d}_{:2x}.png".format(r[0], r[2], r[3])
        # iP.save('Data/' + fn, 'PNG', bits=4)
        enhancer = ImageEnhance.Brightness(iP)
        iE = enhancer.enhance(16)
        iE.save(folder + 'PNG/' + filename.split('/')[-1] + '/' + fn, 'PNG')
        # 0x2A22  0x3042  # HIRAGANA LETTER A