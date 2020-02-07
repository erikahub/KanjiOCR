"""See http://etlcdb.db.aist.go.jp/specification-of-etl-1 for the specification of the format and the data it contains"""

import struct
from collections import defaultdict
from PIL import Image
from os import listdir, mkdir
from os.path import isfile, join, exists
folder = join('Data','DatasetETLCDB','ETL1')
allfiles = [f for f in listdir(folder) if isfile(join(folder, f))]
target = folder+'SPLIT'
# filename = folder + '/ETL1C_07'
for file in allfiles:
    with open(join(folder, file), 'rb') as f:
        while f.readable():
            s = f.read(2052)
            if s == None or len(s) < 2052:
                break
            try:
                mkdir(join(target,file))
            except FileExistsError as err:
                pass
            #if the file already exists skip, else create new file containing all data
            if not exists(join(target,file,'{:1d}'.format(s[6]))):
                with open(join(target,file,'{:1d}'.format(s[6])), 'ab') as wf:
                    wf.write(s)
        
