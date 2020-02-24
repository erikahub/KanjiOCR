import struct
from PIL import Image, ImageEnhance
from os import listdir, mkdir, sep
from os.path import isfile, isdir, join, exists, dirname, abspath
folder = join(dirname(abspath(__file__))[:-4],'Data','DatasetETLCDB')
sourceFolder = join(folder, 'ETL1SPLIT')

allfolders = [join(sourceFolder, f) for f in listdir(sourceFolder) if isdir(join(sourceFolder, f))]
allfiles= list()
for direc in allfolders:
    allfiles += [join(direc,f) for f in listdir(direc) if f != '0' and isfile(join(direc, f))]
# testlist = [allfiles[4]]

data, targets = [], []
for filename in allfiles:
    with open(filename, 'rb') as f:
        # f.seek(skip * 2052)
        while f.readable():
            #get start position of name of the file used from absolute path
            s = f.read(2052)
            if s == None or len(s) < 2052:
                #TODO print error message to see problems
                break
            #19 records, omitted records (x option) with undefined data
            r = struct.unpack('>H2sH6BI4H4B4x2016s4x', s)
            iF = Image.frombytes('F', (64, 63), r[18], 'bit', 4)
            #P maps to 8bit pixels in color, L maps to 8bit pixels black and white
            iP = iF.convert('L')
            # r[2]:=Sheet Index (characters A and B may have the same index, but no A has the same index as another one), 
            # r[3]:=Character Code (JIS X0201)
            enhancer = ImageEnhance.Brightness(iP)
            iE = enhancer.enhance(8)
            
            targets+=['JIS X0201 '+str(r[3])]
            data+=[iE.tobytes()]


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data, targets, test_size=0.3, random_state=0)
# print(len(x_train)/(len(x_train)+len(x_test)))