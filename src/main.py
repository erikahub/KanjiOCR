# import imageio
# import paths
# from os import listdir

# data_path = paths.join(paths.getDBPath(), 'ETL1PNG')
# tmp = listdir(paths.join(data_path, 'train'))
# data_paths = []
# data_
# for folder in tmp:
#     data_paths += listdir(paths.join(data_path, 'train', folder))
# del tmp
# img = imageio.imread(paths.join(data_path, 'train', '0', '0001_0.png'))

import tensorflow as tf
import dataconverter
import matplotlib.pyplot as plt
from PIL import Image

dc = dataconverter.DataConverter()
# dc.split()
dc.load()
# dc.exportPNGOrganised() 
#ds stands for dataset
ds_joint = tf.data.Dataset.from_tensor_slices((dc.features, dc.labels)) #x: features, y: labels

def printSome(dataset, number: int):
    c = 0
    for example in ds_joint:
        c+=1
        print('x:', example[0].numpy()[:10], ' y:', example[1].numpy())
        if c >= number:
            break

tf.random.set_seed(1)
ds = ds_joint.shuffle(buffer_size=100)

# print(dc.features[-1])


# a = Image.open('C:\\Users\\Bjoern_Lewe\\Git Repo\\Data\\DatasetETLCDB\\ETL1PNG\\train\\48\\0001_48.png')

# a.show()


# plt.figure()
# plt.imshow(a)
# plt.colorbar()
# plt.grid(False)
# plt.show()

# printSome(ds, 500)