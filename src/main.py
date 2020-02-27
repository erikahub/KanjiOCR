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
# import tensorflow as tf

import dataconverter
dc = dataconverter.DataConverter()
dc.load()
dc.lables
ds_x = tf.data.Dataset.from_tensor_slices(img)
print(ds_x)