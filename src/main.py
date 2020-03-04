import dataconverter
from cnn_etl1 import CNN_ETL1
import paths

# dc = dataconverter.DataConverter()
# dc.split() #only required once
# dc.load() #only required when using data from memory such as for the export method
# dc.exportPNGOrganised(0.7, 182) #export data from memory to PNGs and split into train and test folders

network = CNN_ETL1()
network.train(epochs=15)
# network.saveModel()
# network.predict(paths.join(paths.getRootPath(), 'Data', 'my_test'))

#network.model.summary()
