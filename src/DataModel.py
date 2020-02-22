import Classificactor
import Regressor

class DataModel():
    def __init__(self, classifier: Classificator, regressor: Regressor):
        self.classifier = classifier
        self.regressor = regressor

    def getOutput(self):
        pass