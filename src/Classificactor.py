import Classifier

class Classificator(Classifier):
    def __init__(self):
        self.labels = []

    def addLabel(self, char:Character):
        self.labels.append(char)
    
    def classify(self):
        pass