import Character
import abc


#Abstract Base Class (abc), simuliert Interface
class Classifier(abc.ABC):
    @abc.abstractmethod
    def classify(self):
        pass