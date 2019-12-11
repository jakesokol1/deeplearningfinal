from abc import ABC, abstractmethod

class Classifier(ABC):

    @abstractmethod
    def train():
        """
        Called by delegator to train the model.
        """
        pass

    @abstractmethod
    def save():
        """
        Called by delegator to save a trained model

        :parameter name: name of model to be saved as
        """
        pass

    @abstractmethod
    def test():
        """
        Called by delegator to test a model saved previously as name.
        """
        pass
