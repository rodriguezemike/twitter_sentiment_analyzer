import AI

class NaiveBayesController:

    def __init__(self, tweetCollection,features):
        self.nbClassifier = AI.NaiveBayesClassifier(tweetCollection,features)

    def prediction_accuracy(self,tweetCollection):
        return self.nbClassifier.accuracy_test(tweetCollection)
