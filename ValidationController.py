import AI

class ValidationController:
    _classifiers = {}
    has_trained = False

    def __init__(self, classifierName = [],classifier = []):
        # e.g.) classifierController(NaiveBayesClassifier(), featureset)
        # Build featureset before passing
        if len(classifierName) == len(classifier) and len(classifier) != 0:
            for name, cls in zip(classifierName,classifier):
                self._classifiers[name] = cls
        else:
            print("ERROR: Classifiers List cannot be empty")


    def score(self, test_data, labelSet):
        if self.has_trained == True:
            toRtn = {}
            for key in self._classifiers:
                toRtn[key] = self._classifiers[key].score(test_data,labelSet)
        else:
            print("ERROR: Classifiers must be trained first!")
        return toRtn

    def train(self, training_data, labelSet):
        if len(self._classifiers) != 0:
            self.has_trained = True
            for key in self._classifiers:
                self._classifiers[key].train(training_data,labelSet)
        else:
            print("ERROR: No Classifiers available")

    def getClassifier(self,name):
        return self._classifiers[name]