import AI

class classifierController:
    _classifiers = {}

    def __init__(self, classifierName,classifier, featureset):
        # e.g.) classifierController(NaiveBayesClassifier(), featureset)
        # Build featureset before passing
        self._classifiers[classifierName] = classifier
        self._classifiers[classifierName].train(featureset)

    def predict(self,classifierName,feature):
        return self._classifiers[classifierName].predict(feature)

    def train(self,classifierName, featureset):
        self._classifiers[classifierName].train(featureset)

