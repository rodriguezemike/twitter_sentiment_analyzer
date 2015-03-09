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

    def score(self, test_data):
        pass

    def train(self, training_data):
        self.has_trained = True
        for key in self._classifiers:
            self._classifiers[key].train(training_data)

