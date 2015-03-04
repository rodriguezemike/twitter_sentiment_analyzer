import nltk
from nltk import classify
import Utils

#------------- General Classifier Class -------------#
class classifier:
    _predictor = None

    def __init__(self,predictor):
        self._predictor = predictor

    def train(self,featureset):
        self._predictor.train(featureset)

    def predict(self,feature):
        self._predictor.predict(feature)


#------------- Naive Bayes Class -------------#
class NaiveBayesClassifier:
    training_data = None
    nbClassifier = None

    def __init__(self):
        pass

    def train(self,featureset):
        self.nbClassifier = nltk.NaiveBayesClassifier.train(featureset)

    def predict(self,feature):
        if self.nbClassifier != None:
            return self.nbClassifier.classify(feature)
        else:
            print('ERROR: Must train classifier before making predictions')

    def predictCollection(self,features):
        if self.nbClassifier != None:
            return self.nbClassifier.classify_many(features)
        else:
            print('ERROR: Must train classifier before making predictions')

#------------- Feature Extractor Class -------------#
class FeatureExtractor:
    def __init__(self,tweetCollection):
        self.tweetCollection = tweetCollection

    def get_feature_most_common(self, tweet, count):
        fdist = Utils.get_frequency_distribution(self.tweetCollection.generate_nltk_text(1))
        features = [feature[0] for feature in fdist.most_common(20)]
        featureset ={}
        tweetTokens = tweet.get_tweet_tokens()
        for feature in self.features:
            featureset[feature] = 'no'
            if feature in tweetTokens:
                featureset[feature] = 'yes'
        return featureset

    def get_feature_set_NB(self):
        tweets = self.tweetCollection.get_tweets()
        toRtn = []
        for tweet in tweets:
            toRtn.append((self.get_feature_NB(tweet),tweet.get_label()))
        return toRtn