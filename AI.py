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
'''
Features - Will change but just connecting it all together.
        featureset["length"] = self.length
        featureset["tokens"] = frozenset(self.tokens)
        featureset["stemmed_tokens"] = frozenset(self.stemmed_tokens)
        featureset["bigrams"] = frozenset(self.bigrams)
        featureset["trigrams"] = frozenset(self.trigrams)
        featureset["pos_tags"] = frozenset(self.pos_tags)
        featureset["pos_tags_count"] = frozenset(self.pos_tags_count)
        featureset["hashtag_count"] = self.hashtag_count
        featureset["negation_count"] = self.negation_count
        featureset["uppercase_count"] = self.uppercase_count
        featureset["lexicon_score"] = self.lexicon_score
        featureset["lexicon_label"] = self.lexicon_label
'''
class FeatureExtractor:
    def __init__(self,tweetCollection):
        self.tweetCollection = tweetCollection

    def get_feature_most_common(self, tweet,count):
        fdist = Utils.get_frequency_distribution(self.tweetCollection.generate_nltk_text(1))
        features = [feature[0] for feature in fdist.most_common(count)]
        featureset = dict()
        tweetTokens = tweet.get_tweet_tokens()
        for feature in features:
            featureset[feature] = 'no'
            if feature in tweetTokens:
                featureset[feature] = 'yes'
        return featureset

    def get_feature_set_NB(self):
        tweets = self.tweetCollection.get_tweets()
        toRtn = [(self.get_feature_most_common(tweet,20), tweet.get_label()) for tweet in tweets]
        return toRtn

    def get_tweet_featuresets(self):
        self.tweetCollection.set_lexicon_features()
        for tweet in self.tweetCollection:
            tweet.generate_features()
        toRtn = [(tweet.get_features(),tweet.get_label()) for tweet in self.tweetCollection]
        return toRtn    
