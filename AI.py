import nltk
import Utils
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer as tfidfTransformer

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


#------------- Logistic Regression Class -------------#

class LogisticRegressionClassifier:
    training_data = None
    lrClassifier = None

    def __init__(self):
        pass

    def train(self,featureset, labelset):
        self.lrClassifier = SGDClassifier(loss='log',shuffle=True).fit(featureset,labelset)

    def predict(self,feature):
        return self.lrClassifier.predict(feature)

    def predictCollection(self,features):
        predictions = []
        for feature in features:
            predictions.append(self.lrClassifier.predict(feature)[0])
        return predictions

    def score(self,features,target):
        self.lrClassifier.score(features,target)

#------------- Feature Extractor Class -------------#
'''
Features - Will change
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
        self.tweetList, self.labelList = self.constructData()

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
        #self.tweetCollection.set_lexicon_features()
        for tweet in self.tweetCollection:
            tweet.generate_features()
        toRtn = [(tweet.get_features(),tweet.get_label()) for tweet in self.tweetCollection]
        return toRtn

    def get_tokenCount_featureset(self):
        #Same featureset as above but using sklearn framework
        count_vect = CountVectorizer()
        transformer = tfidfTransformer()
        X_train_counts = count_vect.fit_transform(self.tweetList)
        return transformer.fit_transform(X_train_counts), count_vect, transformer

    def constructData(self):
        return [TweetObject.get_tweet() for TweetObject in self.tweetCollection.get_tweets()],\
               [TweetObject.get_label() for TweetObject in self.tweetCollection.get_tweets()]

    def get_labels(self):
        return self.labelList

    def get_tweets(self):
        return self.tweetList

    def splitData(self,training_data,target_data):
        return Utils.split_data(training_data,target_data)