import nltk
import Utils
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer as tfidfTransformer
from sklearn.metrics import accuracy_score as meanScore

#------------- Naive Bayes Class -------------#
class NaiveBayesClassifier:
    training_data = None
    nbClassifier = None

    def __init__(self):
        pass

    def train(self,test_data,labelSet):
        self.nbClassifier = MultinomialNB().fit(test_data,labelSet)

    def predict(self,test_datum):
        if self.nbClassifier != None:
            return self.nbClassifier.predict(test_datum)
        else:
            print('ERROR: Must train classifier before making predictions')

    def predictCollection(self,test_data):
        if self.nbClassifier != None:
            return [self.nbClassifier.predict(tweet) for tweet in test_data]
        else:
            print('ERROR: Must train classifier before making predictions')

    def score(self,test_data,labelSet):
        return self.nbClassifier.score(test_data,labelSet)


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

    def score(self,test_data,labelSet):
        return self.lrClassifier.score(test_data,labelSet)

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
    def __init__(self):
        pass

    def get_tokenCount_featureset(self,tweetCollection, test_data=None):
        #Same featureset as above but using sklearn framework
        count_vect = CountVectorizer()
        if test_data != None:
            countVectorizer = count_vect.fit(tweetCollection.get_tweets()[1:100]+test_data.get_tweets()[1:100])
        else:
            countVectorizer = count_vect.fit(tweetCollection.get_tweets())
        if test_data != None:
            return countVectorizer.transform(tweetCollection.get_tweets()), tweetCollection.get_labels(), countVectorizer.transform(test_data.get_tweets()), test_data.get_labels()
        else:
            return countVectorizer.transform(tweetCollection.get_tweets()), tweetCollection.get_labels()


    ##############################
    # ---- BEGIN DEPRECATED ---- #
    ##############################
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
    ##############################
    # ----- END DEPRECATED ----- #
    ##############################