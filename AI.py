import nltk
from nltk import classify


class NaiveBayesClassifier:

    def __init__(self, tweetCollection,features):
        self.features = features
        self.test_data = tweetCollection
        self.nbClassifier = nltk.NaiveBayesClassifier.train(FeatureExtractor(tweetCollection,features).get_feature_set_NB())


    def accuracy_test(self,tweetCollection):
        # Compute Accuracy of classifier
        accuracy = classify.accuracy(self.nbClassifier, FeatureExtractor(tweetCollection,self.features).get_feature_set_NB())
        # Print K most informative features
        print(self.nbClassifier.show_most_informative_features(2))
        return accuracy*100

class FeatureExtractor:
    def __init__(self,tweetCollection,features):
        self.features = features
        self.tweetCollection = tweetCollection

    def get_feature_NB(self, tweet):
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