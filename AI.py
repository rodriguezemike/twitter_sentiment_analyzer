import nltk
from nltk import classify
import Utils


class NaiveBayesClassifier:

    def __init__(self, tweetCollection):
        self.test_data = tweetCollection
        self.nbClassifier = nltk.NaiveBayesClassifier.train(FeatureExtractor(tweetCollection).get_feature_set_NB())

    def classifty(self,tweetCollection):
        features = FeatureExtractor(tweetCollection).get_feature_set_NB()
        result = []
        for oneFeature in features:
            result.append(self.nbClassifier.classify(oneFeature))
        return result

    def accuracy_test(self,tweetCollection):
        # Compute Accuracy of classifier
        accuracy = classify.accuracy(self.nbClassifier, FeatureExtractor(tweetCollection).get_feature_set_NB())
        # Print K most informative features
        print(self.nbClassifier.show_most_informative_features(10))
        return accuracy*100

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