from sklearn import datasets
from AI import *
from tweet import *
from ValidationController import *
from sklearn import metrics
import numpy as np

def validationTest():
    #Construct Features
    filename='training.1600000.processed.noemoticon'
    training_data = TweetCollection()
    training_data.gather_tweets_stanford(count=500,filename=filename)

    featureController = FeatureExtractor(tweetCollection=training_data)

    training_set, count_vect, transformer = featureController.get_tokenCount_featureset()
    target_set = featureController.get_labels()
    train_data, test_data, train_target, test_target = featureController.splitData(training_set, target_set)

    #Construct Classifiers
    classifierName = []
    classifierName.append('LogisticRegress')

    classifiers = []
    classifiers.append(LogisticRegressionClassifier())

    #Do validation Test
    lrClassifier = LogisticRegressionClassifier()
    lrClassifier.train(train_data, train_target)

    predicted = lrClassifier.predictCollection(test_data)
    print(metrics.classification_report(test_target.tolist(),predicted))

def NaiveBayesTest():
    filename='training.1600000.processed.noemoticon'
    training_data = TweetCollection()
    training_data.gather_tweets_stanford(count=100,filename=filename)
    features = FeatureExtractor(tweetCollection=training_data).get_feature_set_NB()
    feature = features[0][0]
    print(feature)
    nbClassifier = NaiveBayesClassifier()
    nbClassifier.train(features)
    print(nbClassifier.predict(feature))

def LogisticRegressionTest():
    filename='training.1600000.processed.noemoticon'
    training_data = TweetCollection()
    training_data.gather_tweets_stanford(count=100,filename=filename)
    featureController = FeatureExtractor(tweetCollection=training_data)
    features, count_vect, transformer = featureController.get_tokenCount_featureset()
    lrClassifier = LogisticRegressionClassifier()
    lrClassifier.train(features,featureController.get_labels())

    sampleTweet = featureController.get_tweets()[11]
    X_train_count = count_vect.transform([sampleTweet])
    testTweet = transformer.transform(X_train_count)

    print(lrClassifier.score(testTweet,["0"]))
    # predicted = lrClassifier.predict(testTweet)
    # for doc, category in zip([sampleTweet], predicted):
    #     print('%r => %s' % (doc, featureController.get_labels()[category]))

def print_tweetCollection(tweetCollection):
    count = 1
    for i in tweetCollection.get_tweets():
        print(str(count)+": "+str(i))
        print('tweet tokens:')
        print(str(count) + str(i.get_tweet_tokens()))
        print('stemmed tweet tokens:')
        print(str(count) + str(i.get_stemmed_tweets()))
        print('no stopwords:')
        print(str(count) + str(i.get_tweet_tokens()))
        count+=1

validationTest()