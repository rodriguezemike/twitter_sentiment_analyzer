from Controller import *
from AI import *
from tweet import *

def davidMain():
    filename='training.1600000.processed.noemoticon'
    # tweetCollection_neg = TweetCollection()
    # tweetCollection_neg.gather_tweets_stanford(count=100,filename=filename,label=kNegTweet)
    # tweetCollection_all = TweetCollection()
    # tweetCollection_all.gather_tweets_stanford(count=4000,filename=filename)
    # fdist = Utils.get_frequency_distribution(tweetCollection_all.generate_nltk_text(1))
    # features = [feature[0] for feature in fdist.most_common(20)]
    # nbController = Controller.NaiveBayesController(tweetCollection_all,features)
    # print(nbController.prediction_accuracy(tweetCollection_neg))
    # tweetCollection_neg.gather_tweets_stanford(count=800,filename=filename)
    # print(nbController.prediction_accuracy(tweetCollection_neg))
    training_data = TweetCollection()
    training_data.gather_tweets_stanford(count=100,filename=filename)
    features = FeatureExtractor(tweetCollection=training_data).get_feature_set_NB()
    feature = features[0][0]
    print(feature)
    nbClassifier = NaiveBayesClassifier()
    nbClassifier.train(features)
    print(nbClassifier.predict(feature))



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


davidMain()