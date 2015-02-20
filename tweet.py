# -*- coding: utf-8 -*-

import Utils
import Controller

#Defined constants
kNegTweet = -1
kPosTweet = 1


class Tweet:
    """Single Tweet Object"""
    def __init__(self, tweet='', label=0):
        self.tweet = tweet.rstrip()
        self.label = label
        self.tokens = Utils.preprocess_tweet_noStem(self.tweet)


    def get_label(self):
        return self.label

    def get_tweet(self):
        return self.tweet

    def get_stemmed_tweets(self):
        tokens = Utils.preprocess_tweet_stem(self.tweet)
        return tokens

    def get_tweet_tokens(self):
        return self.tokens

    def __str__(self):
        return "Tweet: {0}\nLabel: {1}".format(self.tweet,str(self.label))

class TweetCollection:
    """
    Collection of tweets object
    Use this class to manage more than 1 tweet
    """
    def __init__(self,tweets=[]):
        self.tweets = tweets #List of Tweet Object

    def add_tweets(self,tweets):
        if len(tweets) == 0:
            print('TweetCollection: parameter tweets is empty')
        else:
            #Add Tweets that match the labels
            self.tweets.extend(tweets)

    def get_tweets(self):
        return self.tweets

    def gather_tweets_stanford(self,count,filename, label=0):
        '''
        :param label: Tweet Label to gather, if label=0 gather from both labels
        :param count: How many tweets
        :param filename: Name of file to import from
        :return: Tweet Collection with len=count, and label=label
        '''
        self.tweets = []
        tweetCell = -1
        labelCell = 0
        #Other Python can't decode directly, need to specify (utf-8 breaks)
        with open(filename + '.csv',encoding='latin1') as csv_file:
            if label == kNegTweet:
                random_index = Utils._obtain_index(count=count)
                csv_reader = [line.split(',') for line in list(csv_file)[:800000]]
            elif label == kPosTweet:
                random_index = Utils._obtain_index(count=count)
                csv_reader = [line.split(',') for line in list(csv_file)[800001:]]
            else:
                random_index = Utils._obtain_index(count=count,both=1)
                csv_reader = [line.split(',') for line in list(csv_file)] #leave first and last 100k for test
            try:
                for i,row in enumerate(csv_reader):
                    if i in random_index:
                        self.tweets.append(Tweet(row[tweetCell], row[labelCell]))
                        if len(self.tweets) == count:
                            break
            except UnicodeDecodeError as err:
                print("UnicodeDecodeError:{0}".format(err.reason))
            # except:
            #     print("UnexpectedError:", sys.exc_info()[0])

    def generate_nltk_text(self,stopword=0):
        '''
        This function generate nltk.Text object from all the tweets in the collection
        :return: nltk.Text Object
        '''
        return Utils.generate_text_object(self.tweets,stopword)



def test():
    filename='training.1600000.processed.noemoticon'
    tweetCollection_neg = TweetCollection()
    tweetCollection_neg.gather_tweets_stanford(count=100,filename=filename,label=kNegTweet)
    tweetCollection_all = TweetCollection()
    tweetCollection_all.gather_tweets_stanford(count=4000,filename=filename)
    fdist = Utils.get_frequency_distribution(tweetCollection_all.generate_nltk_text(1))
    features = [feature[0] for feature in fdist.most_common(20)]
    nbController = Controller.NaiveBayesController(tweetCollection_all,features)
    print(nbController.prediction_accuracy(tweetCollection_neg))
    tweetCollection_neg.gather_tweets_stanford(count=800,filename=filename)
    print(nbController.prediction_accuracy(tweetCollection_neg))



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


test()


