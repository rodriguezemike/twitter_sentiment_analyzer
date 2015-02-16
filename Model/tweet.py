# -*- coding: utf-8 -*-

import csv
from csv import reader as csvReader
from csv import Sniffer as csvSniffer
from random import randint as randomInteger
import sys
import codecs


kNegTweet = -1
kPosTweet = 1
class Tweet:
    """Single Tweet Object"""
    def __init__(self, tweet='', label=0):
        self.tweet = tweet.rstrip()
        self.label = label

    def get_label(self):
        return self.label

    def get_tweet(self):
        return self.tweet

    def __str__(self):
        return self.tweet + ':' + str(self.label)

class TweetCollection:
    """
    Collection of tweets object
    Use this class to manage more than 1 tweet
    """
    def __init__(self,tweets=[], label=0):
        self.tweets = tweets #List of Tweet Object
        self.label = label   #Label for the collection

    def add_tweets(self,tweets):
        if len(tweets) == 0:
            print('TweetCollection: parameter tweets is empty')
        else:
            #Add Tweets that match the labels
            self.tweets.extend(
                [tweet for tweet in tweets if tweet.get_label()==self.label])

    def get_tweets(self):
        return self.tweets

    def gather_tweets_stanford(self,count,filename):
        '''
        :param label: Tweet Label to gather
        :param count: How many tweets
        :param filename: Name of file to import from
        :return: Tweet Collection with len=count, and label=label
        '''
        self.tweets = []
        tweetCell = -1
        labelCell = 0
        with open(filename + '.csv',encoding='latin1') as csv_file:

            csv_dialect = csvSniffer().sniff(csv_file.read(1024))

            if self.label == kNegTweet:
                random_index = sorted(self._obtain_index(count))
                csv_reader = [line.split(',') for line in list(csv_file)[:800000]]
            elif self.label == kPosTweet:
                random_index = sorted(self._obtain_index(count))
                csv_reader = [line.split(',') for line in list(csv_file)[800001:]]
            try:
                for i,row in enumerate(csv_reader):
                    if i in random_index:
                        self.tweets.append(Tweet(row[tweetCell], row[labelCell]))
                        if len(self.tweets) == count:
                            break
                # #Collect tweets from the selected random indices
                # tweets = [row[tweetCell] for i,row in enumerate(csv_reader) if i in random_index]
                # #Add tweets to collection class
                # tweetCollection.add_tweets(tweets)
            except UnicodeDecodeError as err:
                print("UnicodeDecodeError:{0}".format(err.reason))
            except:
                print("UnexpectedError:", sys.exc_info()[0])

                # finally:
                #     print('One or more tweets could not be imported')
                #     print('Number of imported tweets: ' + str(len(tweetCollection.get_tweets(label))))

    #Helper Functions
    def _obtain_index(self,count):
        random_index = []
        while len(random_index) < count:
            randint = randomInteger(0,800000)
            if randint not in random_index:
                random_index.append(randint)

        return random_index


def test():
    tweetCollection_pos = TweetCollection(label=1)
    tweetCollection_pos.gather_tweets_stanford(100,'training.1600000.processed.noemoticon')
    print('Positive tweets:'+str(len(tweetCollection_pos.get_tweets())))
    count = 1
    for i in tweetCollection_pos.get_tweets():
        print(str(count)+str(i))
        count += 1

    count = 1
    tweetCollection_neg = TweetCollection(label=-1)
    tweetCollection_neg.gather_tweets_stanford(100,'training.1600000.processed.noemoticon')
    print('Negative tweets:' + str(len(tweetCollection_neg.get_tweets())))
    for i in tweetCollection_neg.get_tweets():
        print(str(count)+str(i))
        count+=1

test()


