# -*- coding: utf-8 -*-

import Utils
import Controller

#Defined constants
kNegTweet = -1
kPosTweet = 1

class TweetCollection:
    """
    Collection of tweets object
    Use this class to manage more than 1 tweet
    """
    def __init__(self,tweets=[]):
        self.tweets = tweets #List of Tweet Object
    #Lexicon    
        self.lexicon = Utils.build_Sentiment_Dictionary("positive-words.txt","negative-words.txt")
    # stopword lists         
        self.stopwords_classic = Utils.stopwords
        #Stopword lists generated after tweet collection is made
        #Set with set_stopword_lists
        self.stopwords_tfh = []
        self.stopwords_tf1 = []
        self.stopwords_tfid = []
        
    def add_tweets(self,tweets):
        if len(tweets) == 0:
            print('TweetCollection: parameter tweets is empty')
        else:
            #Add Tweets that match the labels
            self.tweets.extend(tweets)

    def set_stopword_lists(self,ngrams = 1):
        self.stopwords_tf1=Utils.tf1_SWGenerator(self.tweets,ngrams)
        self.stopwords_tfh=Utils.tfh_SWGenerator(self.tweets,ngrams)
        self.stopwords_tfid=Utils.tfid_SWGenerator(self.tweets,ngrams) 

    def get_tweets(self):
        return self.tweets    
    def get_lexicon(self):
        return self.lexicon        
    def get_stopwords_classic(self):
        return self.stopwords_classic    
    def get_stopwords_tf1(self):
        return self.stopwords_tf1    
    def get_topwords_tfh(self):
        return self.stopwords_tfh
    def get_stopwords_tfid(self):
        return self.stopwords_tfid
    
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
        #csv_reader = [line.split(',') for line in csv_file][:80000]
        
        with open(filename + '.csv',encoding='latin1') as csv_file:
            if label == kNegTweet:
                random_index = Utils._obtain_index(count=count)
                csv_reader = [line.split(',') for line in csv_file][:800000]
            elif label == kPosTweet:
                random_index = Utils._obtain_index(count=count)
                csv_reader = [line.split(',') for line in csv_file][800001:]
            else:
                random_index = Utils._obtain_index(count=count,both=1)
                csv_reader = [line.split(',') for line in csv_file] #leave first and last 100k for test
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


    def generate_nltk_text(self,stopword = 0):
        '''
        This function generate nltk.Text object from all the tweets in the collection
        :return: nltk.Text Object
        '''
        return Utils.generate_text_object(self.tweets,stopword)



class Tweet:
    """Single Tweet Object"""
    def __init__(self, tweet='', label=0):
        self.tweet = tweet.rstrip()
        self.length = len(self.tweet)
        self.label = label
        self.tokens = Utils.preprocess_tweet(self.tweet)
        self.stemmed_tokens = Utils.preprocess_tweet(self.tweet,stemming = 1)
        self.bigrams = Utils.bigrams(self.tokens)
        self.trigrams = Utils.trigrams(self.tokens)
        self.pos_tags = Utils.tweet_pos_tags(self.tokens)
        self.pos_tags_count = Utils.count_pos_tags(self.pos_tags)
        self.hashtag_count = Utils.count_hashtags(self.tweet)
        self.negation_count = Utils.count_negation(self.tweet)
        self.uppercase_count = Utils.count_uppercase(self.tweet)
        self.lexicon_score = 99 #No sentiment is ever 99 but we should try to find a default value. 0 = neutral and -1 mean neg
        self.lexicon_label = "NO_LABEL"
        self.features = self.generate_features()

#Feature set support for nltk    
    def generate_features(self):
        featureset = {}
        featureset["tweet"] = self.tweet
        featureset["label"] = self.label
        featureset["length"] = self.length
        featureset["tokens"] = self.tokens
        featureset["stemmed_tokens"] = self.stemmed_tokens
        featureset["bigrams"] = self.bigrams
        featureset["trigrams"] = self.trigrams
        featureset["pos_tags"] = self.pos_tags
        featureset["pos_tags_count"] = self.pos_tags_count
        featureset["hashtag_count"] = self.hashtag_count
        featureset["negation_count"] = self.negation_count
        featureset["uppercase_count"] = self.uppercase_count
        featureset["lexicon_score"] = self.lexicon_score
        featureset["lexicon_label"] = self.lexicon_label      
        return featureset

    def set_tweet_lex_score(self):
        self.lexicon_sore = Utils.uni_LS_score(self.tokens,TweetCollection.get_lexicon())
    def set_tweet_lex_label(self):
        self.lexicon_label = Utils.get_lex_label(self.lexicon_score)
        
    def get_tweet_length(self):
        return self.length        
    def get_label(self):
        return self.label
    def get_tweet(self):
        return self.tweet
    def get_stemmed_tweets(self):
        return self.stemmed_tokens
    def get_tweet_tokens(self):
        return self.tokens
    def get_tweet_bigrams(self):
        return self.bigrams        
    def get_tweet_trigrams(self):
        return self.trigrams    
    def get_tweet_pos_tags(self):
        return self.pos_tags    
    def get_tweet_pos_tags_count(self):
        return self.pos_tags_count    
    def get_tweet_uppercase_count(self):
        return self.uppercase_count 
    def get_tweet_hashtag_count(self):
        return self.hashtag_count
    def get_tweet_negation_count(self):
        return self.tweet_negation_count
    def get_tweet_lexicon_score(self):
        return self.lexicon_score
    def get_tweet_lexicon_label(self):
        return self.lexicon_label
    def get_features(self):
        return self.features
        
    def __str__(self):
        return "Tweet: {0}\nLabel: {1}".format(self.tweet,str(self.label))

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


