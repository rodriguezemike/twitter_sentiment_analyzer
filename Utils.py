from nltk import PorterStemmer as PorterStemmer
from nltk import Text
from nltk import FreqDist
from nltk.corpus import stopwords as _stopwords
from re import search as regexpSearch
from string import punctuation as punctuationList
from random import randint as randomInteger

# Utilities module:

#constants
stopwords = _stopwords.words('english')
stopwords.extend(["i'm","it's",'im','its']) #these tokens are common to both positive and negative

def word_tokenize(toTokenize):
    #Split by space
    _tokens = toTokenize.lower().split(' ')
    #Strip punctuation (leading and trailing)
    return [token.strip(punctuationList) for token in _tokens]

def word_stemmer(tokens):
    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in tokens]

def exclude_stopwords(tokens):
    return [token for token in tokens if token not in stopwords]

def remove_punctuation(tokens):
    #remove token that has only punctuations
    #different than strip from word_tokenize()
    return [token for token in tokens if regexpSearch(r'\w',token)]

def preprocess_tweet_noStem(tweetString):
    '''
    :param tweetString: String
    :return: preprocessed token
    Processing Rule:
    - Split tweetString into tokens
    - Exclude Stopwords
    - Remove Punctuation
    '''
    #tokenize (nltk default tokenizer, Treebank tokenizer)
    _tokens = word_tokenize(tweetString)
    _tokens = exclude_stopwords(_tokens)
    _tokens = remove_punctuation(_tokens)
    return _tokens


def preprocess_tweet_stem(tweetString):
    '''
    :param tweetString: String
    :return: preprocessed token
    Processing Rule:
    - Split tweetString into tokens
    - Exclude Stopwords
    - Remove Punctuation
    - Normalize token using PorterStemmer Algorithm from nltk
    '''
    _tokens = word_tokenize(tweetString)
    _tokens = exclude_stopwords(_tokens)
    _tokens = remove_punctuation(_tokens)
    _tokens = word_stemmer(_tokens)
    return _tokens

def _obtain_index(count,both=0):
    random_index = []
    if both == 1:
        while len(random_index) < count/2:
            randint = randomInteger(0,800000)
            if randint not in random_index:
                random_index.append(randint)
        while len(random_index) < count:
            randint = randomInteger(800001,1600000)
            if randint not in random_index:
                random_index.append(randint)
    else:
        while len(random_index) < count:
            randint = randomInteger(0,800000)
            if randint not in random_index:
                random_index.append(randint)

    return sorted(random_index)

def generate_text_object(tokens, stopword=0):
    _tokens = []
    if stopword == 1:
        for tweet in tokens:
            _tokens.extend(tweet.get_tweet_tokens())
    else:
        for tweet in tokens:
            _tokens.extend(tweet.get_tweet_tokens())
    return Text(_tokens)

def get_frequency_distribution(text):
    return FreqDist(text)

























