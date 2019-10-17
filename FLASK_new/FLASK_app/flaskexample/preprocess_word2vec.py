import re
import sys
from flaskexample.utils import write_status
import nltk
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize, sent_tokenize
from langdetect import detect
import pandas as pd
import spacy
from spacy.lemmatizer import Lemmatizer

lemmatizer = Lemmatizer()
nlp = spacy.load('en')

def preprocess_word(word):
    # Remove punctuation
    word = word.strip('\'",():;')
    # Convert more than 2 letter repetitions to 2 letter
    # funnnnny --> funny
    word = re.sub(r'(.)\1+', r'\1\1', word)
    # Remove - & '
    word = re.sub(r'(-|\')', '', word)
    return word


def is_valid_word(word):
    # Check if word begins with an alphabet
    return (re.search(r'^[a-zA-Z][a-z0-9A-Z\._]*$', word) is not None)


def handle_emojis(tweet):
    # Smile -- :), : ), :-), (:, ( :, (-:, :')
    tweet = re.sub(r'(:\s?\)|:-\)|\(\s?:|\(-:|:\'\))', ' EMO_POS ', tweet)
    # Laugh -- :D, : D, :-D, xD, x-D, XD, X-D
    tweet = re.sub(r'(:\s?D|:-D|x-?D|X-?D)', ' EMO_POS ', tweet)
    # Love -- <3, :*
    tweet = re.sub(r'(<3|:\*)', ' EMO_POS ', tweet)
    # Wink -- ;-), ;), ;-D, ;D, (;,  (-;
    tweet = re.sub(r'(;-?\)|;-?D|\(-?;)', ' EMO_POS ', tweet)
    # Sad -- :-(, : (, :(, ):, )-:
    tweet = re.sub(r'(:\s?\(|:-\(|\)\s?:|\)-:)', ' EMO_NEG ', tweet)
    # Cry -- :,(, :'(, :"(
    tweet = re.sub(r'(:,\(|:\'\(|:"\()', ' EMO_NEG ', tweet)
    return tweet


def preprocess_tweet(tweet,stop_words):
    # use_stemmer = True
    use_lemmatizer = True
    # stemmer = PorterStemmer()
    processed_tweet = []
    # Convert to lower case
    tweet = tweet.lower()
    # Replaces URLs with the word URL
    tweet = re.sub(r'((www\.[\S]+)|(https?://[\S]+))', ' URL ', tweet)
    # Replace @handle with the word USER_MENTION
    tweet = re.sub(r'@[\S]+', 'USER_MENTION', tweet)
    # Replaces #hashtag with hashtag
    tweet = re.sub(r'#(\S+)', r' \1 ', tweet)
    # Remove RT (retweet)
    tweet = re.sub(r'\brt\b', '', tweet)
    # Replace 2+ dots with space
    tweet = re.sub(r'\.{2,}', ' ', tweet)
    # Remove all numbers
    tweet = re.sub('^[0-9]+', '', tweet)
    # Strip space, " and ' from tweet
    tweet = tweet.strip(' "\'')
    # Replace emojis with either EMO_POS or EMO_NEG
    tweet = handle_emojis(tweet)
    # Replace multiple spaces with a single space
    tweet = re.sub(r'\s+', ' ', tweet)

    words = tweet.split()
    lis = ''
    for word in words:
        word = preprocess_word(word)
        if is_valid_word(word):
            lis += word+' '
    lis = nlp(lis)
    for w in lis:
        if w.lemma_ == '-PRON-':
            continue
        else:
            processed_tweet.append(str(w.lemma_))
    # stopwords removal
    # https://www.geeksforgeeks.org/removing-stop-words-nltk-python/
    processed_tweet = [w for w in processed_tweet if not w in stop_words]

    return ' '.join(processed_tweet)

def preprocess_df(structured_file_name):
    processed_file_name = structured_file_name[:-4] + '-processed.csv'

    stop_words = set(stopwords.words('english'))

    with open(structured_file_name, 'r',encoding='utf-8') as csv:
        lines = csv.readlines()
        total = len(lines)
        
        df = pd.DataFrame(columns = ['org_sent', 'processed_sent'])
        for i, line in enumerate(lines):
            sents_pre = nltk.sent_tokenize(line)
            sents = []
            for sent in sents_pre:
                if 'but' in sent.lower():
                    sent = re.split("but", sent, flags=re.IGNORECASE)
                    sents += sent
                else:
                    sents.append(sent)
        
            for sent in sents:
                processed_content = preprocess_tweet(sent,stop_words)
                if len(processed_content.split())<=2:
                    continue
                if detect(processed_content) != 'en':
                    continue
                df = df.append({'org_sent': sent.strip(','), 'processed_sent': processed_content}, ignore_index=True)
            
            write_status(i + 1, total)
            
    df.to_csv(processed_file_name)
    return processed_file_name
