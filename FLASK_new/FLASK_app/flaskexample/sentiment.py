from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from flaskexample.utils import write_status, top_n_words
import numpy as np
from gensim.models import Word2Vec
import pandas as pd


FREQ_DIST_FILE = './flaskexample/static/data/train-structured-service-word2vec-freqdist.pkl'
BI_FREQ_DIST_FILE = './flaskexample/static/data/train-structured-service-word2vec-freqdist-bi.pk'

def get_glove_vectors(vocab):
    # print('Looking for GLOVE vectors')
    w2v_model = Word2Vec.load("./flaskexample/static/data/word2vec-100d.model")

    glove_vectors = {}
    found = 0
    # with open(GLOVE_FILE, 'r') as glove_file:
    for word in w2v_model.wv.vocab:
        #       utils.write_status(i + 1, 0)
        # tokens = line.split()
        # word = tokens[0]
        if vocab.get(word):
            vector = w2v_model.wv[word]
            glove_vectors[word] = vector
            found += 1
    print('\n')
    print('Found %d words in GLOVE' % found)
    return glove_vectors


def get_feature_vector(tweet):
    words = tweet.split()
    feature_vector = []
    for i in range(len(words) - 1):
        word = words[i]
        if vocab.get(word) is not None:
            feature_vector.append(vocab.get(word))
    if len(words) >= 1:
        if vocab.get(words[-1]) is not None:
            feature_vector.append(vocab.get(words[-1]))
    return feature_vector


def process_tweets(service_df,vocab):
    def get_feature_vector(tweet):
            words = tweet.split()
            feature_vector = []
            for i in range(len(words) - 1):
                word = words[i]
                if vocab.get(word) is not None:
                    feature_vector.append(vocab.get(word))
            if len(words) >= 1:
                if vocab.get(words[-1]) is not None:
                    feature_vector.append(vocab.get(words[-1]))
            return feature_vector

    tweets = []
    print('Generating feature vectors')

    total = len(service_df)
    for i in range(total):
        tweet = service_df.iloc[i,2]
        feature_vector = get_feature_vector(tweet)
        tweets.append(feature_vector)
        write_status(i + 1, total)
    return tweets


def pred_sentiment(lda_file_name):
    df = pd.read_csv(lda_file_name)

    np.random.seed(1337)
    vocab_size = 90000
    batch_size = 500
    max_length = 40
    filters = 600
    kernel_size = 3

    vocab = top_n_words(FREQ_DIST_FILE, vocab_size, shift=1)
    glove_vectors = get_glove_vectors(vocab)

    model_service = load_model('./flaskexample/static/data/service-lstm-weighted.h5')
    model_room = load_model('./flaskexample/static/data/lstm-weighted-room.h5')
    model_loc = load_model('./flaskexample/static/data/lstm-weighted-location.h5')

    test_tweets = process_tweets(df,vocab)
    test_tweets = pad_sequences(test_tweets, maxlen=max_length, padding='post')
    
    predictions_service = model_service.predict(test_tweets, batch_size=128, verbose=1)
    labels_service = np.round(predictions_service[:, 0]).astype(int)
    prob_service = predictions_service[:, 0]
    predictions_room = model_room.predict(test_tweets, batch_size=128, verbose=1)
    labels_room = np.round(predictions_room[:, 0]).astype(int)
    prob_room = predictions_room[:, 0]
    predictions_loc = model_loc.predict(test_tweets, batch_size=128, verbose=1)
    labels_loc = np.round(predictions_loc[:, 0]).astype(int)
    prob_loc = predictions_loc[:, 0]

    df['prob_service'] = prob_service
    df['prob_room'] = prob_room
    df['prob_loc'] = prob_loc

    df['pred_service'] = np.multiply(df['service'].values,labels_service)
    df['pred_room'] = np.multiply(df['room'].values,labels_service)
    df['pred_loc'] = np.multiply(df['loc'].values,labels_service)
    df.to_csv('./flaskexample/static/data/results.csv')
    return df

