import gensim
import numpy as np
import nltk
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from spacy.lemmatizer import Lemmatizer
import spacy
import pandas as pd
import re


def categorize(processed_file_name):
    lemmatizer = Lemmatizer()
    nlp = spacy.load('en')
    
    pd.set_option('max_colwidth', 100)
    
    df = pd.read_csv(processed_file_name)
    corpus = []
    
    len_df = len(df)
    
    for i in range(len(df)):
        sent = nltk.sent_tokenize(df.iloc[i,1])
        sent = sent[0].lower().strip()
        sent = re.sub('-PRON-','',sent)
        sent = sent.replace('.', '')
        df.set_value(i,'processed_sent',sent)
    
    lda = gensim.models.ldamodel.LdaModel.load('./flaskexample/static/data/model5-6t.gensim')
    dictionary = gensim.corpora.Dictionary.load('./flaskexample/static/data/dictionary-6t.gensim')

    df['service'] = 0.001
    df['room'] = 0.001
    df['loc'] = 0.001
    
    lda_file_name = './flaskexample/static/data/lda-ed.csv'
    
    for i in range(len(df)):
        processed = []
        lis = nlp(df.iloc[i,1])
        for w in lis:
            processed.append(str(w.lemma_))
        bow = dictionary.doc2bow(processed)
        topics = lda.get_document_topics(bow, minimum_probability=0.0)
        if topics[4][1]>0.25 and len(processed)>1:
            df.at[i,'service'] = topics[4][1] # topics[4][1]
        if topics[5][1]>0.25 and len(processed)>1:
            df.at[i,'room'] = topics[5][1] # topics[5][1]
        if topics[2][1]>0.20 and len(processed)>1:
            df.at[i,'loc'] = topics[2][1] # topics[2][1]
    df.to_csv(lda_file_name)
    return lda_file_name
