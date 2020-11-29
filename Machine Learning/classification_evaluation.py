from __future__ import absolute_import, print_function

import os
import json
import re
import unicodedata
import sys
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from nltk.corpus import stopwords
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from nltk.corpus import stopwords
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.model_selection import KFold,StratifiedKFold
import joblib


class FileStorage():

    def __init__(self,filename):
        self.filename = filename

    def read(self):
        if os.path.exists(self.filename):
            with open(self.filename) as file:
                data = json.load(file)
                return data
        else:
            return {}
        
    def save(self,data):#write data like json file
        try:
            old_data = self.read()
            if len(old_data.keys()) == 0:
                old_data["tweets"] = []
            old_data["tweets"].append(data)
            jsondata = json.dumps(old_data, indent=4, skipkeys=True, sort_keys=True)
            fd = open(self.filename, 'w')
            fd.write(jsondata)
            fd.close()
            print (self.filename + " ha sido escrito exitosamente")
        except Exception as e:
            print (e)
            print ('ERROR writing', self.filename)


def filter_stop_words(text):
    stop_words_list = stopwords.words('spanish')
    stop_words_list += stopwords.words('english')
    text_filtered = [word for word in text.split() if word.lower() not in stop_words_list]
    return text_filtered


def process(tweet):
    tweet = tweet.lower()
    tweet = re.sub(r"(https?)\S+","",tweet)#remove urls
    #tweet = re.sub(r"(\B#)\w*","",tweet)#remove hashtags
    #tweet = re.sub(r"(\B@)\w*","",tweet)#remove mentions
    tweet = re.sub("\n","",tweet)#remove lines separate
    tweet = unicodedata.normalize('NFKD', tweet).encode('ASCII', 'ignore')#remove accents
    tweet = re.sub('[^a-zA-Z ]+', ' ', tweet.decode('ASCII'))#remove punctuactions
    text_no_stop_words = filter_stop_words(tweet)
    tokens = [token for token in text_no_stop_words if len(token) > 2 ]
    tweet = " ".join(tokens)
    return tweet

def load_corpus(filename):
    file_worker = FileStorage(filename)
    data = file_worker.read()
    corpus = []
    raw_tweets = []
    for tweet in data["tweets"]:
        if 'full_text' in tweet.keys():
            text = tweet["full_text"]
        else:
            text = tweet["text"]
        text_filtered = process(text)
        corpus.append(text_filtered)
        raw_tweets.append(text)

    return corpus,raw_tweets


corpus,raw_tweets = load_corpus("ecuarauz.json")

vectorizer = joblib.load("Models\Vectorizer_model.pkl")
k_selector = joblib.load("Models\k_best_model.pkl")
svd = joblib.load("Models\svd_model.pkl")
clf = joblib.load("Models\clf_model.pkl")

#python3 document_prediction.py


document_query_filtered = list(map(process,corpus))
document_vec = vectorizer.transform(document_query_filtered)
document_k  = k_selector.transform(document_vec)
document_reduced = svd.transform(document_k)
label_predicted = clf.predict_proba(document_reduced)

share = 0
threshold = 0.7#umbral
n_docs = len(corpus)

for i,predictions in enumerate(label_predicted):
    print(raw_tweets[i],predictions)
    if (predictions[1] > threshold):
        share += 1
    
print("Nivel de aceptacion: ", 100*share/n_docs )
