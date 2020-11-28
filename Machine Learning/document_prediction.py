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

vectorizer = joblib.load(sys.argv[1])
k_selector = joblib.load(sys.argv[2])
svd = joblib.load(sys.argv[3])
clf = joblib.load(sys.argv[4])

document_query = [
    "viva la revolucion ciudadana",
    'empleo y libertad para todos',
    'defendamos el legado de rafael correa',
    'libre emprendimiento para los ecuatorianos',
    'Gracias a Arauz #Ecuador ya no tendrá que depender del dólar. Serán soberanos',
    'Importante que todos lean esto! Muchos aman al dólar y no creen que Arauz desdolarizará y nos mandará de vuelta al mundo de la inflación descontrolada y la miseria. #DesdolarizacionNO #Desdolarauz',
    'El candidato títere de Correa, se da cuenta que su propuesta de “desdolarización amigable” es rechazada por lo ecuatorianos, y borra, esconde sus opiniones. Para que no olviden, aquí una parte. #Desdolarauz',
    'Rafael Correa inició el camino de la Revolución Ciudadana, la traición de Lenin Moreno será solo un mal recuerdo, con @ecuarauz se viene la Revolución de la Revolución para mejorar la vida',
    'Se está formando un Frente de Apoyo a la Universidad de las Artes. Una de la universidades emblemáticas creadas por la Revolución Ciudadana. Hoy me reuní con jóvenes egresados. Todo el apoyo para estudiantes, egresados y profesores. ¡Apoyemos la Universidad de las Artes!',
    'te amo rafael correa'
]

document_query_filtered = list(map(process,document_query))
document_vec = vectorizer.transform(document_query_filtered)
document_k  = k_selector.transform(document_vec)
document_reduced = svd.transform(document_k)
label_predicted = clf.predict(document_reduced)

for i,x in enumerate(document_query_filtered):
    print(x,label_predicted[i])