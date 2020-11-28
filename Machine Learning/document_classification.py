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

    for tweet in data["tweets"]:
        if 'full_text' in tweet.keys():
            text = tweet["full_text"]
        else:
            text = tweet["text"]
        text_filtered = process(text)
        corpus.append(text_filtered)

    return corpus


#python3 document_classification.py 

filename = "Twitter/ecuarauz.json"
filename_right = "Twitter/LassoGuillermo.json"

left_corpus = load_corpus(filename)
n_docs = len(left_corpus)
print("Documentos de texto de la izquierda ",n_docs)

right_corpus = load_corpus(filename_right)
n_docs_right = len(right_corpus)
print("Documentos de texto de la derecha ",n_docs_right)

y_labels = [ 0 for i in range(0,n_docs) ] #'left'

z_labels = [ 1 for i in range(0,n_docs_right) ] #'right'

corpus = left_corpus + right_corpus

vectorizer = TfidfVectorizer(analyzer='word',ngram_range=(1,3),max_features=1000)
document_term = vectorizer.fit_transform(corpus)
joblib.dump(vectorizer,"./models/vectorizer_model.pkl")

y = y_labels + z_labels

from sklearn.feature_selection import SelectKBest, chi2

k_selector = SelectKBest(chi2, k=500)
document_term_k_best = k_selector.fit_transform(document_term.toarray(), y)
joblib.dump(k_selector,"./models/k_best_model.pkl")

print(document_term_k_best.shape)

n_components = 4
svd = TruncatedSVD(n_components)
X_reduced = svd.fit_transform(document_term_k_best)
print(X_reduced.shape)
joblib.dump(svd,"./models/svd_model.pkl")

features_name = [ "pc_" + str(i) for i in range(0,n_components) ]
df = pd.DataFrame(data=X_reduced,columns=features_name)
df["label"] = y
g = sns.PairGrid(df,hue='label')
g.map(sns.scatterplot)
plt.show()

x_data = df.iloc[:,0:-1]
y_data = df["label"]

C = 1.0 #SVM regularization parameter
kf = StratifiedKFold(n_splits=20,shuffle=True)

clfs = []
scores = []

for i,(train_index, test_index) in enumerate(kf.split(x_data,y_data)):
    X_train, X_test = x_data.iloc[train_index], x_data.iloc[test_index]
    Y_train, Y_test = y_data.iloc[train_index], y_data.iloc[test_index]
    clf = svm.SVC(kernel='linear', C=C)
    clf.fit(X_train,Y_train)
    score = clf.score(X_test, Y_test)
    clfs.append(clf)
    scores.append(score)

best_accuracy = np.argsort(scores)[::-1][0]
clf = clfs[best_accuracy]

joblib.dump(clf,"./models/clf_model.pkl")


