# -*- coding: utf-8 -*-
"""Text Mining - Find collocations.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Ue6AOwbgTl1wFx04A1KhS_KS2zq03-c-
"""

#remove punctuactions
import re

text = "Larga vida al bolón mixto de maduro de Tere. En la vida siempre están los que se quedan criticando y los que dan el.. https://t.co/4AfTqVPVRq"
text_filtered = text.lower()
print(text_filtered)

import re
text_filtered = re.sub(r"(https?)\S+","",text_filtered)#remove urls
text_filtered = re.sub(r"(\B#)\w*","",text_filtered)#remove hashtags
text_filtered = re.sub(r"(\B@)\w*","",text_filtered)#remove mentions
text_filtered = re.sub("\n","",text_filtered)#remove lines separate
print(text_filtered)

#remove accents
import unicodedata
text_filtered = unicodedata.normalize('NFKD', text_filtered).encode('ASCII', 'ignore')#remove accents
print(text_filtered)

text_filtered = re.sub('[^a-zA-Z ]+', ' ', text_filtered.decode('ASCII'))#remove punctuactions
print(text_filtered)

#remove stop words
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords

def filter_stop_words(text):
    stop_words_list = stopwords.words('spanish')
    stop_words_list += stopwords.words('english')
    text_filtered = [word for word in text.split() if word.lower() not in stop_words_list]
    return text_filtered

tokens = filter_stop_words(text_filtered)
print(tokens)

# import these modules 
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize 

def apply_stemmer(text):
    stemmer = SnowballStemmer('spanish')
    words = []
    
    for w in text: 
        word_stemmed = stemmer.stem(w)
        words.append(word_stemmed)

    return words

# choose some words to be stemmed 
tokens_stemmed = apply_stemmer(tokens)
print(tokens_stemmed)

# import these modules 
from nltk.stem import WordNetLemmatizer 
  
lemmatizer = WordNetLemmatizer() 
  
print("rocks :", lemmatizer.lemmatize("rocks")) 
print("corpora :", lemmatizer.lemmatize("corpora")) 
# a denotes adjective in "pos" 
print("better :", lemmatizer.lemmatize("better", pos ="a"))

import os
import json
import re
import sys

def process(tweet):
    tweet = tweet.lower()
    tweet = re.sub(r"(https?)\S+","",tweet)#remove urls
    #tweet = re.sub(r"(\B#)\w*","",tweet)#remove hashtags
    tweet = re.sub(r"(\B@)\w*","",tweet)#remove mentions
    tweet = re.sub("\n","",tweet)#remove lines separate
    tweet = unicodedata.normalize('NFKD', tweet).encode('ASCII', 'ignore')#remove accents
    tweet = re.sub('[^a-zA-Z ]+', ' ', tweet.decode('ASCII'))#remove punctuactions
    text_no_stop_words = filter_stop_words(tweet)
    tokens = [token for token in text_no_stop_words if len(token) > 2 ]
    tweet = " ".join(tokens)
    return tweet

process(text_filtered)

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

file_worker =  FileStorage("E:\Programs Files\JetBrains\PyCharm\PycharmProjects\DataAnalytics-using-MachineLearning\Data Extraction\Twitter\TIMELINE.json")
data = file_worker.read()
tokens = []

for tweet in data["tweets"][0]:
    texto = tweet["full_text"]
    line = "%s\n"%(process(texto))
    tokens += line.split()

#print(tokens)

from nltk import FreqDist
freq = FreqDist(tokens)
freq.plot(20, cumulative=False)

from nltk.collocations import BigramCollocationFinder

def generate_collocations(tokens,score_function,n_top):
    '''
    Given list of tokens, return collocations.
    '''
    finder = BigramCollocationFinder.from_words(tokens, window_size = 2)
    finder.apply_word_filter(lambda w: len(w) < 3)
    finder.apply_freq_filter(1)    
    colls = finder.score_ngrams(score_function)
    #colls = finder.nbest(score_function, n_top)

    return colls

n_top = 20
bigram_measures = nltk.collocations.BigramAssocMeasures()
collocations = generate_collocations(tokens,bigram_measures.raw_freq,n_top)
for col in collocations:
    print(col)

n_top = 20
bigram_measures = nltk.collocations.BigramAssocMeasures()
collocations = generate_collocations(tokens,bigram_measures.likelihood_ratio,n_top)
for col in collocations:
    print(col)

from nltk.collocations import TrigramCollocationFinder

def generate_trigrams(tokens,score_function,n_top):
    '''
    Given list of tokens, return collocations.
    '''
    finder = TrigramCollocationFinder.from_words(tokens, window_size = 3)
    finder.apply_word_filter(lambda w: len(w) < 3)
    finder.apply_freq_filter(1)

    #colls = finder.nbest(score_function, n_top)
    colls = finder.score_ngrams(score_function)
    
    return colls

trigram_measures = nltk.collocations.TrigramAssocMeasures()
collocations = generate_trigrams(tokens,trigram_measures.likelihood_ratio,n_top)
for col in collocations:
    print(col)
