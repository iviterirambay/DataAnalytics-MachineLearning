# -*- coding: utf-8 -*-
"""Analisis Geografico.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1m_xXAxbGfD2bNDtm0Oe9RPYz5BshPY-X
"""
#!pip install numpy scipy scikit-learn==0.19.2 spanish_sentiment_analysis ipython
import os
import json
import re
import sys
import unicodedata
import nltk
from nltk.corpus import stopwords
from classifier import *

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

def process(tweet):
    tweet = tweet.lower()
    tweet = re.sub(r"(https?)\S+","",tweet)#remove urls
    tweet = re.sub(r"(\B#)\w*","",tweet)#remove hashtags
    tweet = re.sub(r"(\B@)\w*","",tweet)#remove mentions
    tweet = re.sub("\n","",tweet)#remove lines separate
    tweet = unicodedata.normalize('NFKD', tweet).encode('ASCII', 'ignore')#remove accents
    tweet = re.sub('[^a-zA-Z ]+', ' ', tweet.decode('ASCII'))#remove punctuactions
    text_no_stop_words = filter_stop_words(tweet)
    tokens = [token for token in text_no_stop_words if len(token) > 2 ]
    tweet = " ".join(tokens)
    return tweet




def filter_stop_words(text):
    stop_words_list = stopwords.words('spanish')
    stop_words_list += stopwords.words('english')
    text_filtered = [word for word in text.split() if word.lower() not in stop_words_list]
    return text_filtered

file_worker = FileStorage("buenos_aires.json")
data = file_worker.read()
print(data['tweets'][10])

c = 0
clf = SentimentClassifier()

latitudes = []
longitudes = []
ids = []
created_at = []
users = []
rts = []
favs = []
texts = []
sent_scores = []
followers = []
links = []
for tweet in data['tweets']:
    if type(tweet['coordinates']) == dict:
        lat = tweet['coordinates']['coordinates'][1]
        lon = tweet['coordinates']['coordinates'][0]
        fecha_hora = tweet['created_at']
        id_ = tweet['id_str']
        text = tweet['text']
        text = process(text)
        favorite_count = tweet['favorite_count']
        retweet_count = tweet['retweet_count']
        followers_count = tweet['user']['followers_count']
        screen_name = tweet['user']['screen_name']
        link = 'https://twitter.com/' + screen_name + '/status/' + id_
        score = clf.predict(text)
        
        latitudes.append(lat)
        longitudes.append(lon)
        ids.append(id_)
        created_at.append(fecha_hora)
        texts.append(text)
        favs.append(favorite_count)
        rts.append(retweet_count)
        followers.append(followers_count)
        users.append(screen_name)
        links.append(link)
        sent_scores.append(score)
        c+=1

f = open('tweets_coordinates.csv', 'w')
f.write('id,created_at,latitude,longitude,user_screen_name,retweets_count,favorites_count,text,followers_count,link,sentiment_score\n')
for i in range(c):
    f.write(str(ids[i]) + ',' + created_at[i] + ',' + str(latitudes[i]) + ',' + str(longitudes[i]) + ',' + users[i] + ',' + str(rts[i]) + ',' + str(favs[i]) + ',' + texts[i] + ',' + str(followers[i]) + ',' + links[i] + ',' + str(sent_scores[i]) +'\n')                   
f.close()

users[i] + ',' + rts[i] + ',' + favs[i] + ',' + texts[i] + ',' + followers[i] + ',' + links[i] + ',' + str(sent_scores[i]) +'\n'



