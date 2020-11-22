import tweepy
import os
import json


class FileStorage():

    def __init__(self, filename):
        self.filename = filename

    def read(self):
        if os.path.exists(self.filename):
            with open(self.filename) as file:
                data = json.load(file)
                return data
        else:
            return {}

    def save(self, data):  # write data like json file
        try:
            old_data = self.read()
            if len(old_data.keys()) == 0:
                old_data["tweets"] = []
            old_data["tweets"].append(data)
            jsondata = json.dumps(old_data, indent=4, skipkeys=True, sort_keys=True)
            fd = open(self.filename, 'w')
            fd.write(jsondata)
            fd.close()
            print(self.filename + " ha sido escrito exitosamente")
        except Exception as e:
            print(e)
            print('ERROR writing', self.filename)


consumer_key = " "
consumer_secret = " "
access_token = " "
access_token_secret = " "

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

file_worker = FileStorage("search.json")
tweets = []
q = "@PlayStation"
num_tweets = 100
for status in tweepy.Cursor(api.search, q=q, tweet_mode="extended").items(num_tweets):
    print(status._json["full_text"])
    tweets.append(status._json)
print(len(tweets))  # print number of tweets
file_worker.save(tweets)