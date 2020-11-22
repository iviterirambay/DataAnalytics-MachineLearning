from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import os
import json


class StdOutListener(StreamListener):
    """ A listener handles tweets that are received from the stream.
    This is a basic listener that just prints received tweets to stdout.
    """

    def __init__(self, filename):
        self.filename = filename

    def on_error(self, status):
        print(status)

    def read_tweets(self):
        if os.path.exists(self.filename):
            with open(self.filename) as file:
                data = json.load(file)
                return data
        else:
            return {}

    def save_tweet(self, tweet):
        try:
            old_data = self.read_tweets()
            if len(old_data.keys()) == 0:
                old_data["tweets"] = []
            old_data["tweets"].append(tweet)
            jsondata = json.dumps(old_data, indent=4, skipkeys=True, sort_keys=True)
            fd = open(self.filename, 'w')
            fd.write(jsondata)
            fd.close()
            print(self.filename + " ha sido escrito exitosamente")
        except Exception as e:
            print(e)
            print('ERROR writing', self.filename)

    def on_data(self, data):
        tweet = json.loads(data)
        print(tweet['text'])
        self.save_tweet(tweet)
        return True


consumer_key = " "
consumer_secret = " "
access_token = " "
access_token_secret = " "

y = -79.876094  # Rectangle y
w = -2.15984  # Rectangle width(which you created with name of x2)

h = -79.849229  # Rectangle height(which you created with name of y2)
x = -2.093846  # Rectangle x


class StdOutListener(StreamListener):
    """ A listener handles tweets that are received from the stream.
    This is a basic listener that just prints received tweets to stdout.
    """

    def __init__(self, filename):
        self.filename = filename

    def on_error(self, status):
        print(status)

    def read_tweets(self):
        if os.path.exists(self.filename):
            with open(self.filename) as file:
                data = json.load(file)
                return data
        else:
            return {}

    def save_tweet(self, tweet):
        try:
            old_data = self.read_tweets()
            if len(old_data.keys()) == 0:
                old_data["tweets"] = []
            old_data["tweets"].append(tweet)
            jsondata = json.dumps(old_data, indent=4, skipkeys=True, sort_keys=True)
            fd = open(self.filename, 'w')
            fd.write(jsondata)
            fd.close()
            print(self.filename + " ha sido escrito exitosamente")
        except Exception as e:
            print(e)
            print('ERROR writing', self.filename)

    def on_data(self, data):
        tweet = json.loads(data)
        print(tweet['text'])
        self.save_tweet(tweet)
        return True


listener = StdOutListener("results.json")

auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
stream = Stream(auth, listener)
stream.filter(locations=[y, w, h, x])