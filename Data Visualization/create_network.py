import os
import json
import itertools
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



file_worker = FileStorage("buenos_aires.json")
data = file_worker.read()

hashtags_count = {}


for tweet in data['tweets']:

    tweet_hashtags = tweet['entities']['hashtags']
    hashtags_list = [hashtag['text'].lower() for hashtag in tweet_hashtags]
    hashtags_list.sort()
    hashtags_comb = list(itertools.combinations(hashtags_list, 2))
    for comb in hashtags_comb:
        if comb not in hashtags_count:
            hashtags_count[comb] = 1
        else:
            hashtags_count[comb] += 1

f = open("hashtags_network.csv", "w", encoding='UTF-8')
f.write('source,target,weight\n')
for comb in hashtags_count:
    count = str(hashtags_count[comb])
    hashtag1 = comb[0]
    hashtag2 = comb[1]
    print(hashtag1, hashtag2, count)
    f.write(hashtag1 + ',' + hashtag2 + ',' + count + '\n')
f.close()


