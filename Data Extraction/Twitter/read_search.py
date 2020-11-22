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


file_worker = FileStorage("search.json")
data = file_worker.read()
print(data)