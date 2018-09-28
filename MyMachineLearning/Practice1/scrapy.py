import re
import urllib.request
import json

url = 'https://api.data.gov.sg/v1/environment/pm25'
hand = urllib.request.urlopen(url).read()

jsonfile = 'middleSpace.txt'
writeJSON = open(jsonfile, 'w')

for line in hand:
    # writeJSON.write(str(line))
    print(line)


