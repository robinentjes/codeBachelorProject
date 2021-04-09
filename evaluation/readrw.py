import re

# read rw
def readRWset():
    rwfile = open('rw/rw.txt', "r", encoding="utf8")
    lines = rwfile.readlines()

    wordpairs = []
    for line in lines:
        array = re.split(r'\t+', line)
        wordpair = (array[0], array[1], float(array[2]))
        wordpairs.append(wordpair)
    rwfile.close()
    
    return wordpairs
