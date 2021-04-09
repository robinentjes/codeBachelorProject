import re

# read SimLex
def readSimLexSet():
    file = open('SimLex-999/SimLex-999.txt', "r", encoding="utf8")
    lines = file.readlines()

    wordpairs = []
    first = True
    for line in lines:
        if first:
            first = False
            continue
        array = re.split(r'\t+', line)
        wordpair = (array[0], array[1], float(array[3]))
        wordpairs.append(wordpair)
    file.close()

    return wordpairs
