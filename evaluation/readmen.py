# read MEN
def readMENset():
    file = open('MEN/MEN_dataset_natural_form_full', "r", encoding="utf8")
    lines = file.readlines()

    wordpairs = []
    for line in lines:
        array = line.split()
        wordpair = (array[0], array[1], float(array[2]))
        wordpairs.append(wordpair)
    file.close()

    return wordpairs
