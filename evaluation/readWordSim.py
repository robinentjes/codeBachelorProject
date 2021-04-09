import csv

# read MEN
def readWordSimSet():
    with open('wordsim353/combined.csv') as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=',')
        wordpairs = []
        for row in csv_reader:
            wordpair = (row['Word 1'], row['Word 2'], float(row['Human (mean)']))
            wordpairs.append(wordpair)

    return wordpairs
