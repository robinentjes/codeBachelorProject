# the binary embeddings needs to be read here
# the datasets for evaluation needs to be read here (MEN, rw, SimLex, wordsim)
# output must be in form word1 word2 score hammingdistance
from typing import List, Tuple
import numpy as np
from scipy import stats
import readrw, readmen, readSimLex, readWordSim
import argparse
from zipfile import ZipFile
import io

parser = argparse.ArgumentParser()
parser.add_argument('dataset', help = "The dataset you would like to use for evaluation (rw, men, simlex, wordsim)")
parser.add_argument('vectorsize', type=int, help = 'The size of the embeddings you want to use (128, 256, 512)')
parser.add_argument('type', help = "Type of encoder that was used for creating binary embeddings (nll, ste)")
args = parser.parse_args()

# returns hamming distance. if one of the words is not found, -1 is returned
def getHamming(word1, word2):
    idx1 = word2idx.get(word1)
    idx2 = word2idx.get(word2)
    if idx1 and idx2:
        emb1 = embeddings[idx1].astype(np.uint)
        emb2 = embeddings[idx2].astype(np.uint)
        return sum(emb1 ^ emb2)
    return -1

if __name__ == "__main__":

    lines = []

    zipFileName = args.type + str(args.vectorsize) + '.zip'
    txtName = args.type + str(args.vectorsize) + '.txt'

    with ZipFile(zipFileName) as zf:
        with io.TextIOWrapper(zf.open(txtName), encoding="utf-8") as f:
            lines = f.readlines()
    words = []
    embeddings: List[np.array] = []
    for i, line in enumerate(lines):
        array = line.split()
        word = array[0]
        words.append(word)
        embedding = np.array(list(map(float, array[1:])))
        embeddings.append(embedding)

    word2idx = {word: idx for idx, word in enumerate(words)}
    idx2word = {idx: word for word, idx in word2idx.items()}

    wordpairs = []
    if args.dataset == 'rw':
        wordpairs = readrw.readRWset()
    elif args.dataset == 'men':
        wordpairs = readmen.readMENset()
    elif args.dataset == 'simlex':
        wordpairs = readSimLex.readSimLexSet()
    elif args.dataset == 'wordsim':
        wordpairs = readWordSim.readWordSimSet()
    else:
        print("no good evaluation set given; choose from: rw, men, simlex, wordsim")

    scores = []
    distances = []
    #loop over wordpairs
    for wordpair in wordpairs:
        distance = getHamming(wordpair[0], wordpair[1])
        if distance == -1:
            continue

        scores.append(wordpair[2])
        distances.append(1 - distance/args.vectorsize)

    print(stats.spearmanr(scores, distances))
