# the binary embeddings needs to be read here
# the datasets for evaluation needs to be read here (MEN, rw, SimLex, wordsim)
# output must be in form word1 word2 score hammingdistance
from typing import List, Tuple
import numpy as np
from scipy import stats, spatial
import readrw, readmen, readSimLex, readWordSim
import argparse
from zipfile import ZipFile
import io

parser = argparse.ArgumentParser()

# TODO: adding flags instead of the arguments index makes it more readable, see:
parser.add_argument('dataset', help = "The dataset you would like to use for evaluation (rw, men, simlex, wordsim)")

args = parser.parse_args()

if __name__ == "__main__":

    # TODO: Please make a small script that retrieves the embeddings and puts it in the correct file
    file = open("../glove.6b/glove.6B.300d.txt", encoding="utf8")
    lines = file.readlines()
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
    similarities = []
    #loop over wordpairs
    for wordpair in wordpairs:
        idx1 = word2idx.get(wordpair[0])
        idx2 = word2idx.get(wordpair[1])
        if idx1 and idx2:
            emb1 = embeddings[idx1].astype(np.float)
            emb2 = embeddings[idx2].astype(np.float)
            cossim = 1 - spatial.distance.cosine(emb1, emb2)
            scores.append(wordpair[2])
            similarities.append(cossim)

    print(stats.spearmanr(scores, similarities))
