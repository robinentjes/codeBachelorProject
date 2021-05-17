import math
from typing import List, Tuple
import numpy as np
import argparse
import readrw, readmen, readSimLex, readWordSim
from zipfile import ZipFile
import io

def bernoulli(n: int, s: int) -> float:
    return math.factorial(n) / (math.factorial(s) * math.factorial(n - s)) * (0.5 ** n)


def calculate_bernoulli_prob(A, B, true) -> float:
    n, s = [0, 0]

    for idx in range(len(A)):
        # We only take the test items into account where the scores are different.
        if A[idx] != B[idx]:
            n += 1

        # We assume success to be the cases where the scores of A is closer to the true value
        if np.abs(A[idx] - true[idx]) < np.abs(B[idx] - true[idx]):
            s += 1

    print(f"Number of examples with different semantic similarity score: {n}")
    print(f"Number of examples where A performs better than B (so closer to the true score): {s}")

    return 2 * np.sum([bernoulli(n, x) for x in np.arange(s, n + 1)]).round(4)

# returns hamming distance. if one of the words is not found, -1 is returned
def getHamming(word1, word2, embtype):
    idx1 = word2idx.get(word1)
    idx2 = word2idx.get(word2)
    if idx1 and idx2:
        if embtype == 'NLL':
            emb1 = embeddingsNLL[idx1].astype(np.uint)
            emb2 = embeddingsNLL[idx2].astype(np.uint)
            return sum(emb1 ^ emb2)
        else:
            emb1 = embeddingsSTE[idx1].astype(np.uint)
            emb2 = embeddingsSTE[idx2].astype(np.uint)
            return sum(emb1 ^ emb2)
    return -1


if __name__ == "__main__":

    #  TODO:  Write code such that the actual scores are loaded, these are just random input
    number_items = 100

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help = "The dataset you would like to use for evaluation (rw, men, simlex, wordsim)")
    parser.add_argument('vectorsize', type=int, help = 'The size of the embeddings you want to use (128, 256, 512)')
    args = parser.parse_args()

    linesNLL = []
    # choose correct files (both nll and ste for a certain vectorsize)
    # first load the nll
    zipFileNLL = "nll" + str(args.vectorsize) + '.zip'
    txtNLL = "nll" + str(args.vectorsize) + '.txt'

    # read the lines NLL
    with ZipFile(zipFileNLL) as zf:
        with io.TextIOWrapper(zf.open(txtNLL), encoding="utf-8") as f:
            linesNLL = f.readlines()

    linesSTE = []
    # then load the STE
    zipFileSTE = "ste" + str(args.vectorsize) + '.zip'
    txtSTE = "ste" + str(args.vectorsize) + '.txt'

    # read the lines STE
    with ZipFile(zipFileSTE) as zf:
        with io.TextIOWrapper(zf.open(txtSTE), encoding="utf-8") as f:
            linesSTE = f.readlines()

    words = []
    embeddingsNLL: List[np.array] = []
    for i, line in enumerate(linesNLL):
        array = line.split()
        word = array[0]
        words.append(word)
        embedding = np.array(list(map(float, array[1:])))
        embeddingsNLL.append(embedding)

    embeddingsSTE: List[np.array] = []
    for i, line in enumerate(linesSTE):
        array = line.split()
        embedding = np.array(list(map(float, array[1:])))
        embeddingsSTE.append(embedding)

    # used for convenience
    word2idx = {word: idx for idx, word in enumerate(words)}
    idx2word = {idx: word for word, idx in word2idx.items()}

    # select the data sets with the wordpairs for evaluation
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

    #scores_true = np.random.uniform(0, 1, number_items)  # The ground truth word similarity scores
    #scores_A = scores_true + np.random.normal(0, 0.3, number_items)  # Similarity scores of classifier A
    #scores_B = np.random.uniform(0, 1, number_items)  # Similarity scores of classifier B
    idx = 0
    scoresNLL = []
    scoresSTE = []
    scoresTrue = []
    for wordpair in wordpairs:
        if idx == 100:
            break

        word1 = wordpair[0]
        word2 = wordpair[1]
        realScore = wordpair[2] / 10 # divided by 10 to get a score between 0 and 1
        hammingNLL = getHamming(word1, word2, 'NLL')
        if hammingNLL == -1:
            continue

        scoreNLL = 1 - (hammingNLL/args.vectorsize)
        scoreSTE = 1 - (getHamming(word1, word2, 'STE')/args.vectorsize)
        scoresTrue.append(realScore)
        scoresNLL.append(scoreNLL)
        scoresSTE.append(scoreSTE)

        #print(word1 + " - " + word2)
        #print("true: " + str(realScore) + " - NLL: " + str(scoreNLL) + " - STE: " + str(scoreSTE))
        idx = idx + 1

    p_value = calculate_bernoulli_prob(scoresSTE, scoresNLL, scoresTrue)
    print(p_value)

    """
    Description: This creates 3 arrays. 1 with the "true" scores, which in our case is number_items numbers ranging
    between 0 and 1. In this test case we assume that A is a little bit better than B so we use the same scores for A
    and add some noise. the scores for B are again a random. Example output:

    Number of examples with different semantic similarity score: 100
    Number of examples where A performs better than B (so closer to the true score): 63
    0.012

    This means that the probability of A performing just as well as B given that A performs 63 times better than B
    in 100 test cases is 0.012 (two-sided). So if you take a P value of 0.01 this is not significant but if you take
    p = 0.05 this is significant.
    """
