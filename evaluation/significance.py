import math

import numpy as np


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


if __name__ == "__main__":

    #  TODO:  Write code such that the actual scores are loaded, these are just random input
    number_items = 100

    scores_true = np.random.uniform(0, 1, number_items)  # The ground truth word similarity scores
    scores_A = scores_true + np.random.normal(0, 0.3, number_items)  # Similarity scores of classifier A
    scores_B = np.random.uniform(0, 1, number_items)  # Similarity scores of classifier B

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

    p_value = calculate_bernoulli_prob(scores_A, scores_B, scores_true)

    print(p_value)
