import nltk
import numpy as np

from hw_pos.build_tables import build_tables
from hw_pos.viterbi import viterbi


def infer():
    """Infer the result."""
    a, b, pi_, word_indices, tag_indices = build_tables();
    # generate decoder
    decoder = {}
    for key in tag_indices:
        decoder[tag_indices[key]] = key
    # pi_i = probability of starting at state i
    pi = np.array(pi_)
    # a_{ij} = probability of transitioning from state i to state j
    A = np.array(a)
    B = np.array(b)
    # coder
    lines = nltk.corpus.brown.tagged_sents(tagset='universal')[10150:10153]
    obs = []
    for line in lines:
        for word_tuple in line:
            word = word_tuple[0].lower()
            if word not in word_indices:
                code = len(b) - 1
            else:
                code = word_indices[word]
            obs.append(code)
    states_guess, prob = viterbi(obs, pi, A, B)
    origin = ""
    for line in lines:
        for word_tuple in line:
            origin += word_tuple[1] + " "

    s = ""
    for i in states_guess:
        s += decoder[i] + " "
    print(tag_indices)
    print(origin)
    print(s)
    print("the probability is : " + str(prob))