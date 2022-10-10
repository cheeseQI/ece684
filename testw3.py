"""Bigram model example.

ECE 684, lecture 6 (language models)
Patrick Wang, 2022
"""
import random
import re

import matplotlib.pyplot as plt
import nltk
import numpy as np


def main():
    """Generate example bigram model."""
    # get corpus text by word
    words = nltk.corpus.gutenberg.words("austen-emma.txt")

    # reduce to lowercase letters only
    words = [w for word in words if re.search("^[a-z]+$", w := word.lower())]

    # build vocabulary
    characters = "".join(words)
    vocabulary = sorted(set(characters))
    token_indices = {char: idx for idx, char in enumerate(vocabulary)}

    vocabulary += [" "]
    vocab_size = len(vocabulary)

    # build bigram table
    bigram_table = np.zeros((vocab_size, vocab_size))
    for word in words:
        for idx in range(len(word) - 1):
            # transition from letter to letter
            bigram_table[token_indices[word[idx]], token_indices[word[idx + 1]]] += 1

        # transition from word break to letter
        bigram_table[-1, token_indices[word[0]]] += 1
        # transition from letter to word break
        bigram_table[token_indices[word[-1]], -1] += 1

    # normalize
    bigram_table = bigram_table / np.sum(bigram_table, axis=1, keepdims=True)

    # evaluate word probability
    word = "quiet"
    # p = p("q") * p("u" | "q") * p("i" | "u") * ...
    p = bigram_table[-1, token_indices[word[0]]]
    for first, second in zip(word[:-1], word[1:]):
        p *= bigram_table[
            token_indices[first],
            token_indices[second],
        ]
    print(p)

    # generate text
    prefix = "q"
    for _ in range(20):
        if prefix[-1] != " ":
            dist = bigram_table[token_indices[prefix[-1]]]
        else:
            dist = bigram_table[-1]
        next_char = random.choices(vocabulary, dist)
        prefix += next_char[0]
        # next_idx = dist.argmax()
        # prefix += vocabulary[next_idx]
    print(prefix)

    # display model
    _, axis = plt.subplots(figsize=(8, 8), dpi=80)
    im = axis.imshow(bigram_table)
    axis.set_title("English character bigram model")
    axis.set_xticks(range(vocab_size))
    axis.set_xticklabels(vocabulary)
    axis.set_yticks(range(vocab_size))
    axis.set_yticklabels(vocabulary)
    plt.colorbar(im)
    plt.show()


if __name__ == "__main__":
    main()