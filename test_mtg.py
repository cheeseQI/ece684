"""Markov Text Generator.

Patrick Wang, 2021

Resources:
Jelinek 1985 "Markov Source Modeling of Text Generation"
"""

import nltk

from mtg import finish_sentence


def test_generator():
    # nltk.download('gutenberg')
    # nltk.download('punkt')
    """Test Markov text generator."""
    corpus = nltk.word_tokenize(nltk.corpus.gutenberg.raw("austen-sense.txt").lower())
    #          eg1  n=3           #
    # deterministic
    words = finish_sentence(
        ["she", "was", "not"],
        3,
        corpus,
        deterministic=True,
    )
    print(words)
    assert words == ["she", "was", "not", "in", "the", "world", "."] or words == [
        "she",
        "was",
        "not",
        "in",
        "the",
        "world",
        ",",
        "and",
        "the",
        "two",
    ]
    # stochastic
    words = finish_sentence(
        ["she", "was", "not"],
        3,
        corpus,
        deterministic=False,
    )
    print(words)
    #       eg2 n=2 with same beginning "
    words = finish_sentence(
        ["she", "was", "not"],
        2,
        corpus,
        deterministic=True,
    )
    print(words)
    words = finish_sentence(
        ["she", "was", "not"],
        2,
        corpus,
        deterministic=False,
    )
    print(words)
    #         eg4  n=5           #
    words = finish_sentence(
        ["you", "always", "say", "somthing", "like", "that"],
        5,
        corpus,
        deterministic=True,
    )
    print(words)
    words = finish_sentence(
        ["you", "always", "say", "somthing", "like"],
        5,
        corpus,
        deterministic=False,
    )
    print(words)
    #          eg5  n=10           #
    words = finish_sentence(
        ["I", "am", "convinced", "within", "myself","that","your","father","had"],
        10,
        corpus,
        deterministic=True,
    )
    print(words)
    words = finish_sentence(
        ["I", "am", "convinced", "within", "myself", "that", "your", "father", "had"],
        10,
        corpus,
        deterministic=False,
    )
    print(words)
if __name__ == "__main__":
    test_generator()
