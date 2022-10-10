import nltk

from mtg import finish_sentence

def test_generator():
    corpus = nltk.word_tokenize(nltk.corpus.gutenberg.raw("austen-sense.txt").lower())
    #         eg3  n=2  with different beginning      #
    words = finish_sentence(
        ["she", "was", "father"],
        4,
        corpus,
        deterministic=True,
    )
    print(words)