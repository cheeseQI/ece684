"""Tests for POS tagging."""
import nltk
from hw_pos.infer import infer

nltk.download('brown')
nltk.download('universal_tagset')

def test_viterbi():
    infer()


if __name__ == "__main__":
    test_viterbi()
