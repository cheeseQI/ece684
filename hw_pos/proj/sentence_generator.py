import format_worker
import pandas as pd
import random
import numpy as np
import format_worker

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
rows = [[]]


def generate_dataset():
    for i in range(200):
        rows.append(generate_sentence())
    format_worker.rewrite_file(format_worker.csv_synthetic_filename, rows)


def generate_sentence():
    labels = ['neg', 'pos']
    pos_words = pd.read_csv(format_worker.csv_pos_word_filename, header=None)
    neg_words = pd.read_csv(format_worker.csv_neg_word_filename, header=None)
    num_words = np.random.randint(50, 100)

    sentiment = np.random.choice(labels)
    words = pos_words if (sentiment == 'pos') else neg_words
    words = list(np.concatenate(words.values.tolist()).flat)
    sentence = ''
    for i in range(num_words):
        sentence += np.random.choice(words) + ' '
    return [sentence, sentiment]


def main():
    generate_dataset()


if __name__ == "__main__":
    main()
