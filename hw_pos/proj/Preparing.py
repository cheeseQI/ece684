import time
import torch
from torchtext.legacy import data
from torchtext.legacy import datasets
import random
import numpy as np

# choose the training and testing data
SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# tokenized
TEXT = data.Field(tokenize='spacy',
                  tokenizer_language='en_core_web_sm',
                  batch_first=True)
LABEL = data.LabelField(dtype=torch.float)

train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)

train_data, valid_data = train_data.split(random_state=random.seed(SEED))

# build word vocab
MAX_VOCAB_SIZE = 2500

TEXT.build_vocab(train_data, max_size=MAX_VOCAB_SIZE)

LABEL.build_vocab(train_data)

# calculate the label vector for training
Y = np.array(
    [
        LABEL.vocab.stoi[data.label]
        for data in train_data.examples
    ]
)

# the label vector for testing
Y_test = np.array(
    [
        LABEL.vocab.stoi[data.label]
        for data in test_data.examples
    ]
)

# the raw count of words in each sentence
X = np.array(
    [
        [
            data.text.count(text)
            for text in TEXT.vocab.stoi
        ]
        for data in train_data.examples
    ]
)

# calculate the positive and negative count of words
neg_idx = []
pos_idx = []

for i in range(len(Y)):
    if Y[i] == 0:
        neg_idx.append(i)
    else:
        pos_idx.append(i)

pos_sum = []
neg_sum = []

for i in range(len(TEXT.vocab.stoi)):
    psum = 0
    nsum = 0
    for j in range(len(pos_idx)):
        psum += X[pos_idx[j]][i]
    for k in range(len(neg_idx)):
        nsum += X[neg_idx[k]][i]
    pos_sum.append(psum)
    neg_sum.append(nsum)

P = np.array(pos_sum)
N = np.array(neg_sum)

# the whole count of each word
freq = np.array(
    [
        TEXT.vocab.freqs[text] + 1
        for text in TEXT.vocab.stoi
    ]
)

# the word vector of points
Z = (P - N) / freq

# the word and points' dictionary
text_dict = dict(zip(TEXT.vocab.stoi, Z))


# function to calculate the score of the sentence
def count_score(sentence, t_dict):
    score = 0
    for t in sentence:
        if t not in t_dict.keys():
            t = "<unk>"
        score += t_dict[t]
    return score


if __name__ == '__main__':
    start_time = round(time.time() * 1000)
    ans = []
    for data in test_data.examples:
        s = count_score(data.text, text_dict)
        if s > 0:
            ans.append(1)
        else:
            ans.append(0)

    correct = (ans == Y_test)
    end_time = round(time.time() * 1000)
    acc = float(correct.sum()) / len(correct)

    print(f'acc: {acc:.3f} | Time: {end_time - start_time}ms')
