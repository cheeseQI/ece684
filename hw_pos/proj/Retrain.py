import time
import Generate
import Preparing
import numpy as np

# used to regenerate dictionary of the word and points
X_new = np.array(
    [
        [
            data.count(text)
            for text in Preparing.TEXT.vocab.stoi
        ]
        for data in Generate.sentences
    ]
)

neg_idx = []
pos_idx = []

for i in range(Generate.len_sentences):
    if Generate.labels[i] == 0:
        neg_idx.append(i)
    else:
        pos_idx.append(i)

pos_sum = []
neg_sum = []

for i in range(len(Preparing.TEXT.vocab.stoi)):
    psum = 0
    nsum = 0
    for j in range(len(pos_idx)):
        psum += X_new[pos_idx[j]][i]
    for k in range(len(neg_idx)):
        nsum += X_new[neg_idx[k]][i]
    pos_sum.append(psum)
    neg_sum.append(nsum)

P = np.array(pos_sum)
N = np.array(neg_sum)

freq = P + N + 1

Z = (P - N) / freq

new_dict = dict(zip(Preparing.TEXT.vocab.stoi, Z))

if __name__ == '__main__':
    start_time = round(time.time() * 1000)
    ans = []
    for data in Preparing.test_data.examples:
        s = Preparing.count_score(data.text, new_dict)
        if s > 0:
            ans.append(1)
        else:
            ans.append(0)

    correct = (ans == Preparing.Y_test)
    end_time = round(time.time() * 1000)
    acc = float(correct.sum()) / len(correct)

    print(f'acc: {acc:.3f} | Time: {end_time - start_time}ms')