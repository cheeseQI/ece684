import Preparing
import random
import pandas as pd

# used to generate sentences
len_sentences = 1000
sentences = []
labels = []
new_sentences = []
new_labels = []
for x in range(len_sentences):
    # randomly pick words for sentences in 10 to 20
    sentence_len = random.randrange(10, 20)
    sentence = random.choices(list(Preparing.text_dict.keys()), k=sentence_len)
    s = Preparing.count_score(sentence, Preparing.text_dict)
    sentences.append(sentence)
    new_sentences.append(" ".join(sentence))
    if s > 0:
        labels.append(1)
        new_labels.append("pos")
    else:
        labels.append(0)
        new_labels.append("neg")

if __name__ == '__main__':
    dataframe = pd.DataFrame({'text':new_sentences, 'label':new_labels})
    dataframe.to_csv("doc.csv", index=False, sep=',')
