import time

from sklearn.metrics import accuracy_score
import format_worker
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# from torchtext.legacy import data
# from torchtext.legacy import datasets
import numpy as np
from sklearn.naive_bayes import MultinomialNB
import pandas as pd


def get_top_2000_words(count_data, count_vect, label_data):
    words = count_vect.get_feature_names_out()
    # freq array for vocab
    pos_counts = np.zeros(len(words))
    neg_counts = np.zeros(len(words))
    for i in range(count_data.shape[0]):
        if label_data[i] == 'pos':
            pos_counts += count_data[i].toarray()[0]
        else:
            neg_counts += count_data[i].toarray()[0]

    pos_count_dict = zip(words, pos_counts)
    pos_count_dict = sorted(pos_count_dict, key=lambda x: x[1], reverse=True)[:2000]
    pos_words = [w[0] for w in pos_count_dict]

    neg_count_dict = zip(words, neg_counts)
    neg_count_dict = sorted(neg_count_dict, key=lambda x: x[1], reverse=True)[:2000]  # delete this ?
    neg_words = [w[0] for w in neg_count_dict]
    return pos_words, neg_words


def remove_intersection(pos_words, neg_words):
    return list(set(pos_words) - set(neg_words)), list(set(neg_words) - set(pos_words))


def train_by_NB(x_train_count, y_train, x_test_count, y_test):
    start_time = round(time.time() * 1000)
    mnb = MultinomialNB()
    mnb.fit(x_train_count, y_train)
    p2 = mnb.predict(x_test_count)
    s2 = accuracy_score(y_test, p2)
    print("Multinomial Naive Bayes Classifier Accuracy :", "{:.2f}%".format(100 * s2))
    end_time = round(time.time() * 1000)
    print(f'Epoch Time: {end_time - start_time}ms')
    # plot_confusion_matrix(mnb, x_test_count, y_test, cmap='Blues')
    # plt.grid(False)
    # plt.show()

def readFromFile(is_real=True):
    test = pd.read_csv(format_worker.csv_test_filename)
    if is_real:
        return pd.read_csv(format_worker.csv_train_filename), test
    else:
        return pd.read_csv(format_worker.csv_synthetic_filename), test

def main():
    train, test = readFromFile(True)
    x_train, y_train = train['text'], train['label']
    x_test, y_test = test['text'], test['label']

    stop_sign = ['very', 'ourselves', 'am', 'through', 'me',
                 'just', 'her', 'ours', 'because', 'is', 'it', 'only',
                 'in', 'such', 'too', 'their', 'if', 'to', 'my',
                 'himself', 'after', 'why', 'while', 'can', 'each', 'itself', 'his', 'all',
                 'once', 'herself', 'more', 'our', 'they', 'on', 'ma', 'them',
                 'its', 'where', 'did', 'you', 'as', 'now', 'before', 'people', 'also',
                 'those', 'yours', 'from', 'who', 'was', 'm', 'been', 'will', 'into', 'film', 'films',
                 'same', 'how', 'some', 'of', 'out', 'with', 's', 'being', 'might', 'watch',
                 'she', 'again', 'be', 'by', 'shan', 'have', 'yourselves', 'and', 'character', 'characters',
                 'are', 'o', 'these', 'further', 'most', 'yourself', 'having', 'are', 'think',
                 'here', 'he', 'were', 'but', 'this', 'myself', 'own', 'we', 'so', 'i',
                 'does', 'both', 'when', 'between', 'd', 'had', 'the', 'y', 'has', 'down',
                 'off', 'than', 'whom', 'should', 've', 'movie', 'film', 'see', 'seen', 'could',
                 'themselves', 'few', 'then', 'what', 'until', 'no', 'about', 'don', 'think',
                 'any', 'that', 'for', 'shouldn', 'do', 'there', 'doing', 'an', 'or',
                 'hers', 'was', 'were', 'above', 'a', 'at', 'your', 'theirs', 'movies',
                 'other', 're', 'him', 'during', 'which', 'br', 'even', 'not', 'would', 'really', 'one']
    count_vect = CountVectorizer(analyzer='word', ngram_range=(1, 2),
                                 stop_words=stop_sign)  # using uni-gram and bi-gram
    x_train_count = count_vect.fit_transform(x_train)
    x_test_count = count_vect.transform(x_test)
    pos_words, neg_words = get_top_2000_words(x_train_count, count_vect, y_train)
    pos_words, neg_words = remove_intersection(pos_words, neg_words)
    f_pos = pd.DataFrame(pos_words)
    # save data of distribution
    f_pos.to_csv(format_worker.csv_pos_word_filename, header=False, index=False)
    f_neg = pd.DataFrame(neg_words)
    f_neg.to_csv(format_worker.csv_neg_word_filename, header=False, index=False)
    train_by_NB(x_train_count, y_train, x_test_count, y_test)

    # use synthetic data from generated sentence
    train, test = readFromFile(False)
    x_train, y_train = train['text'], train['label']
    x_test, y_test = test['text'], test['label']
    x_train_count = count_vect.fit_transform(x_train)
    x_test_count = count_vect.transform(x_test)
    train_by_NB(x_train_count, y_train, x_test_count, y_test)

if __name__ == "__main__":
    main()
