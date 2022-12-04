from sklearn.feature_extraction.text import TfidfVectorizer
import random
import torch
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB

# set up for deterministic results -> why we want deterministic? #
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

def tfidfModel(train_data, test_data):
    print('start feed data!')
    print(f"Number of training examples: {len(train_data)}")
    print(f"Number of testing examples: {len(test_data)}")
    print(vars(train_data.examples[0]))
    tfidf_vect = TfidfVectorizer()  # tfidfVectorizer

    Xtrain, ytrain = train_data['text'], train_data['label']
    Xtest, ytest = test_data['text'], test_data['label']
    Xtrain_tfidf = tfidf_vect.fit_transform(Xtrain)
    Xtest_tfidf = tfidf_vect.transform(Xtest)

    # count_vect = CountVectorizer()  # CountVectorizer
    # Xtrain_count = count_vect.fit_transform(train_data)
    # Xtest_count = count_vect.transform(test_data)
    # mnb = MultinomialNB()
    # mnb.fit(Xtrain_tfidf, ytrain)
    # p2 = mnb.predict(Xtest_tfidf)
    # s2 = accuracy_score(ytest, p2)
    # print("Multinomial Naive Bayes Classifier Accuracy :", "{:.2f}%".format(100 * s2))
    # plot_confusion_matrix(mnb, Xtest_tfidf, ytest, cmap='Blues')
    # plt.grid(False)