from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import metrics
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn import svm
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn import neighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import random

def parse_dataset(fp):
    y = []
    corpus = []
    with open(fp, 'rt') as data_in:
        for line in data_in:
            line = line.rstrip()
            label = int(line.split("\t")[0])
            text = line.split("\t")[1]
            y.append(label)
            corpus.append(text)
    return corpus, y


def featurize(corpus):
    tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True).tokenize
    #vectorizer = CountVectorizer(strip_accents="unicode", analyzer="word", tokenizer=tokenizer, stop_words="english")
    vectorizer = TfidfVectorizer(strip_accents="unicode", analyzer="word", tokenizer=tokenizer, stop_words="english")
    X = vectorizer.fit_transform(corpus)
    #print(vectorizer.get_feature_names()) # to manually check if the tokens are reasonable
    return X

if __name__ == "__main__":

    DATASET_FP = "./slightly_unbalanced_data"

    K_FOLDS = 5 # N-fold crossvalidation
    
    models=[svm.SVC(kernel='linear'),
            svm.SVC(kernel='poly'),
            svm.SVC(kernel='rbf'),
            svm.SVC(kernel='sigmoid'),
            MLPClassifier(solver='lbfgs'),
            MLPClassifier(solver='adam'),
            BernoulliNB(),
            MultinomialNB(),
            LogisticRegression(),
            tree.DecisionTreeClassifier(),
            KNeighborsClassifier()]

    corpus, y = parse_dataset(DATASET_FP)
    X = featurize(corpus)

    class_counts = np.asarray(np.unique(y, return_counts=True)).T.tolist()
    #print class_counts

    for m in models:
        predicted = cross_val_predict(m, X, y, cv=K_FOLDS)
        score = metrics.f1_score(y, predicted, pos_label=1)
        print "F1-score Task", score

