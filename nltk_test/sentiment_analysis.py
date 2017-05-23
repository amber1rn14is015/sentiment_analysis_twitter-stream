import nltk
import numpy as np
import random
import pickle
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.tokenize import word_tokenize
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode


class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf


document_pickle = open('./saved_documents.pickle','rb')
documents = pickle.load(document_pickle)
document_pickle.close()

random.shuffle(documents)

all_words_pickle = open('./saved_all_words.pickle','rb')
all_words = pickle.load(all_words_pickle)
all_words_pickle.close()

word_feature_pickle = open('./saved_word_feature.pickle','rb')
word_features = pickle.load(word_feature_pickle)
word_feature_pickle.close()

def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features


featureset = [(find_features(rev), category) for (rev, category) in documents]
random.shuffle(featureset)
#print(len(featureset))
training_set = featureset[:10100]
testing_set = featureset[10100:]

classifier_pickle = open('./saved_original_classifier.pickle', 'rb')
classifier = pickle.load(classifier_pickle)
classifier_pickle.close()

# MultinomialNB, GaussianNB, BernoulliNB
MNB_classifier_pickle = open('./saved_original_classifier.pickle', 'rb')
MNB_classifier = pickle.load(MNB_classifier_pickle)
MNB_classifier_pickle.close()


BernoulliNB_classifier_pickle = open('./saved_BernoulliNB_classifier.pickle', 'rb')
BernoulliNB_classifier = pickle.load(BernoulliNB_classifier_pickle)
BernoulliNB_classifier_pickle.close()

# LogisticRegression, SGDClassifier
# SVC, LinearSVC, NuSVC
LogisticRegression_classifier_pickle = open('./saved_LogisticRegression_classifier.pickle', 'rb')
LogisticRegression_classifier = pickle.load(LogisticRegression_classifier_pickle)
LogisticRegression_classifier_pickle.close()

LinearSVC_classifier_pickle = open('./saved_LinearSVC_classifier.pickle', 'rb')
LinearSVC_classifier = pickle.load(LinearSVC_classifier_pickle)
LinearSVC_classifier_pickle.close()

SGDClassifier_classifier_pickle = open('./saved_SGDClassifier_classifier.pickle','rb')
SGDClassifier_classifier = pickle.load(SGDClassifier_classifier_pickle)
SGDClassifier_classifier_pickle.close()

NuSVC_classifier_pickle = open('./saved_NuSVC_classifier.pickle','rb')
NuSVC_classifier = pickle.load(NuSVC_classifier_pickle)
NuSVC_classifier_pickle.close()


voted_classifier = VoteClassifier(classifier,
                                  MNB_classifier,
                                  BernoulliNB_classifier,
                                  LogisticRegression_classifier,
                                  LinearSVC_classifier,
                                  SGDClassifier_classifier,
                                  NuSVC_classifier)


def sentiment(text):
    feats = find_features(text)
    return voted_classifier.classify(feats),voted_classifier.confidence(feats)
