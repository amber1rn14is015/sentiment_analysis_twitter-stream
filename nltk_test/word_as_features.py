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
        conf = choice_votes/len(votes)
        return conf


# documents = [(list(movie_reviews.words(fileid)), category)
#             for category in movie_reviews.categories()
#             for fileid in movie_reviews.fileids(category)]
#
# random.shuffle(documents)
#
# #print(documents[1])
#
# all_words = []
# for w in movie_reviews.words():
#     all_words.append(w.lower())
#
# all_words = nltk.FreqDist(all_words)
# #print(all_words.most_common(15))
# #(all_words["journeys"])
#
# word_features = list(all_words.keys())[:3000]

# short_pos = open('short_reviews/positive.txt','r').read()
# short_neg = open('short_reviews/negative.txt','r').read()
#
# documents = []
#
# for w in short_pos.split('\n'):
#     documents.append((w,"pos"))
#
# for w in short_neg.split('\n'):
#     documents.append((w,"neg"))

document_pickle = open('saved_documents.pickle','rb')
documents = pickle.load(document_pickle)
document_pickle.close()
random.shuffle(documents)
# #pickling documents
# documents_save = open('saved_documents.pickle','wb')
# pickle.dump(documents,documents_save)
# documents_save.close()


# short_pos_words = word_tokenize(short_pos)
# short_neg_words = word_tokenize(short_neg)

# all_words = []
# for w in short_pos_words:
#     all_words.append(w.lower())
# for w in short_neg_words:
#     all_words.append(w.lower())
#
# all_words = nltk.FreqDist(all_words)
# #pickling all_words
# all_words_save = open('saved_all_words.pickle','wb')
# pickle.dump(all_words,all_words_save)
# all_words_save.close()
all_words_pickle = open('saved_all_words.pickle','rb')
all_words = pickle.load(all_words_pickle)
all_words_pickle.close()


# word_features = list(all_words.keys())[:5000]
# #pickling word_features
# word_features_save = open('saved_word_feature.pickle','wb')
# pickle.dump(word_features,word_features_save)
# word_features_save.close()
word_feature_pickle = open('saved_word_feature.pickle','rb')
word_features = pickle.load(word_feature_pickle)
word_feature_pickle.close()

def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

#print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))

featureset = [(find_features(rev), category) for (rev, category) in documents]
random.shuffle(featureset)

training_set = featureset[:10100]
testing_set = featureset[10100:]

# classifier = nltk.NaiveBayesClassifier.train(training_set)
# #pickling original nltk classifier
# classifier_save = open('saved_original_classifier.pickle','wb')
# pickle.dump(classifier,classifier_save)
# classifier_save.close()
classifier_pickle = open('saved_original_classifier.pickle','rb')
classifier = pickle.load(classifier_pickle)
classifier_pickle.close()
print("Original Naive Bayes Algo accuracy:", (nltk.classify.accuracy(classifier, testing_set))*100)
#classifier.show_most_informative_features(15)

# MultinomialNB, GaussianNB, BernoulliNB

MNB_classifier = SklearnClassifier(MultinomialNB())
# MNB_classifier.train(training_set)
# #pickling MultinomialNB sklearn classifier
# MNB_classifier_save = open('saved_original_classifier.pickle','wb')
# pickle.dump(MNB_classifier,MNB_classifier_save)
# MNB_classifier_save.close()
MNB_classifier_pickle = open('saved_original_classifier.pickle','rb')
MNB_classifier = pickle.load(MNB_classifier_pickle)
MNB_classifier_pickle.close()
print("MNB_classifier accuracy:", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)


# GaussianNB_classifier = SklearnClassifier(GaussianNB())
# GaussianNB_classifier.train(training_set)
# print("GaussianNB_classifier accuracy:", (nltk.classify.accuracy(GaussianNB_classifier, testing_set)))


BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
# BernoulliNB_classifier.train(training_set)
# #pickling BernoulliNB sklearn classifier
# BernoulliNB_classifier_save = open('saved_BernoulliNB_classifier.pickle','wb')
# pickle.dump(BernoulliNB_classifier,BernoulliNB_classifier_save)
# BernoulliNB_classifier_save.close()
BernoulliNB_classifier_pickle = open('saved_BernoulliNB_classifier.pickle','rb')
BernoulliNB_classifier = pickle.load(BernoulliNB_classifier_pickle)
BernoulliNB_classifier_pickle.close()
print("BernoulliNB_classifier accuracy:", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100)
# LogisticRegression, SGDClassifier
# SVC, LinearSVC, NuSVC

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
# LogisticRegression_classifier.train(training_set)
# #pickling original nltk classifier
# LogisticRegression_classifier_save = open('saved_LogisticRegression_classifier.pickle','wb')
# pickle.dump(LogisticRegression_classifier,LogisticRegression_classifier_save)
# LogisticRegression_classifier_save.close()
LogisticRegression_classifier_pickle = open('saved_LogisticRegression_classifier.pickle','rb')
LogisticRegression_classifier = pickle.load(LogisticRegression_classifier_pickle)
LogisticRegression_classifier_pickle.close()
print("LogisticRegression_classifier accuracy:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)


SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
# SGDClassifier_classifier.train(training_set)
# #pickling original nltk classifier
# SGDClassifier_classifier_save = open('saved_SGDClassifier_classifier.pickle','wb')
# pickle.dump(SGDClassifier_classifier,SGDClassifier_classifier_save)
# SGDClassifier_classifier_save.close()
SGDClassifier_classifier_pickle = open('saved_SGDClassifier_classifier.pickle','rb')
SGDClassifier_classifier = pickle.load(SGDClassifier_classifier_pickle)
SGDClassifier_classifier_pickle.close()
print("SGDClassifier_classifier accuracy:", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set))*100)

#Not reliable
# SVC_classifier = SklearnClassifier(SVC())
# SVC_classifier.train(training_set)
# print("SVC_classifier accuracy:", (nltk.classify.accuracy(SVC_classifier, testing_set)))


LinearSVC_classifier = SklearnClassifier(LinearSVC())
# LinearSVC_classifier.train(training_set)
# #pickling original nltk classifier
# LinearSVC_classifier_save = open('saved_LinearSVC_classifier.pickle','wb')
# pickle.dump(LinearSVC_classifier,LinearSVC_classifier_save)
# LinearSVC_classifier_save.close()
LinearSVC_classifier_pickle = open('saved_LinearSVC_classifier.pickle','rb')
LinearSVC_classifier = pickle.load(LinearSVC_classifier_pickle)
LinearSVC_classifier_pickle.close()
print("LinearSVC_classifier accuracy:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)


NuSVC_classifier = SklearnClassifier(NuSVC())
# NuSVC_classifier.train(training_set)
# #pickling original nltk classifier
# NuSVC_classifier_save = open('saved_NuSVC_classifier.pickle','wb')
# pickle.dump(NuSVC_classifier,NuSVC_classifier_save)
# NuSVC_classifier_save.close()
NuSVC_classifier_pickle = open('saved_NuSVC_classifier.pickle','rb')
NuSVC_classifier = pickle.load(NuSVC_classifier_pickle)
NuSVC_classifier_pickle.close()
print("NuSVC_classifier accuracy:", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)

voted_classifier = VoteClassifier()
# voted_classifier = VoteClassifier(classifier,
#                                   MNB_classifier,
#                                   BernoulliNB_classifier,
#                                   LogisticRegression_classifier,
#                                   SGDClassifier_classifier,
#                                   LinearSVC_classifier,
#                                   NuSVC_classifier)
# #pickling voted classifier
# voted_classifier_save = open('saved_voted_classifier.pickle','wb')
# pickle.dump(voted_classifier,voted_classifier_save)
# voted_classifier_save.close()
voted_classifier_pickle = open('saved_voted_classifier.pickle','rb')
voted_classifier = pickle.load(voted_classifier_pickle)
voted_classifier_pickle.close()
print("voted_classifier accuracy:", (nltk.classify.accuracy(voted_classifier, testing_set))*100)

print("Classification:",voted_classifier.classify(training_set[0][0]),"Confidence:",voted_classifier.confidence(training_set[0][0]))
print("Classification:",voted_classifier.classify(training_set[1][0]),"Confidence:",voted_classifier.confidence(training_set[1][0]))
print("Classification:",voted_classifier.classify(training_set[2][0]),"Confidence:",voted_classifier.confidence(training_set[2][0]))
print("Classification:",voted_classifier.classify(training_set[3][0]),"Confidence:",voted_classifier.confidence(training_set[3][0]))
print("Classification:",voted_classifier.classify(training_set[4][0]),"Confidence:",voted_classifier.confidence(training_set[4][0]))
print("Classification:",voted_classifier.classify(training_set[5][0]),"Confidence:",voted_classifier.confidence(training_set[5][0]))