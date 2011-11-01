#!/usr/bin/python

from __future__ import print_function, division
from itertools import combinations, chain
from numpy import *
import random
import sys

from bayes_classifier import bayes_classifier
from nn_classifier import nn_classifier
from myio import *

def classify(samples, sample_labels, per_class, classifier_type):
    n_samples, n_features = shape(samples);
    classifier = classifier_type(n_features);
    per_train_class = int(0.5*per_class);

    sample_mean = mean(samples, axis=0);
    
    train_samples = [];
    train_labels = [];
    test_samples = [];
    test_labels = [];

    for i in range(n_samples):
        if i % per_class < per_train_class:
            train_samples.append(samples[i]);
            train_labels.append(sample_labels[i]);
        else:
            test_samples.append(samples[i]);
            test_labels.append(sample_labels[i]);

    border = per_train_class;
    train_classes = {};
    while train_samples:

        train_classes[train_labels[0]] = train_samples[:border];
        train_samples = train_samples[border:];
        train_labels = train_labels[border:];

    classifier.train(train_classes);

    correct = 0;
    for i in range(len(test_samples)):
        x = test_samples[i];
        prediction = classifier.predict(x);
        if prediction == test_labels[i]:
            correct += 1;

    return correct/len(test_samples);

if __name__ == '__main__':

    train_file = sys.argv[1];
    feature_file = sys.argv[2];
    per_class = int(sys.argv[3]);

    sample_labels, samples = read_hands(train_file);

    with open(feature_file) as data:
        data.readline();
        indices = [int(i)-1 for i in data.readline().split()];
    print('Using features:', indices);

    print(classify([sample[indices] for sample in samples],
        sample_labels, per_class, bayes_classifier));

    print(classify([sample[indices] for sample in samples],
        sample_labels, per_class, nn_classifier));
