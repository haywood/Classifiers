#!/usr/bin/python

from __future__ import print_function, division
from itertools import combinations, chain
from numpy import *
import random
import sys

from bayes_classifier import bayes_classifier
from nn_classifier import nn_classifier
from myio import *

classifiers = {
        'bayes':bayes_classifier,
        'nn':nn_classifier
        };

if __name__ == '__main__':

    train_file = sys.argv[1];
    feature_file = sys.argv[2];
    test_file = sys.argv[3];
    train_file_out = sys.argv[4];
    test_file_out = sys.argv[5];
    class_type = sys.argv[6];
    per_class = 5;

    sample_labels, samples = read_hands(train_file);
    indices = [];

    with open(feature_file) as data:
        line = data.readline().split();
        line.pop(0);
        indices = [int(f) for f in line];
    indices = [i for i in range(len(samples[0]))];
    print(indices);

    original_samples = samples[:];
    samples = [sample[indices] for sample in samples];
    n_samples, n_features = shape(samples);
    classifier = classifiers[class_type](n_features);
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
    print('%s got %f' % (class_type, correct/len(test_samples)), '%d of %d' % (correct, len(test_samples)));

    output = open(train_file_out, 'w');
    for i in range(n_samples):
        line = ' '.join(str(f) for f in original_samples[i]);
        prediction = classifier.predict(samples[i]);
        print(prediction, line, file=output);
    output.close();

    test_labels, test_samples = read_hands(test_file);
    n_test_samples = shape(test_samples)[0];
    output = open(test_file_out, 'w');
    for i in range(n_test_samples):
        line = ' '.join(str(f) for f in test_samples[i]);
        prediction = classifier.predict(test_samples[i][indices]);
        print(prediction, line, file=output);
    output.close();
