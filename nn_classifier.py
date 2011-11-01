#!/usr/bin/python

from __future__ import print_function, division
from numpy.linalg import norm
from numpy import *

class nn_classifier(object):

    def __init__(self, dim, k=1):
        self._examples = ();
        self._dim = dim;
        self._k = k;

    @property
    def dim(self):
        return self._dim;

    @property
    def examples(self):
        return self._examples;

    def train(self, classes):
        self._examples = ();
        for label in classes:
            for example in classes[label]:
                self._examples = self._examples + ((example, label),);

    def predict(self, x):
        return min(self.examples, key = lambda e: norm(e[0] - x))[1];
