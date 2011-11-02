#!/usr/bin/python

from __future__ import print_function, division
from numpy.linalg import norm
from numpy import *

class nn_classifier(object):

    def __init__(self, dim, k=1):
        self._examples = None;
        self._mean = None;
        self._cov= None;
        self._dim = dim;
        self._k = k;

    @property
    def dim(self):
        return self._dim;

    @property
    def examples(self):
        return self._examples;

    def train(self, classes):
        samples = [];
        for k in classes: samples += classes[k];
        self._mean = mean(samples, axis=0);
        self._cov = cov(transpose(samples));
        if not self._cov.shape:
            self._cov.shape = (1, 1);
        self._examples = ();
        for label in classes:
            for example in classes[label]:
                self._examples = self._examples + (((example - self._mean)/sqrt(diag(self._cov)), label),);

    def predict(self, x):
        x = (x - self._mean)/sqrt(diag(self._cov));
        return min(self.examples, key = lambda e: norm(e[0] - x))[1];
