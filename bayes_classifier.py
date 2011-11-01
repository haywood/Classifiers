from __future__ import print_function
from numpy import *
from multivariate_normal import *

class bayes_classifier(object):

    def __init__(self, dim):
        self._distributions = {};
        self._dim = dim;

    @property
    def dim(self):
        return self._dim;

    @property
    def distributions(self):
        return self._distributions;

    def __getitem__(self, c):
        if self.distributions:
            if c in self.distributions:
                return self.distributions[c];
            else: raise KeyError('No such class.');
        else: raise ValueError("Don't have any classes yet.");
    
    def __setitem__(self, c, v):
        self.distributions[c] = v;

    def train(self, classes):
        self._distributions = {};
        for label in classes:
            X = array(classes[label]);
            self[label] = (mean(X, axis=0), cov(X.T));

    def predict(self, x):
        return max(self.distributions, key=lambda c: mvnpdf(x, self[c][0], self[c][1]));
