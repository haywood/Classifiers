from __future__ import print_function
from numpy.linalg import norm
from numpy import *

def readfile(filename):
    
    samples = [];
    labels = [];
    with open(filename) as sample_file:
        lines = sample_file.readlines();
        for line in lines:
            row = line.split();
            labels.append(row.pop(0));
            samples.append(array([float(s) for s in row]));

    return labels, samples;

def read_hands(filename):
    labels, samples = readfile(filename);
    n_samples, n_features = shape(samples);
# calculate the euclidean distance between each point in a sample and use this as the feature vector
    for i in range(n_samples):
        repl = [norm(samples[i][j:j+2] - samples[i][j+2:j+4]) for j in range(0, n_features-3, 2)];
        samples[i] = array(repl);

    return labels, samples;
