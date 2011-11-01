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
#this transformation on a sample causes its first two points to lie on the same horizontal line
    labels, samples = readfile(filename);
    n_samples, n_features = shape(samples);
    for i in range(n_samples):
        p0 = samples[i][0:2];
        repl = [norm(samples[i][j:j+2] - p0) for j in range(0, n_features-1, 2)];
        samples[i] = array(repl);

    return labels, samples;
