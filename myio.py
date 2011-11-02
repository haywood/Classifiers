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
    for i in range(n_samples):
        p = [samples[i][j:j+2] for j in range(0, n_features-1, 2)];
        repl = [norm(p[0] - p[18])]; # hand width
        repl += [norm(p[j] - p[j+2]) for j in range(2, 15, 4)] + [norm(p[19] - p[21])]; # finger width
        repl += [sum([norm(p[k] - p[k+1]) for k in range(j, j+5)]) for j in range(1, 14, 4)]; # finger length
        repl += [(norm(p[18] - p[20]) + norm(p[20] - p[22]))/2]; # thumb length
        samples[i] = array(repl);

    return labels, samples;
