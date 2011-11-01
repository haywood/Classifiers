#!/usr/bin/python

from __future__ import print_function, division
from itertools import combinations, chain, groupby
from random import sample as native_sample
from random import seed as native_seed
from numpy import *
import os.path
import time
import sys

from classifier import classify
from bayes_classifier import bayes_classifier
from nn_classifier import nn_classifier
from myio import *


def exhaustive(samples, sample_labels, per_class, classifier, alg_args):
    n_features = shape(samples)[1];
    features = tuple(range(n_features));
    ultimate = (-inf, features),;

    for k in range(1, n_features):
        for comb in combinations(features, k):
            s = classify([reshape(sample[list(comb)], (k,)) for sample in samples],
                sample_labels,
                per_class,
                classifier);
            if s > ultimate[0][0]:
                ultimate = (s, comb),;
                print('Ultimate:', *ultimate, sep='\n\t');
            elif s == ultimate[0][0] and len(ultimate[0][1]) == k:
                ultimate = ultimate + ((s, comb),);
                print('Ultimate:', *ultimate, sep='\n\t');

    return ultimate;

def beam_search(samples, sample_labels, per_class, classifier, alg_args):
    initializer, successor_function, selector = alg_args;
    n_features = shape(samples)[1];
    beam_width = n_features;

    features = tuple(range(n_features));
    ultimate = initializer(features);
    frontier = ultimate[:];
   
    while frontier:

        successors = successor_function(frontier, features);
        frontier = selector(samples, frontier, successors, beam_width, classifier);

        if frontier:
            maximal = max(frontier, key = lambda x: x[0]);
            if maximal[0] > ultimate[0][0]:
                ultimate = tuple(filter(lambda x: x[0] == maximal[0], frontier));
                print('Ultimate:', *ultimate, sep='\n\t');

    return ultimate;

def genetic_init(features):
    return tuple((-inf, (f,)) for f in features);

def genetic_successors(frontier, features):
    successors = ();
    for f in combinations([f[1] for f in frontier], 2):
        p = tuple(k for k,_ in groupby(sorted(f[0] + f[1])));
        successors = successors + (p,);
    successors = tuple(filter(lambda x: len(x) < len(features), 
        (k for k,_ in groupby(sorted(successors)))));
    return successors;

def reductive_init(features):
    return features,;

def reductive_successors(frontier, features):
    successors = chain.from_iterable(combinations(x[1], len(x[1]) - 1) for x in frontier);
    return tuple(k for k,_ in groupby(sorted(successors)));

def deterministic_selector(samples, frontier, successors, beam_width, classifier):
    frontier = tuple(
            (classify([reshape(sample[list(s)], (len(s),)) for sample in samples],
                sample_labels,
                per_class,
                classifier),
            s) for s in successors); 
    return sorted(frontier, key = lambda x: x[0], reverse = True)[:beam_width];

def stochastic_selector(samples, frontier, successors, beam_width, classifier):
    successors = tuple(native_sample(successors, min((beam_width, len(successors)))));
    return tuple(
           (classify([reshape(sample[list(s)], (len(s),)) for sample in samples],
                sample_labels,
                per_class,
                classifier),
            s) for s in successors); 

def textures(train_file):
    
    return readfile(train_file);

def hands(train_file):

    return read_hands(train_file);
    
algorithms = {
    'exhaust':exhaustive,
    'beam':beam_search
};

succession_models = {
    'genetic': (genetic_init, genetic_successors),
    'reductive': (reductive_init, reductive_successors)
};

determinism_types = {
    'deterministic': deterministic_selector,
    'stochastic': stochastic_selector
};

classifiers = {
    'bayes': bayes_classifier,
    'nn': nn_classifier
};

modes = {
    'textures': textures,
    'hands': hands
};

if __name__ == '__main__':

    mode = sys.argv[1];
    input_file = sys.argv[2];
    alg = sys.argv[3];
    succ = sys.argv[4];
    det = sys.argv[5];
    classifier = sys.argv[6];
    per_class = int(sys.argv[7]);
    output_file = sys.argv[8];

    sample_labels, samples = modes[mode](input_file);
    init, successor = succession_models[succ];
    select = determinism_types[det];

    best = algorithms[alg](samples, sample_labels, per_class, classifiers[classifier], (init, successor, select));

    accuracy = best[0][0];

    print('Accuracy is %f.' % accuracy);
    best = tuple(tuple(b+1 for b in candidate[1]) for candidate in best);

    print('Here are the best feature sets:', best);

    if os.path.exists(output_file):
        with open(output_file) as feature_file:
            prev_accuracy = float(next(feature_file));
            if prev_accuracy >= accuracy:
                print('Not saving because previous accuracy was at least as good.');
                sys.exit(0);

    with open(output_file, 'w') as feature_file:
        print(accuracy, file=feature_file);
        for b in best: print(*b, file=feature_file); 
