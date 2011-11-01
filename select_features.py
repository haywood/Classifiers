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


def exhaustive(samples, sample_labels, per_class, classifier):
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

def genetic_beam(samples, sample_labels, per_class, classifier):
    n_features = shape(samples)[1];
    ultimate = (-inf, ()),;
    beam_width = n_features;

    features = tuple(range(n_features));
    best = tuple((-inf, (f,)) for f in features);
    frontier = best[:];
   
    while frontier:

# successors is produced by taking pairwise combinations of the current frontier
        successors = ();
        for f in combinations([f[1] for f in frontier], 2):
            p = tuple(k for k,_ in groupby(sorted(f[0] + f[1])));
            successors = successors + (p,);
        successors = tuple(filter(lambda x: len(x) < len(features), 
            (k for k,_ in groupby(sorted(successors)))));

# the frontier is then replaced by the beam_width best of successors
        frontier = tuple(
                (classify([reshape(sample[list(s)], (len(s),)) for sample in samples],
                    sample_labels,
                    per_class,
                    classifier),
                s) for s in successors); 

        frontier = sorted(frontier, key = lambda x: x[0], reverse = True)[:beam_width];
        if frontier:
# get a maximal subset from the frontier and comapre its accuracy to the recorded max
            maximal = max(frontier, key = lambda x: x[0]);
            if maximal[0] > best[0][0]:
                best = tuple(filter(lambda x: x[0] == maximal[0], frontier));
                print('Best:', *best, sep='\n\t');

    return best;

def stochastic_genetic_beam(samples, sample_labels, per_class, classifier):
    
    n_features = shape(samples)[1];
    beam_width = 3*n_features;

    features = tuple(range(n_features));
    native_seed(time.time());

    best = tuple((-inf, (f,)) for f in features);
    frontier = best[:];
 
    while frontier:

# successors is produced by taking pairwise combinations of the current frontier
        successors = ();
        for f in combinations([f[1] for f in frontier], 2):
            p = tuple(k for k,_ in groupby(sorted(f[0] + f[1])));
            successors = successors + (p,);
        successors = tuple(filter(lambda x: len(x) < len(features), 
            (k for k,_ in groupby(sorted(successors)))));
        successors = tuple(native_sample(successors, min((beam_width, len(successors)))));

# the frontier is then replaced by the beam_width best of successors
        frontier = tuple(
               (classify([reshape(sample[list(s)], (len(s),)) for sample in samples],
                    sample_labels,
                    per_class,
                    classifier),
                s) for s in successors); 

        if frontier:
# get a maximal subset from the frontier and comapre its accuracy to the recorded max
            maximal = max(frontier, key = lambda x: x[0]);
            if maximal[0] > best[0][0]:
                best = tuple(filter(lambda x: x[0] == maximal[0], frontier));
                print('Best:', *best, sep='\n\t');

    return best;

def reductive_beam(samples, sample_labels, per_class, classifier):

    n_features = shape(samples)[1];
    features = tuple(range(n_features));
    best = [(-inf, features)];
    frontier = best[:];
    beam_width = n_features;

    while len(frontier[0][1]) > 1:

        successors = chain.from_iterable(combinations(x[1], len(x[1]) - 1) for x in frontier);
        successors = tuple(k for k,_ in groupby(sorted(successors)));
        frontier = tuple(
                (classify([reshape(sample[list(indices)], (len(indices),)) for sample in samples], 
                    sample_labels, 
                    per_class,
                    classifier), 
                indices) for indices in successors);

        frontier = sorted(frontier, key = lambda x: x[0], reverse = True)[:beam_width];

# get a maximal subset from the frontier and comapre its accuracy to the recorded max
        maximal = max(frontier, key = lambda x: x[0]);
        if maximal[0] > best[0][0]:
            best = tuple(filter(lambda x: x[0] == maximal[0], frontier));
            print('Best:', *best, sep='\n\t');

    return best;

def stochastic_reductive_beam(samples, sample_labels, per_class, classifier):

    n_features = shape(samples)[1];
    features = tuple(range(n_features));
    best = [(-inf, features)];
    frontier = best[:];
    beam_width = 3*n_features;

    native_seed(time.time());

    while len(frontier[0][1]) > 1:

        successors = chain.from_iterable(combinations(x[1], len(x[1]) - 1) for x in frontier);
        successors = tuple(k for k,_ in groupby(sorted(successors)));
        successors = tuple(native_sample(successors, min((beam_width, len(successors)))));
        frontier =tuple( 
                (classify([reshape(sample[list(indices)], (len(indices),)) for sample in samples], 
                    sample_labels, 
                    per_class,
                    classifier), 
                indices) for indices in successors);

# get a maximal subset from the frontier and comapre its accuracy to the recorded max
        maximal = max(frontier, key = lambda x: x[0]);
        if maximal[0] > best[0][0]:
            best = tuple(filter(lambda x: x[0] == maximal[0], frontier));
            print('Best:', *best, sep='\n\t');

    return best;

algorithms = {
        'exhaustive':exhaustive,
        'genetic':genetic_beam,
        'stochastic_genetic': stochastic_genetic_beam,
        'reductive': reductive_beam,
        'stochastic_reductive':stochastic_reductive_beam
        };

classifiers = {
        'bayes': bayes_classifier,
        'nn': nn_classifier
        };

def select_for_textures(method, per_class, classifier, train_file):
    
    sample_labels, samples = readfile(train_file);
    best = algorithms[method](samples, sample_labels, per_class, classifiers[classifier]);

    return best;

def select_for_hands(method, per_class, classifier, train_file):

    sample_labels, samples = read_hands(train_file);
    best = algorithms[method](samples, sample_labels, per_class, classifiers[classifier]);

    return best;

modes = {
        'textures': select_for_textures,
        'hands': select_for_hands
        };
    
if __name__ == '__main__':

    mode = sys.argv[1];
    method = sys.argv[2];
    per_class = int(sys.argv[3]);
    output_file = sys.argv[4];
    classifier = sys.argv[5];
    input_file = sys.argv[6];

    best = modes[mode](method, per_class, classifier, input_file);
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
