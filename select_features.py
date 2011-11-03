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


def merit(samples, labels, features):

    sub_samples = [sample[list(features)] for sample in samples];
    sample_corr = corrcoef(sub_samples, rowvar=0);
    bottom = sqrt(sum(sample_corr));
    top = 0;
    for f in features:
        top += corrcoef([sample[f] for sample in samples], labels)[0][1];

    return top/bottom;

def sample_combinations(samples, sample_labels, per_class, classifier, alg_args):

    n_features = shape(samples)[1];
    features = tuple(range(n_features));
    ultimate = (-inf, features),;
    per_k = 4;

    subsets = [];
    for k in range(1, n_features):
        combs = [comb for comb in combinations(features, k)];
        subsets += native_sample(combs, min([per_k, len(combs)]));
    candidates = tuple((merit(samples, sample_labels, comb), comb) for comb in subsets);

    return candidates;

def exhaustive(samples, sample_labels, per_class, classifier, alg_args):
    n_features = shape(samples)[1];
    features = tuple(range(n_features));
    ultimate = (-inf, features),;
    depth_limit = n_features;

    for k in range(depth_limit):
        for comb in combinations(features, k):
            s = merit(samples, sample_labels, comb);
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
    frontier = initializer(features, samples, sample_labels, classifier, per_class);
    ultimate = (-inf, ()),;
    depth_limit = inf;
    depth = 0;
   
    while frontier and depth < depth_limit:

        maximal = max(frontier, key = lambda x: x[0]);
        if maximal[0] > ultimate[0][0]:
            ultimate = tuple(filter(lambda x: x[0] == maximal[0], frontier));
            print('Ultimate:', *ultimate, sep='\n\t');

        successors = successor_function(frontier, features);
        frontier = selector(samples, frontier, successors, beam_width, classifier);
        depth += 1;

    return ultimate;

def genetic_init(features, samples, sample_labels, classifier, per_class):
    return tuple((merit(samples, sample_labels, (f,)), (f,)) for f in features);

def genetic_successors(frontier, features):
    successors = ();
    for f in combinations([f[1] for f in frontier], 2):
        p = tuple(k for k,_ in groupby(sorted(f[0] + f[1])));
        successors = successors + (p,);
    successors = tuple(filter(lambda x: len(x) <= len(features), 
        (k for k,_ in groupby(sorted(successors)))));
    return successors;

def reductive_init(features, samples, sample_labels, classifier, per_class):
    return (merit(samples, sample_labels, features), features),;

def reductive_successors(frontier, features):
    successors = chain.from_iterable(combinations(x[1], len(x[1]) - 1) for x in frontier if len(x[1]) > 1);
    return tuple(k for k,_ in groupby(sorted(successors)));

def deterministic_selector(samples, frontier, successors, beam_width, classifier):
    frontier = tuple((merit(samples, sample_labels, s), s) for s in successors); 
    return sorted(frontier, key = lambda x: x[0], reverse = True)[:beam_width];

def stochastic_selector(samples, frontier, successors, beam_width, classifier):
    successors = tuple(native_sample(successors, min((beam_width, len(successors)))));
    return tuple((merit(samples, sample_labels, s), s) for s in successors); 

def textures(train_file):
    
    return readfile(train_file);

def hands(train_file):

    return read_hands(train_file);
    
algorithms = {
    'sample':sample_combinations,
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

    k = 1;
    for i in range(len(samples)):
        if i < len(samples)-1 and sample_labels[i] != sample_labels[i+1]:
            sample_labels[i] = k;
            k += 1;
        else: sample_labels[i] = k;

    candidates = algorithms[alg](samples, sample_labels, per_class, classifiers[classifier], (init, successor, select));

    candidates = sorted(candidates, reverse = True);
    with open(output_file, 'w') as feature_file:
        for c in candidates: 
            print(c[0], *c[1], file=feature_file); 
            print(c[0], *c[1]);
