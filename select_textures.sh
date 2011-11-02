#!/bin/sh

./select_features.py textures trainfile.txt sample genetic stochastic $1 16 best_texture_features_$1.txt
