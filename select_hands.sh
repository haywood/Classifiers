#!/bin/sh

./select_features.py hands handtrainfile.txt exhaust reductive deterministic $1 5 best_hand_features_$1.txt
