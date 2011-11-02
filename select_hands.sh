#!/bin/sh

./select_features.py hands handtrainfile.txt beam genetic deterministic $1 5 best_hand_features.txt
