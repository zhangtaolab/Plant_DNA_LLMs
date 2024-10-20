#!/bin/bash

for arg in "$*";
do
    exec python3 $RUN_HOME/inference.py $*;
done;
