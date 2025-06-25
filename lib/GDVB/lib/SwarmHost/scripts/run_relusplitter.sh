#!/bin/bash

conda run -p $ReluSplitter/.envs/ReluSplitter python $ReluSplitter/main.py split $@
