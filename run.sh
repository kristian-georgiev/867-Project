#!/bin/bash
# run.sh
# config options can be found in config.yaml
CONFIG=default
python ./main.py --model_training train_new --dataset toy --model two
