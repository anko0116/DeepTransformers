#!/bin/bash

sbatch -c 4 --mem=20G --gres=gpu:1 -t 1-0 --partition=gpu ./run_train.sh

