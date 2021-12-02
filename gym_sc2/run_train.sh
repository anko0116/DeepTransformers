#!/bin/sh

singularity exec --nv -B /users/hahn19:/users/hahn19 /users/hahn19/DeepTransformers/CSCI2951F.simg python3.7 train.py

