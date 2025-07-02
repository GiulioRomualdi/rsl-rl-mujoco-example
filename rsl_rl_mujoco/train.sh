#!/usr/bin/env bash

# python train.py \
#     train.discriminator.use=false\
#     "log_dir= ./logs/400_without_amp" 


python train.py \
    train.discriminator.use=true\
    "log_dir= ./logs/400_with_amp" 