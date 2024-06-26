#!/bin/bash

#param order: epochs, alpha, rev, run_num, Lrate, LAMBDA, gp_sidedness, sess_name
python3 alphagan_cifar_resnet_sl.py $1 5  $2 $3 $4 $5 $6 $7 $8
