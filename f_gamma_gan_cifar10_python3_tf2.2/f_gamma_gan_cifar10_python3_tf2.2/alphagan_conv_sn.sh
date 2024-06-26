#!/bin/bash

#param order: epochs, alpha, rev, run_num, Lrate,  sess_name
python3 alphagan_cifar_conv_spectral_norm_sl.py  $1 5  $2 $3 $4 $5 $6 
