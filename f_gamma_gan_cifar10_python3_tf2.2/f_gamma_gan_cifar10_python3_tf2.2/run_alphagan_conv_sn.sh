#!/bin/bash


epochs=200000
for Lrate in 0.00005 0.0001 #default 0.0002
do
for alpha in 0 #2 10 100
do
for run_num in 1 #1 2 3 4 5
do
for rev in 0
do
job_name="C10a_conv_sn_${alpha}_${rev}_${Lrate}_${run_num}"

sbatch --job-name=$job_name --output="out/${job_name}.out" --partition=1080ti-long --gres=gpu:1 --mem=36000 alphagan_conv_sn.sh $epochs $alpha $rev $run_num $Lrate $job_name

sleep 20s
done
done
done
done
