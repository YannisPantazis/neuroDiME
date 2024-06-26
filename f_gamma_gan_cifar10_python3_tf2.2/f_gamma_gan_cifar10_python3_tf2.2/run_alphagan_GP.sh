#!/bin/bash


epochs=200000
Lrate=0.0002 #default 0.0002
LAMBDA=10 #default 10
gp_sides=2
for alpha in 4 8 #0 2 10 100
do
for run_num in 1 2 3 4 5
do
for rev in 0 1
do
job_name="C10a_${alpha}_${rev}_${gp_sides}_${Lrate}_${run_num}"

sbatch --job-name=$job_name --output="out/${job_name}.out" --partition=1080ti-long --gres=gpu:1 --mem=36000 alphagan.sh $epochs $alpha $rev $run_num $Lrate $LAMBDA $gp_sides $job_name

sleep 20s
done
done
done
