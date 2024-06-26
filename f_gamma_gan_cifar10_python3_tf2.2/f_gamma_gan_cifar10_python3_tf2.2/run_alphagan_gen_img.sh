#!/bin/bash


epochs=200000
Lrate=0.001 #default 0.0002
LAMBDA=10 #default 10
gp_sides=1
for alpha in 0 2 10 100
do
for run_num in 1 2 3 4 5
do
for rev in 0
do
dir_name="C10a_${alpha}_${rev}_${gp_sides}_${Lrate}_${run_num}"	
job_name="C10a_img_${alpha}_${rev}_${gp_sides}_${Lrate}_${run_num}"

sbatch --job-name=$job_name --output="out/${job_name}.out" --partition=2080ti-long --gres=gpu:1 --mem=36000 alphagan_gen_img.sh $epochs $alpha $rev $run_num $Lrate $LAMBDA $gp_sides $dir_name

sleep 20s
done
done
done
