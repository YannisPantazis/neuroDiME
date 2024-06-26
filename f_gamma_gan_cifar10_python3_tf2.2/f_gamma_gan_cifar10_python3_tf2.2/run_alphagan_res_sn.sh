#!/bin/bash


epochs=100000
n_critic=5
beta_1=0
beta_2=0.9
for Lrate in 0.0002 #default 0.0002
do
for alpha in  100 #0 2 10 100
do
for run_num in 2 4 #1 2 3 4 5 
do
for rev in 0
do
job_name="C10a_res_sn_PWC_${alpha}_${rev}_${Lrate}_${run_num}"

sbatch --job-name=$job_name --output="out/${job_name}.out" --partition=2080ti-long --gres=gpu:1 --mem=36000 alphagan_res_sn.sh $epochs $alpha $rev $run_num $Lrate $n_critic $beta_1 $beta_2 $job_name

sleep 20s
done
done
done
done
