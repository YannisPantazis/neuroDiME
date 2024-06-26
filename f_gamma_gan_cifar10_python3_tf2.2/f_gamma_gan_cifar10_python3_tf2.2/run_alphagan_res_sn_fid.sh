#!/bin/bash


epochs=200000
Lrate=0.0002 #default 0.0002
for alpha in 0 2 10 100
do
for run_num in 1 2 3 4 5
do
for rev in 0
do
job_name="C10a_res_sn_fid_${alpha}_${rev}_${Lrate}_${run_num}"
dir_name="C10a_res_sn_${alpha}_${rev}_${Lrate}_${run_num}"	

alpha_fl=$(printf "%.1f" $alpha )
fid_file="cifar_resnet_sn/${dir_name}/fid_alpha_${alpha_fl}_rev_${rev}_set${run_num}_iteration${epochs}.csv"
tmp=$((epochs-1))
gen_data_dir="cifar_resnet_sn/${dir_name}/gen_images_50k_alpha_${alpha_fl}_rev_${rev}_set${run_num}_iteration${tmp}.npy"



sbatch --job-name=$job_name --output="out/${job_name}.out" --partition=1080ti-short --gres=gpu:1 --mem=36000 alphagan_fid.sh $gen_data_dir $fid_file

sleep 20s
done
done
done
