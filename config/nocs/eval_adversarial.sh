#!/bin/bash
cd /data/home/scv2060/run/wby/GPV_Pose-master
module load CUDA/11.3
module load anaconda/2020.11
source activate pytorch1.11.0
python -m evaluation.NF_eval_v6 --per_obj bottle --dataset Real --dataset_dir /data/run01/scv2060/nocs_data \
--keypoint_path /data/run01/scv2060/nocs_data/keypoint.pkl \
--resume_dir /data/run01/scv2060/wby/GPV_Pose-master/output/test_adversarial \
--eval_out output/adversarial \
--fake_grid_num 2000 \
--mask_ratio 0.05 \
--scale_ratio 2 \
--use_sampling_test 1 \
--test_topk 100 \
