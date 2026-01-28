#!/bin/bash
cd /data/home/scv2060/run/wby/GPV_Pose-master
module load CUDA/11.3
module load anaconda/2020.11
source activate pytorch1.11.0
python -m engine.NFtrain_v7 --per_obj laptop --dataset Real --dataset_dir /data/run01/scv2060/nocs_data \
--keypoint_path /data/run01/scv2060/nocs_data/keypoint.pkl \
--batch_size 4 \
--train_steps 2000 \
--backbone gcn_equi2 \
--dim_list 128,128,256,256,512 \
--qnet_config qnet.yaml \
--equi_neighbor_num 10 \
--lr 0.0003 \
--model_save output/gcn_equi2_512_0.0003
