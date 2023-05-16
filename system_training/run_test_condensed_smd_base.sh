#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
ES=48000
DATA=RRG_qtod_data0_times_gtdb_gesa_times-cr
RMN=others/models/RRG/retriever_camrest_cdnet_data0_times_gtdb_gesa_times-cr_retrieve1_seed-111_ep-15_lr-5e-5_wd-0.01_maxlen-128_bs-108_ngpu-_pln-128_tmp-0.05_hnw-0
python test.py \
    --per_gpu_eval_batch_size 32 \
    --model_size base \
    --dataset_name smd \
    --metric_version new1 \
    --retriever_model_name ${RMN} \
    --test_data others/data/smd/data_used/${DATA}/test.json \
    --dbs others/data/smd/data_used/${DATA}/all_db.json \
    --test_model_path others/result/smd/RRG_v3_micro_qtod/new_joint_seed-111_gtml-200_rtml-128_rkml-200_asml-128_gdml-200_rdml-128_kdml-200_dus-100_topd-8_dk_es-48000_gtdb/checkpoint/generator_best_dev \
    --generator_db_maxlength 200 \
    --ranker_db_maxlength 200 \
    --answer_maxlength 128 \
    --top_k_dbs 8 \
    --use_dk True \
    --end_eval_step ${ES} \
    --use_gt_dbs True \
    "$@"