#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
ES=48000
DATA=RRG_cdnet_data0_times_gtdb_gesa_times-cr
RMN=others/models/RRG/retriever_camrest_cdnet_data0_times_gtdb_gesa_times-cr_retrieve1_seed-111_ep-15_lr-5e-5_wd-0.01_maxlen-128_bs-108_ngpu-_pln-128_tmp-0.05_hnw-0
python train.py \
    --name new_joint \
    --model_size base \
    --checkpoint_dir others/result/camrest/RRG_v3_micro_cdnet \
    --retriever_model_name ${RMN} \
    --dataset_name camrest \
    --metric_version new1 \
    --train_data others/data/camrest/data_used/${DATA}/train.json \
    --eval_data others/data/camrest/data_used/${DATA}/val.json \
    --test_data others/data/camrest/data_used/${DATA}/test.json \
    --dbs others/data/camrest/data_used/${DATA}/all_db.json \
    --use_ranker True \
    --rank_attribute_start_step 0 \
    --rank_attribute_pooling avg \
    --ranker_attribute_ways threshold \
    --threshold_attr 0.1 \
    --ranker_times_matrix True \
    --ranker_times_matrix_start_step 0 \
    --ranker_times_matrix_loss_type bce \
    --ranker_times_matrix_query cr \
    --generator_distill_retriever True \
    --generator_distill_retriever_start_step 30000 \
    --use_delex True \
    --use_dk True \
    --dk_mask True \
    --use_checkpoint \
    --total_steps ${ES} \
    --retriever_total_steps ${ES} \
    --ranker_total_steps ${ES} \
    --end_eval_step ${ES} \
    "$@"