#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
ES=32000
DATA=RRG_cdnet_data0_times_gtdb_gesa_times-cr
RMN=others/models/RRG/retriever_camrest_cdnet_data0_times_gtdb_gesa_times-cr_retrieve1_seed-111_ep-15_lr-5e-5_wd-0.01_maxlen-128_bs-108_ngpu-_pln-128_tmp-0.05_hnw-0
python train.py \
    --name new_joint_gt-retrieval_t5-large_glr-7e-5 \
    --lr 7e-5 \
    --model_size large \
    --checkpoint_dir others/result/camrest/RRG_v3_micro_cdnet \
    --retriever_model_name ${RMN} \
    --dataset_name camrest \
    --metric_version new1 \
    --train_data others/data/camrest/data_used/${DATA}/train.json \
    --eval_data others/data/camrest/data_used/${DATA}/val.json \
    --test_data others/data/camrest/data_used/${DATA}/test.json \
    --dbs others/data/camrest/data_used/${DATA}/all_db.json \
    --generator_distill_retriever True \
    --generator_distill_retriever_start_step 24000 \
    --use_delex True \
    --use_dk True \
    --use_checkpoint \
    --total_steps ${ES} \
    --retriever_total_steps ${ES} \
    --ranker_total_steps ${ES} \
    --end_eval_step ${ES} \
    --use_gt_dbs True \
    --use_retriever_for_gt True \
    --top_k_dbs 4 \
    --per_gpu_eval_batch_size 2 \
    "$@"