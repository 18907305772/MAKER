#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
ES=48000
DATA=RRG_data1_times_gtdb_gesa_times-cr-dyn
RMN=others/models/RRG/retriever_train_new_trunc_data_used_new_v0_seed-111_bert-base-uncased_ep-10_lr-5e-5_wd-0.01_maxlen-128_bs-32_ngpu-4_pln-128_tmp-0.05_hnw-0
python train.py \
    --name new_joint_gt-retrieval \
    --model_size base \
    --checkpoint_dir others/result/mwoz_gptke/RRG_v3_micro \
    --retriever_model_name ${RMN} \
    --dataset_name mwoz_gptke \
    --metric_version new1 \
    --train_data others/data/mwoz_gptke/data_used/${DATA}/train.json \
    --eval_data others/data/mwoz_gptke/data_used/${DATA}/val.json \
    --test_data others/data/mwoz_gptke/data_used/${DATA}/test.json \
    --dbs others/data/mwoz_gptke/data_used/${DATA}/all_db.json \
    --retriever_lr 5e-5 \
    --ranker_lr 5e-5 \
    --use_ranker True \
    --rank_attribute_start_step 0 \
    --rank_attribute_pooling avg_wo_context \
    --ranker_attribute_ways threshold \
    --threshold_attr 0.1 \
    --ranker_times_matrix True \
    --ranker_times_matrix_start_step 0 \
    --ranker_times_matrix_loss_type bce \
    --ranker_times_matrix_query cr \
    --generator_distill_retriever True \
    --generator_distill_retriever_start_step 20000 \
    --use_delex True \
    --use_dk True \
    --dk_mask True \
    --use_checkpoint \
    --total_steps ${ES} \
    --retriever_total_steps ${ES} \
    --ranker_total_steps ${ES} \
    --end_eval_step ${ES} \
    --use_gt_dbs True \
    --use_retriever_for_gt True \
    --top_k_dbs 6 \
    "$@"