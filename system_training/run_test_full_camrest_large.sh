#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
ES=96000
DATA=RRG_cdnet_data0_times_gtdb_gesa_times-cr
RMN=others/models/RRG/retriever_camrest_cdnet_data0_times_gtdb_gesa_times-cr_retrieve1_seed-111_ep-15_lr-5e-5_wd-0.01_maxlen-128_bs-108_ngpu-_pln-128_tmp-0.05_hnw-0
python test.py \
    --per_gpu_eval_batch_size 16 \
    --model_size large \
    --dataset_name camrest \
    --metric_version new1 \
    --retriever_model_name ${RMN} \
    --test_data others/data/camrest/data_used/${DATA}/test.json \
    --dbs others/data/camrest/data_used/${DATA}/all_db.json \
    --test_model_path others/result/camrest/RRG_v3_micro_cdnet/new_joint_t5-large_seed-111_gtml-200_rtml-128_rkml-200_asml-64_gdml-100_rdml-128_kdml-100_dus-200_topd-7_tha-0.1_rk-0-avg-tm-0-bce-1.0-cr_gdre-60000-avg_wo_context-kl-1.0_dlx_dk_es-96000/checkpoint/generator_best_dev \
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
    --generator_distill_retriever_start_step 60000 \
    --use_delex True \
    --use_dk True \
    --end_eval_step ${ES} \
    "$@"