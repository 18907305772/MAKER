#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
ES=32000
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
    --test_model_path others/result/camrest/RRG_v3_micro_cdnet/new_joint_gt-retrieval_t5-large_glr-7e-5_seed-111_gtml-200_rtml-128_rkml-200_asml-64_gdml-100_rdml-128_kdml-100_dus-100_topd-4_gdre-24000-avg_wo_context-kl-1.0_dlx_dk_gtdb/checkpoint/generator_best_dev \
    --generator_distill_retriever True \
    --generator_distill_retriever_start_step 24000 \
    --use_delex True \
    --use_dk True \
    --end_eval_step ${ES} \
    --use_gt_dbs True \
    --use_retriever_for_gt True \
    --top_k_dbs 4 \
    "$@"