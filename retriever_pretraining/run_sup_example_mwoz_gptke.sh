#!/bin/bash

# In this example, we show how to train SimCSE on unsupervised Wikipedia data.
# If you want to train it with multiple GPU cards, see "run_sup_example.sh"
# about how to use PyTorch's distributed data parallel.
export CUDA_VISIBLE_DEVICES=0,1,2,3
NUM_GPU=4
PORT_ID=$(expr $RANDOM + 1000)
export OMP_NUM_THREADS=8
SEED=111
MODEL_NAME=bert-base-uncased
EP=10
BS=32
LR=5e-5
MAX_LEN=128
POOLER_NUM=128
TEMP=0.05
WD=0.01
HNW=0
python -m torch.distributed.launch --nproc_per_node $NUM_GPU --master_port $PORT_ID  2.train.py \
    --overwrite_output_dir \
    --seed ${SEED} \
    --model_name_or_path ${MODEL_NAME} \
    --train_file others/data/mwoz_gptke/data_used/RRG_data1_times_gtdb_gesa_retrieve0/train.csv \
    --validation_file others/data/mwoz_gptke/data_used/RRG_data1_times_gtdb_gesa_retrieve0/val.csv \
    --output_dir others/result/mwoz_gptke/RP_v0/retriever_train_new_trunc_data_used_new_v0_seed-${SEED}_${MODEL_NAME}_ep-${EP}_lr-${LR}_wd-${WD}_maxlen-${MAX_LEN}_bs-${BS}_ngpu-${NUM_GPU}_pln-${POOLER_NUM}_tmp-${TEMP}_hnw-${HNW} \
    --num_train_epochs ${EP} \
    --per_device_train_batch_size ${BS} \
    --per_device_eval_batch_size ${BS} \
    --metric_for_best_model eval_acc \
    --learning_rate ${LR} \
    --max_seq_length ${MAX_LEN} \
    --save_total_limit 1 \
    --evaluation_strategy steps \
    --load_best_model_at_end \
    --eval_steps 100 \
    --save_steps 100 \
    --logging_steps 10 \
    --pooler_type cls \
    --pooler_num ${POOLER_NUM} \
    --temp ${TEMP} \
    --do_train \
    --do_eval \
    --weight_decay ${WD} \
    --hard_negative_weight ${HNW} \
    "$@"
