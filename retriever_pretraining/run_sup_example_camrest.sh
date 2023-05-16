#!/bin/bash

# In this example, we show how to train SimCSE on unsupervised Wikipedia data.
# If you want to train it with multiple GPU cards, see "run_sup_example.sh"
# about how to use PyTorch's distributed data parallel.
export CUDA_VISIBLE_DEVICES=0
#NUM_GPU=2
#PORT_ID=$(expr $RANDOM + 1000)
#export OMP_NUM_THREADS=8
for DATA_VERSION in 1
do
for EP in 15
do
for BS in 108
do
for MAX_LEN in 128
do
SEED=111
MODEL_NAME=bert-base-uncased
LR=5e-5
POOLER_NUM=128
TEMP=0.05
WD=0.01
HNW=0
#python -m torch.distributed.launch --nproc_per_node $NUM_GPU --master_port $PORT_ID  2.train.py \
python 2.train.py \
    --overwrite_output_dir \
    --seed ${SEED} \
    --model_name_or_path ${MODEL_NAME} \
    --train_file others/data/camrest/data_used/RRG_cdnet_data0_times_gtdb_gesa_times-cr_retrieve${DATA_VERSION}/train.csv \
    --validation_file others/data/camrest/data_used/RRG_cdnet_data0_times_gtdb_gesa_times-cr_retrieve${DATA_VERSION}/val.csv \
    --output_dir others/result/camrest/RP_v0/retriever_RRG_cdnet_data0_times_gtdb_gesa_times-cr_retrieve${DATA_VERSION}_seed-${SEED}_ep-${EP}_lr-${LR}_wd-${WD}_maxlen-${MAX_LEN}_bs-${BS}_ngpu-${NUM_GPU}_pln-${POOLER_NUM}_tmp-${TEMP}_hnw-${HNW} \
    --num_train_epochs ${EP} \
    --per_device_train_batch_size ${BS} \
    --per_device_eval_batch_size ${BS} \
    --metric_for_best_model eval_acc \
    --learning_rate ${LR} \
    --max_seq_length ${MAX_LEN} \
    --save_total_limit 1 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end \
    --logging_steps 10 \
    --pooler_type cls \
    --pooler_num ${POOLER_NUM} \
    --temp ${TEMP} \
    --do_train \
    --do_eval \
    --weight_decay ${WD} \
    --hard_negative_weight ${HNW} \
    --fp16 \
    --experiment_dataset_name camrest \
    "$@"
done
done
done
done