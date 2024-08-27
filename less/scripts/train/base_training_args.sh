#!/bin/bash

ID=$RANDOM
# export header="torchrun --nproc_per_node 1 --nnodes 1 \
# --rdzv-id=$ID --rdzv_backend c10d \
# -m less.train.train"

# export header="torchrun --master_port=$ID --nproc_per_node 4 --nnodes 1 \
# -m less.train.train"

# export header="python \
# -m less.train.train"

export base_training_args="--do_train True \
--max_seq_length 2048 \
--use_fast_tokenizer True \
--lr_scheduler_type linear \
--warmup_ratio 0.03 \
--weight_decay 0.0 \
--evaluation_strategy steps \
--eval_steps 0.25 \
--logging_steps 1 \
--num_train_epochs 1 \
--save_steps_per_epoch 1 \
--bf16 True \
--tf32 False \
--fp16 False \
--overwrite_output_dir True \
--report_to wandb \
--optim adamw_torch \
--seed 0 \
--percentage 1.0 \
--save_strategy steps \
--save_steps 0.25 \
--lora True \
--lora_r 128 \
--lora_alpha 512 \
--lora_dropout 0.1 \
--lora_target_modules q_proj k_proj v_proj o_proj \
--learning_rate 2e-05 \
--per_device_train_batch_size 1 \
--save_total_limit 15 \
--ddp_find_unused_parameters False \
--gradient_accumulation_steps 32"