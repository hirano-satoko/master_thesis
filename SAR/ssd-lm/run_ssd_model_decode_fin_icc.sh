#!/usr/bin/bash
trap "kill 0" EXIT

script_role="host"
global_seed=2022 # inline param, 2021, 2022, etc
single_device_cuda="0" # inline param, "0", "1", etc
multi_device_cuda="0,1,2,3" # inline param, "0,1,2,3", "0", etc
hf_cache="/home/hirano/.cache/huggingface"
core_lm_name="roberta-large"
main_log_dir="/home/hirano/ssd-lm/logging" #/Storage2/hirano/container_target/ssd-lm/models"

interpret_dataset_tokenized_path="${main_log_dir}/openwebtext_processed_pct100_blk200"

# data hyperparameters
global_max_seq_len=200
####

# slurm_run_ssd_model_decode_fin_icc3.sh

# retrain
retrain_num_train_epochs=10000
retrain_per_device_train_batch_size=1
retrain_per_device_eval_batch_size=1
retrain_learning_rate=1e-4
retrain_weight_decay=0.01
retrain_gradient_accumulation_steps=1
retrain_num_warmup_steps=2000
retrain_max_train_steps=100000

sigma_num_steps=2000 # 1000 or 2500 or 5000
loss_mode="n/a"
remove_noise_mode="fin|/private/home/xhan77/ssd-lm/logging/pplm_discrim_prompts.jsonl"
pa=5
cs=1 # placeholder
dbs=25
noise_manual_scale=1.0
subdir=20250203 # どのモデルか
decode_context_size=0
decode_truncate_len=150 # 150 or 180 or 188
decode_depth=2 # 2 or 1
# decode_ctr_lr=100 #0.0 100.0 500.0 2000.0 # 勾配更新の時の係数
projection_top_p=0.2 #0.2 0.5 0.9
out_fn=ssd_ctrsa_gen.jsonl

# 制御方法
decode_ctr_lr=100 # 勾配更新の時の係数

#
# for ((outer=700; outer<=1300; outer+=50))
# do
#     limit=$((outer < 1000 ? outer : 1000))
#     for ((inner=250; inner<limit; inner+=50))
#     do
decode_ctr_tmax=1000
decode_ctr_tmin=800

CUDA_VISIBLE_DEVICES=${multi_device_cuda} torchrun --nproc_per_node=4 \
ssd_model_decode_icc.py \
--max_seq_length ${global_max_seq_len} \
--model_name_or_path ${core_lm_name} \
--num_train_epochs ${retrain_num_train_epochs} \
--per_device_train_batch_size ${retrain_per_device_train_batch_size} \
--per_device_eval_batch_size ${retrain_per_device_eval_batch_size} \
--learning_rate ${retrain_learning_rate} \
--weight_decay ${retrain_weight_decay} \
--gradient_accumulation_steps ${retrain_gradient_accumulation_steps} \
--num_warmup_steps ${retrain_num_warmup_steps} \
--max_train_steps ${retrain_max_train_steps} \
--seed ${global_seed} \
--use_slow_tokenizer \
--output_dir ${main_log_dir}/${subdir} \
--subdir ${subdir} \
--loss_mode ${loss_mode} \
--remove_noise_mode ${remove_noise_mode} \
--hardcoded_pseudo_diralpha ${pa} \
--context_size ${cs} \
--decoding_block_size ${dbs} \
--sigma_num_steps ${sigma_num_steps} \
--tokenized_data_file_path ${interpret_dataset_tokenized_path} \
--if_create_tokenized_data_file "no" \
--decode_context_size ${decode_context_size} \
--decode_truncate_len ${decode_truncate_len} \
--decode_depth ${decode_depth} \
--train_mode decode \
--decode_ctr_lr ${decode_ctr_lr} \
--decode_ctr_tmax ${decode_ctr_tmax} \
--decode_ctr_tmin ${decode_ctr_tmin} \
--projection_top_p ${projection_top_p} \
--projection_alg "even" \
--ctr_opt_label_idx 2 \
--out_fn ${out_fn}

#     done
# done