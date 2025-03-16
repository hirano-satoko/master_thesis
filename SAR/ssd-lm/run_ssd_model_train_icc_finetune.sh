#!/usr/bin/bash
trap "kill 0" EXIT

script_role="host"
global_seed=2022 # inline param, 2021, 2022, etc
single_device_cuda="0" # inline param, "0", "1", etc
multi_device_cuda="0,1,2" # inline param, "0,1,2,3", "0", etc
hf_cache="/home/hirano/.cache/huggingface"
core_lm_name="/home/hirano/ssd-lm/logging/20250131" # finetune
main_log_dir="/home/hirano/ssd-lm/logging"

# data hyperparameters
global_max_seq_len=200
####

# slurm_run_ssd_model_train_iic_finetune.sh

# retrain
retrain_num_train_epochs=10000 # just a placeholder, use max train steps
retrain_per_device_train_batch_size=24
retrain_per_device_eval_batch_size=24
retrain_learning_rate=5e-5
retrain_weight_decay=0.01
retrain_gradient_accumulation_steps=4
retrain_num_warmup_steps=2000
retrain_max_train_steps=15000

sigma_num_steps=5000
loss_mode="xe"
remove_noise_mode="no_dir"
pa=5
cs=0 # placeholder
dbs=25
precision="fp16"
noise_manual_scale=1.0
train_mode="resume"

################ START ################

available_port=29510
main_node_name=$(hostname)
main_ip_address=$(python3 -c 'import sys; import socket; ip=socket.gethostbyname(sys.argv[1]); print(ip)' ${main_node_name})

CUDA_VISIBLE_DEVICES=${multi_device_cuda} HF_HOME=${hf_cache} accelerate launch \
    --multi_gpu --mixed_precision ${precision} \
    --num_processes 3 --num_machines 1 --machine_rank 0 \
    --main_process_ip ${main_ip_address} --main_process_port ${available_port} \
    --num_cpu_threads_per_process 2 \
    ssd_model_train_iic_finetune.py \
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
    --output_dir ${main_log_dir}/ssd_dbs${dbs} \
    --loss_mode ${loss_mode} \
    --remove_noise_mode ${remove_noise_mode} \
    --hardcoded_pseudo_diralpha ${pa} \
    --context_size ${cs} \
    --decoding_block_size ${dbs} \
    --sigma_num_steps ${sigma_num_steps} \
    --noise_manual_scale ${noise_manual_scale} \
    --if_create_tokenized_data_file "no" \
    --train_mode ${train_mode}