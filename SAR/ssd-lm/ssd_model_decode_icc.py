'''
20241112

multi gpu
'''

#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for masked language modeling (BERT, ALBERT, RoBERTa...)
on a text file or a dataset without using HuggingFace Trainer.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=masked-lm
"""
# You can also adapt this script on your own mlm task. Pointers for this are left as comments.

import argparse
from cmath import exp
import logging
import math
from multiprocessing.sharedctypes import Value
import os
import socket
import random
from pathlib import Path

import datasets
import torch
from datasets import load_dataset
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from tqdm.auto import tqdm

import transformers
from huggingface_hub import Repository
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AdamW,
    AutoConfig,
    AutoModel,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    SchedulerType,
    get_scheduler,
    set_seed,
)
from transformers.file_utils import get_full_repo_name
from transformers.utils.versions import require_version

import numpy as np
import pickle
from filelock import FileLock
import dill
from datasets import load_from_disk
from datasets import concatenate_datasets
from termcolor import colored
import time
import json
from flufl.lock import Lock
import datetime

from text_datasets_ln import load_data_text
import time

import torch.distributed as dist

logger = logging.getLogger(__name__)
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

def setup_dist():
    """
    Setup a distributed process group.
    """
    if dist.is_initialized():
        return

    backend = "gloo" if not torch.cuda.is_available() else "nccl"

    if backend == "gloo":
        hostname = "localhost"
    else:
        hostname = socket.gethostbyname(socket.getfqdn())

    if os.environ.get("LOCAL_RANK") is None:
        os.environ["MASTER_ADDR"] = hostname
        os.environ["RANK"] = str(0)
        os.environ["WORLD_SIZE"] = str(1)
        port = _find_free_port()
        os.environ["MASTER_PORT"] = str(port)
        os.environ['LOCAL_RANK'] = str(0)
    
    dist.init_process_group(backend=backend, init_method="env://")

    if torch.cuda.is_available():  # This clears remaining caches in GPU 0
        torch.cuda.set_device(dev())
        torch.cuda.empty_cache()

def dev():
    """
    Get the device to use for torch.distributed.
    """
    if torch.cuda.is_available():
        print(f"cuda:{os.environ['LOCAL_RANK']}")
        return torch.device(f"cuda:{os.environ['LOCAL_RANK']}")
    return torch.device("cpu")

def get_time_variables(t, total_t, device): # according to https://arxiv.org/pdf/2102.09672.pdf

    def ft(small_t, big_t, s=1e-4):
        return torch.cos((small_t / big_t + s) / (1 + s) * math.pi / 2) ** 2

    alpha_t_bar = ft(t, total_t) / ft(torch.zeros(t.shape).to(device), total_t)
    alpha_t_minus_bar = ft(t-1, total_t) / ft(torch.zeros(t.shape).to(device), total_t)
    beta_t = 1 - (alpha_t_bar / alpha_t_minus_bar)
    beta_t_til = (1 - alpha_t_minus_bar) / (1 - alpha_t_bar) * beta_t
    alpha_t = 1 - beta_t
    return alpha_t_bar, alpha_t_minus_bar, beta_t, beta_t_til, alpha_t


def apply_controlling_drift(args, perturbed_inputs_diralpha):
    if args.decode_ctr_lr <= 0:
        args.ctr_loss = -1
        return perturbed_inputs_diralpha

    if args.ctr_model is None:
        ctr_model_name = "cardiffnlp/twitter-roberta-base-sentiment"
        args.ctr_model = AutoModelForSequenceClassification.from_pretrained(ctr_model_name).to(args.accelerator.device)
    optimizing_label_index = args.ctr_opt_label_idx

    for ctr_i in range(1):
        with torch.enable_grad():
            perturbed_inputs_diralpha_4ctr = perturbed_inputs_diralpha.clone()
            perturbed_inputs_diralpha_4ctr.requires_grad_()
            perturbed_inputs_simplex_4ctr = torch.nn.functional.softmax(perturbed_inputs_diralpha_4ctr, dim=-1)
            perturbed_inputs_embeds_4ctr = torch.nn.functional.linear(perturbed_inputs_simplex_4ctr, args.ctr_model.get_input_embeddings().weight.t())
            ctr_loss = -torch.nn.functional.log_softmax(args.ctr_model(inputs_embeds=perturbed_inputs_embeds_4ctr).logits, dim=-1)[:,optimizing_label_index].mean()
            args.ctr_loss = ctr_loss
            ctr_delta = -torch.autograd.grad(ctr_loss, perturbed_inputs_diralpha_4ctr)[0] # indexing 0 because the return is a tuple

        perturbed_inputs_diralpha = perturbed_inputs_diralpha + args.decode_ctr_lr * ctr_delta 
    
    return perturbed_inputs_diralpha

def apply_controlling_drift_classifier(
    args, 
    model,
    classifier_layer,
    diffusion_embeds,
    unit_context_input_ids,
    image_vector,
    coef,
    ):

    with torch.enable_grad():
        diffusion_embeds.requires_grad_()

        outputs = model(inputs_embeds=diffusion_embeds, output_hidden_states=True)
        language_feature = outputs.hidden_states[-1][:, unit_context_input_ids.size(1):]
        
        language_feature = language_feature.to(torch.float64) # [batch_size, seqlen, hidden_size]
        image_vector = image_vector.to(torch.float64) # [batch, 768]

        ctr_loss = torch.nn.functional.mse_loss(
            input=classifier_layer(torch.mean(language_feature,dim=1)), 
            target=image_vector,
            )
        args.ctr_loss = ctr_loss
        ctr_delta = -torch.autograd.grad(ctr_loss, diffusion_embeds, allow_unused=True)[0] # indexing 0 because the return is a tuple

    updated_diffusion_embeds = diffusion_embeds + coef * ctr_delta
        
    return updated_diffusion_embeds

def apply_controlling_drift_classifier_optimizer(
    args,
    language_feature, # torch.float32
    classifier_layer,
    image_vector,
    coef,
    K,
    lr,
    ):
    language_feature_param = torch.nn.Parameter(language_feature)

    with torch.enable_grad():
        for k in range(K):
            optimizer = torch.optim.AdamW(
                params=[language_feature_param], 
                lr=lr,
                )#, weight_decay=1e-5)
            optimizer.zero_grad()

            language_feature_param = language_feature_param.to(torch.float64) # [batch_size, seqlen, hidden_size]
            image_vector = image_vector.to(torch.float64) # [batch, 768]

            # ctr_loss = args.decode_ctr_lr * torch.nn.functional.mse_loss(
            ctr_loss = torch.nn.functional.mse_loss(
                input=classifier_layer(torch.mean(language_feature_param,dim=1)), 
                target=image_vector,
                )
            ctr_loss *= coef
            args.ctr_loss = ctr_loss  # torch.float64
            ctr_loss.backward()

            optimizer.step()
            print(f'diff of language feature before and after gradient update : {torch.sum(language_feature - language_feature_param.data)}')
            language_feature_param = torch.nn.Parameter((language_feature_param.data).detach())
    
    # 勾配更新によって、NaNやInfが発生していないか
    if torch.isnan(language_feature_param.data).any() or torch.isinf(language_feature_param.data).any():
        print("Warning: NaN or Inf detected in language_feature_param")

    return language_feature_param.data


def logits_projection(logits, top_p, one_hot_value):
    # print(f'logits size : {logits.size()}')
    assert len(logits.size()) == 3
    very_low_value = -10000

    # get top-p indices
    probs = torch.nn.functional.softmax(logits, dim=-1)
    sorted_probs, indices = torch.sort(probs, dim=-1, descending=True)
    cum_sum_probs = torch.cumsum(sorted_probs, dim=-1)
    nucleus = cum_sum_probs < top_p
    nucleus = torch.cat([nucleus.new_ones(nucleus.shape[:-1] + (1,)), nucleus[..., :-1]], dim=-1)
    valid_indices = nucleus.scatter(2, indices, nucleus)

    logits = logits.masked_fill(valid_indices == 0, very_low_value - one_hot_value)
    return torch.clamp(logits, max=very_low_value + one_hot_value) - very_low_value


def logits_uneven_projection(logits, top_p, one_hot_value):
    assert len(logits.size()) == 3
    very_low_value = -10000

    # get top-p indices
    probs = torch.nn.functional.softmax(logits, dim=-1)
    sorted_probs, indices = torch.sort(probs, dim=-1, descending=True)
    cum_sum_probs = torch.cumsum(sorted_probs, dim=-1)
    nucleus = cum_sum_probs < top_p
    nucleus = torch.cat([nucleus.new_ones(nucleus.shape[:-1] + (1,)), nucleus[..., :-1]], dim=-1)
    valid_indices = nucleus.scatter(2, indices, nucleus)

    filtered_logits = logits.masked_fill(valid_indices == 0, very_low_value)
    max_logits = torch.max(filtered_logits, -1, keepdim=True)[0]
    filtered_logits = filtered_logits - max_logits + one_hot_value # max logit gets +5, others keep the same diff with max logit
    return torch.clamp(filtered_logits, min=-one_hot_value)


def logits_sampling_projection(logits, top_p, one_hot_value):
    assert len(logits.size()) == 3
    very_low_value = -10000

    # get top-p indices
    probs = torch.nn.functional.softmax(logits, dim=-1)
    sorted_probs, indices = torch.sort(probs, dim=-1, descending=True)
    cum_sum_probs = torch.cumsum(sorted_probs, dim=-1)
    nucleus = cum_sum_probs < top_p
    nucleus = torch.cat([nucleus.new_ones(nucleus.shape[:-1] + (1,)), nucleus[..., :-1]], dim=-1)
    valid_indices = nucleus.scatter(2, indices, nucleus)

    filtered_logits = logits.masked_fill(valid_indices == 0, -float('Inf'))
    m = torch.distributions.categorical.Categorical(logits=filtered_logits)
    selected = m.sample()
    return 2 * one_hot_value * torch.nn.functional.one_hot(selected, logits.size(2)) - one_hot_value


def decode(
    args, 
    step,
    predict_indexes,
    image_vectors,
    batch_input_ids, 
    dec_depth, 
    total_t, 
    skip_t,
    model_embedding_lut, 
    embedding_sum_layer, 
    timestep_layer, 
    classifier_layer,
    model, 
    tokenizer,
    ):
    batch_size = 10 #args.per_device_eval_batch_size
    # args.decode_truncate_len = 
    batch_input_ids = batch_input_ids.unsqueeze(0).repeat(batch_size, 1)
    image_vectors = image_vectors.repeat(batch_size, 1, 1)
    assert image_vectors.size(0) == batch_size
    device = batch_input_ids.device
    
    if args.context_size > 0:
        unit_context_input_ids = batch_input_ids[:, :args.context_size].clone()
    else:
        unit_context_input_ids = None
    history_decode_ids = None
    history_decode_probs = None

    for dep_step, (start_index, end_index) in enumerate(predict_indexes):
        unit_seq_len = end_index - start_index

        unit_noise = args.noise_manual_scale * args.one_hot_value * torch.normal(0, 1, size=(batch_size, unit_seq_len, args.vocab_size)).to(device)
        xt = unit_noise

        if unit_context_input_ids is not None:
            context_inputs_embeds = model_embedding_lut(unit_context_input_ids)
        else:
            context_inputs_embeds = None

        t_range = list(range(1, args.sigma_num_steps+1, skip_t))
        t_range.reverse()
        
        for t in tqdm(t_range):
            selected_t = torch.FloatTensor([t]).repeat(batch_size).to(device)
            alpha_t_bar, alpha_t_minus_bar, beta_t, beta_t_til, alpha_t = get_time_variables(selected_t, total_t, device)
            beta_t_til = beta_t_til.view(batch_size, 1, 1)
            zt = args.noise_manual_scale * args.one_hot_value * torch.normal(0, 1, size=(batch_size, unit_seq_len, args.vocab_size)).to(device)
            
            perturbed_inputs_diralpha = xt
            
            mean_or_protect_for_nan = True # (HACK: for the nan issue)
            if mean_or_protect_for_nan:
                perturbed_inputs_simplex = torch.nn.functional.softmax(perturbed_inputs_diralpha, dim=-1) # HACK: only for mean of dirichlet, not for sample
            else:
                perturbed_inputs_diralpha = torch.exp(perturbed_inputs_diralpha)
                dir_model = torch.distributions.dirichlet.Dirichlet(perturbed_inputs_diralpha)
                perturbed_inputs_simplex = dir_model.sample()

            # pass to the model, conditioned on the timestep as well
            perturbed_inputs_embeds = embedding_sum_layer(perturbed_inputs_simplex)
            t_progress = selected_t / total_t
            timestep_embeds = timestep_layer(t_progress.view(batch_size,1,1).repeat(1,unit_seq_len,1))

            diffusion_embeds = perturbed_inputs_embeds + timestep_embeds
            if context_inputs_embeds is not None:
                diffusion_embeds = torch.cat((context_inputs_embeds, diffusion_embeds), dim=1)
            
            # 条件に応じて潜在変数を勾配更新
            # print(f'step{step},t{t},depstep{dep_step},updated_diffusion_embeds diff: {torch.sum(updated_diffusion_embeds - diffusion_embeds)}')
            if t>args.decode_ctr_tmax:
                updated_diffusion_embeds = apply_controlling_drift_classifier(
                    args, 
                    model,
                    classifier_layer,
                    diffusion_embeds,
                    unit_context_input_ids,
                    image_vectors[:, dep_step], # [batch_size, 19-???, 768] -> [batch_size, 768]
                    args.decode_ctr_lr,
                    )
            elif t>args.decode_ctr_tmin:
                updated_diffusion_embeds = apply_controlling_drift_classifier(
                    args, 
                    model,
                    classifier_layer,
                    diffusion_embeds,
                    unit_context_input_ids,
                    image_vectors[:, dep_step], # [batch_size, 19-???, 768] -> [batch_size, 768]
                    args.decode_ctr_lr/10,
                    )
            else:
                # updated_diffusion_embeds = diffusion_embeds
                updated_diffusion_embeds = apply_controlling_drift_classifier(
                    args, 
                    model,
                    classifier_layer,
                    diffusion_embeds,
                    unit_context_input_ids,
                    image_vectors[:, dep_step], # [batch_size, 19-???, 768] -> [batch_size, 768]
                    args.decode_ctr_lr/100,
                    )
            
            outputs = model(inputs_embeds=updated_diffusion_embeds, output_hidden_states=True)
            equivalent_score = outputs.logits
            if unit_context_input_ids is not None:
                equivalent_score = equivalent_score[:, unit_context_input_ids.size(1):].contiguous()

            if t > 1:
                sigma_t = torch.sqrt(beta_t_til)
            else:
                sigma_t = 0
            
            # xt-1 を求める
            if args.loss_mode == "l2_on_z":
                raise NotImplementedError("l2_on_z samping is not implemented yet")
            else:
                if args.projection_alg == "even":
                    projected_logits = logits_projection(
                        equivalent_score, 
                        top_p=args.projection_top_p, 
                        one_hot_value=args.one_hot_value,
                        )
                elif args.projection_alg == "sampling":
                    projected_logits = logits_sampling_projection(equivalent_score, top_p=args.projection_top_p, one_hot_value=args.one_hot_value)
                else:
                    raise ValueError("Unknown projection algorithm")
                
                xtminus1 = torch.sqrt(alpha_t_minus_bar).view(-1, 1, 1) * projected_logits
                xtminus1 = xtminus1 + torch.sqrt(1 - alpha_t_minus_bar).view(-1, 1, 1) * zt
                
                xt = xtminus1

            # 何回かに一回表示するのと、タイムステップ最後
            if (t-1) % 200 == 0 or t == 1:
                simplex = torch.nn.functional.softmax(xt, dim=-1) # HACK: only for mean of dirichlet, not for sample

                if unit_context_input_ids is not None:
                    context_sequences = tokenizer.batch_decode(unit_context_input_ids.detach().to('cpu'), skip_special_tokens=True)
                    # logger.info(f"context: {context_sequences}")
                
                real_token_ids_list = torch.argmax(simplex, dim=-1).view(batch_size, unit_seq_len)
                sampled_sequences = tokenizer.batch_decode(real_token_ids_list.clone().detach().to('cpu'), skip_special_tokens=True)
                # logger.info(f"t={t}: {colored(str(sampled_sequences), 'red')}")

                # 
                simplex = equivalent_score
                topk_values, topk_indices = torch.topk(simplex, k=1, dim=-1)
                real_token_ids_list = topk_indices.view(batch_size, unit_seq_len)
                # real_token_ids_list = torch.argmax(simplex, dim=-1).view(batch_size, unit_seq_len)
                real_token_probs_list = topk_values.view(batch_size, unit_seq_len)
                sampled_sequences = tokenizer.batch_decode(real_token_ids_list.clone().detach().to('cpu'), skip_special_tokens=True)
                # logger.info(f"t={t} (before +z): {colored(str(sampled_sequences), 'green')}")
                
                logger.info(f"ctr loss: {args.ctr_loss}")

                if simplex.size(0) > 0 and simplex.size(1) > 0:
                    logger.info(f"non-zero vocab: {torch.count_nonzero(projected_logits > -args.one_hot_value+0.0001) / simplex.size(0) / simplex.size(1)} out of {torch.numel(projected_logits) / simplex.size(0) / simplex.size(1)}")
                else:
                    logger.warning("simplex.size(0) or simplex.size(1) is zero.")
                # logger.info(f"non-zero vocab: {torch.count_nonzero(projected_logits > -args.one_hot_value+0.0001) / simplex.size(0) / simplex.size(1)} out of {torch.numel(projected_logits) / simplex.size(0) / simplex.size(1)}")
        print(f'{dep_step} : {sampled_sequences}')
        # T回終わった
        unit_context_input_ids = torch.cat((unit_context_input_ids, real_token_ids_list), dim=1)
        if history_decode_ids is None: # 初回だね
            history_decode_ids = real_token_ids_list
            history_decode_probs = real_token_probs_list
        else:
            history_decode_ids = torch.cat((history_decode_ids, real_token_ids_list), dim=1)
            history_decode_probs = torch.cat((history_decode_probs, real_token_probs_list), dim=1)

    sampled_sequences = tokenizer.batch_decode(history_decode_ids.clone().detach().to('cpu'), skip_special_tokens=True)

    return history_decode_ids, history_decode_probs, sampled_sequences


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a Masked Language Modeling task")
    # Han: many arguments below will not be used, but keeping for future edits
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library). For example, Wikipedia.",
    )
    parser.add_argument(
        "--additional_dataset_name",
        type=str,
        default=None,
        help="The name of the additional dataset to use (via the datasets library). For example, BookCorpus.",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--validation_split_percentage",
        default=5,
        help="The percentage of the train set used as validation set in case there's no validation split",
    )
    parser.add_argument(
        "--raw_data_percentage",
        default=100,
        help="The percentage of raw data used as the train set",
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--data_cap",
        type=int,
        default=2,
        help="Max number of data for which we will save graidents.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--subdir", type=str, default=None, help="model name")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=None,
        help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated.",
    )
    parser.add_argument(
        "--line_by_line",
        type=bool,
        default=False,
        help="Whether distinct lines of text in the dataset are to be handled as distinct sequences.",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", type=bool, default=False, help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--mlm_probability", type=float, default=0.0, help="Ratio of tokens to mask for masked language modeling loss"
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument("--no_save_grads", action="store_true", help="Whether to save gradients to a file.")
    # for computing influence scores w.r.t. the querying file
    parser.add_argument(
        "--query_file", type=str, default=None, help="A pickle file containing gradient information from the querying data."
    )
    parser.add_argument(
        "--query_data_cap", type=int, default=None, help="Max number of data for which we will save gradients.",
    )
    parser.add_argument("--influence_metric", type=str, default=None, help="Metric for computing the gradients.")
    parser.add_argument("--init_blank_language_model", action="store_true", help="Whether or not to use a completely blank LM.")
    parser.add_argument(
        "--tokenized_data_file_path", type=str, default=None, help="Path of the tokenized data file."
    )
    parser.add_argument(
        "--if_create_tokenized_data_file", type=str, default=None, help="Whether to create a new tokenized data file (yes or no)."
    )
    parser.add_argument(
        "--sigma_start_value", type=float, default=-1, help="",
    )
    parser.add_argument(
        "--sigma_end_value", type=float, default=-1, help="",
    )
    parser.add_argument(
        "--sigma_num_steps", type=int, default=1000, help="",
    )
    parser.add_argument(
        "--loss_mode", type=str, default="", help="",
    )
    parser.add_argument(
        "--remove_noise_mode", type=str, default="", help="",
    )
    parser.add_argument(
        "--hardcoded_pseudo_diralpha", type=float, default=3, help="",
    )
    parser.add_argument(
        "--context_size", type=int, default=0, help="",
    )
    parser.add_argument(
        "--decoding_block_size", type=int, default=25, help="",
    )
    parser.add_argument(
        "--train_mode", type=str, default="", help="",
    )
    parser.add_argument(
        "--noise_manual_scale", type=float, default=1, help="",
    )
    parser.add_argument(
        "--decode_context_size", type=int, default=25, help="",
    ) # how many to cut from left
    parser.add_argument(
        "--decode_truncate_len", type=int, default=50, help="",
    ) # how many to cut from right
    parser.add_argument(
        "--decode_depth", type=int, default=2, help="",
    )
    parser.add_argument(
        "--decode_ctr_lr", type=float, default=0.0, help="",
    )
    parser.add_argument(
        "--decode_ctr_tmax", type=float, default=0.0, help="",
    )
    parser.add_argument(
        "--decode_ctr_tmin", type=float, default=0.0, help="",
    )
    parser.add_argument(
        "--out_fn", type=str, default="_sample_gen.jsonl", help="",
    )
    parser.add_argument(
        "--projection_top_p", type=float, default=0.2, help="",
    )
    parser.add_argument(
        "--projection_alg", type=str, default="even", help="",
    ) # even, sampling
    parser.add_argument(
        "--ctr_opt_label_idx", type=int, default=0, help="",
    ) # 0 (neg in sentiment), 2 (pos in sentiment)
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    setup_dist()
    world_size = dist.get_world_size() or 1
    rank = dist.get_rank() or 0

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
        
    # Handle the repository creation
    if rank==0:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # under decode mode, will load model from the output_dir
    if args.train_mode == "decode":
        args.model_name_or_path = args.output_dir
        logger.info(f"Overwriting model_name_or_path ({args.model_name_or_path}) with {args.output_dir}")
    else:
        raise ValueError(
            "train_mode should be decode to perform this file"
        )

    # Load pretrained model and tokenizer
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    assert args.use_slow_tokenizer == True
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    test_dataset = load_data_text(
        tokenizer=tokenizer,
        split='test',
    )
    test_dataloader = iter(DataLoader(
        test_dataset, 
        batch_size=1,
        drop_last=False,
        shuffle=False,
        ))

    all_test_data = []
    idx = 0
    try:
        while True:
            cond = next(test_dataloader)
            if idx % world_size == rank:  # Split data per nodes
                all_test_data.append(cond)
            idx += 1
    except StopIteration:
        print('### End of reading iteration...')

    if idx % world_size and rank >= idx % world_size:
        all_test_data.append({})  # Dummy data for Remainder : for dist.barrier()

    if rank == 0:
        from tqdm import tqdm
        iterator = tqdm(all_test_data)
    else:
        iterator = iter(all_test_data)

    ########

    # # If we want to use a non-existing architecture, we can do it here
    # config.hidden_size = 1600
    # config.intermediate_size = 4096
    # config.max_position_embeddings = 128
    # config.num_attention_heads = 25
    # config.num_hidden_layers = 48

    if args.init_blank_language_model:
        model = AutoModelForMaskedLM.from_config(config)
    elif args.model_name_or_path:
        model = AutoModelForMaskedLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        )
    else:
        raise ValueError("specify --init_blank_language_model")
        logger.info("Training new model from scratch")
        model = AutoModelForMaskedLM.from_config(config)

    model.resize_token_embeddings(len(tokenizer))

    vocab_size = model.get_input_embeddings().weight.size(0)
    hidden_size = model.get_input_embeddings().weight.size(1)
    embedding_sum_layer = torch.nn.Linear(vocab_size, hidden_size, bias=False)
    with torch.no_grad():
        embedding_sum_layer.weight.copy_(torch.transpose(model.get_input_embeddings().weight.clone(), 0, 1))
    timestep_layer = torch.nn.Linear(1, hidden_size, bias=True)
    classifier_layer = torch.nn.Linear(hidden_size, 768, bias=False).double()

    model_embedding_lut = model.get_input_embeddings()

    total_t = args.sigma_num_steps
    one_hot_value = args.hardcoded_pseudo_diralpha # for a pseudo one-hot encoding for alpha

    args.remove_noise_mode = args.remove_noise_mode.split('|')
    args.noise_analysis_list = list()

    if args.train_mode == "train" or args.train_mode == "resume":
        raise ValueError("Training or resuming is disabled here")

    ##########################################

    # out_json_fn = os.path.join(args.output_dir, f"ctx{args.decode_context_size}_trunc{args.decode_truncate_len}_depth{args.decode_depth}_ctrlr{args.decode_ctr_lr}_step{args.sigma_num_steps}_topp{args.projection_top_p}_decalg{args.projection_alg}_ctridx{args.ctr_opt_label_idx}_" + args.out_fn)
    # out_json_fn = os.path.join("/home/hirano/ssd-lm", f"ctx{args.decode_context_size}_trunc{args.decode_truncate_len}_depth{args.decode_depth}_ctrlr{args.decode_ctr_lr}_step{args.sigma_num_steps}_topp{args.projection_top_p}_decalg{args.projection_alg}_ctridx{args.ctr_opt_label_idx}_" + args.out_fn)
    out_json_fn = os.path.join("/home/hirano/abel_home/container_target/appendix", f"model{args.subdir}_average_t>{args.decode_ctr_tmax}-ctrlr{args.decode_ctr_lr}_t>{args.decode_ctr_tmin}-ctrlr{args.decode_ctr_lr/10}_t>0-ctrlr{args.decode_ctr_lr/100}" + args.out_fn)
    if rank==0:
        if os.path.exists(out_json_fn):
            os.remove(out_json_fn)
            logger.info(f"Cleaning existing {out_json_fn}")

    # Decoding, includes hardcode for now
    if args.train_mode == "decode":
        _stdict = torch.load(os.path.join(args.output_dir, "embed_sum_layer.pt"))
        # _stdict = dict((f"module.{_k}", _stdict[_k]) if not _k.startswith("module.") else (_k, _stdict[_k]) for _k in _stdict)
        embedding_sum_layer.load_state_dict(_stdict)

        _stdict = torch.load(os.path.join(args.output_dir, "timestep_layer.pt"))
        # _stdict = dict((f"module.{_k}", _stdict[_k]) if not _k.startswith("module.") else (_k, _stdict[_k]) for _k in _stdict)
        timestep_layer.load_state_dict(_stdict)

        _stdict = torch.load(os.path.join(args.output_dir, "classifier_layer.pt"))
        # _stdict = dict((f"module.{_k}", _stdict[_k]) if not _k.startswith("module.") else (_k, _stdict[_k]) for _k in _stdict)
        classifier_layer.load_state_dict(_stdict)

        model.eval().to(dev())
        model_embedding_lut.eval().to(dev())
        embedding_sum_layer.eval().to(dev())
        timestep_layer.eval().to(dev())
        classifier_layer.eval().to(dev())

        args.sigma_noise_scale = 1.0
        args.interpolation_with_prev = 0.0

        args.context_size = args.decode_context_size
        args.one_hot_value = one_hot_value
        args.vocab_size = vocab_size
        args.ctr_model = None

        # annotatorごとの発語速度
        json_path = '/Storage2/hirano/container_target/localized_narratives/karpathy/20240618/annotator_utterancespeed_average.json'
        with open(json_path, 'r', encoding='utf-8') as file:
            annotator_utterancespeed_dict = json.load(file)

        task_start_time = time.time()

        if "fin" in args.remove_noise_mode:
            args.orig_decode_truncate_len = args.decode_truncate_len
            with torch.no_grad():
                for step, batch in enumerate(iterator):
                    if not batch:  # Barrier for Remainder
                        for i in range(world_size):
                            dist.barrier()
                        continue
                    
                    # print(f"{batch['image_id'].item()} {batch['annotator_id'].item()}")
                    # 予測単語数作成
                    durations = batch['durations']
                    assert durations.shape == torch.Size([1,19])
                    durations = torch.squeeze(durations,0)
                    assert durations.shape == torch.Size([19])
                    
                    predict_indexes = []
                    start_index = 0
                    end_index = 0
                    for adopt_index, duration in enumerate(durations):
                        if duration==-1:
                            break
                        token_num = torch.ceil(duration * annotator_utterancespeed_dict[str(batch['annotator_id'].item())]).int().item()
                        # token_num = torch.ceil(duration * 2.3).int().item()
                        end_index+=token_num
                        predict_indexes.append((start_index, end_index))
                        start_index = end_index

                    image_vectors = batch['image_vectors'][:, :adopt_index, :].to(dev()) # torch.Size([1, 19, 768]) -> torch.Size([1, 19-???, 768])
                    assert len(predict_indexes)==image_vectors.size(1)

                    assert args.per_device_eval_batch_size == 1

                    input_ids = torch.LongTensor([tokenizer.bos_token_id]).to(dev())
                    args.context_size = len(input_ids)
                    args.decode_truncate_len = args.orig_decode_truncate_len - args.context_size 
                    
                    # 生成
                    history_decode_ids, history_decode_probs, sampled_sequences = \
                            decode(
                                args, 
                                step,
                                predict_indexes,
                                image_vectors,
                                input_ids, 
                                args.decode_depth, 
                                total_t, 
                                10,#skip_t
                                model_embedding_lut, 
                                embedding_sum_layer, 
                                timestep_layer, 
                                classifier_layer,
                                model, 
                                tokenizer,
                                )
                    div_num=1
                    # print(history_decode_probs)
                    history_decode_ppl = torch.prod(history_decode_probs, dim=-1) # [batch_size]
                    history_decode_ppl_topk_values, history_decode_ppl_topk_indices = torch.topk(history_decode_ppl, k=10)
                    while torch.isinf(history_decode_ppl_topk_values).any().item():
                        div_num*=10
                        history_decode_ppl = torch.prod(history_decode_probs/div_num, dim=-1) # [batch_size]

                        history_decode_ppl_topk_values, history_decode_ppl_topk_indices = torch.topk(history_decode_ppl, k=10)
                    print(f'div_num : {div_num}')
                    logger.info(f"history_decode_ppl_topk_values {history_decode_ppl_topk_values}")

                    export_dict = dict()
                    export_dict['image_id'] = batch['image_id'].item()
                    export_dict['annotator_id'] = batch['annotator_id'].item()
                    export_dict['indexes'] = batch['index'].tolist()
                    export_dict['gold_string'] = tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True)
                    export_dict['string'] = []
                    for decode_indice in history_decode_ppl_topk_indices:
                        export_dict['string'].append(sampled_sequences[decode_indice])

                    for i in range(world_size):
                        if i == rank:  # Write files sequentially
                            with open(out_json_fn, mode='a') as f_out:
                                f_out.write(json.dumps(export_dict))
                                f_out.write("\n")
                        dist.barrier()
                    
                    print(f'END of BATCH{step}　on device{torch.cuda.current_device()}')
        
        else:
            raise ValueError("correct mode should be included in remove_noise_mode, a string separated by pipes")
        
        print('### Total takes {:.2f}s .....'.format(time.time() - task_start_time))
        print(f'### Written the decoded output to {out_json_fn}')


if __name__ == "__main__":
    main()
