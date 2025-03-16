import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer, default_data_collator, PreTrainedTokenizerFast, \
    PreTrainedTokenizer
from datasets import load_dataset
import sys, os
import torch

from collections import Counter, defaultdict
from functools import partial
from itertools import chain

from PIL import Image

import json
import pandas as pd


def helper_tokenize_encode(
    data_lst, 
    tokenizer,
    split, 
    ):
    
    result_train_lst = [] # return value
    group_lst = defaultdict(list)

    # 長さを得るための準備
    special_tokens_ids = []
    special_tokens_dict = tokenizer.special_tokens_map
    for token in special_tokens_dict:
        special_tokens_ids.append(tokenizer.convert_tokens_to_ids(special_tokens_dict[token]))

    # 画像特徴量の準備
    if split == 'train':
        print('loading karpathy blip feature TRAIN set')
        feature_path = '/Storage2/hirano/container_target/blip_feature/image_feature/karpathy_train_blipfeature.txt'
    elif split == 'valid':
        print('loading karpathy blip feature VALID set')
        feature_path = '/Storage2/hirano/container_target/blip_feature/image_feature/karpathy_val_blipfeature.txt'
    elif split == 'test':
        print('loading karpathy blip feature TEST set')
        feature_path = '/Storage2/hirano/container_target/blip_feature/image_feature/karpathy_test_blipfeature.txt'
    with open(feature_path, 'r') as f:
        id2feature_dict = json.load(f)

    with torch.no_grad():
        for image_id, caption in data_lst:
            # image_id
            group_lst['image_id'].append(image_id)
            # caption
            caption = ' '.join(caption)
            input_ids = tokenizer.encode(
                caption,                  # List of sentences to tokenize
                padding='max_length',     # Pads to max_length
                truncation=True,          # Truncates to max_length if longer
                max_length=200,           # Set the length to 200
                return_tensors=None,      # Returns lists instead of tensors
                )
            group_lst['word_ids'].append(input_ids)
            # length
            num_tokens_before_padding = len([input_id for input_id in input_ids if input_id not in special_tokens_ids])
            group_lst['length'].append(num_tokens_before_padding)
            # imagefeatures
            imagefeatures = id2feature_dict[image_id]
            group_lst['image_vectors'].append(imagefeatures)

        print(len(group_lst['word_ids']))
        assert len(group_lst['word_ids']) == len(group_lst['length'])
        assert len(group_lst['word_ids']) == len(group_lst['image_vectors'])
        assert len(group_lst['word_ids']) == len(group_lst['image_id']), f"{len(group_lst['word_ids'])}vs{len(group_lst['image_id'])}"

        print('get hidden states')
        for image_id, input_ids, length, image_vectors in zip(group_lst['image_id'], group_lst['word_ids'], group_lst['length'], group_lst['image_vectors']):
            assert len(input_ids)==200

            result_train_lst.append({'image_id':image_id, 'input_ids': input_ids, 'length':length, 'image_vectors':image_vectors})
    
    print('helper_tokenize_encode ends.')
    # print(hirano.shape)
    return result_train_lst

def get_corpus_rocstory(
    tokenizer,
    data_path,
    split='train', 
    ):

    print('loading dataset from simple e2e dataset')
    data_lst = []

    if split == 'train':
        print('loading form the TRAIN set')
        path = f'{data_path}/karpathy_train.txt'
    elif split == 'valid':
        print('loading form the VALID set')
        path = f'{data_path}/karpathy_val.txt'
    elif split == 'test':
        print('loading form the TEST set')
        path = f'{data_path}/karpathy_test.txt'

    index=0
    with open(path, 'r') as ff:
        for row in ff:
            dataDir, filename, imageid, captions = row.split('||')
            caption_lst = captions.rstrip().split('|')
            for caption in caption_lst:
                data_lst.append([imageid, caption])
            index+=1
            # if index>100:
            #     break

    print('length of data_lst : ', len(data_lst))
            
    result_train_lst = helper_tokenize_encode(
        data_lst=data_lst, 
        tokenizer=tokenizer,
        split=split,
        )
    
    print('length of result_train_lst : ',len(result_train_lst))
    return {'train': result_train_lst}

def load_data_text(
    tokenizer,
    split='train', 
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    :param data_dir: a dataset directory.
    """
    print('hello loading text data. ')

    training_data = get_corpus_rocstory(
        tokenizer = tokenizer,
        data_path = '/Storage/hirano/mscoco_dataset/karpathy',
        split=split,
        )
    
    dataset = TextDataset(
        training_data,
    )

    return dataset


class TextDataset(Dataset):
    def __init__(
        self, 
        text_datasets, 
        ):
        
        super().__init__()
        self.text_datasets = text_datasets
        self.length = len(self.text_datasets['train'])
       
    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        out_dict = {}
        out_dict['image_id'] = self.text_datasets['train'][idx]['image_id']
        out_dict['input_ids'] = np.array(self.text_datasets['train'][idx]['input_ids'])
        out_dict['length'] = np.array(self.text_datasets['train'][idx]['length'])
        out_dict['image_vectors'] = np.array(self.text_datasets['train'][idx]['image_vectors'])

        assert out_dict['image_vectors'] is not None, "out_dict[image_vectors]には有効なデータが格納されていません。"
        
        return out_dict