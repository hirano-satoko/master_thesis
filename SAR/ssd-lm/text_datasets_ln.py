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

    with torch.no_grad():
        total_length=0
        for image_id, annotator_id, caption, indexes, imagefeatures, durations in data_lst:
            caption = ' '.join(caption)
            input_ids = tokenizer.encode(
                caption,                  # List of sentences to tokenize
                padding='max_length',     # Pads to max_length
                truncation=True,          # Truncates to max_length if longer
                max_length=200,           # Set the length to 200
                return_tensors=None,      # Returns lists instead of tensors
                )
            group_lst['word_ids'].append(input_ids)

            max_image_vectors_len = 19
            while len(indexes)!=max_image_vectors_len:
                indexes.append((-1,-1))
            
            while len(imagefeatures)!=max_image_vectors_len:
                imagefeatures.append([0]*768)
            
            while len(durations)!=max_image_vectors_len:
                durations.append(-1)
            
            group_lst['index'].append(indexes)
            assert imagefeatures is not None, "imagefeaturesには有効なデータが格納されていません。"
            group_lst['image_vectors'].append(imagefeatures)
            assert group_lst['image_vectors'] is not None, "group_lst[image_vectors]には有効なデータが格納されていません。"
            group_lst['durations'].append(durations)

            group_lst['image_id'].append(image_id)
            group_lst['annotator_id'].append(annotator_id)

        print(len(group_lst['word_ids']))
        assert len(group_lst['word_ids']) == len(group_lst['index'])
        assert len(group_lst['word_ids']) == len(group_lst['image_vectors'])
        assert len(group_lst['word_ids']) == len(group_lst['durations'])
        assert len(group_lst['word_ids']) == len(group_lst['image_id'])
        assert len(group_lst['word_ids']) == len(group_lst['annotator_id']), f"{len(group_lst['word_ids'])}vs{len(group_lst['annnotator_id'])}"

        print('get hidden states')
        for image_id, annotator_id, input_ids, indexes, image_vectors, durations in zip(group_lst['image_id'], group_lst['annotator_id'], group_lst['word_ids'], group_lst['index'], group_lst['image_vectors'],  group_lst['durations']):
            assert len(input_ids)==200
            assert len(image_vectors)==19
            assert len(indexes)==19
            assert len(durations)==19

            assert image_vectors is not None, "変数には有効なデータが格納されていません。"
            
            result_train_lst.append({'image_id':image_id, 'annotator_id':annotator_id, 'input_ids': input_ids, 'index':indexes, 'image_vectors':image_vectors, 'durations':durations})
    
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
        path = f'{data_path}/localized_narratives_train.h5'
    elif split == 'valid':
        print('loading form the VALID set')
        path = f'{data_path}/localized_narratives_valid.h5'
    elif split == 'test':
        print('loading form the TEST set')
        path = f'{data_path}/localized_narratives_test.h5'

    print('data load path : ',path)
    for i in range(5):
        df = pd.read_hdf(path, f'data{i}')
        for index, row in df.iterrows():
            # if split == 'test':
            #     if index>11:
            #         break
            if str(row['image_id']) in ['263211', '128140', '570521']:
                data_lst.append([row['image_id'], row['annotator_id'], row['caption'], row['indexes'], row['imagefeatures'], row['durations']])

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
        data_path = '/Storage2/hirano/container_target/localized_narratives/karpathy/20240618',
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
        out_dict['annotator_id'] = self.text_datasets['train'][idx]['annotator_id']
        out_dict['input_ids'] = np.array(self.text_datasets['train'][idx]['input_ids'])
        out_dict['index'] = np.array(self.text_datasets['train'][idx]['index'])
        out_dict['image_vectors'] = np.array(self.text_datasets['train'][idx]['image_vectors'])
        out_dict['durations'] = np.array(self.text_datasets['train'][idx]['durations'])

        assert out_dict['image_vectors'] is not None, "out_dict[image_vectors]には有効なデータが格納されていません。"
        
        return out_dict