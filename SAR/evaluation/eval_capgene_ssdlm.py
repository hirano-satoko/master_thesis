'''
20241015
ssd-lm評価
'''

import os, sys, glob, json
import numpy as np
import pandas as pd
import argparse
from torch import torch

from torchmetrics.text.rouge import ROUGEScore
rougeScore = ROUGEScore()

from bert_score import BERTScorer
# モデルを事前にロード
bertscorer = BERTScorer(model_type='microsoft/deberta-xlarge-mnli', lang='en')

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk

from tqdm import tqdm
import csv

from scipy.spatial.distance import cosine
from scipy.stats import pearsonr

from openai import OpenAI
# os.environ['OPENAI_API_KEY'] = ''
client = OpenAI()
embedding_model = "text-embedding-3-small"

def get_seg_bleu(recover, reference, start_index, end_index, weights=(1/4,1/4,1/4,1/4)):
    return sentence_bleu([reference.split()[start_index:end_index]], recover.split()[start_index:end_index],  weights=weights, smoothing_function=SmoothingFunction().method4,)

def get_bleu(recover, reference, weights=(1/4,1/4,1/4,1/4)):
    return sentence_bleu([reference.split()], recover.split(),  weights=weights, smoothing_function=SmoothingFunction().method4,)

def get_embedding(text, model=embedding_model):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding

def eval_BRB(
    outputpath, 
    mbrby, 
    save,
    ):
    image_ids = []
    annotator_ids = []
    bleu1 = []
    bleu2 = []
    bleu3 = []
    bleu4 = []
    rougel = []
    avg_len = []
    dist1 = []
    bertscore = []

    seg_bleu1 = []
    seg_bleu2 = []
    seg_bleu3 = []
    seg_bleu4 = []
    seg_rougel = []
    seg_bertscore = []
    seg_cosine = []
    seg_pearson = []

    export_list = []

    counter=0

    with open(outputpath, 'r') as file:
        for line in tqdm(file):
            item = json.loads(line)
            image_id = item['image_id']
            annotator_id = item['annotator_id']
            
            max_score = 0

            # 生成データ
            recovers = item['string']    
            # 正解データ
            reference = item['gold_string']
            assert len(reference)==1
            reference = reference[0]
            # インデックス
            indexes = item['indexes']

            if mbrby == 'rouge':
                for recover in recovers:
                    # get_bleu(mbr_recover, reference)
                    score = rougeScore(recover, reference)['rougeL_fmeasure']
                    # P, R, F1 = bertscorer.score([recover], [reference])
                    # if F1.item() > max_score:
                    if score > max_score:
                        mbr_recover = recover
            elif mbrby == 'perplexity':
                mbr_recover = recovers[0]

            avg_len.append(len(mbr_recover.split(' ')))

            seg_bleu1_lst = []
            seg_bleu2_lst = []
            seg_bleu3_lst = []
            seg_bleu4_lst = []
            seg_rougel_lst = []
            seg_bertscore_lst = []
            seg_cosine_lst = []
            seg_pearson_lst = []

            # 保存のために各データごとに初期化
            export_dict = dict()
            export_indexes = []

            # bboxごとの
            for (s,e) in indexes[0]:
                if (s,e)==(-1,-1):
                    break
                if s==e:
                    continue
                if not mbr_recover[s:e] or not reference[s:e]:
                    continue
                if isinstance(mbr_recover[s:e], str) and not mbr_recover[s:e].strip():
                    continue
                if isinstance(reference[s:e], str) and not reference[s:e].strip():
                    continue

                if len(mbr_recover[s:e]) == 0 or len(reference[s:e]) == 0:
                    continue

                seg_bleu1_lst.append(get_seg_bleu(mbr_recover, reference, s, e, (1,0,0,0)))
                seg_bleu2_lst.append(get_seg_bleu(mbr_recover, reference, s, e, (1/2, 1/2, 0,0)))
                seg_bleu3_lst.append(get_seg_bleu(mbr_recover, reference, s, e, (1/3, 1/3, 1/3, 0)))
                seg_bleu4_lst.append(get_seg_bleu(mbr_recover, reference, s, e))
                seg_rougel_lst.append(rougeScore(mbr_recover[s:e], reference[s:e])['rougeL_fmeasure'].tolist())
                ### bertscore
                P, R, F1 = bertscorer.score([mbr_recover[s:e]], [reference[s:e]])
                seg_bertscore_lst.append(F1.item())
                ### openai embedding
                mbr_recover_embedding = get_embedding(' '.join(mbr_recover[s:e]))
                reference_embedding = get_embedding(' '.join(reference[s:e]))
                # (3). コサイン類似度 (Cosine Similarity)
                cos_sim = 1 - cosine(mbr_recover_embedding, reference_embedding)  # コサイン距離は1から引く
                seg_cosine_lst.append(cos_sim)
                # (4). ピアソン相関係数 (Pearson Correlation Coefficient)
                pearson_corr, _ = pearsonr(mbr_recover_embedding, reference_embedding)
                seg_pearson_lst.append(pearson_corr)

                export_indexes.append((s,e))
            
            image_ids.append(image_id)
            annotator_ids.append(annotator_id)
            # segごと
            seg_bleu1.append(np.mean(seg_bleu1_lst))
            seg_bleu2.append(np.mean(seg_bleu2_lst))
            seg_bleu3.append(np.mean(seg_bleu3_lst))
            seg_bleu4.append(np.mean(seg_bleu4_lst))
            seg_rougel.append(np.mean(seg_rougel_lst))
            seg_bertscore.append(np.mean(seg_bertscore_lst))
            seg_cosine.append(np.mean(seg_cosine_lst))
            seg_pearson.append(np.mean(seg_pearson_lst))
            
            ## 保存用
            export_dict['image_id'] = image_id
            export_dict['annotator_id'] = annotator_id
            export_dict['indexes'] = export_indexes
            export_dict['gold_string'] = item['gold_string']
            export_dict['string'] = mbr_recover
            export_list.append(export_dict)

            # キャプション全体
            bleu1.append(get_bleu(mbr_recover, reference, (1,0,0,0)))
            bleu2.append(get_bleu(mbr_recover, reference, (1/2, 1/2, 0,0)))
            bleu3.append(get_bleu(mbr_recover, reference, (1/3, 1/3, 1/3, 0)))
            bleu4.append(get_bleu(mbr_recover, reference))
            rougel.append(rougeScore(mbr_recover, reference)['rougeL_fmeasure'].tolist())
            ### bertscore
            P, R, F1 = bertscorer.score([mbr_recover], [reference])
            bertscore.append(F1.item())

            counter+=1
            # if counter>10:
            #     break

    if save:
        out_json_fn = f'{os.path.splitext(outputpath)[0]}_mbr{os.path.splitext(outputpath)[1]}'
        with open(out_json_fn, mode="w") as f_out:
            for export in export_list:
                f_out.write(json.dumps(export))
                f_out.write("\n")

    print('*'*30)
    print('avg seg_BLEU1 score', np.mean(seg_bleu1))
    print('avg seg_BLEU2 score', np.mean(seg_bleu2))
    print('avg seg_BLEU3 score', np.mean(seg_bleu3))
    print('avg seg_BLEU4 score', np.mean(seg_bleu4))
    print('avg seg_ROUGE-L score', np.mean(seg_rougel))
    print('avg seg BERTSore', np.mean(seg_bertscore))
    print('avg seg COSINE', np.mean(seg_cosine))
    print('avg seg PEARSONR', np.mean(seg_pearson))
    print('avg len', np.mean(avg_len))


    # 書き込み対象のCSVファイル名
    csv_file = "/home/hirano/interactive_captiongeneration/evaluation/output.csv"
    # ヘッダー行
    headers = ["image_id", "annotator_id", "seg_bleu1", "seg_rougel", "seg_bertscore"]
    # データ行を作成
    rows = zip(image_ids, annotator_ids, bleu1, rougel, bertscore)
    # CSVに書き込み
    with open(csv_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(headers)  # ヘッダーを書き込み
        writer.writerows(rows)   # データ行を書き込み

    print('*'*30)
    print('avg BLEU1 score', np.mean(bleu1))
    print('avg BLEU2 score', np.mean(bleu2))
    print('avg BLEU3 score', np.mean(bleu3))
    print('avg BLEU4 score', np.mean(bleu4))
    print('avg ROUGE-L score', np.mean(rougel))
    print('avg BERTScore score', np.mean(bertscore))

    return np.mean(seg_bleu1), np.mean(seg_rougel), np.mean(seg_bertscore), np.mean(avg_len)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='decoding args.')
    parser.add_argument('--outputpath', type=str, default='', help='path to the folder of decoded texts')
    parser.add_argument('--mbrby', type=str, default='', help='path to the folder of decoded texts')
    parser.add_argument('--save', action='store_true', help='save or not')
    
    args = parser.parse_args()    

    seg_bleu1, seg_rougel, seg_bertscore, avg_len = eval_BRB(
        args.outputpath, 
        args.mbrby, 
        args.save,
        )

    
    
    