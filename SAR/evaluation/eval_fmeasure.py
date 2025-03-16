import argparse
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import torch
import re
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import csv

def NAR(
    outputpath,
    reffilepath,
    reffiletype,
):
    gene_dict = {}
    with open(outputpath, 'r') as f:
        json_data = json.load(f)
        for item in tqdm(json_data):
            # 生成データ
            image_id = item["image_id"]
            try:
                generated = item["caption"]
            except KeyError:
                generated = " "
            generated_lst = re.findall(r'\b\w+\b|[.,]', generated)
            
            gene_dict[f'{image_id}'] = generated_lst

    fmeasure_lst=[]
    precision_lst=[]
    recall_lst=[]
    tp_lst = []
    fp_lst = []
    fn_lst = []
    tn_lst = []

    with open(reffilepath, 'r') as file:
        for line in tqdm(file):
            item = json.loads(line)
            counter = item['counter']
            image_id = item['image_id']
            od_dict = item['object_detection']

            try:
                generated_lst = gene_dict[f'{counter}_{image_id}']
            except KeyError:
                pass
            
            # print(f'------------------{annotator_id}_{image_id}------------------')

            # 検出ラベル集める
            all_elements = []
            for key, value in od_dict.items():
                all_elements.extend(value)
            all_elements = list(dict.fromkeys(all_elements))

            # 結果のリストを作成
            true_result = []
            pred_result = []
            # 各要素が辞書内のどのセグメントに含まれているかを確認
            for element in all_elements:
                for key, values in od_dict.items():
                    true_found = 0  # 初期状態では見つからない
                    pred_found = 0
                    s,e = eval(key)
                    seg_gene = generated_lst[s:e]
                    if element in values:
                        true_found = 1
                    if element in seg_gene:
                        pred_found = 1
                    
                    true_result.append(true_found)
                    pred_result.append(pred_found)

            # F-measureを計算
            assert len(true_result)==len(pred_result), "Mismatch between true_result and pred_result"
            # print(f'正解配列:{true_result}')
            # print(f'予測配列:{pred_result}')
            if len(true_result)>0:
                fmeasure = f1_score(true_result, pred_result)
                # print(f"F-measure: {fmeasure}")
                fmeasure_lst.append(fmeasure)
                # Precision（適合率）
                precision = precision_score(true_result, pred_result, zero_division=0)
                precision_lst.append(precision)
                # Recall（再現率）
                recall = recall_score(true_result, pred_result)
                recall_lst.append(recall)
                # Confusion Matrix から TP, FP, FN, TN を計算
                cm = confusion_matrix(true_result, pred_result, labels=[0, 1])
                if cm.shape == (2, 2):
                    tn, fp, fn, tp = cm.ravel()
                else:
                    tn = fp = fn = tp = 0
                    print(f"Warning: Unexpected confusion matrix shape {cm.shape}. Defaulting to 0s.")
                tp_lst.append(tp)
                fp_lst.append(fp)
                fn_lst.append(fn)
                tn_lst.append(tn)
    
    print(f"{reffiletype}")
    print(f"mean F-measure: {np.mean(fmeasure_lst)}")
    print(f"mean Precision: {np.mean(precision_lst)}")
    print(f"mean Recall: {np.mean(recall_lst)}")
    print(f"TP: {np.mean(tp_lst)}, FP: {np.mean(fp_lst)}, FN: {np.mean(fn_lst)}, TN: {np.mean(tn_lst)}")
    print('*'*10)

def SAR(
    outputpath,
    reffilepath,
    reffiletype,
    ):

    ref_dict = {}
    with open(reffilepath, 'r') as file:
        for line_num, line in enumerate(file, 1):
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Error in line {line_num}: {line}")
                print(f"Error message: {e}")
                continue
            annotator_id = item['annotator_id']
            image_id = item['image_id']
            od_dict = item['object_detection']

            ref_dict[f'{image_id}_{annotator_id}'] = od_dict

    fmeasure_lst=[]
    fmeasure_all_lst = []
    precision_lst=[]
    recall_lst=[]
    tp_lst = []
    fp_lst = []
    fn_lst = []
    tn_lst = []

    with open(outputpath, 'r') as file:
        for line in tqdm(file):
            item = json.loads(line)
            generated = item['string']
            if len(item['string'])!=1:
                generated = generated[0]
            generated_lst = re.findall(r'\b\w+\b|[.,]', generated)
            
            annotator_id = item['annotator_id']
            image_id = item['image_id']

            od_dict = ref_dict[f'{image_id}_{annotator_id}']

            # 検出ラベル集める
            all_elements = []
            for key, value in od_dict.items():
                all_elements.extend(value)
            all_elements = list(dict.fromkeys(all_elements))

            # 結果のリストを作成
            true_result = []
            pred_result = []
            # 各要素が辞書内のどのセグメントに含まれているかを確認
            for element in all_elements:
                for key, values in od_dict.items():
                    true_found = 0  # 初期状態では見つからない
                    pred_found = 0
                    s,e = eval(key)
                    seg_gene = generated_lst[s:e]
                    if element in values:
                        true_found = 1
                    if element in seg_gene:
                        pred_found = 1
                    
                    true_result.append(true_found)
                    pred_result.append(pred_found)

            # F-measureを計算
            assert len(true_result)==len(pred_result), "Mismatch between true_result and pred_result"
            if len(true_result)>0:
                # Fmeasure (F値)
                fmeasure = f1_score(true_result, pred_result)
                fmeasure_lst.append(fmeasure)
                fmeasure_all_lst.append(fmeasure)
                # print(f'------------------{annotator_id}_{image_id}------------------')
                # print(f'正解配列:{true_result}')
                # print(f'予測配列:{pred_result}')
                # print(f"F-measure: {fmeasure}")
                
                # Precision（適合率）
                precision = precision_score(true_result, pred_result, zero_division=0)
                precision_lst.append(precision)

                # Recall（再現率）
                recall = recall_score(true_result, pred_result)
                recall_lst.append(recall)

                # Confusion Matrix から TP, FP, FN, TN を計算
                cm = confusion_matrix(true_result, pred_result, labels=[0, 1])
                if cm.shape == (2, 2):
                    tn, fp, fn, tp = cm.ravel()
                else:
                    tn = fp = fn = tp = 0
                    print(f"{image_id}_{annotator_id} Warning: Unexpected confusion matrix shape {cm.shape}. Defaulting to 0s.")

                tp_lst.append(tp)
                fp_lst.append(fp)
                fn_lst.append(fn)
                tn_lst.append(tn)
            else:
                fmeasure_all_lst.append(-1)
                
    
    print(f"{reffiletype}")
    print(f"mean F-measure: {np.mean(fmeasure_lst)}")
    top_indices = np.argsort(fmeasure_lst)[-10:][::-1]  # 上位10個を降順に取得
    print(f"fmeasure top10 index : {top_indices}")
    print(f"mean Precision: {np.mean(precision_lst)}")
    print(f"mean Recall: {np.mean(recall_lst)}")
    print(f"TP: {np.mean(tp_lst)}, FP: {np.mean(fp_lst)}, FN: {np.mean(fn_lst)}, TN: {np.mean(tn_lst)}")
    print('*'*10)

    # 書き込み対象のCSVファイル名
    csv_file = "/home/hirano/interactive_captiongeneration/evaluation/output.csv"
    # CSVを読み込む
    df = pd.read_csv(csv_file)
    # 元の行数を取得
    num_rows = len(df)
    # 新しい列を追加する
    print(f'len of fmeasure_all_lst : {len(fmeasure_all_lst)}')
    df[f"{reffiletype}_fmeasure"] = fmeasure_all_lst[:num_rows]
    # CSVファイルに保存する
    df.to_csv(csv_file, index=False)

    return np.mean(fmeasure_lst), np.mean(precision_lst), np.mean(recall_lst)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='decoding args.')
    parser.add_argument('--outputpath', type=str, default='', help='path to the folder of decoded texts')
    parser.add_argument('--method', type=str, default='', help='NAR or SAR')
    parser.add_argument('--reffiletype', type=str, default=0.9, help='0.5 or 0.9')
    args = parser.parse_args()

    object_detection_path = f'/Storage2/hirano/container_source/localized_narratives/karpathy/20240618/object_detection_test_{args.reffiletype}.jsonl'

    if args.method == 'NAR':
        NAR(
            args.outputpath,
            object_detection_path,
            args.reffiletype,
        )
    elif args.method == 'SAR':
        fmeasure, precision, recall = SAR(
            args.outputpath,
            object_detection_path,
            args.reffiletype,
        )