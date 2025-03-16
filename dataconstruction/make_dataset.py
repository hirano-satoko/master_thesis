#20240516
#04と違うところは、durationの計算のところ
#make_divided とか make_one_dataのところとかを消したところ

import pandas as pd
import numpy as np
import itertools
import pathlib
from tqdm import tqdm
import time
from random import randint
import urllib.request

import re

from PIL import Image
import requests
import torch as th
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import json

from models.blip import blip_feature_extractor

import math

import os, sys, glob
sys.setrecursionlimit(1680000)

from torchmetrics.multimodal.clip_score import CLIPScore
metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16")

def load_demo_image(raw_image,
                    image_size,
                    device):

    transform = transforms.Compose([
        transforms.Resize((image_size,image_size),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])
    image = transform(raw_image).unsqueeze(0).to(device)
    return image


def load_blip(image_size, device):
    model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base.pth'
    model = blip_feature_extractor(pretrained=model_url, image_size=image_size, vit='base')
    model.eval()
    model = model.to(device)

    return model

def analyse_karpathy(
    path,
    ):
    ids = []

    with open(path, 'r') as f:
        for i, row in enumerate(f):
            dataDir, filename, imageid, captions = row.rstrip().split('||')
            ids.append(imageid)
    
    return ids

### xyt_lst を返す
def analyse_tracedata(
                    caption_stime,
                    caption_etime,
                    tracedata):
    
    xyt_lst =[]
    data_num=0
    
    for data in tracedata:
        for i, d in enumerate(data):
            # キャプション時間外のところはのぞく
            if d['t'] < caption_stime or d['t'] > caption_etime:
                continue
            
            # assert d['x']>0
            # assert d['y']>0
            # assert d['t']>0
                
            xyt_lst.append((d['x'], d['y'], d['t'] ))
            data_num+=1
            
    return xyt_lst, data_num

### nタイムステップ前からの座標変化量リストを返す
def analyse_delta(
                    xyt_lst,
                    N):
    
    tdelta_lst = []
    s = 0
    # length = 0
    v_lst=[]
    
    # print(f'beforenstep:{N}')
    
    for ((x0,y0,t0),(x1,y1,t1)) in zip(xyt_lst[:-N],xyt_lst[N:]):
                
        delta_x = x0 - x1
        delta_y = y0 - y1
        delta = np.sqrt(delta_x**2 + delta_y**2)
        s+=delta
        v_lst.append(delta/abs(t1-t0))
        # length+=1
        
        tdelta_lst.append((t1,delta))
    
    time_span = xyt_lst[-1][2] - xyt_lst[0][2]
    try:
        ave_v = s / time_span
    except ZeroDivisionError:
        ave_v = 0
    # print(f'time_span : {time_span}, sum : {s}, ave_v : {ave_v}')
    ave_v = np.mean(v_lst)
    # print(f'time_span : {time_span}, sum : {s}, ave_v : {ave_v}')
            
    return tdelta_lst, s, ave_v

### 閾値を返す（４分位範囲）
def analyse_threshold(tdelta_lst, p):
    delta_lst = [d for (t,d) in tdelta_lst]
    q0, q1, q2, q3, threshold, q4 = np.percentile(delta_lst, q=[0, 25, 50, 75, p, 100])
    
    outlier_high = (q3-q1)*1.5 + q3
    outlier_low = q1 - (q3-q1)*1.5
    
    return threshold, outlier_high, outlier_low

### 採用区間タイムステップリストを返す
def analyse_over_timestep(
    tdelta_lst,
    threshold,
    ):
    
    under_t_lst = []
    over=True
    
    for t, delta in tdelta_lst:   
        # 変化量が閾値を下回っている
        if delta<threshold:
            # 下回りはじめ
            if over:
                under_s_time = t
                over = False
            # 下回りの続き
            under_e_time = t
                
        # 変化量が閾値を超えてる→不採用
        else:
            #  超え始め
            if not(over):
                # そこまでを返り値リストに追加
                under_t_lst.append((under_s_time, under_e_time))
                over = True
    
    # 下回ったまま終わった
    if not(over):
        under_t_lst.append((under_s_time, under_e_time))
            
    return under_t_lst

### 不採用区間のうち、区間内の座標変化量が小さいものは採用キャプションに変更
def concat(
    under_t_lst, #　採用区間
    over_d_lst, over_s_lst, over_e_lst,
    min_index,
    ):

    N =len(under_t_lst)
    insert_index = min_index-1
    
    # 不採用の不採用区間をpop
    if min_index==N:
        # s_N-1, e_N-1
        s_nminus1, e_nminus1 = under_t_lst.pop(min_index-1)
        assert len(under_t_lst)==N-1
    elif min_index==0:
        # s_0, e_0
        s_n, e_n = under_t_lst.pop(min_index)
        insert_index=min_index
        assert len(under_t_lst)==N-1
    else:
        # s_n, e_n
        s_n, e_n = under_t_lst.pop(min_index)
        # s_n-1, e_n-1
        s_nminus1, e_nminus1 = under_t_lst.pop(min_index-1)
        assert len(under_t_lst)==N-2
    
    ## 採用区間を更新
    if min_index==0:
        # トレース開始時間
        new_s = s_n #over_s_lst[min_index]
    else:
        # s_n-1
        new_s = s_nminus1 #over_e_lst[min_index-1]
        
    if min_index==N:
        # トレース終了時間
        new_e = e_nminus1 #over_e_lst[min_index]
    else:
        # e_n
        new_e = e_n #over_s_lst[min_index+1]
        
    # 更新
    under_t_lst.insert(insert_index, (new_s, new_e))
    
    over_d_lst.pop(min_index)
    over_s_lst.pop(min_index)
    over_e_lst.pop(min_index)
    
    return under_t_lst, over_d_lst, over_s_lst, over_e_lst

def concat_under_t_lst(
    under_t_lst,
    tdelta_lst1,
    ave_v1,
    ):
    
    N = len(under_t_lst)
    over_s_lst = [tdelta_lst1[0][0]] +  [e for (s,e) in under_t_lst]
    over_e_lst = [s for (s,e) in under_t_lst] + [tdelta_lst1[-1][0]]
    assert N+1 == len(over_s_lst)
    assert N+1 == len(over_e_lst)
    over_d_lst=[]
    
    # 不採用区間内の合計移動量を計算
    start_index=0
    for n in range(N+1): # 0<=n<=N
        sum_d=0
        for i, (t, d) in enumerate(tdelta_lst1[start_index:]):
            if over_s_lst[n] < t and t<over_e_lst[n]:
                sum_d+=d
            elif over_e_lst[n]<=t:
                start_index+=i
                over_d_lst.append(sum_d)
                break
        
    assert N+1 == len(over_d_lst)
    
    ## 不採用区間ごとの座標変化量を小さい順に並べる
    values, indices = th.topk(th.tensor(over_d_lst), k=N+1, largest=False)
    rm_num=0
    try:
        min_index=indices[0].item()
    except IndexError:
        return under_t_lst
    
    while math.ceil(values[0].item()) <= (ave_v1*(over_e_lst[min_index] - over_s_lst[min_index])):
        
        # 一個減ったものが返ってくる
        under_t_lst, over_d_lst, over_s_lst, over_e_lst = concat(
            under_t_lst, #　採用区間
            over_d_lst,
            over_s_lst,
            over_e_lst,
            min_index,
        )
        rm_num+=1
#         assert len(under_t_lst)==N-rm_num ## 変わらない場合もある n=0 or n=N
        values, indices = th.topk(th.tensor(over_d_lst), k=N+1-rm_num, largest=False)
        try:
            min_index=indices[0].item()
        except IndexError:
            return under_t_lst
    
    return under_t_lst

### 採用区間のうち、区間内の座標変化量が平均速度より小さいものは除く
def remove_bydelta(under_t_lst,
                    tdelta_lst1,
                    ave_v1):
    
    min_delta = (0,1000000)
    min_exist = False
    
    ## 区間内の合計移動量が一番小さい区間を探す
    start_index=0
    for i, (s_t, e_t) in enumerate(under_t_lst):
        sum_d=0
        
        # 区間内の合計移動量を計算
        for j, (t, d) in enumerate(tdelta_lst1[start_index:]):
            if s_t <= t and t<=e_t:
                sum_d+=d
            elif t>e_t:
                time_span = t - tdelta_lst1[start_index][0]
                start_index+=j
                break
        # 一番小さい区間かチェック
        if sum_d < min_delta[1]:
            min_delta = (i, sum_d)
            # 閾値
            th2 = ave_v1# * time_span
            min_exist = True
    
    # 小さい区間があった
    if min_exist:
        # 合計移動量が閾値を下回っている
        # print(f'min_delta : {min_delta}')
        if min_delta[1] < th2:
            under_t_lst.pop(min_delta[0])
        else:
            min_exist = False
    
    return under_t_lst, min_exist

def remove_under_t_lst(
    under_t_lst,
    xyt_lst,
    tdelta_lst1, sum1, ave_v1,
    ):
    
    under_t_lst, min_exist = remove_bydelta(under_t_lst, tdelta_lst1, ave_v1)
    while min_exist:
        under_t_lst, min_exist = remove_bydelta(under_t_lst, tdelta_lst1, ave_v1)

    return under_t_lst

def make_bbox(
    blip_model,
    img, width, height,
    image_size,
    device,
    grouped_xyt_lst,
    ):

    bbox_lst=[]
    feature_lst=[]
    # duration_lst=[]

    for coords_list in grouped_xyt_lst:
        # 座標リストが空の場合はエラーを返す
        if not coords_list:
            raise ValueError("座標リストが空です")

        # 与えられた座標リスト内の点の最小・最大x座標およびy座標を取得
        min_x = min(coord[0] for coord in coords_list)
        max_x = max(coord[0] for coord in coords_list)
        min_y = min(coord[1] for coord in coords_list)
        max_y = max(coord[1] for coord in coords_list)

        # 右下の座標を含むように四角形の座標を調整
        left = int(max(0,min_x)*width)
        top = int(max(0,min_y)*height)
        right = int(min(max_x+1,width)*width)
        bottom = int(min(max_y+1, height)*height)

        assert left<right
        assert top<bottom

        ## bbox抽出
        bbox = img.crop((left, top, right, bottom))

        ## 切り出したのを使う場合
        # bbox_trans = transforms.functional.to_tensor(bbox)
        # bbox_lst.append(bbox_trans)

        # 黒い画像を作成
        black_image = Image.new('RGB', (width, height), (0, 0, 0))
        # # 指定範囲の部分を元の画像から取り出して、黒い画像の同じ範囲に貼り付ける
        black_image.paste(bbox, (left, top))
        # clipscore用に変換
        black_trans = transforms.functional.to_tensor(black_image)
        bbox_lst.append(black_trans)

        # blip feature
        th.cuda.empty_cache()
        load_bbox = load_demo_image(
            raw_image = bbox, 
            image_size = image_size, 
            device = device,)
        th.cuda.empty_cache()
        bbox_feature = blip_model(load_bbox, caption='', mode='image')[0,0] # tensor torch.Size([768])
        assert not bbox_feature.isnan().any().item()
        feature_lst.append(bbox_feature.tolist())
        th.cuda.empty_cache()

        # # duration
        # duration = coords_list[-1][-1] - coords_list[0][-1]
        # duration_lst.append(duration)

    assert len(grouped_xyt_lst)==len(bbox_lst)
    assert len(grouped_xyt_lst)==len(feature_lst)
    # assert len(grouped_xyt_lst)==len(duration_lst)

    return bbox_lst, feature_lst#, duration_lst

def group_coordinates_by_time(
    xyt_lst, 
    under_t_lst, 
    timed_caption,
    ):

    # もし時間範囲のリストが空の場合
    if not under_t_lst:
        start_time = timed_caption[0]['start_time']  # 開始時間
        end_time = timed_caption[-1]['end_time']    # 終了時間
        # 開始時間から終了時間までの座標を格納するリスト
        grouped_xyt_lst = [[coord for coord in xyt_lst if start_time <= coord[-1] <= end_time]]

        assert len(grouped_xyt_lst)==1
        assert len(grouped_xyt_lst[0])!=0
        assert len(under_t_lst)==0
        return grouped_xyt_lst, under_t_lst

    # 時間範囲ごとのxy座標を格納するリスト
    grouped_xyt_lst = [[] for _ in range(len(under_t_lst))]

    # xy座標を時間範囲ごとに分類
    for x, y, time in xyt_lst:
        # 時間範囲を特定し、適切なリストにxy座標を追加
        for i, (s_t, e_t) in enumerate(under_t_lst):
            if s_t <= time and time <= e_t:
                grouped_xyt_lst[i].append((x, y, time))
                break

    assert len(grouped_xyt_lst)==len(under_t_lst)
    for idx, (grouped_xyt, under_t) in enumerate(zip(grouped_xyt_lst, under_t_lst)):
        # もし区間内の座標が空だったら
        if not grouped_xyt:
            grouped_xyt_lst.pop(idx)
            under_t_lst.pop(idx)
    assert len(grouped_xyt_lst)==len(under_t_lst)

    return grouped_xyt_lst, under_t_lst

### 各採用区間に応じたキャプションの start_indexとend_indexを返す
def make_caption(
    under_t_lst,
    timed_caption,):
    
    caption_lst = []
    grouped_words=[]
    
    if not under_t_lst:  # time_intervalsが空の場合
        for t_c in timed_caption:
            utterance = t_c['utterance']
            utterance_lst = re.findall(r'\b\w+\b|[.,]', utterance)
            
            grouped_words.append(utterance_lst)
            caption_lst.extend(utterance_lst)
        
        index_lst=[(0, len(caption_lst))]

        # duration
        duration_lst = [timed_caption[-1]['end_time'] - timed_caption[0]['start_time']]

        return caption_lst, index_lst, duration_lst
    
    # 各時間範囲ごとの発話単語を格納するリスト
    grouped_words = [[] for _ in range(len(under_t_lst)+1)]
    grouped_timed_caption = [[] for _ in range(len(under_t_lst)+1)]

    # 発話単語を時間範囲ごとに分類
    for k, t_c in enumerate(timed_caption):
        start_time = t_c['start_time']
        end_time = t_c['end_time']
        utterance = t_c['utterance']
        utterance_lst = re.findall(r'\b\w+\b|[.,]', utterance)
        caption_lst.extend(utterance_lst)
        
        # 時間範囲を特定し、適切なリストに発話単語を追加
        for i, (s_t0, e_t0) in enumerate(under_t_lst):
            if end_time < s_t0:
                grouped_words[i].extend(utterance_lst)
                grouped_timed_caption[i].append(t_c)
                break
        else:
            grouped_words[-1].extend(utterance_lst)  # 最後の範囲に追加
            grouped_timed_caption[-1].append(t_c)
            
    # 新しいリストを作成する
    new_grouped_words = []
    # 最初の2つの要素を結合して新しいリストに追加
    new_grouped_words.append(grouped_words[0] + grouped_words[1])
    # 残りの要素をそのまま新しいリストに追加
    new_grouped_words.extend(grouped_words[2:])

    new_grouped_timed_caption = []
    new_grouped_timed_caption.append(grouped_timed_caption[0] + grouped_timed_caption[1])
    new_grouped_timed_caption.extend(grouped_timed_caption[2:])
    
    assert len(new_grouped_words)==len(new_grouped_timed_caption)

    ## 各bboxに対応するindex範囲を計算
    start_index=0
    index_lst=[]
    duration_lst=[]
    for group_w, group_t_c in zip(new_grouped_words, new_grouped_timed_caption):
        # index
        end_index = start_index + len(group_w)
        index_lst.append((start_index, end_index))
        start_index = end_index + 1

        # duration
        try:
            duration = group_t_c[-1]['end_time'] - group_t_c[0]['start_time']
        except IndexError: # 採用区間に割り当てられたキャプションなし
            duration = 0
        duration_lst.append(duration)

    assert len(index_lst)==len(under_t_lst)
    assert len(index_lst)==len(duration_lst)

    return caption_lst, index_lst, duration_lst


### 各採用区間に応じた画像とキャプションを作成
def visualize_divide(
                    blip_model,
                    img, width, height, 
                    image_size,
                    device,
                    xyt_lst, # キャプションがついていないところはすでに除かれている
                    under_t_lst,
                    timed_caption,
                    ):

    # xyt_lst 分割
    grouped_xyt_lst, under_t_lst = group_coordinates_by_time(
        xyt_lst,
        under_t_lst,
        timed_caption,
    )

    # 画像分割
    image_lst, feature_lst = make_bbox(
        blip_model,
        img, width, height,
        image_size,
        device,
        grouped_xyt_lst,
    )

    # キャプション分割
    caption_lst, index_lst, duration_lst = make_caption(
        under_t_lst,
        timed_caption,)

    ## clipscore
    assert len(index_lst)==len(image_lst)
    clipscore_lst=[]
    for idx, ((s,e), image) in enumerate(zip(index_lst, image_lst)):
        clipscore = metric(image, ' '.join(caption_lst[s:e+1])).detach().item()
        clipscore_lst.append(clipscore)

    assert len(index_lst)==len(clipscore_lst)
    
    return index_lst, feature_lst, duration_lst, caption_lst, clipscore_lst

### 画像全体の特徴量抽出
def image_to_vector(
                    blip_model,
                    img, 
                    image_size,
                    device,):

    image = load_demo_image(
            raw_image = img, 
            image_size = image_size, 
            device = device,)
    image_vector = blip_model(image, caption='', mode='image')[0,0].tolist() # tensor torch.Size([768])

    return image_vector


### 大元
def analyse_dataset(dataset, image_size, device, blip_model, ids_invalid, ids_intest,
                    df_train, df_valid, df_test):

    max_index_feature_lst_len=0
    
    for index, row in dataset.iterrows():
        
        image_id = str(row['image_id'])
        # print(image_id)
        # if image_id!='366711':
        #     continue
        while len(image_id)!=6:
            image_id='0'+image_id
        dataset_id = row['dataset_id']
        dataDir = dataset_id[7:-1]
        
        timed_caption = row['timed_caption']
        caption_stime = timed_caption[0]['start_time']
        caption_etime = timed_caption[-1]['end_time']
    
        ## トレースデータからキャプション時間外のものを除いた xyt_lst を返す
        xyt_lst, data_num = analyse_tracedata(
                                                caption_stime, 
                                                caption_etime,
                                                row['traces'],
        )
        if not xyt_lst:
            continue
        
        ## 1タイムステップ前からの座標変化量リストを返す
        N = 1
        tdelta_lst1, sum1, ave_v1  = analyse_delta(
                                                    xyt_lst,
                                                    N,
        )

        ## nタイムステップ前からの座標変化量リストを返す
        N = int(data_num/10)
        tdelta_lst_n, sum_n, ave_vn  = analyse_delta(
                                                        xyt_lst,
                                                        N,
        )
        if not tdelta_lst_n:
            continue
        ## 変化量の閾値を決める
        threshold, high, low = analyse_threshold(tdelta_lst_n, 90)
        
        ## nタイムステップ前からの座標変化量が閾値を超えた区間を除いた採用タイムステップリストを返す
        under_t_lst = analyse_over_timestep(
            tdelta_lst_n,
            threshold,
            )

        ## 不採用区間の取捨選択(分割数減)
        ## 不採用区間の合計移動量が小さいものを、前後の採用区間と連結
        # under_t_lst = concat_under_t_lst(
        #     under_t_lst,
        #     tdelta_lst1,
        #     ave_v1)

        ## 採用区間の合計移動量が小さいものをのぞく(分割数減)
        under_t_lst = remove_under_t_lst(under_t_lst,
                                            xyt_lst,
                                            tdelta_lst1, sum1, ave_v1)
        
        ## 画像をロード
        image_url = f'/Storage/hirano/mscoco_dataset/' \
                        f'{dataDir}4/COCO_{dataDir}4_000000{image_id}.jpg'
        try:
            img = Image.open(image_url).convert('RGB')
        except FileNotFoundError:
            if dataDir[0]=='t':
                image_url = f'/Storage/hirano/mscoco_dataset/' \
                        f'val2014/COCO_val2014_000000{image_id}.jpg'
                img = Image.open(image_url).convert('RGB')
            else:
                image_url = f'/Storage/hirano/mscoco_dataset/' \
                        f'train2014/COCO_train2014_000000{image_id}.jpg'
                img = Image.open(image_url).convert('RGB')
        width, height = img.size
        
        ## 閾値によって分けたトレースと画像を表示
        index_lst, feature_lst, duration_lst, caption_lst, clipscore_lst = visualize_divide(
            blip_model,
            img, width, height,
            image_size,
            device,
            xyt_lst,
            under_t_lst,
            timed_caption,
            )
        # print('*'*30)
        ## 最大分割数
        if len(index_lst)>max_index_feature_lst_len:
            max_index_feature_lst_len = len(index_lst)
        
        ## 画像全体の特徴量計算
        image_vector = image_to_vector(
                    blip_model,
                    img, 
                    image_size,
                    device,)

        wholeimage_clipscore = metric(transforms.functional.to_tensor(img), ' '.join(caption_lst)).detach().item()

        result_dict = {'dataset_id':dataset_id,
                        'image_id':row['image_id'],
                        'annotator_id':row['annotator_id'],
                        'caption':caption_lst,
                        'image_vector':image_vector,
                        'wholeimage_clipscore':wholeimage_clipscore,
                        'indexes':index_lst,
                        'imagefeatures':feature_lst,
                        'durations':duration_lst,
                        'clipscores':clipscore_lst,}
        
        if str(row['image_id']) in ids_invalid:
            df_valid = pd.concat([df_valid, pd.DataFrame([result_dict])], ignore_index=True)
        elif str(row['image_id']) in ids_intest:
            df_test = pd.concat([df_test, pd.DataFrame([result_dict])], ignore_index=True)
        else:
            df_train = pd.concat([df_train, pd.DataFrame([result_dict])], ignore_index=True)
        
    
    print(f'index:{index}, max_index_feature_lst_len:{max_index_feature_lst_len}')
    return df_train, df_valid, df_test


if __name__ == '__main__':
    image_size = 224
    device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    blip_model = load_blip(image_size, device)

    ids_invalid = analyse_karpathy(
        '/Storage/hirano/mscoco_dataset/karpathy/karpathy_val.txt',
    )
    ids_intest = analyse_karpathy(
        '/Storage/hirano/mscoco_dataset/karpathy/karpathy_test.txt',
    )

    save_path_train = '/home/hirano/blip/BLIP/localized_narratives_train.h5'
    save_path_valid = '/home/hirano/blip/BLIP/localized_narratives_valid.h5'
    save_path_test = '/home/hirano/blip/BLIP/localized_narratives_test.h5'

    path_lst = glob.glob('/Storage2/Dataset/localized_narratives/COCO/*jsonl')
    assert len(path_lst)==5

    for i, path in enumerate(path_lst):
        dataset = pd.read_json(
            path, 
            orient='records',
            lines=True)

        df_train = pd.DataFrame([])
        df_valid = pd.DataFrame([])
        df_test = pd.DataFrame([])
            
        df_train, df_valid, df_test = analyse_dataset(
            dataset, 
            image_size, 
            device, 
            blip_model, 
            ids_invalid, 
            ids_intest,
            df_train, df_valid, df_test,
            )

        print(df_train.shape)
        print(df_valid.shape)
        print(df_test.shape)
        df_train.to_hdf(save_path_train, key=f'data{i}', mode='a')
        df_valid.to_hdf(save_path_valid, key=f'data{i}', mode='a')
        df_test.to_hdf(save_path_test, key=f'data{i}', mode='a')