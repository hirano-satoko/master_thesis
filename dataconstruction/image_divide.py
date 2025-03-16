# import
import argparse
import pandas as pd
import numpy as np
import torch as th
import time
import urllib.request
import re
from PIL import Image
import requests
import json
from matplotlib import pyplot as plt
import  japanize_matplotlib
import os, sys, glob
import math
import cv2
import random

# url_to_image
def url_to_image(url):
    image = cv2.imread(url, cv2.IMREAD_COLOR)
    return image


# analyse_tracedata : create xyt_lst
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
                
            xyt_lst.append((d['x'], d['y'], d['t'] ))
            data_num+=1
            
    return xyt_lst, data_num

# visualize_all : show image with traces
def visualize_all(
    image_url,
    save_path,
    xyt_lst):
    
    image = url_to_image(image_url)
    height, width, channel = image.shape
    
    for ((x0,y0,t0),(x1,y1,t1)) in zip(xyt_lst[:-1],xyt_lst[1:]):
       
        cv2.line(image, 
                 (int(x0 * width), int(y0 * height)), 
                 (int(x1 * width), int(y1 * height)),
                 (0,0,255), 
                 thickness=2, 
                 lineType=cv2.LINE_AA)
    
    # imgはnp.uint8型のnp.ndarray
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # cv2.imwrite(f'./figure_20240523/{annotator_id}_{image_id}/overall_picture.jpg', image)
    cv2.imwrite(f'{save_path}/overall_picture.jpg', image)

# analyse_delta : create tdelta_lst
def analyse_delta(
    xyt_lst,
    N,
    ):
    
    tdelta_lst = []
    s = 0
    v_lst=[]
    
    for ((x0,y0,t0),(x1,y1,t1)) in zip(xyt_lst[:-N],xyt_lst[N:]):
                
        delta_x = x0 - x1
        delta_y = y0 - y1
        delta = np.sqrt(delta_x**2 + delta_y**2)
        s+=delta
        v_lst.append(delta/abs(t1-t0))
        
        tdelta_lst.append((t1,delta))
    
    time_span = xyt_lst[-1][2] - xyt_lst[0][2]
    try:
        ave_v = s / time_span
    except ZeroDivisionError:
        ave_v = 0
   
    ave_v = np.mean(v_lst)
            
    return tdelta_lst, s, ave_v

# analyse_threshold : calculate threshold
def analyse_threshold(tdelta_lst, p):
    delta_lst = [d for (t,d) in tdelta_lst]
    q0, q1, q2, q3, threshold, q4 = np.percentile(delta_lst, q=[0, 25, 50, 75, p, 100])
    
    outlier_high = (q3-q1)*1.5 + q3
    outlier_low = q1 - (q3-q1)*1.5
    
    return threshold, outlier_high, outlier_low

# analyse_over_timestep : create under_t_lst based on threshold from tdelta_lst
def analyse_over_timestep(
    tdelta_lst,
    threshold,
    ):
    under_t_lst = []
    over=True
    
    for t, delta in tdelta_lst:
        # 変化量が閾値を下回っている
        if delta<threshold:
            # over==True -> 下回りはじめ
            if over:
                under_s_time = t
                over = False
            # 下回りの続き
            under_e_time = t 
        # 変化量が閾値を超えてる→不採用
        else:
            #  over==false -> 超え始め
            if not(over):
                # そこまでを返り値リストに追加
                under_t_lst.append((under_s_time, under_e_time))
                over = True
    
    # 下回ったまま終わった
    if not(over):
        under_t_lst.append((under_s_time, under_e_time))
            
    return under_t_lst

def visualize_alldelta(
    tdelta_lst, 
    data_num,
    save_path,message,
    ):
    fig, ax = plt.subplots(figsize=(6.4*1.5, 4.8*1.2))
    ax.set_ylabel('△Coordinate from N steps prior', fontsize=20)
    ax.set_xlabel('Timestep', fontsize=20)

    under_xlst = [t for (t,d) in tdelta_lst]
    under_ylst = [d for (t,d) in tdelta_lst]

    plt.plot(under_xlst, under_ylst, color='blue', linestyle='solid', lw=2.5)

    plt.legend(fontsize=14)
    plt.savefig(f'{save_path}/delta_graph_{message}.jpg')


# visualize_delta : show under_t_lst
def visualize_delta(
    threshold,
    tdelta_lst, 
    data_num,
    under_t_lst,
    with_threshold,
    save_path, message,
    ):
    
    fig, ax = plt.subplots(figsize=(6.4*1.5, 4.8*1.2))
    ax.set_ylabel('△Coordinate from N steps prior', fontsize=20)
    ax.set_xlabel('Timestep', fontsize=20)
    
    start_index=0
    over_xlst=[]
    over_ylst=[]
    under_xlst=[]
    under_ylst=[]
        
    for i, (s_t, e_t) in enumerate(under_t_lst):
        for j, (t, d) in enumerate(tdelta_lst[start_index:]):
            
            if t<s_t:
                over_xlst.append(t)
                over_ylst.append(d)
            elif s_t<=t and t<=e_t:
                under_xlst.append(t)
                under_ylst.append(d)
            else:
                if i==0:
                    plt.plot(under_xlst, under_ylst, color='blue', linestyle='solid', lw=2.5, label='Adopted Segment')
                    plt.plot(under_xlst[0], under_ylst[0] ,color='blue', marker='o', markersize=8)
                    plt.plot(under_xlst[-1], under_ylst[-1] ,color='blue', marker='o', markersize=8)
                    
                    plt.plot(over_xlst, over_ylst, color='blue', linestyle='dotted', lw=1, label='Rejected Segment')
                else:
                    plt.plot(under_xlst, under_ylst, color='blue', linestyle='solid', lw=2.5)
                    plt.plot(under_xlst[0], under_ylst[0] ,color='blue', marker='o', markersize=8)
                    plt.plot(under_xlst[-1], under_ylst[-1] ,color='blue', marker='o', markersize=8)
                    
                    plt.plot(over_xlst, over_ylst, color='blue', linestyle='dotted', lw=1)
            
                start_index+=(j+1)
                over_xlst=[t]
                over_ylst=[d]
                under_xlst=[]
                under_ylst=[]
                break
    
    if under_xlst:
        plt.plot(under_xlst, under_ylst,color='blue', linestyle='solid', lw=2.5)
        plt.plot(under_xlst[0], under_ylst[0] ,color='blue', marker='o', markersize=8)
        plt.plot(under_xlst[-1], under_ylst[-1] ,color='blue', marker='o', markersize=8)
    
    if over_xlst:
        plt.plot(over_xlst, over_ylst, color='blue', linestyle='dotted', lw=1)
    
    if start_index<data_num:
        over_xlst=[]
        over_ylst=[]
        for j, (t, d) in enumerate(tdelta_lst[start_index:]):
                over_xlst.append(t)
                over_ylst.append(d)
        plt.plot(over_xlst, over_ylst, color='blue', linestyle='dotted', lw=1)
    
    ## 閾値
    if with_threshold:
        ax.axhline(y=threshold,color='crimson',linestyle='dashed', lw=5, label='threshold')
    
    plt.legend(fontsize=14)
    
    # plt.savefig(f'./figure_20240523/{image_id}/{annotator_id}/delta_graph_{message}.jpg')
    plt.savefig(f'{save_path}/delta_graph_{message}.jpg')

# concat : 不採用区間の不採用
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

# concat_under_t_lst
def concat_under_t_lst(
    under_t_lst,
    tdelta_lst1,
    ave_v1):
    
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
        values, indices = th.topk(th.tensor(over_d_lst), k=N+1-rm_num, largest=False)
        try:
            min_index=indices[0].item()
        except IndexError:
            return under_t_lst
    
    return under_t_lst

# remove_bydelta : 採用区間の不採用
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
            th2 = ave_v1#*time_span
            min_exist = True
    
    # 小さい区間があった
    if min_exist:
        # 合計移動量が閾値を下回っている
        if  min_delta[1]<th2:
            under_t_lst.pop(min_delta[0])
        else:
            min_exist = False
    
    return under_t_lst, min_exist

# remove_under_t_lst
def remove_under_t_lst(
    under_t_lst,
    xyt_lst,
    tdelta_lst1, sum1, ave_v1
    ):
    
    under_t_lst, min_exist = remove_bydelta(under_t_lst, tdelta_lst1, ave_v1)
    while min_exist:
        under_t_lst, min_exist = remove_bydelta(under_t_lst, tdelta_lst1, ave_v1)

    return under_t_lst

# group_coordinates_by_time : xyt_lst を under_t_lst に応じて分割
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

# make_bbox
def make_bbox(
    img, width, height,
    grouped_xyt_lst,
    ):

    bbox_lst=[]

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
        bbox_lst.append(bbox)

    assert len(grouped_xyt_lst)==len(bbox_lst)

    return bbox_lst

def make_bbox_traces(
    img_url,
    save_path,
    grouped_xyt_lst,
    ):
    
    for index, coords_list in enumerate(grouped_xyt_lst):
        image = url_to_image(img_url)
        height, width, channel = image.shape

        # 与えられた座標リスト内の点の最小・最大x座標およびy座標を取得
        min_x = min(coord[0] for coord in coords_list)
        max_x = max(coord[0] for coord in coords_list)
        min_y = min(coord[1] for coord in coords_list)
        max_y = max(coord[1] for coord in coords_list)

        # 右下の座標を含むように四角形の座標を調整
        left = int(max(0,min_x)*width)
        top = int(max(0,min_y)*height)
        right = min(int(max_x*width)+1, width)
        bottom = min(int(max_y*height)+1, height)
        # right = int(min(max_x+1,width)*width)
        # bottom = int(min(max_y+1, height)*height)
        print(f'bottom : {bottom}')

        try:
            for (x0,y0,t0),(x1,y1,t1) in zip(coords_list[:-1],coords_list[1:]):
                cv2.line(image, 
                        (int(x0 * width), int(y0 * height)), 
                        (int(x1 * width), int(y1 * height)),
                        (0,0,255), 
                        thickness=2, 
                        lineType=cv2.LINE_AA)
        except IndexError:
            image[coords_list[0][1]*height, coords_list[0][0]*width] = [0,0,255]

        cv2.imwrite(f'{save_path}/bboxtrace{index}.jpg', image)

        crop_image = image.copy()
        bbox = crop_image[top:bottom, left:right]
        cv2.imwrite(f'{save_path}/bboxtracecrop{index}.jpg', bbox)

# make_caption
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
        
        index_lst=[(0, len(caption_lst)-1)]
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
            
    start_index=0
    index_lst=[]
    duration_lst=[]
    for group_w, group_t_c in zip(new_grouped_words, new_grouped_timed_caption):
        # index
        end_index = start_index + len(group_w)
        index_lst.append((start_index, end_index))
        start_index = end_index
        
        # duration
        try:
            duration = group_t_c[-1]['end_time'] - group_t_c[0]['start_time']
        except IndexError: # 採用区間に割り当てられたキャプションなし
            duration = 0
        duration_lst.append(duration)
        
    print(len(caption_lst), end_index)
    assert len(caption_lst)==end_index
    print(caption_lst)
    
    assert len(index_lst)==len(under_t_lst)
    assert len(index_lst)==len(duration_lst)
    
    return caption_lst, index_lst, duration_lst

# visualize_divide
def visualize_divide(
    img_url,
    img, width, height,
    save_path,
    xyt_lst, # キャプションがついていないところはすでに除かれている
    under_t_lst,
    timed_caption,
    genecap,
):
    
    # xyt_lst 分割
    grouped_xyt_lst, under_t_lst = group_coordinates_by_time(
        xyt_lst,
        under_t_lst,
        timed_caption,
    )

    # 画像分割
    bbox_lst = make_bbox(
        img, width, height,
        grouped_xyt_lst,
    )

    make_bbox_traces(
        img_url,
        save_path,
        grouped_xyt_lst,
    )

    # キャプション分割
    caption_lst, index_lst, duration_lst = make_caption(
        under_t_lst,
        timed_caption,)

    assert len(index_lst)==len(bbox_lst)
    
    ## 描画
    g_s=0
    genecap_lst = re.findall(r'\b\w+\b|[.,]', genecap)
    with open(f'{save_path}/caption.txt', mode='w') as f:
        for index, ((s,e), bbox, duration) in enumerate(zip(index_lst, bbox_lst, duration_lst)):
            bbox.save(f'{save_path}/bbox{index}.jpg')
            
            print(f'滞在時間：{duration}')
            print(f"キャプション:{' '.join(caption_lst[s:e])}")
            f.write(f"{s}~{e+1}:{' '.join(caption_lst[s:e])}\n")
            
            token_num = round(duration*2.1)
            g_e = g_s+token_num
            f.write(f"{g_s}~{g_e}:{' '.join(genecap_lst[g_s:g_e])}\n\n")
            g_s=g_e
        
        f.write(f"{g_e}~:{' '.join(genecap_lst[g_e:])}\n\n")
    
    return index_lst, bbox_lst, duration_lst, caption_lst

# analyse_dataset
def analyse_dataset(
    dataset, 
    save_dir,
    search_lst, 
    genecap_lst,
    ):
    
    max_index_feature_lst_len=0
    
    for index, row in dataset.iterrows():
        ## データをロード
        image_id = str(row['image_id'])
        annotator_id = str(row['annotator_id'])
        timed_caption = row['timed_caption']
        caption_stime = timed_caption[0]['start_time']
        caption_etime = timed_caption[-1]['end_time']
        
#         if not image_id in ['70426', '133380', '138599', '204930', '230964', '238272', '260966', '356708', '360181', '537053']:
#             continue
        
        passorno=True
        for s_index, s in enumerate(search_lst):
            if (row['annotator_id'], row['image_id']) == s:
                passorno = False
                break
                
        if passorno:
            continue
        
        print(f'-----{index}_{annotator_id}_{image_id}-----')
        
        save_path = f'./{save_dir}/{image_id}/{annotator_id}'
        try:
            os.makedirs(save_path)
        except FileExistsError:
            pass
        
        image_id6 = image_id
        while len(image_id6)!=6:
            image_id6='0'+image_id6
        dataset_id = row['dataset_id']
        dataDir = dataset_id[7:-1]
        
        ## 画像をロード
        img_url = f'/Storage/hirano/mscoco_dataset/' \
                        f'{dataDir}4/COCO_{dataDir}4_000000{image_id6}.jpg'
        try:
            img = Image.open(img_url).convert('RGB')
        except FileNotFoundError:
            if dataDir[0]=='t':
                img_url = f'/Storage/hirano/mscoco_dataset/' \
                        f'val2014/COCO_val2014_000000{image_id6}.jpg'
                img = Image.open(img_url).convert('RGB')
            else:
                img_url = f'/Storage/hirano/mscoco_dataset/' \
                        f'train2014/COCO_train2014_000000{image_id6}.jpg'
                img = Image.open(img_url).convert('RGB')
        width, height = img.size
        
    
        ## トレースデータからキャプション時間外のものを除いた xyt_lst を返す
        xyt_lst, data_num = analyse_tracedata(
            caption_stime, 
            caption_etime,
            row['traces'],
        )
        if not xyt_lst:
            continue
            
        # トレースつき画像を表示
        visualize_all(
            img_url,
            save_path,
            xyt_lst,
        )
        
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
            threshold,)
        # 変化量グラフを表示
        visualize_delta(
            threshold,
            tdelta_lst_n,
            data_num, 
            under_t_lst, 
            False,
            save_path,
            '1', #file classifier
            )
        # 変化量グラフを表示
        visualize_delta(
            threshold,
            tdelta_lst_n,
            data_num, 
            under_t_lst,
            True,
            save_path,
            '1', #file classifier
        )
        
        ### 取捨選択
        ## 不採用区間の合計移動量が小さいものを、前後の採用区間と連結
        under_t_lst = concat_under_t_lst(
            under_t_lst,
            tdelta_lst1,
            ave_v1,
        )
        # 変化量グラフを表示
        visualize_delta(
            threshold,
            tdelta_lst_n,
            data_num, 
            under_t_lst,
            True,
            save_path,
            '2',
        )
        
        ## 採用区間の合計移動量が小さいものをのぞく
        under_t_lst = remove_under_t_lst(
            under_t_lst,
            xyt_lst,
            tdelta_lst1, sum1, ave_v1)
        # 最終的な変化量グラフを表示
        visualize_delta(
            threshold,
            tdelta_lst_n, 
            data_num,
            under_t_lst,
            True,
            save_path,
            '3',
        )
        
        ## 閾値によって分けたトレースと画像を表示
        index_lst, bbox_lst, duration_lst, caption_lst = visualize_divide(
            img_url,
            img, width, height,
            save_path,
            xyt_lst,
            under_t_lst,
            timed_caption,
            genecap_lst[s_index],
        )
        print('*'*30)
        ## 最大分割数
        if len(index_lst)>max_index_feature_lst_len:
            max_index_feature_lst_len = len(index_lst)
    
    print(f'index:{index}, max_index_feature_lst_len:{max_index_feature_lst_len}')

# visualize_divide
def visualize_divide_imgid(
    img_url,
    img, width, height,
    save_path,
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
    bbox_lst = make_bbox(
        img, width, height,
        grouped_xyt_lst,
    )

    make_bbox_traces(
        img_url,
        save_path,
        grouped_xyt_lst,
    )

    # キャプション分割
    caption_lst, index_lst, duration_lst = make_caption(
        under_t_lst,
        timed_caption,)

    assert len(index_lst)==len(bbox_lst)
    
    ## 描画
    with open(f'{save_path}/caption.txt', mode='w') as f:
        for index, ((s,e), bbox, duration) in enumerate(zip(index_lst, bbox_lst, duration_lst)):
            bbox.save(f'{save_path}/bbox{index}.jpg')
            
            print(f'滞在時間：{duration}')
            print(f"キャプション:{' '.join(caption_lst[s:e])}")
            f.write(f"{s}~{e+1}:{' '.join(caption_lst[s:e])}\n")
    
    return index_lst, bbox_lst, duration_lst, caption_lst

# analyse_dataset
def analyse_dataset_imgid(
    dataset, 
    save_dir,
    search_imgid_lst,
    ):
    
    max_index_feature_lst_len=0
    
    for index, row in dataset.iterrows():
        ## データをロード
        image_id = str(row['image_id'])
        annotator_id = str(row['annotator_id'])
        timed_caption = row['timed_caption']
        caption_stime = timed_caption[0]['start_time']
        caption_etime = timed_caption[-1]['end_time']
        
        if not image_id in search_imgid_lst:
            continue
        
        print(f'-----{index}_{annotator_id}_{image_id}-----')
        
        save_path = f'./{save_dir}/{image_id}/{annotator_id}'
        try:
            os.makedirs(save_path)
        except FileExistsError:
            pass
        
        image_id6 = image_id
        while len(image_id6)!=6:
            image_id6='0'+image_id6
        dataset_id = row['dataset_id']
        dataDir = dataset_id[7:-1]
        
        ## 画像をロード
        img_url = f'/Storage/hirano/mscoco_dataset/' \
                        f'{dataDir}4/COCO_{dataDir}4_000000{image_id6}.jpg'
        try:
            img = Image.open(img_url).convert('RGB')
        except FileNotFoundError:
            if dataDir[0]=='t':
                img_url = f'/Storage/hirano/mscoco_dataset/' \
                        f'val2014/COCO_val2014_000000{image_id6}.jpg'
                img = Image.open(img_url).convert('RGB')
            else:
                img_url = f'/Storage/hirano/mscoco_dataset/' \
                        f'train2014/COCO_train2014_000000{image_id6}.jpg'
                img = Image.open(img_url).convert('RGB')
        width, height = img.size
        
    
        ## トレースデータからキャプション時間外のものを除いた xyt_lst を返す
        xyt_lst, data_num = analyse_tracedata(
            caption_stime, 
            caption_etime,
            row['traces'],
        )
        if not xyt_lst:
            continue
            
        # トレースつき画像を表示
        visualize_all(
            img_url,
            save_path,
            xyt_lst,
        )
        
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
            threshold,)
        visualize_alldelta(
            tdelta_lst_n, 
            data_num,
            save_path,'beforecut',
        )
        # 変化量グラフを表示
        visualize_delta(
            threshold,
            tdelta_lst_n,
            data_num, 
            under_t_lst, 
            False,
            save_path,
            '1_w', #file classifier
        )
        # 変化量グラフを表示
        visualize_delta(
            threshold,
            tdelta_lst_n,
            data_num, 
            under_t_lst,
            True,
            save_path,
            '1_wo', #file classifier
        )
        
        ### 取捨選択
        ## 不採用区間の合計移動量が小さいものを、前後の採用区間と連結
        under_t_lst = concat_under_t_lst(
            under_t_lst,
            tdelta_lst1,
            ave_v1
        )
        # 変化量グラフを表示
        visualize_delta(
            threshold,
            tdelta_lst_n,
            data_num, 
            under_t_lst,
            True,
            save_path,
            'afterconcat',
        )
        
        ## 採用区間の合計移動量が小さいものをのぞく
        under_t_lst = remove_under_t_lst(
            under_t_lst,
            xyt_lst,
            tdelta_lst1, sum1, ave_v1)
        # 最終的な変化量グラフを表示
        visualize_delta(
            threshold,
            tdelta_lst_n, data_num,
            under_t_lst,
            True,
            save_path,
            'afterremove',
        )
        
        ## 閾値によって分けたトレースと画像を表示
        index_lst, bbox_lst, duration_lst, caption_lst = visualize_divide_imgid(
            img_url,
            img, width, height,
            save_path,
            xyt_lst,
            under_t_lst,
            timed_caption,
        )
        print('*'*30)
        ## 最大分割数
        if len(index_lst)>max_index_feature_lst_len:
            max_index_feature_lst_len = len(index_lst)
    
    print(f'index:{index}, max_index_feature_lst_len:{max_index_feature_lst_len}')

# analyse_search : search image id from index
def analyse_search_byindexes(hdf_path, search_indexes):
    index=0
    look_index=0
    search_lst=[]
    
    for j in range(5):
        dataset = pd.read_hdf(
            hdf_path, f'data{j}',
        )
        for k, row in dataset.iterrows():
            if index==search_indexes[look_index]:
                search_lst.append((index, row['annotator_id'], row['image_id']))
                look_index+=1
                if look_index==len(search_indexes):
                    return search_lst
            index+=1
            
    return search_lst

# 
def bring_genecaption(output_path, search_lst):
    look_index=0
    genecap_lst=[]

    with open(output_path, 'r') as fj:
        output_json = json.load(fj)
        
        for item in output_json:
            image_id = item["image_id"]
            
            if f'{search_lst[look_index][0]}_{search_lst[look_index][2]}' == image_id:
                genecap_lst.append(item["caption"])
                
                look_index+=1
                if look_index==len(search_lst):
                    break
    
    assert len(search_lst)==len(genecap_lst)
    return genecap_lst

def bring_genecaption_SAR(output_path, search_lst):
    look_index=0
    genecap_lst=[]

    with open(outputpath, 'r') as file:
        for line in file:
            item = json.loads(line)
            generated = item['string']
            if len(item['string'])!=1:
                generated = generated[1]

        for item in output_json:
            image_id = item["image_id"]
            
            if f'{search_lst[look_index][0]}_{search_lst[look_index][2]}' == image_id:
                genecap_lst.append(item["caption"])
                
                look_index+=1
                if look_index==len(search_lst):
                    break
    
    assert len(search_lst)==len(genecap_lst)
    return genecap_lst


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='decoding args.')
    parser.add_argument('--searchby', type=str, default='imageid', help='path to the folder of decoded texts')
    args = parser.parse_args()

    path_lst = glob.glob('/Storage2/Dataset/localized_narratives/COCO/*jsonl')

    if args.searchby == 'index':
        for path in path_lst:
            dataset = pd.read_json(
                    path, 
                    orient='records',
                    lines=True,
            )
            hdf_test_path = '/Storage2/hirano/container_source/localized_narratives/karpathy/20240618/localized_narratives_test.h5'
            # search_indexes = [53,142,4696]
            search_indexes = random.sample(range(6090), 50)
            search_indexes.sort()
            search_lst = analyse_search_byindexes(hdf_test_path, search_indexes) # (annotator_id, image_id)
            
            date='20240508'
            model='0428'
            coef='005'
            seed='102'
            output_path = f'/Storage2/hirano/container_source/ln_results/' \
                            f'{date}/{date}_model{model}_coef{coef}_seed{seed}_step200_notkarpathy.json'
            genecap_lst = bring_genecaption(output_path, search_lst)
            
            ## 大元
            analyse_dataset(
                dataset, 
                'figure0727',
                [(x[1], x[2]) for x in search_lst],
                genecap_lst,
            )
    
    elif args.searchby == 'imageid':
        for path in path_lst:
            dataset = pd.read_json(
                    path, 
                    orient='records',
                    lines=True,
            )
            search_imgids = ['570521', '159044', '358090', '263211', '128140', '448627', '444722']
            ## 大元
            analyse_dataset_imgid(
                dataset, 
                'figure_20250306',
                search_imgids,
            )
