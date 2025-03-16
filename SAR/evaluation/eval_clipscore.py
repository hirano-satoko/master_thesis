import argparse
import os
from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import re
from torchvision import transforms

from torchmetrics.multimodal.clip_score import CLIPScore
metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16")

def NAR(
    outputpath,
    reffilepath,
):
    # {annotatorid_imageid : corner}
    corner_dict = {}
    # {annotatorid_imageid : indexes}
    index_dict = {}
    j=0
    ## 正解文の辞書作成
    for i in range(5):
        dataset = pd.read_hdf(
                reffilepath, f'data{i}',
        )
        for index, row in dataset.iterrows():
            image_id = row['image_id']
            assert len(row['corners'])==len(row['indexes'])
            corner_dict[f'{j}_{image_id}'] = row['corners']
            index_dict[f'{j}_{image_id}'] = row['indexes']
            j+=1
    
    seg_clipscore_lst = []
    clipscore_lst = []
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
            
            index_lst = index_dict[image_id]
            corner_lst = corner_dict[image_id]

            image = load_image(image_id.split("_")[1])

            seg_clipscore = []
            for (s,e), (left, top, right, bottom) in zip(index_lst, corner_lst):
                clipscore = metric(transforms.functional.to_tensor(image.crop((left, top, right, bottom))), ' '.join(generated_lst[s:e])).detach().item()
                seg_clipscore.append(clipscore)
            seg_clipscore_lst.append(np.mean(seg_clipscore))

            clipscore = metric(transforms.functional.to_tensor(image), generated).detach().item()
            clipscore_lst.append(clipscore)

    print('avg seg CLIPSore', np.mean(seg_clipscore_lst))
    print('avg CLIPSore', np.mean(clipscore_lst))

def SAR(
    outputpath,
    reffilepath,
    ):
    
    # {annotatorid_imageid : corner}
    corner_dict = {}
    # {annotatorid_imageid : indexes}
    index_dict = {}
    for i in range(5):
        dataset = pd.read_hdf(
                reffilepath, f'data{i}',
        )
        for index, row in dataset.iterrows():
            annotator_id = row['annotator_id']
            image_id = row['image_id']
            assert len(row['corners'])==len(row['indexes'])
            corner_dict[f'{annotator_id}_{image_id}'] = row['corners']
            index_dict[f'{annotator_id}_{image_id}'] = row['indexes']

    seg_clipscore_lst = []
    clipscore_lst = []
    with open(outputpath, 'r') as file:
        for line in tqdm(file):
        # for line in file:
            item = json.loads(line)
            generated = item['string']
            if len(item['string'])!=1:
                generated = generated[0]
            generated_lst = re.findall(r'\b\w+\b|[.,]', generated)

            annotator_id = item['annotator_id']
            image_id = item['image_id']
            
            index_lst = index_dict[f'{annotator_id}_{image_id}']
            corner_lst = corner_dict[f'{annotator_id}_{image_id}']

            image = load_image(str(image_id))

            seg_clipscore = []
            for (s,e), (left, top, right, bottom) in zip(index_lst, corner_lst):
                clipscore = metric(transforms.functional.to_tensor(image.crop((left, top, right, bottom))), ' '.join(generated_lst[s:e])).detach().item()
                seg_clipscore.append(clipscore)
            seg_clipscore_lst.append(np.mean(seg_clipscore))

            clipscore = metric(transforms.functional.to_tensor(image), generated).detach().item()
            clipscore_lst.append(clipscore)

    print('avg seg CLIPSore', np.mean(seg_clipscore_lst))
    print('avg CLIPSore', np.mean(clipscore_lst))

    # 書き込み対象のCSVファイル名
    csv_file = "/home/hirano/interactive_captiongeneration/evaluation/output.csv"
    # CSVを読み込む
    df = pd.read_csv(csv_file)
    # 元の行数を取得
    num_rows = len(df)
    # 新しい列を追加する
    df["seg_clipscore"] = seg_clipscore_lst[:num_rows]
    # CSVファイルに保存する
    df.to_csv(csv_file, index=False)

    return np.mean(seg_clipscore_lst), np.mean(clipscore_lst)


def load_image(image_id):
    while len(image_id)!=6:
        image_id='0'+image_id

    # 画像をロード
    base_path = '/Storage/hirano/mscoco_dataset'
    image_paths = [
        f'{base_path}/val2014/COCO_val2014_000000{image_id}.jpg',
        f'{base_path}/train2014/COCO_train2014_000000{image_id}.jpg'
    ]
    for path in image_paths:
        if os.path.exists(path):
            img = Image.open(path).convert('RGB')
            break
    else:
        raise FileNotFoundError(f"Image with ID {image_id} not found in val2014 or train2014.")
    
    return img

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='decoding args.')
    parser.add_argument('--outputpath', type=str, default='', help='path to the folder of decoded texts')
    parser.add_argument('--method', type=str, default='', help='NAR or SAR')
    args = parser.parse_args()

    if args.method == 'NAR':
        NAR(
            args.outputpath,
            '/Storage2/hirano/container_source/localized_narratives/karpathy/20240618/localized_narratives_test.h5',
        )
    elif args.method == 'SAR':
        seg_clipscore, all_clipscore = SAR(
            args.outputpath,
            '/Storage2/hirano/container_source/localized_narratives/karpathy/20240618/localized_narratives_test.h5',
        )