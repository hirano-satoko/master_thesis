import argparse
import json
from collections import Counter
import pandas as pd
import numpy as np
import torch 
import os
from PIL import Image
from tqdm import tqdm

def analyse_vocab_length(path):
    words_lst=[]

    length_lst=[]
    sum_length=0
    max_length=0
    min_length=100

    data_num=0

    for i in range(5):
        dataset = pd.read_hdf(
                path, f'data{i}',
        )
        print(i, ' : ', dataset.shape)

        for index, row in dataset.iterrows():
            caption = row['caption']

            caption_length = len(caption)
            length_lst.append(caption_length)
            sum_length+=caption_length
            if caption_length>max_length:
                max_length=caption_length
            elif caption_length<min_length:
                min_length=caption_length
            
            data_num+=1

            words_lst.extend(caption)

    print(f'data num : {data_num}')
    print(f'average legnth : {sum_length/data_num}, max legnth : {max_length}, min legnth : {min_length}')
    print(np.percentile(length_lst, q=[0, 25, 50, 75, 100]))

    print(f'words : {len(words_lst)}')
    print(words_lst[:10])

    counter = Counter(words_lst)

    # for word in words_lst:
    #     print(word)
    #     counter.update(word)
    print(len(counter))

    vocab_dict = {'START': 0, 'END': 1, 'UNK':2, 'PAD':3}
    important_vocab = list(vocab_dict.keys())

    for k, v in counter.items():
        if v > 3 and not(k in important_vocab):
            vocab_dict[k] = len(vocab_dict)

    print(len(vocab_dict))
    print(list(vocab_dict.keys())[:30])

def analyse_head(path):
    for i in range(5):
        dataset = pd.read_hdf(
                path, f'data{i}',
        )
        print(i, ' : ', dataset.shape)
        for index, row in dataset.iterrows():
            print(f"{row['dataset_id']}, {row['image_id']}, {row['annotator_id']}")
            print(' '.join(row['caption']))
            if index>2:
                break

def analyse_duration(path):
    lst=[]

    for i in range(5):
        dataset = pd.read_hdf(
                path, f'data{i}',
        )
        print(i, ' : ', dataset.shape)

        for index, row in dataset.iterrows():
            caption = row['caption']
            caption_length = len(caption)

            durations = row['durations']
            duration = sum(durations)

            # assert not (0 in durations)
            assert duration != 0
            lst.append(caption_length/duration)

    print(f'average : {np.mean(lst)}')
    print(f'四分位範囲 : {np.percentile(lst, q=[0, 25, 50, 75, 100])}')

def analyse_duration_individual(path):
    duration_dict = {}

    for i in range(5):
        dataset = pd.read_hdf(
                path, f'data{i}',
        )
        print(i, ' : ', dataset.shape)

        for index, row in dataset.iterrows():
            annotator_id = row['annotator_id']

            caption = row['caption']
            caption_length = len(caption)

            durations = row['durations']
            duration = sum(durations)
            assert duration != 0

            duration_dict.setdefault(annotator_id, []).append(caption_length/duration)
    
    medium_dict = {}
    ave_dict = {}
    for k,v in duration_dict.items():
        medium_dict[k] = np.percentile(v, q=[50])[0]
        ave_dict[k] = np.mean(v)
        print(f'---annotator{k}---')
        print(f'average : {np.mean(v)}')
        print(f'四分位範囲 : {np.percentile(v, q=[0, 25, 50, 75, 100])}')
    
    print(f'number of annotators : {k}')
    
    save_path = os.path.join(os.path.dirname(path), 'annotator_utterancespeed_medium.json')
    print(f'save_path : {save_path}')
    with open(save_path, 'w') as f:
        json.dump(medium_dict, f)
    print('*'*20)
    print(medium_dict)

    save_path = os.path.join(os.path.dirname(path), 'annotator_utterancespeed_average.json')
    print(f'save_path : {save_path}')
    with open(save_path, 'w') as f:
        json.dump(ave_dict, f)
    print('*'*20)
    print(ave_dict)

def lists_match(l1, l2):
    if len(l1) != len(l2):
        return False
    return all(x == y and type(x) == type(y) for x, y in zip(l1, l2))

def analyse_order(refpath, resulstpath):

    result_lst = []
    with open(resulstpath, 'r') as f:
        json_data = json.load(f)
        for i, item in enumerate(json_data):
            result_lst.append(item["image_id"])
    
    ref_lst=[]
    for j in range(5):
        dataset = pd.read_hdf(
                refpath, f'data{j}',
        )
        for k, row in dataset.iterrows():
            ref_lst.append(str(row['image_id']))

    print(f'equality : {lists_match(result_lst, ref_lst)}')

def analyse_search(path, search_index):
    
    index=0
    for j in range(5):
        dataset = pd.read_hdf(
            path, f'data{j}',
        )
        for k, row in dataset.iterrows():
            if index==search_index:
                print(f'data{j}, {index}, {search_index}')
                return row
            index+=1
    return row

def analyse_clipscore(path):
    whole_scores = []
    divided_scores = []
    max_score = 0
    max_row = None
    min_score = 100
    min_row = None
    dividednum_lst = []

    for j in range(5):
        dataset = pd.read_hdf(
            path, f'data{j}',
        )
        print(f'-----dataset{j}-----')
        for k, row in dataset.iterrows():
            wholeimage_clipscore = row['wholeimage_clipscore']
            clipscores_array = np.array(row['black_clipscores'])
            mean_clipscores = np.mean(clipscores_array)

            whole_scores.append(wholeimage_clipscore)
            divided_scores.append(mean_clipscores)

            dividednum_lst.append(len(clipscores_array))

            # if mean_clipscores > wholeimage_clipscore:
            #     print(row)
            
            if mean_clipscores > max_score:
                max_score = mean_clipscores
                max_row = row
            if mean_clipscores < min_score:
                min_score = mean_clipscores
                min_row = row
            if mean_clipscores > 27 and len(clipscores_array)!=1:
                print(f'{mean_clipscores}\n{row}')
                print('*'*10)
    
    print('-'*30)
    print(f'whole image clipscore : {np.mean(np.array(whole_scores))}')
    print(f'divided image clipscore : {np.mean(np.array(divided_scores))}')
    print(f'分割数平均 : {np.mean(np.array(dividednum_lst))}')
    print(f'max score: {max_score}\n{max_row}')
    print(f'min score: {min_score}\n{min_row}')


def analyse_index(path):
    for j in range(5):
        dataset = pd.read_hdf(
            path, f'data{j}',
        )
        for k, row in dataset.iterrows():
            indexes = row['indexes']
            for (s,e) in indexes:
                try:
                    assert s<=e   
                except AssertionError:
                    print(s, e)
                    return row

    return row

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

def analyse_object_detection_detr(
    reffile_path,
    output_path,
    ):
    from transformers import AutoImageProcessor, DetrForObjectDetection
    # モデルとプロセッサのロード
    processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

    counter = 0
    
    for i in range(5):
        dataset = pd.read_hdf(
                reffile_path, f'data{i}',
        )
        for index, row in tqdm(dataset.iterrows()):
            annotator_id = row['annotator_id']
            image_id = row['image_id']

            # if annotator_id==31 and ismage_id==31636:
            gt_lst = row['caption']
            index_lst = row['indexes']
            corner_lst = row['corners']
            # BBOX
            image = load_image(str(image_id))
            cropped_images = []
            for (left, top, right, bottom) in corner_lst:
                cropped_image = image.crop((left, top, right, bottom))
                cropped_images.append(cropped_image)

            # 推論
            inputs = processor(images=cropped_images, return_tensors="pt")
            outputs = model(**inputs)
            results = processor.post_process_object_detection(
                outputs, 
                # threshold=0.9, # defalutは0.5
                )
            # 正解配列作成
            od_dict = {}
            assert len(index_lst) == len(results), "Mismatch between index list and detection results"
            for step, ((s,e), result) in enumerate(zip(index_lst, results)):
                key = f"({s},{e})"  # タプルを文字列に変換
                gt = ' '.join(gt_lst[s:e])
                if 'labels' in result and len(result['labels']) > 0:
                    od_dict[key] = list({
                        label_name
                        for label in result['labels']
                        if (label_name := model.config.id2label[label.item()]) in gt
                    })
                else:
                    od_dict[key] = []
                    
            export_dict = {
                "image_id" : image_id,
                "annotator_id" : annotator_id,
                "counter" : counter,
                "object_detection" : od_dict,
            }
            with open(output_path, "a") as f:
                # 新しい要素が生成されるたびに保存
                f.write(json.dumps(export_dict) + "\n")

            counter+=1

def analyse_object_detection_segformer(
    reffile_path,
    output_path,
    ):
    from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
    # SegFormer のロード
    processor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")

    counter = 0
    
    for i in range(5):
        dataset = pd.read_hdf(
                reffile_path, f'data{i}',
        )
        for index, row in tqdm(dataset.iterrows()):
            annotator_id = row['annotator_id']
            image_id = row['image_id']

            # if annotator_id==31 and ismage_id==31636:
            gt_lst = row['caption']
            index_lst = row['indexes']
            corner_lst = row['corners']
            # BBOX
            image = load_image(str(image_id))
            cropped_images = []
            for (left, top, right, bottom) in corner_lst:
                cropped_image = image.crop((left, top, right, bottom))
                cropped_images.append(cropped_image)

            # 推論
            inputs = processor(images=cropped_images, return_tensors="pt")
            outputs = model(**inputs)
            # 出力結果のアーギュメント処理
            logits = outputs.logits  # [batch_size, num_labels, height, width]
            # 各ピクセルごとに最大スコアを持つクラスを取得
            predicted_classes = torch.argmax(logits, dim=1)  # [batch_size, height, width]

            # 正解配列作成
            od_dict = {}
            assert len(index_lst) == len(predicted_classes), "Mismatch between index list and detection results"
            for step, ((s,e), predicted_class) in enumerate(zip(index_lst, predicted_classes)):
                key = f"({s},{e})"  # タプルを文字列に変換
                gt = ' '.join(gt_lst[s:e])
                # 一意なクラスIDを取得
                unique_classes = torch.unique(predicted_class)  

                if len(unique_classes) > 0:
                    od_dict[key] = list({
                        label_name
                        for unique_class in unique_classes
                        if (label_name := model.config.id2label[unique_class.item()]) in gt
                    })
                else:
                    od_dict[key] = []
                    
            export_dict = {
                "image_id" : image_id,
                "annotator_id" : annotator_id,
                "counter" : counter,
                "object_detection" : od_dict,
            }
            with open(output_path, "a") as f:
                # 新しい要素が生成されるたびに保存
                f.write(json.dumps(export_dict) + "\n")

            counter+=1

# set を返す
def get_label_names_from_classes(classes, id2label, gt):
    return {
        id2label[class_item.item()]
        for class_item in classes
        if id2label[class_item.item()] in gt
    }

def analyse_object_detection_detrsegformer(
    reffile_path,
    output_path,
    ):
    from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation, AutoImageProcessor, DetrForObjectDetection
    # SegFormer のロード
    segformer_processor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    segformer_model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    # モデルとプロセッサのロード
    detr_processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
    detr_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

    counter = 0
    
    for i in range(5):
        dataset = pd.read_hdf(
                reffile_path, f'data{i}',
        )
        for index, row in tqdm(dataset.iterrows()):
            annotator_id = row['annotator_id']
            image_id = row['image_id']

            # if annotator_id==31 and ismage_id==31636:
            gt_lst = row['caption']
            index_lst = row['indexes']
            corner_lst = row['corners']
            # BBOX
            image = load_image(str(image_id))
            cropped_images = []
            for (left, top, right, bottom) in corner_lst:
                cropped_image = image.crop((left, top, right, bottom))
                cropped_images.append(cropped_image)

            # SEGFORMER 推論
            inputs = segformer_processor(images=cropped_images, return_tensors="pt")
            outputs = segformer_model(**inputs)
            logits = outputs.logits  # [batch_size, num_labels, height, width]
            predicted_classes = torch.argmax(logits, dim=1)  # [batch_size, height, width]

            # DETR 推論
            inputs = detr_processor(images=cropped_images, return_tensors="pt")
            outputs = detr_model(**inputs)
            results = detr_processor.post_process_object_detection(
                outputs, 
                # threshold=0.9, # defalutは0.5
                )

            # 正解配列作成
            od_dict = {}
            assert len(index_lst) == len(predicted_classes), "Mismatch between index list and detection results"
            assert len(index_lst) == len(results), "Mismatch between index list and detection results"
            for step, ((s,e), predicted_class, result) in enumerate(zip(index_lst, predicted_classes, results)):
                key = f"({s},{e})"
                gt = ' '.join(gt_lst[s:e])

                # segformer 処理
                unique_classes = torch.unique(predicted_class)
                segformer_labels = get_label_names_from_classes(unique_classes, segformer_model.config.id2label, gt)

                # DETR 結果の処理
                detr_labels = get_label_names_from_classes(result['labels'], detr_model.config.id2label, gt)

                od_dict[key] = list(segformer_labels.union(detr_labels))
                    
            export_dict = {
                "image_id" : image_id,
                "annotator_id" : annotator_id,
                "counter" : counter,
                "object_detection" : od_dict,
            }
            with open(output_path, "a") as f:
                # 新しい要素が生成されるたびに保存
                f.write(json.dumps(export_dict) + "\n")

            counter+=1

def analyse_length_comparison(path):
    import numpy as np
    from scipy.spatial.distance import cosine
    from scipy.stats import pearsonr
    from scipy.special import kl_div

    # annotatorごとの発語速度
    json_path = '/Storage2/hirano/container_source/localized_narratives/karpathy/20240618/annotator_utterancespeed_medium.json'
    with open(json_path, 'r', encoding='utf-8') as file:
        annotator_utterancespeed_dict = json.load(file)

    # 初期化
    mae_list= []
    mre_list = []
    cos_sim_list = []
    pearson_corr_list = []
    kl_divergence_list= []
    
    counter = 0
    # 処理開始
    for i in range(5):
        dataset = pd.read_hdf(
                path, f'data{i}',
        )
        for index, row in dataset.iterrows():
            # 正解単語数配列
            index_lst = row['indexes']
            gt_length_list = [e - s for (s, e) in index_lst]

            # 予測単語数配列
            durations = row['durations']
            annotator_id = str(row['annotator_id'])
            utterancespeed = annotator_utterancespeed_dict[annotator_id]
            pred_length_list = []
            for duration in durations:
                token_num = torch.ceil(torch.tensor(duration * utterancespeed)).int().item()
                pred_length_list.append(token_num)
            
            assert len(gt_length_list)==len(pred_length_list), "!mismatch!"
            assert len(gt_length_list)!=0, "!length is 0!"
            # print(f'gt_length_list : {gt_length_list}')
            # print(f'pred_length_list : {pred_length_list}')
            # print('*'*3)

            gt_length_list = np.array(gt_length_list)
            pred_length_list = np.array(pred_length_list)

            # (1). 平均誤差 (Mean Absolute Error)
            mae = np.mean(np.abs(gt_length_list - pred_length_list))
            mae_list.append(mae.item())

            # (2). 平均相対誤差 (Mean Relative Error)
            if not np.any(gt_length_list == 0):
                mre = np.mean(np.abs((gt_length_list - pred_length_list) / gt_length_list))
                mre_list.append(mre.item())

            # (3). コサイン類似度 (Cosine Similarity)
            cos_sim = 1 - cosine(gt_length_list, pred_length_list)  # コサイン距離は1から引く
            cos_sim_list.append(cos_sim)

            # (4). ピアソン相関係数 (Pearson Correlation Coefficient)
            if len(gt_length_list) >= 2 and len(np.unique(gt_length_list)) > 1 and len(np.unique(pred_length_list)) > 1 :
                pearson_corr, _ = pearsonr(gt_length_list, pred_length_list)
                pearson_corr_list.append(pearson_corr)

            # (5). KLダイバージェンス (Kullback-Leibler Divergence)
            # 1. 整数を確率分布に変換
            sum_array1 = np.sum(gt_length_list)
            sum_array2 = np.sum(pred_length_list)
            # 0で割るエラーを防ぐために、各配列をその合計で割る
            prob1 = gt_length_list / sum_array1
            prob2 = pred_length_list / sum_array2
            # 2. 0の要素に小さい値を足す (ゼロ除算を防ぐ)
            prob1 = np.where(prob1 == 0, 1e-10, prob1)
            prob2 = np.where(prob2 == 0, 1e-10, prob2)
            # 3. KLダイバージェンスの計算
            kl_divergence = np.sum(kl_div(prob1, prob2))
            kl_divergence_list.append(kl_divergence.item())

            counter+=1
            # if counter>10:
            #     break
    
    print(f"Mean Absolute Error: {np.mean(mae_list)}")
    print(f"Mean Relative Error: {np.mean(mre_list)}")
    print(f"Cosine Similarity: {np.mean(cos_sim_list)}")
    print(f"Pearson Correlation Coefficient: {np.mean(pearson_corr_list)}")
    print(f"KL Divergence: {np.mean(kl_divergence_list)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='decoding args.')
    parser.add_argument('--analyse', type=str, default='vocab_length', help='path to the folder of decoded texts')
    parser.add_argument('--search_index', type=int, default=1, help='index you want to search')
    
    args = parser.parse_args()

    # path = '/home/hirano/blip/BLIP/localized_narratives_train.h5'
    path = '/Storage2/hirano/container_source/localized_narratives/karpathy/20240618/localized_narratives_test.h5'
    # path = '/Storage2/hirano/container_source/localized_narratives/karpathy/localized_narratives_test.h5'

    if args.analyse == 'vocab_length':
        analyse_vocab_length(path)
    elif args.analyse == 'head':
        analyse_head(path)
    elif args.analyse == 'duration':
        analyse_duration(path)
    elif args.analyse == 'duration_individual':
        analyse_duration_individual(path)
    elif args.analyse == 'order':
        resultpath = '/Storage2/hirano/container_source/ln_results/20240125/20240125_model0123_coef0005_seed100_step200.json'
        analyse_order(path, resultpath)
    elif args.analyse == 'search':
        row = analyse_search(path, args.search_index)
        print(f"dataset_id : {row['dataset_id']}")
        print(f"image_id : {row['image_id']}")
        print(f"annotator_id : {row['annotator_id']}")
        print(f"caption : {row['caption']}")
        print(f"caption : {' '.join(row['caption'])}")
        print(f"indexes : {row['indexes']}")
        print(f"durations : {row['durations']}")
        print('*'*30)
    elif args.analyse == 'clipscore':
        analyse_clipscore(path)
    elif args.analyse == 'index':
        row = analyse_index(path)
        print(row)
    elif args.analyse == 'object_detection':
        # analyse_object_detection_detr(
        # analyse_object_detection_segformer(
        analyse_object_detection_detrsegformer(
        '/Storage2/hirano/container_source/localized_narratives/karpathy/20240618/localized_narratives_test.h5',
        '/home/hirano/interactive_captiongeneration/evaluation/object_detection_test_detr0.5segformer.jsonl',
        # '/Storage2/hirano/container_source/localized_narratives/karpathy/20240618/object_detection_test.jsonl',
        )
    elif args.analyse == 'length_comparison':
        analyse_length_comparison(
            '/Storage2/hirano/container_source/localized_narratives/karpathy/20240618/localized_narratives_test.h5',
        )
