#!/bin/bash
outputfolder='/Storage2/hirano/container_source/ln_results/20241113/'
reffilepath='/Storage2/hirano/container_source/localized_narratives/karpathy/localized_narratives_test.h5'

outputpath='/home/hirano/container_source/appendix/model20250203_average_t>1000.0-ctrlr100.0_t>800.0-ctrlr10.0_t>0-ctrlr1.0ssd_ctrsa_gen.jsonl'
# '/Storage2/hirano/container_source/ln_results/SAR/20250107/model20250104_average_t>700.0-ctrlr100.0_t>700.0-ctrlr10.0_t>0-ctrlr1.0ssd_ctrsa_gen.jsonl'

mbr_outputpath='/home/hirano/container_source/appendix/model20250203_average_t>1000.0-ctrlr100.0_t>800.0-ctrlr10.0_t>0-ctrlr1.0ssd_ctrsa_gen.jsonl'
# '/Storage2/hirano/container_source/ln_results/20241113/20241113_model0428_coef005_seed102_step200_perplexity_notkarpathy.json'
# mbr_outputpath='/Storage2/hirano/container_source/ln_results/SAR/20250107/model20250104_average_t>700.0-ctrlr100.0_t>700.0-ctrlr10.0_t>0-ctrlr1.0ssd_ctrsa_gen.jsonl'

method='SAR'
mbrby='perplexity'

# エラー時にスクリプトを停止するオプション
set -xe

# スクリプト開始メッセージ
echo "Starting to execute Python scripts with Conda environment..."

# Python ファイルを順番に実行
if [ "$method" == 'NAR' ]; then
    echo "Running eval_capgene_20240803.py..."
    python eval_capgene_20240803.py --outputfolder ${outputfolder} --reffilepath ${reffilepath}
elif [ "$method" == 'SAR' ]; then
    echo "Running eval_capgene_ssd-lm.py..."
    python eval_capgene_ssdlm.py --outputpath ${outputpath} --mbrby ${mbrby}
else
    echo "Invalid method: $method"
    exit 1
fi
echo "--------------------------------------------------------------------------------"

echo "Running eval_clipscore.py..."
python eval_clipscore.py --outputpath ${mbr_outputpath} --method ${method}
echo "--------------------------------------------------------------------------------"

echo "Running eval_fmeasure.py..."
python eval_fmeasure.py --outputpath ${mbr_outputpath} --method ${method} --reffiletype threshold0.9
python eval_fmeasure.py --outputpath ${mbr_outputpath} --method ${method} --reffiletype threshold0.5
python eval_fmeasure.py --outputpath ${mbr_outputpath} --method ${method} --reffiletype segformer
python eval_fmeasure.py --outputpath ${mbr_outputpath} --method ${method} --reffiletype detr0.5segformer

# 終了メッセージ
echo "All scripts executed successfully in Conda environment."