
model_dir=$1
data_dir=$2
cuda_device="0"

for split_idx in 0 1 2 3 4;
do

  allennlp predict ${model_dir}/xwlp-split-${split_idx}/model.tar.gz \
    ${data_dir}/dygiepp/split_${split_idx}/test.json \
    --predictor dygie \
    --include-package dygie \
    --use-dataset-reader \
    --output-file ${model_dir}/xwlp-split-${split_idx}/test_pred.jsonl \
    --cuda-device $cuda_device
done