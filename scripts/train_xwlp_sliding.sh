

cuda_device="0,1,2,3"
data_dir=$1
output_dir=$2
config_file="./training_config/xwlp_sliding.jsonnet"

for split_idx in 0 1 2 3 4;
do
  experiment_name="xwlp-split-${split_idx}"
  data_root="${data_dir}/dygiepp/split_${split_idx}"
  serial_dir="${output_dir}/${experiment_name}"
  cache_dir="${output_dir}/${experiment_name}/cached"

  # Train model.
  ie_train_data_path=$data_root/train.json \
      ie_dev_data_path=$data_root/dev.json \
      ie_test_data_path=$data_root/test.json \
      cuda_device=$cuda_device \
      allennlp train $config_file \
      --cache-directory $cache_dir \
      --serialization-dir  $serial_dir \
      --include-package dygie
done
