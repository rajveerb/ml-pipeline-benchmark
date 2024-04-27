# !/bin/bash

root_dir_="/home/hice1/mrao70/scratch/special_problems"
home_dir_="/home/hice1/mrao70"
# /home/hice1/mrao70/scratch/datasets
dataset_dir="${home_dir_}/scratch/datasets/imagenet"
# result_dir_="${root_dir_}/results"
# python_path="${home_dir_}/.conda/envs/ml-profiling/bin/python"
# # program_path="/mydata/rbachkaniwala3/code/rajveerb-ml-pipeline-benchmark/code/image_classification/code/pytorch_main.py"
# program_path="${root_dir_}/ml-pipeline-benchmark/code/image_classification/code/pytorch_main.py"
# time_binary="/bin/time"
# dataset_dir="${root_dir_}/tiny-imagenet-200"


result_dir_="${root_dir_}/pytorch_profiles_imagenet_dataset"
result_dir_2="${root_dir_}/results"
# result_dir_="/mydata/pytorch_profiles_imagenet_dataset"
python_path="${home_dir_}/.conda/envs/ml-profiling/bin/python"
program_path="${root_dir_}/ml-pipeline-benchmark/code/image_classification/code/pytorch_main.py"
profiler_steps=100

batch_sizes=("128" "256" "512" "1024")
num_gpus=("1" "2" "3" "4")

# batch_sizes=("512" "1024")
# num_gpus=("2" "3" "4")

num_epochs=1


# check if result directory exists
if [ ! -d ${result_dir_} ]; then
    mkdir ${result_dir_}
fi

for batch_size in "${batch_sizes[@]}"
do
    for num_gpu in "${num_gpus[@]}"
    do
        result_dir=${result_dir_}/b${batch_size}_gpu${num_gpu}
        result_dir2=${result_dir_2}/b${batch_size}_gpu${num_gpu}
        # check if result directory exists
        if [ ! -d ${result_dir2} ]; then
            mkdir -p ${result_dir2}
        fi

        # TORCH_DATALOADER_PIN_CORE=1 ${python_path} ${program_path} ${dataset_dir} -b ${batch_size} --gpus ${num_gpu} -j ${num_gpu} --epochs ${num_epochs} --val-loop 0 --profile --profile-log-prefix ${result_dir};
        # ${python_path} ${program_path} ${dataset_dir} -b ${batch_size} --gpus ${num_gpu} -j ${num_gpu} --epochs ${num_epochs} --val-loop 0 --profile --profile-log-prefix ${result_dir} --profiler-steps ${profiler_steps} --log-train-file ${result_dir2}/custom_log --gpu-idle-times ${result_dir2}/gpu_idle_times;
        TORCH_DATALOADER_PIN_CORE=1 ${python_path} ${program_path} ${dataset_dir} -b ${batch_size} --gpus ${num_gpu} -j ${num_gpu} --epochs ${num_epochs} --val-loop 0 --log-train-file ${result_dir2}/custom_log --gpu-idle-times ${result_dir2}/gpu_idle_times;
    done
done