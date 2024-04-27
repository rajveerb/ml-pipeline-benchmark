# !/bin/bash
root_dir_="/home/hice1/mrao70/scratch/special_problems"
home_dir_="/home/hice1/mrao70"
result_dir_="${root_dir_}/results"
python_path="${home_dir_}/.conda/envs/ml-profiling/bin/python"
# program_path="/mydata/rbachkaniwala3/code/rajveerb-ml-pipeline-benchmark/code/image_classification/code/pytorch_main.py"
program_path="${root_dir_}/ml-pipeline-benchmark/code/image_classification/code/pytorch_main.py"
time_binary="/bin/time"
dataset_dir="${home_dir_}/scratch/datasets/imagenet"


# batch_sizes=("128" "256" "512" "1024")
# num_gpus=("1" "2" "3" "4")
batch_sizes=("1024")
num_gpus=("1")
num_epochs=1
time_format="wall(s),user(s),kernel(s),max_rss(KB)\n%e,%U,%S,%M"
e2e_log_dir="${result_dir_}/e2e"

# check if result directory exists
if [ ! -d ${result_dir_} ]; then
    mkdir -p ${result_dir_}
fi

# check if e2e log directory exists
if [ ! -d ${e2e_log_dir} ]; then
    mkdir -p ${e2e_log_dir}
fi

for batch_size in "${batch_sizes[@]}"
do
    for num_gpu in "${num_gpus[@]}"
    do

        # if batch size is 1024 and num_gpu is 1 then skip
        # Below is done due to CUDA out of memory error for V100 GPUs
        # if [ ${batch_size} -eq 1024 ] && [ ${num_gpu} -eq 1 ]; then
        #     continue
        # fi

        result_dir="${result_dir_}/b${batch_size}_gpu${num_gpu}"

        # if result directory exists, clear it
        if [ -d ${result_dir} ]; then
            rm -rf ${result_dir}/*
        fi

        # check if result directory exists
        if [ ! -d ${result_dir} ]; then
            mkdir -p ${result_dir}
        fi

        # TORCH_DATALOADER_PIN_CORE=1 as env variable to pin each data loader to a specific core 
        TORCH_DATALOADER_PIN_CORE=1 ${time_binary} --format=${time_format} -o "${e2e_log_dir}/custom_log_b${batch_size}_gpu${num_gpu}.log" ${python_path} ${program_path} ${dataset_dir} -b ${batch_size} --gpus ${num_gpu} -j ${num_gpu} --epochs ${num_epochs} --log-train-file ${result_dir}/custom_log --val-loop 0 --gpu-idle-times ${result_dir}/gpu_idle_times;
    done
done