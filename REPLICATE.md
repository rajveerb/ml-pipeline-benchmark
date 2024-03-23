We provide the code/scripts to replicate P3Tracer experiment results in the cloudlab testbed using a c4130 node available in Wisconsin cluster.

### Steps

1. Use our [`P3Tracer_c4130_cloudlab.profile`](P3Tracer_c4130_cloudlab.profile) as the geni script for cloudlab experiment profile. This profile uses a long term cloudlab dataset as a mounted remote filesystem with 916 GB storage. Mounted on `/mydata`.

2. Grow root partition of c4130 node on cloudlab (If this step is not performed, there will be no space left in root for installations):
    ```bash
    export RESIZEROOT=100
    sudo scripts/cloudlab/grow-rootfs.sh
    export RESIZEROOT=
    ```
3. Install Intel VTune:
    ```bash
    sudo bash scripts/cloudlab/install_vtune.sh
    # Make sure to run below commands 
    echo "source /opt/intel/oneapi/setvars.sh" > ~/.bashrc
    sudo usermod -aG vtune $USER && newgrp vtune
    ```
    Note: we used `Intel(R) VTune(TM) Profiler 2024.0.1 (build 627177)`
4. Install CUDA 11.8:
    ```bash
    sudo bash scripts/cloudlab/cuda_installer_script.sh
    ```
5. Install CuDNN 8.7.0:
    ```bash
    sudo bash scripts/cloudlab/cudnn_installer_script.sh
    ```
6. Reboot the machine:
    ```bash
    sudo reboot
    ```
7. Install [`vmtouch`](https://linux.die.net/man/8/vmtouch):
    ```bash
    sudo apt install vmtouch
    ```
8. Clone this repository!
9. Get submodules:

    ```git
    git submodule update --init --recursive
    ```
10. Install anaconda from [here](https://conda.io/projects/conda/en/latest/user-guide/install/linux.html): 
11. Create a conda environment
    ```bash
    conda create -n P3Tracer python=3.10
    conda activate P3Tracer
    ```
12. Follow the **P3Torch** build instructions in `code/P3Torch/README.md`
13. Follow the **itt-python** build instructions in `code/itt-python/README.md`
14. Follow the **torchvision** build instructions in `code/torchvision/README.md`
15. Get the mapping logs for the preprocessing operations:
    ```bash
    bash code/image_classification/P3Map/P3Map.sh
    ```
16. Generate JSON file with mapping info by running all cells in [`code/image_classification/P3Map/logsToMapping.ipynb`](code/image_classification/P3Map/logsToMapping.ipynb)
17. You have successfully obtained the mapping ([`code/image_classification/P3Map/mapping_funcs.json`](code/image_classification/P3Map/mapping_funcs.json)) using **P3Map**!
18. 