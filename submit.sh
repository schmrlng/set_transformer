#!/bin/bash
#SBATCH --array=1-5
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=2
#SBATCH --mem-per-cpu=2G
#SBATCH --time=120

module load python/3.9.0 cuda/11.2.0 cudnn/8.1.1.33
source env/bin/activate
case $SLURM_ARRAY_TASK_ID in
    1)
        python3 main.py --config.input_encoding=fourier_feature --config.input_encoding_scale=0.01
        ;;
    2)
        python3 main.py --config.input_encoding=fourier_feature --config.input_encoding_scale=0.1
        ;;
    3)
        python3 main.py --config.input_encoding=fourier_feature --config.input_encoding_scale=1.0
        ;;
    4)
        python3 main.py --config.input_encoding=sinusoidal
        ;;
    5)
        python3 main.py --config.input_encoding=affine
        ;;
esac
