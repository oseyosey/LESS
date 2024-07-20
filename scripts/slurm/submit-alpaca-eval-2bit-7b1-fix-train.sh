#!/bin/bash
#SBATCH -J bbh-llama2-7b-2bit-5%-random  # Job name
#SBATCH -o slurm_out/bbbh-llama2-7b-2bit-5%-rando.o%j    # Name of stdout output file (%j expands to jobId)
#SBATCH -e slurm_out/bbh-llama2-7b-2bit-5%-rando.e%j    # Name of stderr output file
#SBATCH -N 1   # Total number of CPU nodes requested
#SBATCH -n 16   # Total number of CPU cores requrested
#SBATCH --mem=120gb    # CPU Memory pool for all cores
#SBATCH -t 72:00:00    # Run time (hh:mm:ss)
#SBATCH --partition=rush --gres=gpu:a6000:2   # Which queue to run on, and what resources to use
                                               # --partition=<queue> - Use the `<queue>` queue
                                               # --gres=gpu:1 - Use 1 GPU of any type
                                               # --gres=gpu:1080ti:1 - Use 1 GTX 1080TI GPU

nvidia-smi

cd /share/kuleshov/jy928/compute_optimal_data_selection/LESS && conda run -p /share/kuleshov/jy928/envs/LESS-2 bash scripts/full_data_revised_lora_train_mmlu.sh

# LESS/scripts/full_data_revised_lora_train_mmlu.sh

