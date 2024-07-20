#!/bin/bash
#SBATCH -J bbh-llama1-13b-2bit-fix-train-eval    # Job name
#SBATCH -o slurm_out/bbh-llama1-13b-2bit-fix-train-eval.o%j    # Name of stdout output file (%j expands to jobId)
#SBATCH -e slurm_out/bbh-llama1-13b-2bit-fix-train-eval.e%j    # Name of stderr output file
#SBATCH -N 1   # Total number of CPU nodes requested
#SBATCH -n 16   # Total number of CPU cores requrested
#SBATCH --mem=100gb    # CPU Memory pool for all cores
#SBATCH -t 72:00:00    # Run time (hh:mm:ss)
#SBATCH --partition=rush --gres=gpu:a6000:2   # Which queue to run on, and what resources to use
                                               # --partition=<queue> - Use the `<queue>` queue
                                               # --gres=gpu:1 - Use 1 GPU of any type
                                               # --gres=gpu:1080ti:1 - Use 1 GTX 1080TI GPU

nvidia-smi
cd /share/kuleshov/jy928/compute_optimal_data_selection/LESS
conda activate -p /share/kuleshov/jy928/envs/LESS 

export HF_HOME=/share/kuleshov/jy928/hf_data

DATA_DIR=data
MODEL_PATH=meta-llama/Llama-2-7b-hf
PERCENTAGE=0.05 # percentage of the full data to train, you can specify the training file you want to use in the script
DATA_SEED=6
JOB_NAME=llama2-7b-p${PERCENTAGE}-lora-seed${DATA_SEED}
VALIDATION_TASK=mmlu
include_validation=false

./less/scripts/train/full_data_lora_train.sh "$DATA_DIR" "$MODEL_PATH" "$PERCENTAGE" "$DATA_SEED" "$JOB_NAME" "$VALIDATION_TASK" "$include_validation"

# cd /share/kuleshov/jy928/two_bit_quant/hadamard_cuda && conda run -p /share/kuleshov/jy928/envs/llmtools-quip-d4 --no-capture-output python setup.py install
# cd /share/kuleshov/jy928/llmtools-2bit/quip/quiptools && conda run -p /share/kuleshov/jy928/envs/llmtools-quip-d4 --no-capture-output python setup.py install
# cd /share/kuleshov/jy928/llmtools-2bit && conda run -p /share/kuleshov/jy928/envs/llmtools-quip-d4 --no-capture-output python setup.py install
# cd /share/kuleshov/jy928/llmtools-2bit/experiment_bbh && conda run -p /share/kuleshov/jy928/envs/llmtools-quip-d4 --no-capture-output python alpaca_quip/train_bbh_quip.py --model_name /share/kuleshov/jy928/llmtools-2bit/quip/quantized_weights/llama1-quip-13B-D4 --adapter llama1_adapters/llama1-bbh-2bit-13b-fix-seed42-mb4-old --seed 42 --mbatch_size 4
#cd /share/kuleshov/jy928/llmtools-2bit/experiment_bbh && conda run -p /share/kuleshov/jy928/envs/llmtune-new --no-capture-output python mnli_quip_fix/eval_mnli_llama_quip.py --model_name /share/kuleshov/jy928/llmtools-2bit/quip/quantized_weights/llama1-quip-7b-D4 --adapter /share/kuleshov/jy928/llmtools-2bit/experiment_mnli/llama1_adapters/llama1-mnli-2bit-7b-fix-seed65-mb32-old --seed 42 --file_name llama1_mnli_2bit_7b1_fix_65.txt --start_index 0 --end_index 9815 --checkpoint_name /share/kuleshov/jy928/llmtools-2bit/experiment_mnli/mnli_output/llama1-quip-7b-fix-output-seed65-old

