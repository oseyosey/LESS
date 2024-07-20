export HF_HOME=/share/kuleshov/jy928/hf_data

DATA_DIR=data
MODEL_PATH=meta-llama/Llama-2-7b-hf
PERCENTAGE=0.05 # percentage of the full data to train, you can specify the training file you want to use in the script
DATA_SEED=6
JOB_NAME=llama2-7b-p${PERCENTAGE}-lora-seed${DATA_SEED}
VALIDATION_TASK=mmlu
include_validation=false

./less/scripts/train/full_data_lora_train.sh "$DATA_DIR" "$MODEL_PATH" "$PERCENTAGE" "$DATA_SEED" "$JOB_NAME" "$VALIDATION_TASK" "$include_validation"