source eval.sh

DIR=$1 # model directory
DATA_DIR=$2 # evaluation data directory
OUTPUT_DIR=$3 # output directory (if needed)

# main evaluation function
#* this is with chain of thought (COT)
#* increase the maximum number of examples from 200 to 1000.
eval_gsm8k() {
    mdir=$DIR
    set_save_dir $mdir gsm8k $OUTPUT_DIR
    echo $save_dir
    mkdir -p $save_dir
    cmd="python -m eval.gsm.run_eval \
    --data_dir $DATA_DIR/gsm/ \
    --n_shot 8 \
    --max_num_examples 200 \
    --save_dir $save_dir \
    --model $mdir \
    --tokenizer $mdir \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format"
    eval "$cmd" 2>&1 | tee $save_dir/log-200.txt
}

# eval_gsm8k() {
#     mdir=$DIR
#     set_save_dir $mdir gsm8k $OUTPUT_DIR
#     echo $save_dir
#     mkdir -p $save_dir
#     cmd="python -m eval.gsm.run_eval \
#     --data_dir $DATA_DIR/gsm/ \
#     --n_shot 8 \
#     --max_num_examples 200 \
#     --save_dir $save_dir \
#     --model $mdir \
#     --tokenizer $mdir \
#     --use_chat_format \
#     --no_cot \
#     --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format"
#     eval "$cmd" 2>&1 | tee $save_dir/log-200.txt
# }


# extract the results
extract_gsm8k() {
    mdir=$DIR
    set_save_dir $mdir gsm8k $OUTPUT_DIR
    result=$(jq .exact_match $save_dir/metrics-200.json)
    result=$(echo "$result * 100" | bc)
    echo $result
}

export -f eval_gsm8k
export -f extract_gsm8k

