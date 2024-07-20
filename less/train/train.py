#!/usr/bin/env python
# coding=utf-8
import logging
import os
import random
import sys
import time

import datasets
import torch
import torch.distributed as dist
import transformers
# from instruction_tuning.train.lora_trainer import LoRAFSDPTrainer, Trainer
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          DataCollatorForSeq2Seq, HfArgumentParser, Trainer,
                          set_seed)
from huggingface_hub import login

from accelerate import Accelerator
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory

from less.data_selection.get_training_dataset import get_training_dataset, get_training_dataset_with_validation
from less.train.data_arguments import DataArguments, get_data_statistics
from less.train.model_arguments import ModelArguments, add_padding_to_tokenizer
from less.train.training_arguments import TrainingArguments


logger = logging.getLogger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["WANDB_DISABLED"] = "true"

def main():
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training parameters {training_args}")
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Dataset parameters {data_args}")
    logger.info(f"Number of GPUS (cuda) {torch.cuda.device_count()}")

    # login set up
    login(token = "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")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Device map
    device_index = Accelerator().process_index
    device_map = {"": device_index}

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, device_map=device_map)

    # Load training dataset
    ##TODO: Set-up hypothesis 1
    if training_args.include_validation:
        train_dataset = get_training_dataset_with_validation(data_args.train_files,
                                            tokenizer=tokenizer,
                                            max_seq_length=data_args.max_seq_length,
                                            sample_percentage=data_args.percentage,
                                            seed=data_args.sample_data_seed)
    else:
        train_dataset = get_training_dataset(data_args.train_files,
                                            tokenizer=tokenizer,
                                            max_seq_length=data_args.max_seq_length,
                                            sample_percentage=data_args.percentage,
                                            seed=data_args.sample_data_seed)

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, torch_dtype=model_args.torch_dtype, device_map=device_map)
    add_padding_to_tokenizer(tokenizer)

    # resize embeddings if needed (e.g. for LlamaTokenizer)
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
        # if you load lora model and resize the token embeddings, the requires_grad flag is set to True for embeddings
        if isinstance(model, PeftModel):
            model.get_input_embeddings().weight.requires_grad = False
            model.get_output_embeddings().weight.requires_grad = False

    if not isinstance(model, PeftModel) and model_args.lora:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=model_args.lora_target_modules,
        )
        model = get_peft_model(model, lora_config)
        logger.info(
            f"Applied LoRA to model."
        )
        model.print_trainable_parameters()

        # for checkpointing
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    get_data_statistics(train_dataset)

    if "dataset" in train_dataset.features:
        train_dataset = train_dataset.remove_columns(
            ["dataset", "id", "messages"])
            

    for index in random.sample(range(len(train_dataset)), 1):
        logger.info(
            f"Sample {index} of the training set: {train_dataset[index]}.")

    model_params = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    logger.info(f"trainable model_params: {model_params}")

    #TODO: Add support for evaluation dataset druing training. 
    analysis_dataset = None
    if training_args.analysis_mode:
        from less.data_selection.get_validation_dataset import get_dataset
        analysis_dataset = get_dataset(training_args.analysis_dataset,
                                       data_dir=data_args.data_dir,
                                       tokenizer=tokenizer,
                                       max_length=data_args.max_seq_length)

    # for testing if the model can go through full length
    # import torch
    # from datasets import Dataset

    # input_ids = [torch.randint(0, 32000, (2048, )) for _ in range(10000)]
    # attention_mask = [torch.ones(2048, ) for _ in range(10000)]
    # train_dataset = Dataset.from_dict({"input_ids": input_ids, "labels": input_ids, "attention_mask": attention_mask})

    if dist.is_initialized() and dist.get_rank() == 0:
        print(model)
    elif not dist.is_initialized():
        print(model)

    # accelerator = Accelerator()
    # model = accelerator.prepare(model)

    # # Data Parallel Training
    # world_size = int(os.environ.get("WORLD_SIZE", 1))
    # ddp = world_size != 1
    # if not ddp and torch.cuda.device_count() > 1:
    #     model.is_parallelizable = True
    #     model.model_parallel = True
    
    # #* Enable Naive Pipeline Parallel *#
    # num_of_gpus = torch.cuda.device_count()
    # if num_of_gpus > 1:
    #     print("Enabling Naive Pipeline Parallel")
    #     max_memory = get_balanced_memory(
    #         model,
    #         max_memory=None,
    #         no_split_module_classes=["LlamaDecoderLayer", "LlamaMLP"],
    #         dtype=model_args.torch_dtype,
    #         low_zero=False,
    #     )

    # device_map = infer_auto_device_map(
    #     model,
    #     max_memory=max_memory,
    #     no_split_module_classes=["LlamaDecoderLayer", "LlamaMLP"],
    #     dtype=model_args.torch_dtype
    # )

    # model = dispatch_model(model, device_map=device_map)


    ##TODO: inspect if they evaluate it based on their performance. 
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=analysis_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer, model=model, padding="longest")
    )

    # Training
    train_result = trainer.train()
    trainer.save_model()  # Saves the tokenizer too for easy upload

    metrics = train_result.metrics

    metrics["train_samples"] = len(train_dataset)

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # remove the full model in the end to save space, only adapter is needed
    if isinstance(model, PeftModel):
        pytorch_model_path = os.path.join(
            training_args.output_dir, "pytorch_model_fsdp.bin")
        os.remove(pytorch_model_path) if os.path.exists(
            pytorch_model_path) else None


if __name__ == "__main__":
    main()
