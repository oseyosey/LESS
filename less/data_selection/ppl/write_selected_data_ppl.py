import argparse
import os

import torch
import random

from less.data_selection.get_training_dataset import load_raw_dataset
from datasets import Dataset

def parse_args():
    argparser = argparse.ArgumentParser(
        description='Script for selecting the data for training')
    argparser.add_argument('--train_file_names', type=str, # this could be just a scoring from different data selection methods
                           nargs='+', help='The path to the score file')
    argparser.add_argument('--train_files', type=str, nargs='+',
                           help='The path of the training file that corresponds to the score file')
    argparser.add_argument('--target_task_names', type=str,
                           nargs='+', help='The name of the target task')
    argparser.add_argument('--output_path', type=str,
                           default="selected_data", help='The path to the output')
    argparser.add_argument('--max_samples', type=int,
                           default=None, help='The maximum number of samples')
    argparser.add_argument('--percentage', type=float, default=None,
                           help='The percentage of the data to be selected')
    argparser.add_argument('--epochs_num', type=int, 
                          help='The number of epochs for training the model')
    argparser.add_argument('--data_seed', type=int, default=42,
                           help='The data seed for data permutation for each perplexity region')

    args = argparser.parse_args()

    return args


def count_lines(filename):
    with open(filename, 'r', encoding='utf-8', errors='ignore') as file:
        line_count = 0
        for line in file:
            line_count += 1
    return line_count


if __name__ == "__main__":
    args = parse_args()
    assert len(args.train_file_names) == len(args.train_files)
    assert args.percentage is not None or args.max_samples is not None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_train_files = len(args.train_file_names)

    lm_datasets = load_raw_dataset(args.train_files, sample_percentage=1.0)
    lm_datasets_dict = lm_datasets.to_dict()
    
    # for key in lm_datasets_dict.keys():
    #     lm_datasets_dict[key].extend(val_datasets_dict[key])

    for target_task in args.target_task_names:
        output_path = os.path.join(args.output_path, target_task)

        score_paths = os.path.join(output_path, f"{target_task}_ppl_score-epoch{args.epochs_num}.pt") 
        
        ppl_scores = torch.load(score_paths, map_location=device)
        ppl_scores_tensor = torch.from_numpy(ppl_scores)

        total_samples = ppl_scores.shape[0]

        if args.percentage is not None:
            args.max_samples = int(args.percentage * total_samples)
            data_amount_name = f"p{args.percentage}"
            print(f"Selecting {args.max_samples} samples")
        else:
            data_amount_name = f"num{args.max_samples}"

        # sort the scores and output the corresponding data indexf from lowest ppl scores to highest ppl scores
        topk_scores, topk_indices = torch.topk(ppl_scores_tensor, args.max_samples, largest=False) #* largest=False, as we want to select the lowest PPL scores. 

        #* findings from "When Less is More: Investigating Data Pruning for Pretrained Language Models" paper
            # The middle percentile ppl (keep 30%) section performs the best
            #* makes no assumptions about the specific order, so we have to do some kind of random sampling. 

        num_bins = 5000 
        hist = torch.histc(ppl_scores_tensor, bins=num_bins)
        cumulative_hist = torch.cumsum(hist, dim=0)

        # Step 2: Determine the thresholds based on the cumulative distribution
        total_samples = cumulative_hist[-1].item()
        percentile_33_index = torch.searchsorted(cumulative_hist, total_samples * 0.33)
        percentile_66_index = torch.searchsorted(cumulative_hist, total_samples * 0.66)

        threshold_33 = torch.linspace(ppl_scores_tensor.min(), ppl_scores_tensor.max(), num_bins)[percentile_33_index]
        threshold_66 = torch.linspace(ppl_scores_tensor.min(), ppl_scores_tensor.max(), num_bins)[percentile_66_index]

        middle_third_indices = topk_indices[(topk_scores > threshold_33) & (topk_scores <= threshold_66)]
        first_third_indices = topk_indices[topk_scores <= threshold_33]
        bottom_third_indices = topk_indices[topk_scores > threshold_66]

        # Step 3: randomly shuffle the data between each perpelxity region.
        random.seed(args.data_seed)
        random.shuffle(middle_third_indices)
        random.shuffle(first_third_indices)
        random.shuffle(bottom_third_indices)

        # Step 4: Reorder the indices: middle third, first third, and bottom third
        reordered_indices = torch.cat([middle_third_indices, first_third_indices, bottom_third_indices])

        # Create a subset of lm_datasets based on the topk_indices
        selected_lm_datasets_dict = {key: [lm_datasets_dict[key][i] for i in reordered_indices] for key in lm_datasets_dict.keys()}

        selected_lm_datasets = Dataset.from_dict(selected_lm_datasets_dict)

        # Save in JSON Lines format
        selected_lm_datasets.to_json(f"{output_path}/{target_task}-train-p{args.percentage}-ppl-epoch{args.epochs_num}-mid-k-{args.data_seed}.jsonl")
        