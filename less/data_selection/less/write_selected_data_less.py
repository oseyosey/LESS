import argparse
import os

import torch

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
    argparser.add_argument('--less_seed', type=int, default=3,
                           help='Seed number used in LESS data selection')

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

        score_paths = os.path.join(output_path, f"{target_task}_less_score_seed{args.less_seed}.pt") 
        
        less_scores = torch.load(score_paths, map_location=device)
        less_scores_tensor = torch.from_numpy(less_scores)

        total_samples = less_scores.shape[0]

        if args.percentage is not None:
            args.max_samples = int(args.percentage * total_samples)
            data_amount_name = f"p{args.percentage}"
            print(f"Selecting {args.max_samples} samples")
        else:
            data_amount_name = f"num{args.max_samples}"

        # sort the scores and output the corresponding data index
        topk_scores, topk_indices = torch.topk(less_scores_tensor, args.max_samples, largest=True)

        # Create a subset of lm_datasets based on the topk_indices
        selected_lm_datasets_dict = {key: [lm_datasets_dict[key][i] for i in topk_indices] for key in lm_datasets_dict.keys()}

        selected_lm_datasets = Dataset.from_dict(selected_lm_datasets_dict)

        # Save in JSON Lines format
        selected_lm_datasets.to_json(f"{output_path}/{target_task}-train-p{args.percentage}-less-seed{args.less_seed}.jsonl")
        

        # file_specific_index = torch.cat(
        #     [torch.arange(line_num) for line_num in num_samples]).to(device)
        # data_from = torch.cat([torch.ones(line_num, dtype=torch.long)
        #                       * i for i, line_num in enumerate(num_samples)]).to(device)
        # sorted_scores, sorted_index = torch.sort(
        #     all_scores, dim=0, descending=True)
        # sorted_score_file = os.path.join(output_path, f"sorted.csv")

        # data_from = data_from[sorted_index]
        # sorted_index = file_specific_index[sorted_index]
        

        # if not os.path.exists(sorted_score_file):
        #     with open(sorted_score_file, 'w', encoding='utf-8') as file:
        #         file.write("file name, index, score\n")
        #         for score, index, name in zip(sorted_scores, sorted_index, data_from):
        #             file.write(
        #                 f"{args.train_file_names[name.item()]}, {index.item()}, {round(score.item(), 6)}\n")

        # topk_scores, topk_indices = torch.topk(
        #     all_scores.float(), args.max_samples, dim=0, largest=True)

        # all_lines = []
        # for i, train_file in enumerate(args.train_files):
        #     with open(train_file, 'r', encoding='utf-8', errors='ignore') as file:
        #         all_lines.append(file.readlines()[:num_samples[i]])

        # final_index_list = sorted_index[:args.max_samples].tolist()
        # final_data_from = data_from[:args.max_samples].tolist()
        # with open(os.path.join(output_path, f"top_{data_amount_name}.jsonl"), 'w', encoding='utf-8', errors='ignore') as file:
        #     for index, data_from in zip(final_index_list, final_data_from):
        #         try:
        #             file.write(all_lines[data_from][index])
        #         except:
        #             import pdb
        #             pdb.set_trace()
