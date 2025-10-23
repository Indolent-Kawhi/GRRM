import argparse
import os
from typing import List
from utils.utils import set_seed, load_datasets, parse_dataset_args
from tqdm import tqdm
from datasets import Dataset

def main(args):
    set_seed(args.seed)
    
    data = []

    for _ in range(3):
        train_data, valid_data = load_datasets(args)
        for row in tqdm(train_data):
            data.append({
                "conversations": [
                    {"from": "human",
                    "value": row["input_ids"]},
                    {"from": "gpt",
                    "value": row["labels"]}
                ]
            })

    Dataset.from_list(data).to_parquet(os.path.join(args.data_path, args.dataset, f'{args.dataset}-align.parquet'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LLMRec')
    parser = parse_dataset_args(parser)
    args = parser.parse_args()

    main(args)
