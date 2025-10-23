import json
import random
import re
from pathlib import Path
import numpy as np
import argparse
from datasets import load_dataset, Dataset
from prompt import seqrec_prompt, SFT_SYSTEM_PROMPT

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Process recommender system dataset and generate training data')
    
    # Dataset related arguments
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name')
    parser.add_argument('--root', type=str, default='data',
                        help='Root directory path for data')
    parser.add_argument('--output_file', type=str, default=None,
                        help='Full path for output file')
    
    # Result file related arguments
    parser.add_argument('--result_files', type=str, nargs='*', default=None,
                        help='Specify result file list to load (without extension), e.g.: result_gemini result_gpt5. '
                             'If not specified, will automatically scan all result_*.jsonl files under root/{dataset}')
    parser.add_argument('--align_data', type=str, default=None,
                        help='Full path for alignment data file (default: {dataset}.parquet in same directory as output_file)')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size (default: 8)')
    
    return parser.parse_args()

def scan_result_files(data_dir):
    """Automatically scan all result_*.jsonl files under data_dir"""
    result_files = []
    pattern = "result_*.jsonl"
    
    for file_path in Path(data_dir).glob(pattern):
        file_name = file_path.stem
        result_files.append(file_name)
    
    result_files.sort()
    return result_files

def load_json_file(file_path):
    """Load JSON or JSONL file"""
    file_path = Path(file_path)
    if file_path.suffix == '.jsonl':
        data = []
        with open(file_path, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        return data
    with open(file_path, 'r') as f:
        return json.load(f)

def load_dataset_files(dataset, data_dir, result_files):
    """Load all dataset files"""
    datasets = {
        'inters': load_json_file(data_dir / f'{dataset}.inter.json'),
        'items': load_json_file(data_dir / f'{dataset}.item.json'),
        'reviews': load_json_file(data_dir / f'{dataset}.review.json'),
        'indexs': load_json_file(data_dir / f'{dataset}.index.json')
    }
    
    # Dynamically load all result files
    results = []
    print(f"\nResult files to load: {result_files}")
    
    for result_file in result_files:
        file_path = data_dir / f'{result_file}.jsonl'
        if file_path.exists():
            loaded_data = load_json_file(file_path)
            results.extend(loaded_data)
            print(f"✓ Loaded: {file_path} (samples: {len(loaded_data)})")
        else:
            print(f"✗ Warning: File does not exist {file_path}")
    
    print(f"\nTotal loaded result samples: {len(results)}")
    datasets['results'] = results
    return datasets

def delete_step4(response):
    """Delete content after Step 4"""
    match = re.search(r'^(.*)\..*?Step 4', response, re.DOTALL)
    if match:
        return match.group(1) + '.'
    return None

def process_results_to_conversations(results, inters, indexs):
    """Process result data into conversation format"""
    conversations_data = []
    loss_count = 0
    
    for row in results:
        start_index = row["history_start_index"]
        end_index = row["history_end_index"]
        his = inters[row['user_id']][start_index:end_index+2]
        target = his[-1]
        his = his[:-1]
        
        response = delete_step4(row['assistant_response'])
        if response is None:
            loss_count += 1
            continue
        
        # Generate prompt
        history_items = ', '.join([''.join(indexs[str(item)]) for item in his])
        prompt = random.choice(seqrec_prompt).format(inters=history_items)
        
        # Generate reply
        response_text = (
            f"<think>\n{response}\n</think>\n\n"
            f"Let's rewrite user's purchase history: {history_items}.\n\n"
            f"Based on the above analysis, the recommended product is "
            f"<answer>{''.join(indexs[str(target)])}</answer>"
        )
        
        conversations_data.append({
            "conversations": [
                {"from": "human", "value": prompt},
                {"from": "gpt", "value": response_text}
            ],
            "system": SFT_SYSTEM_PROMPT
        })
    
    print(f"Processing finished, lost samples: {loss_count}")
    return conversations_data

def load_original_data(file_path):
    """Load original training data"""
    ori_data = load_dataset('parquet', data_files=str(file_path), split='train').to_list()
    for row in ori_data:
        row['system'] = "You are a helpful assistant."
    return ori_data

def shuffle_data_splits(ori_parts, reason_data, seed):
    """Shuffle splits of data"""
    rng = random.Random(seed)
    for part in ori_parts:
        rng.shuffle(part)
    rng.shuffle(reason_data)

def create_batched_shuffled_data(ori_part, reason_part, batch_size, seed=42, curriculum=False):
    """Create batches and mix data"""
    random.seed(seed)
    
    ori_copy = list(ori_part)
    reason_copy = list(reason_part)

    ori_batches = [ori_copy[i:i + batch_size] for i in range(0, len(ori_copy), batch_size)]
    reason_batches = [reason_copy[i:i + batch_size] for i in range(0, len(reason_copy), batch_size)]

    if curriculum:
        # Curriculum learning strategy
        all_batches = []
        num_ori_batches = len(ori_batches)
        
        for i, ori_batch in enumerate(ori_batches):
            all_batches.append(ori_batch)
            prob_insert = (i / num_ori_batches) * len(reason_batches) / num_ori_batches * 1.5
            if random.random() < prob_insert and reason_batches:
                all_batches.append(reason_batches.pop(0))
        
        if reason_batches:
            all_batches.extend(reason_batches)
    else:
        # Fully random mixing
        all_batches = ori_batches + reason_batches
        random.shuffle(all_batches)

    final_data = [item for batch in all_batches for item in batch]
    return final_data

def create_mixed_dataset(ori_data, reason_data, batch_size):
    """Create mixed dataset"""
    # Split into three parts
    ori_splits = np.array_split(ori_data, 3)
    ori_part1, ori_part2, ori_part3 = [list(arr) for arr in ori_splits]
    
    # Shuffle
    post_split_shuffle_seed = 20241006
    seeds = [42, 43, 44]
    shuffle_data_splits([ori_part1, ori_part2, ori_part3], reason_data, 
                        post_split_shuffle_seed)
    
    # Create three mixed data parts
    data_part1 = create_batched_shuffled_data(
        ori_part1, reason_data, batch_size, 
        seed=seeds[0], curriculum=True
    )
    data_part2 = create_batched_shuffled_data(
        ori_part2, reason_data, batch_size, 
        seed=seeds[1], curriculum=True
    )
    data_part3 = create_batched_shuffled_data(
        ori_part3, reason_data, batch_size, 
        seed=seeds[2], curriculum=False
    )
    
    # Print statistics
    print(f"Original 'ori' length: {len(ori_data)}")
    print(f"Original 'reason' length: {len(reason_data)}")
    print("-" * 40)
    print(f"Part 1 (curriculum): ori={len(ori_part1)}, mixed={len(data_part1)}")
    print(f"Part 2 (curriculum): ori={len(ori_part2)}, mixed={len(data_part2)}")
    print(f"Part 3 (random): ori={len(ori_part3)}, mixed={len(data_part3)}")
    
    return data_part1 + data_part2 + data_part3

def main():
    """Main function"""
    # Parse command line arguments
    args = parse_args()
    
    # Set paths
    data_dir = Path(f"{args.root}/{args.dataset}")
    
    if args.output_file:
        output_path = Path(args.output_file)
    else:
        output_path = data_dir / f'{args.dataset}-mix.parquet'
    
    # Scan or use specified result files
    if args.result_files:
        result_files = args.result_files
    else:
        result_files = scan_result_files(data_dir)
    
    # Alignment data path
    if args.align_data:
        align_data_path = Path(args.align_data)
    else:
        align_data_path = data_dir / f'{args.dataset}-align.parquet'
    
    # Print config info
    print("=" * 60)
    print("Config Info:")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Data directory: {data_dir}")
    print(f"Output file: {output_path}")
    print(f"Alignment data: {align_data_path}")
    print(f"Batch size: {args.batch_size}")
    print(f"Result files: {result_files}")
    
    # 1. Load all dataset files
    print("\n" + "=" * 60)
    print("Step 1: Load dataset files")
    print("=" * 60)
    datasets = load_dataset_files(args.dataset, data_dir, result_files)
    
    # 2. Process results to conversation format
    print("\n" + "=" * 60)
    print("Step 2: Process result data")
    print("=" * 60)
    reason_data = process_results_to_conversations(
        datasets['results'], 
        datasets['inters'], 
        datasets['indexs']
    )
    
    # 3. Load original data
    print("\n" + "=" * 60)
    print("Step 3: Load aligned training data")
    print("=" * 60)
    ori_data = load_original_data(align_data_path)
    print(f"Aligned data samples: {len(ori_data)}")
    
    # 4. Create mixed dataset
    print("\n" + "=" * 60)
    print("Step 4: Create mixed dataset")
    print("=" * 60)
    mixed_data = create_mixed_dataset(ori_data, reason_data, args.batch_size)
    
    # 5. Save results
    print("\n" + "=" * 60)
    print("Step 5: Save dataset")
    print("=" * 60)
    Dataset.from_list(mixed_data).to_parquet(output_path)
    print(f"✓ Dataset saved to: {output_path}")
    print(f"✓ Total samples: {len(mixed_data)}")
    print("=" * 60)

if __name__ == "__main__":
    main()