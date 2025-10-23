import json
import argparse
import random
from datasets import Dataset
from prompt import seqrec_prompt, SFT_SYSTEM_PROMPT

def load_json_file(file_path):
    if file_path.endswith('.jsonl'):
        data = []
        with open(file_path, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        return data
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def write_json_file(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)
        
def main():
    parser = argparse.ArgumentParser(description='Transform data for RL training')
    parser.add_argument('--root', type=str, default='data', help='Root directory for data')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    args = parser.parse_args()
    
    root = args.root
    dataset = args.dataset

    inters = load_json_file(f'{root}/{dataset}/{dataset}.inter.json')
    items = load_json_file(f'{root}/{dataset}/{dataset}.item.json')
    reviews = load_json_file(f'{root}/{dataset}/{dataset}.review.json')
    indexs = load_json_file(f'{root}/{dataset}/{dataset}.index.json')

    random.seed(42)

    rl = []
    val = []

    for user, item_ids in inters.items():
        for i in range(4, len(item_ids)-1):
            one_data = dict()
            target = item_ids[i]
            one_data["reward_model"] = {"style": "rule", "ground_truth": ''.join(indexs[str(target)])}
            history = item_ids[:i]
            history = history[-20:]
            one_data["data_source"] = f"{dataset}"
            one_data["prompt"] = [
                {"role": "system", "content": SFT_SYSTEM_PROMPT},
                {"role": "user", "content": random.choices(seqrec_prompt)[0].format(inters=', '.join([''.join(indexs[str(item)]) for item in history]))}
            ]
            one_data["ability"] = "rec"
            one_data["extra_info"] = {"title": items[str(target)]["title"], "type": "reason"}
            if i == len(item_ids) - 2:
                val.append(one_data)
            else:
                rl.append(one_data)
    
    random.seed(42)
    random_indices = random.sample(range(len(val)), len(val))

    Dataset.from_list(rl).to_parquet(f'{root}/{dataset}/{dataset}-rl.parquet')
    Dataset.from_list([val[i] for i in random_indices[:1000]]).to_parquet(f'{root}/{dataset}/{dataset}-val.parquet')

if __name__ == '__main__':
    main()