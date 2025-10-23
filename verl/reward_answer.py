import re
from verl.model_judge import model_based_verify

score_level = [0.5, 0.707, 0.866,  0.95]

def get_index_list(item_str: str):
    return item_str.strip('<|>').split('|><|')

def compute_format(solution_str, type='reason'):
    if type == 'reason':
        think_match = re.search(r'<think>.*?</think>', solution_str, re.DOTALL)
        answer_match = re.search(r'<answer>(<\|\w_\d+\|>)+</answer>', solution_str, re.DOTALL)
        if think_match and answer_match:
            return 1.0
        return 0.0
    elif type == 'direct':
        answer_match = re.search(r'(<\|\w_\d+\|>){4,5}', solution_str, re.DOTALL)
        if answer_match:
            return 1.0
        return 0.0

def compute_acc(solution_str, ground_truth, type='reason'):
    if type == 'reason':
        match = re.search(r'<answer>((<\|\w_\d+\|>)+)</answer>', solution_str, re.DOTALL)
    elif type == 'direct':
        match = re.search(r'((<\|\w_\d+\|>){4,5})', solution_str, re.DOTALL)
    if match:
        answer = match.group(1).strip()
        if answer == ground_truth:
            return 1.0
        sol_index, gt_index = get_index_list(answer), get_index_list(ground_truth)
        score = 0.0
        for i in range(min(len(sol_index), len(gt_index))):
            if sol_index[i] == gt_index[i]:
                score = score_level[min(i, 3)]
            else:
                break  
        return score
    return 0.0

def compute_score(data_source, solution_str, ground_truth, extra_info):
    solution_str = solution_str.replace("<|endoftext|>","").replace("<|im_end|>","")
    type = extra_info.get("type", "reason")
    format = compute_format(solution_str, type=type)
    match = compute_acc(solution_str, ground_truth, type=type)
    score = match if format == 1.0 else 0.0
    return {
        "format": format,
        "acc": 1.0 if match == 1.0 else 0.0,
        "score": score,
        "match": match,
    }
