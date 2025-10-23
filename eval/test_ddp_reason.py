import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoTokenizer

# Import the OpenAI client library
import openai

from utils.utils import set_seed, parse_dataset_args, parse_test_args, load_test_dataset
from evaluate import get_topk_results, get_metrics_results
import re
from data_process.prompt import SFT_SYSTEM_PROMPT


def construct_prompt(tokenizer, user_prompt):
    messages = [
        {"role": "system", "content": SFT_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + "<think>\n"

def parse_vllm_api_args(parser):
    """Adds arguments for connecting to the VLLM API server."""
    group = parser.add_argument_group("VLLM API Arguments")
    group.add_argument('--vllm_host', type=str, default='localhost', help='Hostname of the VLLM API server.')
    group.add_argument('--vllm_base_port', type=int, default=10010, help='Base port for VLLM API servers. Rank `i` will connect to `base_port + i`.')
    group.add_argument('--vllm_api_key', type=str, default='EMPTY', help='API key for the VLLM server (usually not needed for local instances).')
    group.add_argument('--vllm_model_name', type=str, default='qwen3', help='The model name identifier used by the VLLM server.')
    return parser

def test_vllm_api_ddp(args):
    """
    Test function refactored to use VLLM API endpoints for generation.
    Each DDP rank acts as a client to a dedicated VLLM server instance.
    """
    set_seed(args.seed)
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK") or 0)
    torch.cuda.set_device(local_rank) # Still useful for tensors on this rank
    if local_rank == 0:
        print(vars(args))

    dist.init_process_group(backend="nccl", world_size=world_size, rank=local_rank)
    device = torch.device("cuda", local_rank)

    # 1. Configure the API client for this rank
    api_url = f"http://{args.vllm_host}:{args.vllm_base_port + local_rank}/v1"
    if local_rank == 0:
        print(f"World Size: {world_size}. Rank 0 connecting to {api_url}")
    
    # Each rank gets its own client pointing to its dedicated server
    client = openai.OpenAI(
        api_key=args.vllm_api_key,
        base_url=api_url,
    )
    # VLLM requires a model name argument, even if it's hosting only one.
    # We get this from the server startup log or set it via --served-model-name.
    # For simplicity, we pass it as a script argument.
    vllm_model_name = args.ckpt_path if args.vllm_model_name == 'default-model' else args.vllm_model_name


    # 2. Load Tokenizer and distribute data (same as before)
    tokenizer = AutoTokenizer.from_pretrained(args.ckpt_path, padding_side="left")
    if tokenizer.chat_template is None:
        tokenizer.chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{'<|im_start|>assistant\n'}}{% endif %}"

    test_data = load_test_dataset(args)
    ddp_sampler = DistributedSampler(test_data, num_replicas=world_size, rank=local_rank, drop_last=True, seed=42)
    rank_indices = list(iter(ddp_sampler))
    rank_test_data = [test_data[i] for i in rank_indices]
    all_items = test_data.get_all_items()
    
    if local_rank == 0:
        print(f"Total data num: {len(test_data)}, Data per rank: {len(rank_test_data)}")

    metrics = args.metrics.split(",")
    
    # RESUME LOGIC: Initialize variables for tracking progress and state
    start_idx = 0
    metrics_results = {m: 0 for m in metrics}
    total = 0
    
    if local_rank == 0:
        output_file = args.results_file
        
        # Create parent directory if it doesn't exist
        output_dir = Path(output_file).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Output will be saved to: {output_file}")
        
        # RESUME LOGIC: Check for existing output file to resume from
        if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
            print("Found existing output file. Attempting to resume...")
            with open(output_file, 'r', encoding='utf-8') as f:
                processed_lines = f.readlines()
            
            if processed_lines:
                processed_items = [json.loads(line) for line in processed_lines]
                
                # Restore total count
                total = len(processed_items)
                
                # Restore metrics by recalculating from the stored topk_res
                all_topk_res = [res for item in processed_items for res in item['topk_res']]
                if total > 0:
                    restored_metrics = get_metrics_results(all_topk_res, metrics)
                    for m, res in restored_metrics.items():
                        # The original logic sums the scores, so we multiply the restored average by the total count
                        metrics_results[m] = res * total
                
                # Determine the starting point for the loop
                last_step = max(item['step'] for item in processed_items)
                start_idx = (last_step + 1) * args.test_batch_size
                
                print(f"Resumed state: Processed {total} items up to step {last_step}.")
                print(f"Starting process from data index {start_idx}.")
                temp = {m: metrics_results[m] / total for m in metrics_results} if total > 0 else {}
                print(f"Restored metrics: {temp}")
        else:
            print("No existing file found or file is empty. Starting from scratch.")
            # Only clear the file when starting fresh
            open(output_file, 'w').close()

    # RESUME LOGIC: Broadcast the starting index from rank 0 to all other ranks
    start_tensor = torch.tensor([start_idx], dtype=torch.long, device=device)
    dist.broadcast(start_tensor, src=0)
    start_idx = start_tensor.item()


    # 4. Main Inference Loop using API calls
    batch_size = args.test_batch_size
    # RESUME LOGIC: Adjust the loop range and tqdm progress bar to start from the correct index
    for i in tqdm(range(start_idx, len(rank_test_data), batch_size), 
                  disable=(local_rank!=0), 
                  initial=start_idx // batch_size, 
                  total=len(rank_test_data) // batch_size):
        dist.barrier()
        
        batch = rank_test_data[i:i + batch_size]
        if not batch:
            continue
        
        prompts = [construct_prompt(tokenizer, item['input_ids']) for item in batch for i in range(args.num_beams)]
        targets = [item['labels'] for item in batch]
        bs = len(targets)
        
        # Generate responses using the VLLM API server
        try:
            response = client.completions.create(
                model=vllm_model_name,
                prompt=prompts,
                n=1,  # Number of completions per prompt
                temperature=0.7,
                top_p=0.9,
                max_tokens=1560,
                logprobs=1, # Request logprobs for scoring
                extra_body={
                    "skip_special_tokens": False,
                    "stop_token_ids": [tokenizer.eos_token_id],
                }
            )
        except openai.APIConnectionError as e:
            print(f"Rank {local_rank} could not connect to {api_url}. Please ensure the VLLM server is running. Error: {e}")
            # Optionally, break or implement retry logic
            sys.exit(1)
        
        # 5. Process API response
        predictions = []
        scores = []
        raw_outputs = []
        
        # The response.choices list contains (len(prompts) * n) items.
        # They are ordered by prompt, e.g., [p0_n0, p0_n1, p1_n0, p1_n1, ...]
        for prompt_idx, prompt_text in enumerate(prompts):
            start_choice_idx = prompt_idx * args.num_beams
            end_choice_idx = start_choice_idx + args.num_beams
            prompt_choices = response.choices[start_choice_idx:end_choice_idx]

            for choice in prompt_choices:
                generated_text = choice.text
                full_text = prompt_text + generated_text
                raw_outputs.append(full_text)

                # match = re.search(r"</think>\s*((<\|\w_\d+\|>)+)", generated_text, re.DOTALL)
                match = re.search(r"<answer>((<\|\w_\d+\|>)+)", generated_text, re.DOTALL)
                if match:
                    predictions.append(match.group(1).strip())
                else:
                    predictions.append("")
                    print(f"Rank {local_rank}: No match found in output: {generated_text}")
                
                # Calculate sequence score from token_logprobs
                seq_logprob = sum(choice.logprobs.token_logprobs)
                scores.append(seq_logprob)

        scores = torch.tensor(scores, device=device, dtype=torch.float)
        
        topk_res = get_topk_results(predictions, scores, targets, args.num_beams,
                                    all_items=all_items if args.filter_items else None)

        # 6. Gather results and evaluate (same as before)
        batch_data = []
        for j in range(bs):
            batch_predictions = predictions[j * args.num_beams:(j + 1) * args.num_beams]
            batch_raw_outputs = raw_outputs[j * args.num_beams:(j + 1) * args.num_beams]
            sample_topk_res = [topk_res[j]]
            batch_data.append({
                "step": i // batch_size,
                "topk_res": sample_topk_res,
                "target": targets[j],
                "predictions": batch_predictions,
                "raw_outputs": batch_raw_outputs,
            })
        
        # ... (The rest of the distributed gathering and metric calculation is identical)
        batch_data_gather_list = [None for _ in range(world_size)]
        dist.all_gather_object(obj=batch_data, object_list=batch_data_gather_list)
        
        if local_rank == 0:
            with open(output_file, 'a', encoding='utf-8') as f:
                for device_batch_data in batch_data_gather_list:
                    for item in device_batch_data:
                        f.write(json.dumps(item, ensure_ascii=False) + '\n')

        bs_gather_list = [None for _ in range(world_size)]
        dist.all_gather_object(obj=bs, object_list=bs_gather_list)
        total += sum(bs_gather_list)
        res_gather_list = [None for _ in range(world_size)]
        dist.all_gather_object(obj=topk_res, object_list=res_gather_list)

        if local_rank == 0:
            all_device_topk_res = [item for sublist in res_gather_list for item in sublist]
            batch_metrics_res = get_metrics_results(all_device_topk_res, metrics)
            for m, res in batch_metrics_res.items():
                metrics_results[m] = metrics_results.get(m, 0) + res * sum(bs_gather_list)

            if (i // batch_size + 1) % 10 == 0:
                temp = {m: metrics_results[m] / total for m in metrics_results}
                print(f"Step {i // batch_size + 1}: {temp}")

    dist.barrier()
    
    if local_rank == 0:
        # Final calculation remains the same
        if total > 0:
            for m in metrics_results:
                metrics_results[m] /= total
        else:
            print("Warning: Total processed items is zero. No final metrics to calculate.")
        print("======================================================")
        print("Final Results: ", metrics_results)
        print("======================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLMRec_test_vllm_api")
    parser = parse_dataset_args(parser)
    parser = parse_test_args(parser)
    # Add the new arguments for connecting to the API server
    parser = parse_vllm_api_args(parser)

    args = parser.parse_args()

    test_vllm_api_ddp(args)