import argparse
import os
import os.path as osp
import random
import time
import re
from logging import getLogger
import openai
import httpx
from utils import load_json, intention_prompt_1, intention_prompt_2, preference_prompt_1, preference_prompt_2, amazon18_dataset2fullname, write_json_file, preference_prompt_2_1, preference_prompt_2_2, item_feature_prompt
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import itertools
import collections

# Global variables
client_pool = {}
port_iterator = None
port_lock = threading.Lock()
use_api_key = False
api_client = None

def init_ports(ports):
    """Initialize port list and client pool"""
    global client_pool, port_iterator, use_api_key
    use_api_key = False
    port_iterator = itertools.cycle(ports)
    
    # Create a client instance for each port
    for port in ports:
        client_pool[port] = openai.OpenAI(
            base_url=f"http://localhost:{port}/v1",
            api_key="EMPTY",
            http_client=httpx.Client(
                # limits=httpx.Limits(max_connections=16, max_keepalive_connections=8),
                timeout=30.0
            )
        )

def init_api_key(api_key, base_url=None, model_name=None):
    """Initialize API Key authentication"""
    global api_client, use_api_key
    use_api_key = True
    
    client_kwargs = {
        "api_key": api_key,
        "http_client": httpx.Client(timeout=32.0)
    }
    
    if base_url:
        client_kwargs["base_url"] = base_url
    
    api_client = openai.OpenAI(**client_kwargs)

def get_openai_client():
    """Get OpenAI client - supports API Key and port modes"""
    global port_iterator, port_lock, client_pool, use_api_key, api_client
    
    if use_api_key:
        return api_client, "api_key"
    else:
        with port_lock:
            port = next(port_iterator)
        return client_pool[port], port

def cleanup_clients():
    """Clean up client connections"""
    global client_pool, api_client, use_api_key
    
    if use_api_key and api_client:
        if hasattr(api_client._client, 'close'):
            api_client._client.close()
    else:
        for client in client_pool.values():
            if hasattr(client._client, 'close'):
                client._client.close()

def extract(response):
    try:
        idx = response.index(':')
        return response[idx+1:]
    except ValueError:
        print(f"Error extracting response: {response}")
        return ""

def get_res_single(prompt, max_tokens, model_name, max_retries=5, base_delay=1.0):
    """Single request with retry and exponential backoff"""
    client, identifier = get_openai_client()
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                reasoning_effort='low',
                # temperature=0.7,
                # top_p=0.8,
            )
            result = response.choices[0].message.content.strip()
            # print(f"Prompt:\n{prompt}\nResponse:\n{result}\n\n" + "="*50)
            return result
        except Exception as e:
            print(f"Error with {identifier} (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                delay = min(delay, 32)  
                time.sleep(delay)
            else:
                return ""

def get_res_batch(prompt_list, max_tokens, model_name, max_workers=None):
    """Batch request processing"""
    global use_api_key, client_pool
    
    if max_workers is None:
        if use_api_key:
            max_workers = min(len(prompt_list), 10)  # API concurrency limit
        else:
            max_workers = min(len(prompt_list), len(client_pool) * 2)
    
    results = [None] * len(prompt_list)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(get_res_single, prompt, max_tokens, model_name): i 
            for i, prompt in enumerate(prompt_list)
        }
        
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                results[index] = future.result()
            except Exception as e:
                print(f"Error processing prompt {index}: {e}")
                results[index] = ""
    
    return results

def get_intention_train(args, inters, item2feature, reviews):
    intention_train_output_file = os.path.join(args.root,"intention_train.json")

    # Suggest modifying the prompt based on different datasets
    prompts=[]
    prompts.append(intention_prompt_1)
    prompts.append(intention_prompt_2)
    dataset_full_name = amazon18_dataset2fullname[args.dataset]
    dataset_full_name = dataset_full_name.replace("_", " ").lower()

    prompt_list = [[],[]]
    inter_data = []

    for (user,item_list) in inters.items():
        user = int(user)
        item = int(item_list[-3])
        history = item_list[:-3]

        inter_data.append((user,item,history))

        review = reviews[str((user, item))]["review"]
        item_title = item2feature[str(item)]["title"]
        for i in range(2):
            prompt_list[i].append(prompts[i].format(item_title=item_title,dataset_full_name=dataset_full_name,review=review))
    
    st = 0
    with open(intention_train_output_file, mode='a') as f:
        with tqdm(total=len(prompt_list[0])) as pbar:
            while st < len(prompt_list[0]):

                res_1 = get_res_batch(prompt_list[0][st:st+args.batchsize], args.max_tokens, args.model_name, args.max_workers)
                res_2 = get_res_batch(prompt_list[1][st:st+args.batchsize], args.max_tokens, args.model_name, args.max_workers)

                for i, answer in enumerate(zip(res_1,res_2)):
                    user, item, history = inter_data[st+i]

                    if len(answer) == 1:
                        print(answer)
                        user_preference = item_character = answer[0]
                    elif len(answer) >= 3:
                        print(answer)
                        answer = answer[-1]
                        user_preference = item_character = answer
                    else:
                        user_preference, item_character = answer

                    user_preference = extract(user_preference)
                    user_preference = user_preference.strip().replace('\n','')

                    item_character = extract(item_character)
                    item_character = item_character.strip().replace('\n','')

                    dict = {"user":user, "item":item, "inters": history,
                            "user_related_intention":user_preference, "item_related_intention": item_character}

                    json.dump(dict, f)
                    f.write("\n")
                pbar.update(args.batchsize)
                st += args.batchsize

    return intention_train_output_file

def get_intention_test(args, inters, item2feature, reviews):
    intention_test_output_file = os.path.join(args.root,"intention_test.json")

    # Suggest modifying the prompt based on different datasets
    prompts=[]
    prompts.append(intention_prompt_1)
    prompts.append(intention_prompt_2)
    dataset_full_name = amazon18_dataset2fullname[args.dataset]
    dataset_full_name = dataset_full_name.replace("_", " ").lower()

    prompt_list = [[],[]]
    inter_data = []

    for (user,item_list) in inters.items():
        user = int(user)
        item = int(item_list[-1])
        history = item_list[:-1]

        inter_data.append((user,item,history))

        review = reviews[str((user, item))]["review"]
        item_title = item2feature[str(item)]["title"]
        for i in range(2):
            prompt_list[i].append(prompts[i].format(item_title=item_title,dataset_full_name=dataset_full_name,review=review))
    
    st = 0
    with open(intention_test_output_file, mode='a') as f:
        with tqdm(total=len(prompt_list[0])) as pbar:
            while st < len(prompt_list[0]):

                res_1 = get_res_batch(prompt_list[0][st:st+args.batchsize], args.max_tokens, args.model_name, args.max_workers)
                res_2 = get_res_batch(prompt_list[1][st:st+args.batchsize], args.max_tokens, args.model_name, args.max_workers)

                for i, answer in enumerate(zip(res_1,res_2)):
                    user, item, history = inter_data[st+i]

                    if len(answer) == 1:
                        print(answer)
                        user_preference = item_character = answer[0]
                    elif len(answer) >= 3:
                        print(answer)
                        answer = answer[-1]
                        user_preference = item_character = answer
                    else:
                        user_preference, item_character = answer

                    user_preference = extract(user_preference)
                    user_preference = user_preference.strip().replace('\n','')

                    item_character = extract(item_character)
                    item_character = item_character.strip().replace('\n','')

                    dict = {"user":user, "item":item, "inters": history,
                            "user_related_intention":user_preference, "item_related_intention": item_character}

                    json.dump(dict, f)
                    f.write("\n")
                pbar.update(args.batchsize)
                st += args.batchsize

    return intention_test_output_file

def get_user_preference(args, inters, item2feature, reviews):
    preference_output_file = os.path.join(args.root,"user_preference.jsonl")

    # Suggest modifying the prompt based on different datasets
    prompt_1 = preference_prompt_1
    prompt_2 = preference_prompt_2_1
    prompt_3 = preference_prompt_2_2

    dataset_full_name = amazon18_dataset2fullname[args.dataset]
    dataset_full_name = dataset_full_name.replace("_", " ").lower()
    print(dataset_full_name)

    prompt_list_1 = []
    prompt_list_2 = []
    prompt_list_3 = []

    users = []

    for (user,item_list) in inters.items():
        users.append(user)
        history = item_list[:-3]
        item_titles = []
        for j, item in enumerate(history):
            item_titles.append(str(j+1) + '.' + item2feature[str(item)]["title"])
        if len(item_titles) > args.max_his_len:
            item_titles = item_titles[-args.max_his_len:]
        item_titles = ", ".join(item_titles)
        
        input_prompt_1 = prompt_1.format(dataset_full_name=dataset_full_name, item_titles=item_titles)
        input_prompt_2 = prompt_2.format(dataset_full_name=dataset_full_name, item_titles=item_titles)
        input_prompt_3 = prompt_3.format(dataset_full_name=dataset_full_name, item_titles=item_titles)

        prompt_list_1.append(input_prompt_1)
        prompt_list_2.append(input_prompt_2)
        prompt_list_3.append(input_prompt_3)

    st = 0
    with open(preference_output_file, mode='a') as f:
        with tqdm(total=len(prompt_list_1)) as pbar:
            while st < len(prompt_list_1):

                res_1 = get_res_batch(prompt_list_1[st:st + args.batchsize], args.max_tokens, args.model_name, args.max_workers)
                res_2 = get_res_batch(prompt_list_2[st:st + args.batchsize], args.max_tokens, args.model_name, args.max_workers)
                res_3 = get_res_batch(prompt_list_3[st:st + args.batchsize], args.max_tokens, args.model_name, args.max_workers)
                for i, answers in enumerate(zip(res_1, res_2, res_3)):
                    
                    user = users[st + i]

                    answer_1, answer_2, answer_3 = answers

                    if answer_1 == '':
                        print("answer null error")
                        answer_1 = "I enjoy high-quality item."
                        
                    if answer_2 == '':
                        print("answer null error")
                        answer_2 = "I enjoy high-quality item."

                    if answer_3 == '':
                        print("answer null error")
                        answer_3 = "I enjoy high-quality item."

                    short_preference=answer_3
                    long_preference=answer_2

                    long_preference = extract(long_preference)
                    long_preference = long_preference.strip().replace('\n','')
                    
                    short_preference = extract(short_preference)
                    short_preference = short_preference.strip().replace('\n','')

                    dict = {"user":user,"user_preference":[answer_1, long_preference, short_preference]}
                    json.dump(dict, f)
                    f.write("\n")
                pbar.update(args.batchsize)
                st += args.batchsize

    return preference_output_file

def get_item_features(args, inters, item2feature, reviews):
    """Generate objective item features based on item description and sampled user reviews."""
    feature_output_file = os.path.join(args.root, f"item_features.jsonl")
    
    # 1. Build item -> user_ids mapping from training set (only use [:-2] part)
    item_to_train_users = collections.defaultdict(list)
    for user_id, item_list in inters.items():
        # Only use training set part ([:-2]), avoid test data leakage
        train_items = item_list[:-2]
        for item_id in train_items:
            item_to_train_users[str(item_id)].append(user_id)
    
    # 2. Build item -> reviews mapping (only use reviews from training set users)
    item_to_reviews = collections.defaultdict(list)
    for key, value in reviews.items():
        try:
            # key is like "('user_id', 'item_id')"
            user_id, item_id = eval(key)
            user_id = str(user_id)
            item_id = str(item_id)
            
            # Only use this review if the user bought the item in training set
            if user_id in item_to_train_users[item_id]:
                item_to_reviews[item_id].append(value['review'])
        except:
            continue

    # 3. Collect items to process (all items in training set with reviews)
    items_to_process = [item for item in item_to_reviews.keys() if item_to_reviews[item]]
    
    prompt_list = []
    item_ids = []
    
    # 4. Prepare prompts
    for item_id in items_to_process:
        if item_id not in item2feature:
            continue
        
        item_info = item2feature[item_id]
        item_title = item_info.get("title", "N/A")
        item_description = item_info["description"]
        if item_description == "." or item_description == "":
            item_description = "No description available."
        
        # Sample up to 8 reviews (only from training set)
        available_reviews = item_to_reviews[item_id]
        if not available_reviews:
            # If no training reviews, skip or use empty review
            user_reviews_text = "No user reviews available."
        else:
            available_reviews = [review for review in available_reviews if len(review) >= 20]
            sampled_reviews = random.sample(available_reviews, min(len(available_reviews), 8))
            user_reviews_text = "\n".join([f"User {i+1}: {r}" for i, r in enumerate(sampled_reviews)])
        
        prompt = item_feature_prompt.format(
            item_title=item_title,
            item_description=item_description,
            user_reviews=user_reviews_text
        )
        prompt_list.append(prompt)
        item_ids.append(item_id)

    print(f"Example prompt for item {item_id}:")
    print(prompt)

    # 5. Batch call LLM to generate features and write to jsonl in real time
    st = 0
    item_features = {}
    with open(feature_output_file, "w") as fout, tqdm(total=len(prompt_list), desc="Generating Item Features") as pbar:
        while st < len(prompt_list):
            batch_prompts = prompt_list[st:st + args.batchsize]
            batch_item_ids = item_ids[st:st + args.batchsize]
            
            results = get_res_batch(batch_prompts, args.max_tokens, args.model_name, args.max_workers)
            
            for item_id, feature_text in zip(batch_item_ids, results):
                if feature_text:
                    # feature = extract(feature_text).strip()
                    feature = feature_text.strip()
                    item_features[item_id] = feature
                    # Write to jsonl in real time
                    json.dump({"item_id": item_id, "feature": feature}, fout, ensure_ascii=False)
                    fout.write("\n")
            st += len(batch_prompts)
            pbar.update(len(batch_prompts))
            
    return feature_output_file

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Instruments', help='Instruments / Beauty / Sports')
    parser.add_argument('--root', type=str, default='data')
    
    # Local port config
    parser.add_argument('--ports', type=str, default=None, help='Comma-separated list of ports')
    
    # API Key config
    parser.add_argument('--api_key', type=str, default=None, help='OpenAI API Key for external API usage')
    parser.add_argument('--base_url', type=str, default=None, help='Base URL for API (optional, for custom endpoints)')
    parser.add_argument('--model_name', type=str, default='gpt-5', help='Model name to use')
    
    parser.add_argument('--max_tokens', type=int, default=512)
    parser.add_argument('--batchsize', type=int, default=128)
    parser.add_argument('--max_his_len', type=int, default=20)
    parser.add_argument('--max_workers', type=int, default=32, help='Max workers for threading')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Choose initialization method based on config
    if args.ports:
        print("Using local port authentication")
        # Parse port list and initialize load balancing
        ports = [int(p.strip()) for p in args.ports.split(',')]
        init_ports(ports)
        if args.max_workers is None:
            args.max_workers = len(ports) * 2
    else:
        print(f"Using API Key authentication with model: {args.model_name}")
        init_api_key(args.api_key, args.base_url, args.model_name)
        if args.max_workers is None:
            args.max_workers = 10  # Reduce concurrency for API mode

    args.root = os.path.join(args.root, args.dataset)

    inter_path = os.path.join(args.root, f'{args.dataset}.inter.json')
    inters = load_json(inter_path)

    item2feature_path = os.path.join(args.root, f'{args.dataset}.item.json')
    item2feature = load_json(item2feature_path)

    reviews_path = os.path.join(args.root, f'{args.dataset}.review.json')
    reviews = load_json(reviews_path)

    try:
        preference_output_file = get_user_preference(args, inters, item2feature, reviews)
        intention_train_output_file = get_intention_train(args, inters, item2feature, reviews)
        intention_test_output_file = get_intention_test(args, inters, item2feature, reviews)
        feature_output_file = get_item_features(args, inters, item2feature, reviews)

        intention_train = {}
        user_preference = {}

        with open(intention_train_output_file, "r") as f:
            for line in f:
                content = json.loads(line)
                if content["user"] not in intention_train:
                    intention_train[content["user"]] = {"item":content["item"],
                                                    "inters":content["inters"],
                                                    "querys":[ content["user_related_intention"], content["item_related_intention"] ]}

        with open(preference_output_file, "r") as f:
            x = 0
            for line in f:
                content = json.loads(line)
                user_preference[content["user"]] = content["user_preference"]

        user_dict = {
            "user_explicit_preference": user_preference,
            "user_vague_intention": 
                {
                    "train": intention_train, 
                },
        }

        write_json_file(user_dict, os.path.join(args.root, f'{args.dataset}.user.json'))
            
        item_features = {}
        with open(feature_output_file, "r") as f:
            for line in f:
                content = json.loads(line)
                item_features[content["item_id"]] = content["feature"]
                
        os.rename(item2feature_path, os.path.join(args.root, f'{args.dataset}.item_old.json'))
        
        for item in item_features:
            if item in item2feature:
                item2feature[item]["description"] = item_features[item]
                
        write_json_file(item2feature, os.path.join(args.root, f'{args.dataset}.item.json'))
    finally:
        # Ensure resource cleanup
        cleanup_clients()
