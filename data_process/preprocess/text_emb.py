import argparse
import collections
import gzip
import html
import json
import os
import random
import re
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import numpy as np
from utils import clean_text, load_json, set_device, load_plm
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig, AutoTokenizer, AutoModel


def init_distributed():
    """Initialize distributed environment"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu_id = int(os.environ['LOCAL_RANK'])
    else:
        print('Not using distributed mode')
        return False, 0, 1, 0
    
    torch.cuda.set_device(gpu_id)
    dist.init_process_group(backend='nccl')
    return True, rank, world_size, gpu_id


def split_data_for_rank(item_text_list, rank, world_size):
    """Split data by rank"""
    total_items = len(item_text_list)
    items_per_rank = (total_items + world_size - 1) // world_size
    start_idx = rank * items_per_rank
    end_idx = min(start_idx + items_per_rank, total_items)
    
    return item_text_list[start_idx:end_idx], start_idx


def load_data(args):

    item2feature_path = os.path.join(args.root, f'{args.dataset}.item.json')
    item2feature = load_json(item2feature_path)

    return item2feature

def generate_text(item2feature, features):
    item_text_list = []

    for item in item2feature:
        data = item2feature[item]
        text = []
        for meta_key in features:
            if meta_key in data:
                meta_value = clean_text(data[meta_key])
                text.append(meta_value.strip())

        item_text_list.append([int(item), text])

    return item_text_list

def preprocess_text(args):
    print('Process text data: ')
    print(' Dataset: ', args.dataset)

    item2feature = load_data(args)
    # load item text and clean
    item_text_list = generate_text(item2feature, ['title', 'description'])
    # item_text_list = generate_text(item2feature, ['title'])
    # return: list of (item_ID, cleaned_item_text)
    return item_text_list

def generate_item_embedding(args, item_text_list, tokenizer, model, word_drop_ratio=-1, rank=0, world_size=1):
    print(f'Generate Text Embedding on GPU {rank}: ')
    print(' Dataset: ', args.dataset)

    # Split data
    local_item_text_list, start_idx = split_data_for_rank(item_text_list, rank, world_size)
    
    items, texts = zip(*local_item_text_list)
    
    # Create local order_texts to keep original index
    max_item_id = max([item for item, _ in item_text_list])
    local_order_texts = {}
    for item, text in zip(items, texts):
        local_order_texts[item] = text

    embeddings = []
    items_processed = []
    start, batch_size = 0, 8
    
    with torch.no_grad():
        # Process local data
        local_items = list(local_order_texts.keys())
        local_texts = [local_order_texts[item] for item in local_items]
        
        for start in tqdm(range(0, len(local_texts), batch_size), 
                         desc=f"Embedding on GPU {rank}", 
                         disable=(rank != 0)):  # Show progress bar only on main process
            field_texts = local_texts[start: start + batch_size]
            current_items = local_items[start: start + batch_size]
            
            field_texts = list(zip(*field_texts))
    
            field_embeddings = []
            for sentences in field_texts:
                sentences = list(sentences)
                if word_drop_ratio > 0:
                    if rank == 0:
                        print(f'Word drop with p={word_drop_ratio}')
                    new_sentences = []
                    for sent in sentences:
                        new_sent = []
                        sent = sent.split(' ')
                        for wd in sent:
                            rd = random.random()
                            if rd > word_drop_ratio:
                                new_sent.append(wd)
                        new_sent = ' '.join(new_sent)
                        new_sentences.append(new_sent)
                    sentences = new_sentences
                
                encoded_sentences = tokenizer(sentences, max_length=args.max_sent_len,
                                              truncation=True, return_tensors='pt',padding="longest").to(args.device)
                outputs = model(input_ids=encoded_sentences.input_ids,
                                attention_mask=encoded_sentences.attention_mask)
    
                masked_output = outputs.last_hidden_state * encoded_sentences['attention_mask'].unsqueeze(-1)
                mean_output = masked_output.sum(dim=1) / encoded_sentences['attention_mask'].sum(dim=-1, keepdim=True)
                mean_output = mean_output.detach().cpu()
                field_embeddings.append(mean_output)
    
            field_mean_embedding = torch.stack(field_embeddings, dim=0).mean(dim=0)
            embeddings.append(field_mean_embedding)
            items_processed.extend(current_items)

    if embeddings:
        local_embeddings = torch.cat(embeddings, dim=0).numpy()
    else:
        local_embeddings = np.array([]).reshape(0, model.config.hidden_size)
    
    if rank == 0:
        print(f'Local embeddings shape on GPU {rank}: ', local_embeddings.shape)

    return local_embeddings, items_processed


def collect_embeddings(local_embeddings, items_processed, total_items, rank, world_size, device):
    """Collect embedding results from all GPUs"""
    if world_size == 1:
        return local_embeddings
    
    # Convert numpy array to CUDA tensor for communication
    local_embeddings_tensor = torch.from_numpy(local_embeddings).to(device)
    items_tensor = torch.tensor(items_processed, device=device, dtype=torch.long)
    
    # Gather embedding dimension info
    embedding_dim = local_embeddings_tensor.shape[1] if local_embeddings_tensor.numel() > 0 else 0
    local_size = torch.tensor([local_embeddings_tensor.shape[0]], device=device)
    
    # Gather size info from all processes
    all_sizes = [torch.zeros(1, device=device, dtype=torch.long) for _ in range(world_size)]
    dist.all_gather(all_sizes, local_size)
    
    if rank == 0:
        # Get max size for padding
        max_size = max([size.item() for size in all_sizes])
        embedding_dim_tensor = torch.tensor([embedding_dim], device=device)
        
        # Broadcast embedding dimension
        dist.broadcast(embedding_dim_tensor, src=0)
        embedding_dim = embedding_dim_tensor.item()
    else:
        # Receive embedding dimension
        embedding_dim_tensor = torch.zeros(1, device=device, dtype=torch.long)
        dist.broadcast(embedding_dim_tensor, src=0)
        embedding_dim = embedding_dim_tensor.item()
        
        max_size = max([size.item() for size in all_sizes])
    
    # Pad local data to max size
    if local_embeddings_tensor.shape[0] < max_size:
        if embedding_dim > 0:
            padding = torch.zeros(max_size - local_embeddings_tensor.shape[0], embedding_dim, device=device)
            local_embeddings_tensor = torch.cat([local_embeddings_tensor, padding], dim=0)
        else:
            local_embeddings_tensor = torch.zeros(max_size, 768, device=device)  # default dimension
    
    if items_tensor.shape[0] < max_size:
        padding = torch.full((max_size - items_tensor.shape[0],), -1, device=device, dtype=torch.long)
        items_tensor = torch.cat([items_tensor, padding], dim=0)
    
    # Gather all data
    all_embeddings = [torch.zeros_like(local_embeddings_tensor) for _ in range(world_size)]
    all_items = [torch.zeros_like(items_tensor) for _ in range(world_size)]
    
    dist.all_gather(all_embeddings, local_embeddings_tensor)
    dist.all_gather(all_items, items_tensor)
    
    if rank == 0:
        # Reorganize embeddings by original item order
        final_embeddings = np.zeros((total_items, embedding_dim if embedding_dim > 0 else 768))
        
        for gpu_idx in range(world_size):
            gpu_embeddings = all_embeddings[gpu_idx].cpu().numpy()
            gpu_items = all_items[gpu_idx].cpu().numpy()
            valid_size = all_sizes[gpu_idx].item();
            
            for i in range(valid_size):
                item_id = gpu_items[i]
                if item_id >= 0:  # ignore padding -1
                    final_embeddings[item_id] = gpu_embeddings[i]
        
        return final_embeddings
    
    return None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Sports', help='Instruments / Arts / Games')
    parser.add_argument('--root', type=str, default="data")
    parser.add_argument('--gpu_id', type=int, default=2, help='ID of running GPU')
    parser.add_argument('--plm_checkpoint', type=str,
                        default='Qwen/Qwen3-4B-Instruct-2507')
    parser.add_argument('--max_sent_len', type=int, default=2048)
    parser.add_argument('--word_drop_ratio', type=float, default=-1, help='word drop ratio, do not drop by default')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # Initialize distributed environment
    is_distributed, rank, world_size, local_gpu_id = init_distributed()
    
    if is_distributed:
        args.device = torch.device(f'cuda:{local_gpu_id}')
        if rank == 0:
            print(f"Using distributed mode with {world_size} GPUs")
    else:
        device = set_device(args.gpu_id)
        args.device = device
        rank, world_size = 0, 1

    args.root = os.path.join(args.root, args.dataset)

    # Only main process handles data
    if rank == 0:
        item_text_list = preprocess_text(args)
        print("Example item text:")
        for item, text in item_text_list[:5]:
            print(f"Item: {item}, Text: {text}")
    else:
        item_text_list = None
    
    # Broadcast data to all processes
    if is_distributed:
        item_text_list = [item_text_list]
        dist.broadcast_object_list(item_text_list, src=0)
        item_text_list = item_text_list[0]

    # Load model
    plm_tokenizer, plm_model = load_plm(args.plm_checkpoint)
    if plm_tokenizer.pad_token_id is None:
        plm_tokenizer.pad_token_id = 0
    plm_model = plm_model.to(args.device)
    
    # Wrap as distributed model
    if is_distributed and world_size > 1:
        plm_model = DDP(plm_model, device_ids=[local_gpu_id])

    # Generate embedding
    local_embeddings, items_processed = generate_item_embedding(
        args, item_text_list, plm_tokenizer, plm_model, 
        word_drop_ratio=args.word_drop_ratio, rank=rank, world_size=world_size
    )
    
    # Collect results and save
    if is_distributed:
        final_embeddings = collect_embeddings(
            local_embeddings, items_processed, len(item_text_list), rank, world_size, args.device
        )
    else:
        final_embeddings = local_embeddings
    
    # Only main process saves results
    if rank == 0:
        print('Final embeddings shape: ', final_embeddings.shape)
        file = os.path.join(args.root, args.dataset + '.emb.npy')
        np.save(file, final_embeddings)
        print(f"Embeddings saved to {file}")
    
    # Clean up distributed environment
    if is_distributed:
        dist.destroy_process_group()


