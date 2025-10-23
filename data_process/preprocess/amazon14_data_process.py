import argparse
import collections
import gzip
import html
import json
import os
import random
import re
import torch
from tqdm import tqdm
import numpy as np
import ast
from utils import check_path, clean_text, amazon18_dataset2fullname, write_json_file, write_remap_index

def load_ratings(file):
    users, items, inters = set(), set(), set()
    with open(file, 'r') as fp:
        for line in tqdm(fp, desc='Load ratings'):
            try:
                user, item, rating, time = line.strip().split(',')
                users.add(user)
                items.add(item)
                inters.add((user, item, float(rating), int(time)))
            except ValueError:
                print(line)
    return users, items, inters


def load_meta_items(file):
    items = {}
    with gzip.open(file, "r") as fp:
        for line in tqdm(fp, desc="Load metas"):
            data = ast.literal_eval(line.decode('utf-8'))
            item = data["asin"]
            title = clean_text(data.get("title", ""))
            descriptions = data.get("description", "")
            descriptions = clean_text(descriptions)

            brand = data.get("brand", "").replace("by\n", "").strip()

            categories = data.get("categories", "")
            categories = ",".join(categories[0]).strip()
            price = data.get("price", "")

            items[item] = {"title": title, "description": descriptions, "brand": brand, "categories": categories, "price": price}
        return items


def load_review_data(args, user2id, item2id):

    dataset_full_name = amazon18_dataset2fullname[args.dataset]
    review_file_path = os.path.join(args.input_path, 'Review', 'reviews_' + dataset_full_name + '.json.gz')

    reviews = {}

    with gzip.open(review_file_path, "r") as fp:

        for line in tqdm(fp,desc='Load reviews'):
            inter = json.loads(line)
            try:
                user = inter['reviewerID']
                item = inter['asin']
                if user in user2id and item in item2id:
                    uid = user2id[user]
                    iid = item2id[item]
                else:
                    continue
                if 'reviewText' in inter:
                    review = clean_text(inter['reviewText'])
                else:
                    review = ''
                if 'summary' in inter:
                    summary = clean_text(inter['summary'])
                else:
                    summary = ''
                reviews[str((uid,iid))]={"review":review, "summary":summary}

            except ValueError:
                print(line)

    return reviews


def get_user2count(inters):
    user2count = collections.defaultdict(int)
    for unit in inters:
        user2count[unit[0]] += 1
    return user2count


def get_item2count(inters):
    item2count = collections.defaultdict(int)
    for unit in inters:
        item2count[unit[1]] += 1
    return item2count


def generate_candidates(unit2count, threshold):
    cans = set()
    for unit, count in unit2count.items():
        if count >= threshold:
            cans.add(unit)
    return cans, len(unit2count) - len(cans)


def filter_inters(inters, can_items=None,
                  user_k_core_threshold=0, item_k_core_threshold=0):
    new_inters = []
    
    print("inters sample: ", inters[:2])
    print(f"can_items sample: {list(can_items.items())[:2]}")

    # filter by meta items
    if can_items:
        print('\nFiltering by meta items: ')
        for unit in inters:
            if unit[1] in can_items.keys():
                new_inters.append(unit)
        inters, new_inters = new_inters, []
        print('    The number of inters: ', len(inters))

    # filter by k-core
    if user_k_core_threshold or item_k_core_threshold:
        print('\nFiltering by k-core:')
        idx = 0
        user2count = get_user2count(inters)
        item2count = get_item2count(inters)

        while True:
            new_user2count = collections.defaultdict(int)
            new_item2count = collections.defaultdict(int)
            users, n_filtered_users = generate_candidates( # users is set
                user2count, user_k_core_threshold)
            items, n_filtered_items = generate_candidates(
                item2count, item_k_core_threshold)
            if n_filtered_users == 0 and n_filtered_items == 0:
                break
            for unit in inters:
                if unit[0] in users and unit[1] in items:
                    new_inters.append(unit)
                    new_user2count[unit[0]] += 1
                    new_item2count[unit[1]] += 1
            idx += 1
            inters, new_inters = new_inters, []
            user2count, item2count = new_user2count, new_item2count
            print('    Epoch %d The number of inters: %d, users: %d, items: %d'
                    % (idx, len(inters), len(user2count), len(item2count)))
    return inters


def make_inters_in_order(inters):
    user2inters, new_inters = collections.defaultdict(list), list()
    for inter in inters:
        user, item, rating, timestamp = inter
        user2inters[user].append((user, item, rating, timestamp))
    for user in user2inters:
        user_inters = user2inters[user]
        user_inters.sort(key=lambda d: d[3])
        interacted_item = set()
        for inter in user_inters:
            if inter[1] in interacted_item: # 过滤重复交互
                continue
            interacted_item.add(inter[1])
            new_inters.append(inter)
    return new_inters


def preprocess_rating(args):
    dataset_full_name = amazon18_dataset2fullname[args.dataset]

    print('Process rating data: ')
    print(' Dataset: ', args.dataset)

    # load ratings
    rating_file_path = os.path.join(args.input_path, 'Ratings', f"ratings_{dataset_full_name}.csv")
    rating_users, rating_items, rating_inters = load_ratings(rating_file_path)

    # load item IDs with meta data
    meta_file_path = os.path.join(args.input_path, 'Metadata', f'meta_{dataset_full_name}.json.gz')
    meta_items = load_meta_items(meta_file_path)

    # 1. Filter items w/o meta data;
    # 2. K-core filtering;
    print('The number of raw inters: ', len(rating_inters))

    rating_inters = make_inters_in_order(rating_inters)

    rating_inters = filter_inters(rating_inters, can_items=meta_items,
                                  user_k_core_threshold=args.user_k,
                                  item_k_core_threshold=args.item_k)

    # sort interactions chronologically for each user
    rating_inters = make_inters_in_order(rating_inters)
    print('\n')

    # return: list of (user_ID, item_ID, rating, timestamp)
    return rating_inters, meta_items

def convert_inters2dict(inters):
    user2items = collections.defaultdict(list)
    
    # 首先收集所有唯一的用户和物品ID
    all_users = set()
    all_items = set()
    for inter in inters:
        user, item, rating, timestamp = inter
        all_users.add(user)
        all_items.add(item)
    
    # 对用户和物品ID进行排序，确保固定的映射
    sorted_users = sorted(list(all_users))
    sorted_items = sorted(list(all_items))
    
    # 基于排序后的顺序生成固定映射
    user2index = {user: idx for idx, user in enumerate(sorted_users)}
    item2index = {item: idx for idx, item in enumerate(sorted_items)}
    
    # 构建用户-物品交互字典
    for inter in inters:
        user, item, rating, timestamp = inter
        user2items[user2index[user]].append(item2index[item])
    
    return user2items, user2index, item2index

def generate_data(args, rating_inters):
    print('Split dataset: ')
    print(' Dataset: ', args.dataset)

    # generate train valid temp
    user2items, user2index, item2index = convert_inters2dict(rating_inters)
    train_inters, valid_inters, test_inters = dict(), dict(), dict()
    for u_index in range(len(user2index)):
        inters = user2items[u_index]
        # leave one out
        train_inters[u_index] = [str(i_index) for i_index in inters[:-2]]
        valid_inters[u_index] = [str(inters[-2])]
        test_inters[u_index] = [str(inters[-1])]
        assert len(user2items[u_index]) == len(train_inters[u_index]) + \
               len(valid_inters[u_index]) + len(test_inters[u_index])
    return user2items, train_inters, valid_inters, test_inters, user2index, item2index

def convert_to_atomic_files(args, train_data, valid_data, test_data):
    print('Convert dataset: ')
    print(' Dataset: ', args.dataset)
    uid_list = list(train_data.keys())
    uid_list.sort(key=lambda t: int(t))

    with open(os.path.join(args.output_path, args.dataset, f'{args.dataset}.train.inter'), 'w') as file:
        file.write('user_id:token\titem_id_list:token_seq\titem_id:token\n')
        for uid in uid_list:
            item_seq = train_data[uid]
            seq_len = len(item_seq)
            for target_idx in range(1, seq_len):
                target_item = item_seq[-target_idx]
                seq = item_seq[:-target_idx][-50:]
                file.write(f'{uid}\t{" ".join(seq)}\t{target_item}\n')

    with open(os.path.join(args.output_path, args.dataset, f'{args.dataset}.valid.inter'), 'w') as file:
        file.write('user_id:token\titem_id_list:token_seq\titem_id:token\n')
        for uid in uid_list:
            item_seq = train_data[uid][-50:]
            target_item = valid_data[uid][0]
            file.write(f'{uid}\t{" ".join(item_seq)}\t{target_item}\n')

    with open(os.path.join(args.output_path, args.dataset, f'{args.dataset}.test.inter'), 'w') as file:
        file.write('user_id:token\titem_id_list:token_seq\titem_id:token\n')
        for uid in uid_list:
            item_seq = (train_data[uid] + valid_data[uid])[-50:]
            target_item = test_data[uid][0]
            file.write(f'{uid}\t{" ".join(item_seq)}\t{target_item}\n')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Arts', help='Instruments / Arts / Games')
    parser.add_argument('--user_k', type=int, default=5, help='user k-core filtering')
    parser.add_argument('--item_k', type=int, default=5, help='item k-core filtering')
    parser.add_argument('--input_path', type=str, default='')
    parser.add_argument('--output_path', type=str, default='')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # load interactions from raw rating file
    rating_inters, meta_items = preprocess_rating(args)


    # split train/valid/temp
    all_inters,train_inters, valid_inters, test_inters, user2index, item2index = generate_data(args, rating_inters)

    check_path(os.path.join(args.output_path, args.dataset))

    write_json_file(all_inters, os.path.join(args.output_path, args.dataset, f'{args.dataset}.inter.json'))
    convert_to_atomic_files(args, train_inters, valid_inters, test_inters)

    item2feature = collections.defaultdict(dict)
    for item, item_id in item2index.items():
        item2feature[item_id] = meta_items[item]

    reviews = load_review_data(args, user2index, item2index)

    print("user:",len(user2index))
    print("item:",len(item2index))

    write_json_file(item2feature, os.path.join(args.output_path, args.dataset, f'{args.dataset}.item.json'))
    write_json_file(reviews, os.path.join(args.output_path, args.dataset, f'{args.dataset}.review.json'))


    write_remap_index(user2index, os.path.join(args.output_path, args.dataset, f'{args.dataset}.user2id'))
    write_remap_index(item2index, os.path.join(args.output_path, args.dataset, f'{args.dataset}.item2id'))