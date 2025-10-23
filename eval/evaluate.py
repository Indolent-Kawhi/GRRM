import math

def get_topk_results(predictions, scores, targets, k, all_items=None):
    results = []
    B = len(targets)

    if all_items is not None:
        for i, seq in enumerate(predictions):
            if seq not in all_items:
                scores[i] = -1000

    for b in range(B):
        batch_seqs = predictions[b * k: (b + 1) * k]
        batch_scores = scores[b * k: (b + 1) * k]

        pairs = [(a, b) for a, b in zip(batch_seqs, batch_scores)]
        sorted_pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
        
        # Support multi-label: if targets[b] is a string, convert to list; if already a list, keep as is
        target_items = targets[b]
        if isinstance(target_items, str):
            target_items = [target_items]
        elif not isinstance(target_items, list):
            # If other iterable type, convert to list
            target_items = list(target_items)
        
        one_results = []
        for sorted_pred in sorted_pairs:
            if sorted_pred[0] in target_items:
                one_results.append(1)
            else:
                one_results.append(0)

        results.append(one_results)

    return results

def get_metrics_results(topk_results, metrics, targets=None):
    """
    Calculate evaluation metrics, support multi-target
    Args:
        topk_results: top-k results
        metrics: list of metrics
        targets: target items list, used for metrics that require target length
    """
    res = {}
    for m in metrics:
        if m.lower().startswith("hit") or m.lower().startswith("recall"):
            k = int(m.split("@")[1])
            if m.lower().startswith("hit"):
                res[m] = hit_k(topk_results, k)
            else:  # recall
                res[m] = recall_k(topk_results, k, targets)
        elif m.lower().startswith("ndcg"):
            k = int(m.split("@")[1])
            res[m] = ndcg_k(topk_results, k, targets)
        elif m.lower().startswith("precision"):
            k = int(m.split("@")[1])
            res[m] = precision_k(topk_results, k)
        elif m.lower().startswith("mrr"):
            k = int(m.split("@")[1])
            res[m] = mrr_k(topk_results, k)
        else:
            raise NotImplementedError(f"Metric {m} not implemented")

    return res

def hit_k(topk_results, k):
    """Hit@K: Whether any target item is hit"""
    hit = 0.0
    for row in topk_results:
        res = row[:k]
        if sum(res) > 0:
            hit += 1
    return hit / len(topk_results)

def recall_k(topk_results, k, targets=None):
    """Recall@K: Number of hit target items / total target items"""
    if targets is None:
        # If targets not provided, assume each user has one target item
        return hit_k(topk_results, k)
    
    total_recall = 0.0
    for i, row in enumerate(topk_results):
        res = row[:k]
        hits = sum(res)
        
        # Calculate number of target items
        target_items = targets[i]
        if not isinstance(target_items, (list, tuple)):
            target_items = [target_items]
        num_targets = len(target_items)
        
        if num_targets > 0:
            total_recall += hits / num_targets
    
    return total_recall / len(topk_results)

def precision_k(topk_results, k):
    """Precision@K: Number of hit items / k"""
    total_precision = 0.0
    for row in topk_results:
        res = row[:k]
        hits = sum(res)
        total_precision += hits / k
    
    return total_precision / len(topk_results)

def mrr_k(topk_results, k):
    """MRR@K: Mean Reciprocal Rank"""
    total_rr = 0.0
    for row in topk_results:
        res = row[:k]
        for i, hit in enumerate(res):
            if hit == 1:
                total_rr += 1.0 / (i + 1)
                break  # Only calculate the first hit position
    
    return total_rr / len(topk_results)

def ndcg_k(topk_results, k, targets=None):
    """NDCG@K: Normalized Discounted Cumulative Gain"""
    total_ndcg = 0.0
    
    for i, row in enumerate(topk_results):
        res = row[:k]
        
        # Calculate IDCG (ideal DCG)
        if targets is not None:
            target_items = targets[i]
            if not isinstance(target_items, (list, tuple)):
                target_items = [target_items]
            num_targets = len(target_items)
        else:
            # Assume only one target item
            num_targets = 1
        
        idcg = 0.0
        for j in range(min(num_targets, k)):
            idcg += 1.0 / math.log2(j + 2)
        
        # Calculate DCG
        dcg = 0.0
        for j, hit in enumerate(res):
            if hit == 1:
                dcg += 1.0 / math.log2(j + 2)
                if num_targets == 1:
                    break
        
        # Calculate NDCG
        if idcg > 0:
            total_ndcg += dcg / idcg
    
    return total_ndcg / len(topk_results)