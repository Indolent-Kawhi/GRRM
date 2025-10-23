import os
import numpy as np
import polars as pl
import time
import argparse
import json
from collections import defaultdict
from k_means_constrained import KMeansConstrained

def balanced_kmeans_level_constrained(X, K, max_iter=100, tol=1e-7, random_state=None, verbose=False):
    """Balanced K-means implemented with k-means-constrained"""
    start_time = time.time()
    n, d = X.shape
    X = X.astype(np.float32, copy=False)
    
    # Calculate min and max cluster size
    min_size = max(1, n // K - 1)  # allow some imbalance
    max_size = n // K + 1
    
    if verbose:
        print(f"    Starting constrained K-means with K={K}, n={n}, d={d}")
        print(f"    Cluster size constraints: [{min_size}, {max_size}]")
    
    # Use k-means-constrained
    kmeans = KMeansConstrained(
        n_clusters=K,
        size_min=min_size,
        size_max=max_size,
        max_iter=max_iter,
        tol=tol,
        random_state=random_state,
        n_init=3,
        verbose=verbose,
        n_jobs=16
    )
    
    # Train and get labels
    labels = kmeans.fit_predict(X)
    centroids = kmeans.cluster_centers_
    
    print(f"[Time] balanced_kmeans_level_constrained (K={K}): {time.time() - start_time:.2f}s")
    
    if verbose:
        # Check cluster size distribution
        unique, counts = np.unique(labels, return_counts=True)
        print(f"    Cluster sizes: min={counts.min()}, max={counts.max()}, mean={counts.mean():.1f}")
    
    return labels, centroids

def residual_kmeans_constrained(X, K, L, max_iter=300, tol=1e-4, random_state=None, verbose=False):
    """Residual K-means implemented with k-means-constrained"""
    total_start = time.time()
    n, d = X.shape
    Ks = ([K] * L) if isinstance(K, int) else list(K)
    assert len(Ks) == L

    X = X.astype(np.float32, copy=False)
    R = X.copy()
    codes_all = np.empty((L, n), dtype=np.int32)
    codebooks = []

    for l in range(L):
        level_start = time.time()
        k_l = Ks[l]
        if verbose:
            mse_before = np.mean(R ** 2)
            print(f"\n=== Level {l+1}/{L} | K={k_l} ===")
            print(f"  Residual MSE before clustering: {mse_before:.6f}")

        # Generate random seed for sub-level
        seed_l = None if random_state is None else int(np.random.RandomState(random_state + l).randint(0, 2**31 - 1))
        
        codes_l, C_l = balanced_kmeans_level_constrained(
            R, k_l, max_iter=max_iter, tol=tol, random_state=seed_l, verbose=verbose
        )

        codes_all[l] = codes_l
        codebooks.append(C_l)
        
        # Subtract reconstructed part from residual
        R -= C_l[codes_l]

        print(f"[Time] Level {l+1}: {time.time() - level_start:.2f}s")
        if verbose:
            mse_after = np.mean(R ** 2)
            print(f"  Residual MSE after Level {l+1}: {mse_after:.6f}")

    recon = X - R
    print(f"[Time] residual_kmeans_constrained total: {time.time() - total_start:.2f}s")
    
    if verbose:
        total_mse = np.mean((X - recon) ** 2)
        print(f"\nFinal reconstruction MSE: {total_mse:.6f}")
    
    return codes_all, codebooks, recon

def check_constrained_availability():
    """Check k-means-constrained availability"""
    try:
        from k_means_constrained import KMeansConstrained
        print("k-means-constrained is available")
        return True
    except ImportError as e:
        print(f"k-means-constrained not available: {e}")
        print("Please install it with: pip install k-means-constrained")
        return False
    
def deal_with_dedupilcate(df):
    """Handle duplicates"""
    df_with_index = df.with_row_index()

    # Process the DataFrame using older-compatible syntax
    result_df = df_with_index.with_columns(
        pl.when(pl.len().over("codes") > 1)
        .then(
            pl.col("codes").list.concat(
                # This line is changed to work with older Polars versions
                pl.col("index").rank(method="ordinal").over("codes").cast(pl.Int64)
            )
        )
        .otherwise(pl.col("codes"))
        .alias("codes")
    ).drop("index")

    return result_df

def parse_args():
    parser = argparse.ArgumentParser(description="Constrained K-means clustering")
    parser.add_argument('--root', type=str, default="data")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the input data file")
    parser.add_argument("--k", type=int, default=256, help="Number of clusters")
    parser.add_argument("--l", type=int, default=4, help="Number of levels")
    parser.add_argument("--max_iter", type=int, default=100, help="Maximum number of iterations")
    parser.add_argument("--min_cluster_size", type=int, default=None, help="Minimum cluster size")
    parser.add_argument("--max_cluster_size", type=int, default=None, help="Maximum cluster size")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Check k-means-constrained availability
    if not check_constrained_availability():
        exit(1)
    
    # Load data
    t0 = time.time()
    data_path = os.path.join(args.root, args.dataset + '.emb.npy')
    embeddings = np.load(data_path).astype(np.float32)
    print(f"Loaded embeddings with shape {embeddings.shape}. [Time: {time.time()-t0:.2f}s]")

    K_values = [args.k] * args.l
    # Run residual K-means
    t1 = time.time()
    codes_all, codebooks, recon = residual_kmeans_constrained(
        embeddings, K=K_values, L=args.l, random_state=42, verbose=True, max_iter=args.max_iter
    )
    print(f"[Time] residual_kmeans_constrained finished: {time.time()-t1:.2f}s")

    # Save codebooks
    output_dir = os.path.join(args.root, args.dataset)

    t2 = time.time()
    np.savez_compressed(os.path.join(output_dir, 'codebooks_constrained.npz'), 
                       **{f'codebook_{i}': cb for i, cb in enumerate(codebooks)})
    print(f"[Time] save codebooks finished: {time.time()-t2:.2f}s")

    codes_plus_one = codes_all.T + 1
    codes_df = pl.DataFrame({'codes': [list(c) for c in codes_plus_one]})
    
    # Deduplication
    t4 = time.time()
    codes_dedup = deal_with_dedupilcate(codes_df)
    print(f"[Time] deduplication finished: {time.time()-t4:.2f}s")
    
    # Save original codes (not deduplicated)
    np.save(os.path.join(output_dir, 'codes_constrained.npy'), codes_all.T)
    print(f"Codes saved to {os.path.join(output_dir, 'codes_constrained.npy')}")
    
    # Generate JSON index
    t5 = time.time()
    codes_json = {}
    for id, row in enumerate(codes_dedup.iter_rows(named=True)):
        codes_ = []
        for i, code in enumerate(row['codes']):
            codes_.append(f'<|{chr(97+i)}_{code}|>')
        codes_json[str(id)] = codes_
    
    # Save JSON index
    json_path = os.path.join(output_dir, f'{args.dataset}.index.json')
    with open(json_path, 'w') as f:
        json.dump(codes_json, f, indent=2)
    print(f"[Time] JSON index generation finished: {time.time()-t5:.2f}s")
    print(f"JSON index saved to {json_path}")
    
    # Print final statistics
    print(f"\nFinal statistics:")
    print(f"- Original data shape: {embeddings.shape}")
    print(f"- Number of levels: {args.l}")
    print(f"- K values per level: {K_values}")
    print(f"- Final reconstruction error: {np.mean((embeddings - recon) ** 2):.6f}")
    
    # Deduplication statistics
    codes_str = codes_df.with_columns(
        pl.col("codes").map_elements(lambda x: ','.join(map(str, x)), return_dtype=pl.Utf8).alias("codes_str")
    )
    duplicates = (codes_str
                  .group_by("codes_str")
                  .count()
                  .filter(pl.col("count") > 1)
                  .sort("count", descending=True))
    
    if len(duplicates) > 0:
        print(f"\nDeduplication statistics:")
        print(f"- Number of duplicate groups: {len(duplicates)}")
        print(f"- Total duplicates: {duplicates['count'].sum() - len(duplicates)}")