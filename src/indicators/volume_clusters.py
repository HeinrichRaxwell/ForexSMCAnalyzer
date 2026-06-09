import numpy as np
import pandas as pd

def calculate_volume_clusters(
    df: pd.DataFrame,
    lookback: int = 200,
    k: int = 5,
    iterations: int = 50,
    rows: int = 20
):
    """
    Implements the K-Means clustering and Volume Profile calculation
    from the LuxAlgo Clusters Volume Profile indicator.
    
    Args:
        df (pd.DataFrame): DataFrame containing ['High', 'Low', 'Close', 'Volume']
        lookback (int): Number of bars to analyze from the end.
        k (int): Number of clusters.
        iterations (int): K-Means refinement iterations.
        rows (int): Bins per cluster volume profile.
        
    Returns:
        dict: A dictionary containing details of the computed clusters:
            - 'centroids': List of final centroid prices
            - 'pocs': List of POC levels for each cluster
            - 'total_volumes': List of total volume for each cluster
            - 'current_cluster': Cluster index of the current bar
            - 'current_poc': POC price of the current bar's cluster
            - 'current_dist_to_poc': Distance from close to current cluster's POC in percent
    """
    n = min(len(df), lookback)
    if n < k:
        return {}
        
    # Slice the last lookback bars
    sub_df = df.iloc[-n:].copy()
    
    # hl2 median price and volume
    hl2 = ((sub_df['High'] + sub_df['Low']) / 2).values
    volume = sub_df['Volume'].values
    high = sub_df['High'].values
    low = sub_df['Low'].values
    close_curr = sub_df['Close'].values[-1]
    
    min_p = hl2.min()
    max_p = hl2.max()
    
    # 1. Initialize centroids
    step = (max_p - min_p) / (k + 1)
    centroids = np.zeros(k)
    for j in range(k):
        centroids[j] = min_p + (j + 1) * step
        
    assignments = np.zeros(n, dtype=int)
    
    # 2. K-Means Iterations
    for iter_idx in range(iterations):
        # Assign each bar to nearest centroid (vectorized)
        assignments = np.argmin(np.abs(hl2[:, np.newaxis] - centroids), axis=1)
            
        # Recalculate centroids (VWAP)
        for j in range(k):
            cluster_mask = (assignments == j)
            if cluster_mask.any():
                sum_pv = np.sum(hl2[cluster_mask] * volume[cluster_mask])
                sum_v = np.sum(volume[cluster_mask])
                if sum_v > 0:
                    centroids[j] = sum_pv / sum_v
                    
    # 3. Calculate Volume Profile per cluster
    pocs = np.zeros(k)
    total_volumes = np.zeros(k)
    
    for c_id in range(k):
        cluster_mask = (assignments == c_id)
        if not cluster_mask.any():
            continue
            
        c_highs = high[cluster_mask]
        c_lows = low[cluster_mask]
        c_vols = volume[cluster_mask]
        c_prices = hl2[cluster_mask]
        
        c_min = np.min(c_lows)
        c_max = np.max(c_highs)
        total_volumes[c_id] = np.sum(c_vols)
        
        bin_vols = np.zeros(rows)
        bin_size = (c_max - c_min) / rows
        if bin_size == 0:
            bin_size = 0.00001
            
        for i in range(len(c_prices)):
            b_h = c_highs[i]
            b_l = c_lows[i]
            b_v = c_vols[i]
            wick_range = max(b_h - b_l, 0.00001)
            
            for b_idx in range(rows):
                bin_b = c_min + b_idx * bin_size
                bin_t = bin_b + bin_size
                
                intersect_l = max(b_l, bin_b)
                intersect_h = min(b_h, bin_t)
                
                if intersect_h > intersect_l:
                    bin_vols[b_idx] += b_v * (intersect_h - intersect_l) / wick_range
                    
        # POC bin
        poc_bin_idx = np.argmax(bin_vols)
        poc_b = c_min + poc_bin_idx * bin_size
        poc_t = poc_b + bin_size
        pocs[c_id] = (poc_b + poc_t) / 2
        
    # Get current bar cluster details
    curr_cluster = assignments[-1]
    curr_poc = pocs[curr_cluster]
    curr_dist = (close_curr - curr_poc) / curr_poc if curr_poc > 0 else 0.0
    
    return {
        'centroids': list(centroids),
        'pocs': list(pocs),
        'total_volumes': list(total_volumes),
        'current_cluster': int(curr_cluster),
        'current_poc': float(curr_poc),
        'current_dist_to_poc': float(curr_dist)
    }
