import numpy as np
import pandas as pd

# ==========================================
# --- MA SELECTOR & HELPER FUNCTIONS ---
# ==========================================
def sma(s: pd.Series, l: int) -> pd.Series:
    return s.rolling(l, min_periods=1).mean()

def ema(s: pd.Series, l: int) -> pd.Series:
    return s.ewm(span=l, adjust=False, min_periods=1).mean()

def rma(s: pd.Series, l: int) -> pd.Series:
    return s.ewm(alpha=1/l, adjust=False, min_periods=1).mean()

def dema(s: pd.Series, l: int) -> pd.Series:
    e1 = ema(s, l)
    e2 = ema(e1, l)
    return 2 * e1 - e2

def tema(s: pd.Series, l: int) -> pd.Series:
    e1 = ema(s, l)
    e2 = ema(e1, l)
    e3 = ema(e2, l)
    return 3 * (e1 - e2) + e3

def wma(s: pd.Series, l: int) -> pd.Series:
    if l <= 1:
        return s.copy()
    denominator = l * (l + 1) / 2
    sum_val = s * l
    for i in range(1, l):
        sum_val = sum_val + s.shift(i) * (l - i)
    return sum_val / denominator

def linreg(s: pd.Series, l: int) -> pd.Series:
    if l <= 1:
        return s.copy()
    sum_x = l * (l - 1) / 2
    sum_x2 = l * (l - 1) * (2 * l - 1) / 6
    denominator = l * sum_x2 - sum_x**2
    sum_y = s.rolling(l).sum()
    sum_xy = wma(s, l) * (l * (l + 1) / 2) - sum_y
    slope = (l * sum_xy - sum_x * sum_y) / denominator
    intercept = (sum_y - slope * sum_x) / l
    return slope * (l - 1) + intercept

def hma(s: pd.Series, l: int) -> pd.Series:
    half_len = int(l / 2)
    sqrt_len = int(np.sqrt(l))
    wma_half = wma(s, half_len)
    wma_full = wma(s, l)
    diff = 2 * wma_half - wma_full
    return wma(diff, sqrt_len)

def zlsma(s: pd.Series, l: int) -> pd.Series:
    lr = linreg(s, l)
    return lr + (lr - sma(s, l))

def thma(s: pd.Series, l: int) -> pd.Series:
    l_3 = max(1, int(np.round(l / 3)))
    l_2 = max(1, int(np.round(l / 2)))
    diff = wma(s, l_3) * 3 - wma(s, l_2) - wma(s, l)
    return wma(diff, l)

def calcMA(ma_type: str, s: pd.Series, l: int) -> pd.Series:
    l = max(1, l)
    if ma_type == "SMA":
        return sma(s, l)
    elif ma_type == "EMA":
        return ema(s, l)
    elif ma_type == "DEMA":
        return dema(s, l)
    elif ma_type == "TEMA":
        return tema(s, l)
    elif ma_type == "LSMA":
        return linreg(s, l)
    elif ma_type == "WMA":
        return wma(s, l)
    elif ma_type == "HMA":
        return hma(s, l)
    elif ma_type == "ZLSMA":
        return zlsma(s, l)
    elif ma_type == "SMMA":
        return rma(s, l)
    elif ma_type == "THMA":
        return thma(s, l)
    else:
        return sma(s, l)

# CHOP Calculation Function
def calculate_tr(df: pd.DataFrame) -> pd.Series:
    high = df['High']
    low = df['Low']
    close_prev = df['Close'].shift(1)
    
    tr1 = high - low
    tr2 = (high - close_prev).abs()
    tr3 = (low - close_prev).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

def calculate_chop(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr = calculate_tr(df)
    sum_tr = tr.rolling(period).sum()
    
    highest_high = df['High'].rolling(period).max()
    lowest_low = df['Low'].rolling(period).min()
    close_prev = df['Close'].shift(1)
    
    max_high = pd.concat([highest_high, close_prev], axis=1).max(axis=1)
    min_low = pd.concat([lowest_low, close_prev], axis=1).min(axis=1)
    
    range_high_low = max_high - min_low
    chop = 100 * np.log10(sum_tr / range_high_low.replace(0, 0.00001)) / np.log10(period)
    return chop

# Normalization
def normalize_series(s: pd.Series, l: int) -> pd.Series:
    s_prev = s.shift(1)
    mean = s_prev.rolling(l, min_periods=1).mean()
    std = s_prev.rolling(l, min_periods=1).std()
    return (s - mean) / np.maximum(std, 0.00001)

# RSI
def rsi(s: pd.Series, period: int = 14) -> pd.Series:
    delta = s.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = gain / loss.replace(0, 0.00001)
    return 100 - (100 / (1 + rs))

# SuperTrend
def calculate_supertrend(df: pd.DataFrame, period: int = 10, factor: float = 2.0):
    high = df['High'].values
    low = df['Low'].values
    close = df['Close'].values
    
    tr = calculate_tr(df).values
    atr = calculate_tr(df).ewm(alpha=1/period, min_periods=1, adjust=False).mean().values
    
    hl2 = (high + low) / 2
    basic_ub = hl2 + factor * atr
    basic_lb = hl2 - factor * atr
    
    final_ub = basic_ub.copy()
    final_lb = basic_lb.copy()
    
    # Loop using NumPy arrays
    for i in range(1, len(df)):
        # Upper band
        if basic_ub[i] < final_ub[i-1] or close[i-1] > final_ub[i-1]:
            final_ub[i] = basic_ub[i]
        else:
            final_ub[i] = final_ub[i-1]
            
        # Lower band
        if basic_lb[i] > final_lb[i-1] or close[i-1] < final_lb[i-1]:
            final_lb[i] = basic_lb[i]
        else:
            final_lb[i] = final_lb[i-1]
            
    # Calculate SuperTrend line and direction
    direction = np.ones(len(df))  # 1 = Bearish (downtrend), -1 = Bullish (uptrend)
    supertrend = np.zeros(len(df))
    
    for i in range(1, len(df)):
        prev_dir = direction[i-1]
        if prev_dir == -1:
            if close[i] < final_lb[i]:
                direction[i] = 1
                supertrend[i] = final_ub[i]
            else:
                direction[i] = -1
                supertrend[i] = final_lb[i]
        else:
            if close[i] > final_ub[i]:
                direction[i] = -1
                supertrend[i] = final_lb[i]
            else:
                direction[i] = 1
                supertrend[i] = final_ub[i]
                
    return pd.Series(supertrend, index=df.index), pd.Series(direction, index=df.index)

# KNN Core Engine
def calculate_knn_probabilities(
    pc1: pd.Series, pc2: pd.Series, pc3: pd.Series, pc4: pd.Series, target: pd.Series,
    k: int = 10, sampling_window: int = 1000, stride: int = 10,
    p_param: float = 2.0, w_param: float = 2.0
):
    n = len(pc1)
    prob_up = np.zeros(n)
    prob_down = np.zeros(n)
    
    v1 = pc1.values
    v2 = pc2.values
    v3 = pc3.values
    v4 = pc4.values
    tgt = target.values
    
    # Warm up: need at least sampling_window + stride bars of history
    start_idx = sampling_window + stride
    
    for t in range(start_idx, n):
        # Get past indices based on stride
        past_indices = t - np.arange(stride, sampling_window + stride + 1, stride)
        past_indices = past_indices[past_indices >= 0]
        
        # Only keep indices where target is not 0
        valid_indices = past_indices[tgt[past_indices] != 0]
        
        if len(valid_indices) < k:
            continue
            
        h1 = v1[valid_indices]
        h2 = v2[valid_indices]
        h3 = v3[valid_indices]
        h4 = v4[valid_indices]
        
        # Minkowski distance
        d1 = np.abs(v1[t] - h1)
        d2 = np.abs(v2[t] - h2)
        d3 = np.abs(v3[t] - h3)
        d4 = np.abs(v4[t] - h4)
        
        dists = (d1**p_param + d2**p_param + d3**p_param + d4**p_param) ** (1.0 / p_param)
        
        # Sort and get nearest neighbors
        nearest_indices = np.argsort(dists)[:k]
        k_dists = dists[nearest_indices]
        k_labels = tgt[valid_indices[nearest_indices]]
        
        sigma = k_dists[min(k // 2, len(k_dists) - 1)]
        sigma = max(sigma, 0.0001)
        
        # Gaussian weights
        weights = np.exp(-(k_dists**w_param) / (2 * sigma**2))
        total_w = weights.sum()
        
        if total_w > 0:
            prob_up[t] = np.sum(weights[k_labels == 1]) / total_w
            prob_down[t] = np.sum(weights[k_labels == -1]) / total_w
            
    return pd.Series(prob_up, index=pc1.index), pd.Series(prob_down, index=pc1.index)

# Single-bar KNN probability calculator for lazy/optimized execution
def calculate_knn_probability_at_bar(
    t: int, pc1_vals: np.ndarray, pc2_vals: np.ndarray, pc3_vals: np.ndarray, pc4_vals: np.ndarray, tgt_vals: np.ndarray,
    k: int = 10, sampling_window: int = 1000, stride: int = 10,
    p_param: float = 2.0, w_param: float = 2.0
):
    # Get past indices based on stride
    past_indices = t - np.arange(stride, sampling_window + stride + 1, stride)
    past_indices = past_indices[past_indices >= 0]
    
    # Only keep indices where target is not 0
    valid_indices = past_indices[tgt_vals[past_indices] != 0]
    
    if len(valid_indices) < k:
        return 0.0, 0.0
        
    h1 = pc1_vals[valid_indices]
    h2 = pc2_vals[valid_indices]
    h3 = pc3_vals[valid_indices]
    h4 = pc4_vals[valid_indices]
    
    # Minkowski distance
    d1 = np.abs(pc1_vals[t] - h1)
    d2 = np.abs(pc2_vals[t] - h2)
    d3 = np.abs(pc3_vals[t] - h3)
    d4 = np.abs(pc4_vals[t] - h4)
    
    dists = (d1**p_param + d2**p_param + d3**p_param + d4**p_param) ** (1.0 / p_param)
    
    # Sort and get nearest neighbors
    nearest_indices = np.argsort(dists)[:k]
    k_dists = dists[nearest_indices]
    k_labels = tgt_vals[valid_indices[nearest_indices]]
    
    sigma = k_dists[min(k // 2, len(k_dists) - 1)]
    sigma = max(sigma, 0.0001)
    
    # Gaussian weights
    weights = np.exp(-(k_dists**w_param) / (2 * sigma**2))
    total_w = weights.sum()
    
    prob_up = 0.0
    prob_down = 0.0
    if total_w > 0:
        prob_up = np.sum(weights[k_labels == 1]) / total_w
        prob_down = np.sum(weights[k_labels == -1]) / total_w
        
    return float(prob_up), float(prob_down)

# Main entry point to compute features and KNN probabilities
def run_knn_classifier(
    df: pd.DataFrame,
    atr_period: int = 10, factor: float = 2.0,
    k_neighbors: int = 10, sampling_window_size: int = 1000, momentum_window: int = 10,
    feat_ma_type: str = "SMA", rsi_len: int = 20, ma_len: int = 20,
    signal_len: int = 10, chop_len: int = 14, normalizing_window_size: int = 1000,
    p_param: float = 2.0, w_param: float = 2.0, use_pca: bool = True,
    lazy: bool = False
):
    df = df.copy()
    close = df['Close']
    
    # 1. Target Labeling (SuperTrend direction)
    _, direction = calculate_supertrend(df, atr_period, factor)
    
    # Target = direction * -1 (Pine Script translation: -1 = uptrend/bull, 1 = downtrend/bear)
    # direction is -1 for bull, 1 for bear. So target is 1 for bull, -1 for bear.
    target = direction * -1
    
    # Filter target changes: target is 0 if direction has flipped within last 5 bars
    target_clean = target.copy()
    for i in range(1, 6):
        target_clean = target_clean.where(direction.shift(i) == direction, 0)
        
    # 2. Features Calculation
    f_rsi_s = rsi(close, rsi_len)
    f_rsi_m = f_rsi_s.shift(momentum_window)
    f_rsi_l = f_rsi_s.shift(momentum_window * 2)
    
    ma_ref = calcMA(feat_ma_type, close.shift(1), ma_len)
    f_ma_s_dev = (close - ma_ref) / ma_ref.replace(0, 0.00001)
    f_ma_m_dev = f_ma_s_dev.shift(momentum_window)
    f_ma_l_dev = f_ma_s_dev.shift(momentum_window * 2)
    
    rsi_sma = sma(f_rsi_s.shift(1), signal_len)
    f_rsi_s_sig_dist = (f_rsi_s - rsi_sma) / rsi_sma.replace(0, 0.00001)
    f_rsi_m_sig_dist = f_rsi_s_sig_dist.shift(momentum_window)
    f_rsi_l_sig_dist = f_rsi_s_sig_dist.shift(momentum_window * 2)
    
    f_chop_s = calculate_chop(df, chop_len)
    f_chop_m = f_chop_s.shift(momentum_window)
    f_chop_l = f_chop_s.shift(momentum_window * 2)
    
    # Z-Score Normalization
    f_rsi_s_z = normalize_series(f_rsi_s, normalizing_window_size)
    f_rsi_m_z = normalize_series(f_rsi_m, normalizing_window_size)
    f_rsi_l_z = normalize_series(f_rsi_l, normalizing_window_size)
    
    f_ma_s_dev_z = normalize_series(f_ma_s_dev, normalizing_window_size)
    f_ma_m_dev_z = normalize_series(f_ma_m_dev, normalizing_window_size)
    f_ma_l_dev_z = normalize_series(f_ma_l_dev, normalizing_window_size)
    
    f_rsi_s_sd_z = normalize_series(f_rsi_s_sig_dist, normalizing_window_size)
    f_rsi_m_sd_z = normalize_series(f_rsi_m_sig_dist, normalizing_window_size)
    f_rsi_l_sd_z = normalize_series(f_rsi_l_sig_dist, normalizing_window_size)
    
    f_chop_s_z = normalize_series(f_chop_s, normalizing_window_size)
    f_chop_m_z = normalize_series(f_chop_m, normalizing_window_size)
    f_chop_l_z = normalize_series(f_chop_l, normalizing_window_size)
    
    # PCA compression / Combination
    if use_pca:
        pc1 = normalize_series(f_rsi_s_z + f_ma_s_dev_z + f_rsi_s_sd_z * 0.5, normalizing_window_size)
        pc2 = normalize_series(f_rsi_m_z + f_ma_m_dev_z + f_rsi_m_sd_z * 0.5, normalizing_window_size) * 0.9
        pc3 = normalize_series(f_rsi_l_z + f_ma_l_dev_z + f_rsi_l_sd_z * 0.5, normalizing_window_size) * 0.8
        pc4 = normalize_series(f_chop_s_z + f_chop_m_z * 0.9 + f_chop_l_z * 0.8, normalizing_window_size) * 0.8
    else:
        pc1 = f_rsi_m_z
        pc2 = f_ma_m_dev_z
        pc3 = f_rsi_m_sd_z
        pc4 = f_chop_m_z
        
    if lazy:
        return pc1, pc2, pc3, pc4, target_clean
        
    # KNN Engine execution
    prob_up, prob_down = calculate_knn_probabilities(
        pc1, pc2, pc3, pc4, target_clean,
        k=k_neighbors, sampling_window=sampling_window_size, stride=momentum_window,
        p_param=p_param, w_param=w_param
    )
    
    return prob_up, prob_down
