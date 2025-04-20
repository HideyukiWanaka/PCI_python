# symmetry_indices.py
import numpy as np
import pandas as pd # isnullチェック用

def calculate_symmetry_index(value_L, value_R):
    """左右の値から対称性指数 (Symmetry Index, SI) を計算"""
    if pd.isna(value_L) or pd.isna(value_R): return np.nan
    denominator = 0.5 * (value_L + value_R)
    if np.isclose(denominator, 0): return np.nan
    si = np.abs(value_L - value_R) / np.abs(denominator) * 100.0
    return si

# def calculate_symmetry_ratio(value_L, value_R): ... (必要なら)