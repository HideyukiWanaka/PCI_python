# preprocessing.py
import pandas as pd
import numpy as np
from scipy.signal import find_peaks # find_peaks を使うので import
from pathlib import Path

# ★★★ find_peaks のパラメータを引数に追加 ★★★
def preprocess_angular_velocity(data_file, rows_to_skip=11, sampling_interval_ms=5,
                                right_prefix='R', left_prefix='L', trunk_prefix='T',
                                sync_col_suffix='_Acc_Y', align_col_suffix='_Gyro_Z',
                                peak_height=None, peak_prominence=None, peak_distance=None): # 引数を追加
    """
    (関数説明は同じ)
    Returns:
        tuple: (pd.DataFrame, float, float, int, int, np.ndarray, np.ndarray)
            (戻り値の構成は同じ)
    """
    print(f"--- [Function@preprocessing] 1. 角速度データの前処理開始 (find_peaks: h={peak_height}, p={peak_prominence}, d={peak_distance}) ---")
    sampling_rate_hz = 1000.0 / sampling_interval_ms
    error_return = (None, None, None, -1, -1, None, None)

    # --- データの読み込みと列名設定 ---
    # (変更なし - 前回のコードと同じ)
    try:
        data_file_path = Path(data_file) if not isinstance(data_file, Path) else data_file
        try: df = pd.read_csv(data_file_path, skiprows=rows_to_skip, encoding='cp932')
        except UnicodeDecodeError: df = pd.read_csv(data_file_path, skiprows=rows_to_skip, encoding='shift_jis')
        new_column_names = [ # ... (省略) ...
            'Time', f'{right_prefix}_Acc_X', f'{right_prefix}_Acc_Y', f'{right_prefix}_Acc_Z', f'{right_prefix}_Ch4_V',
            f'{right_prefix}_Gyro_X', f'{right_prefix}_Gyro_Y', f'{right_prefix}_Gyro_Z', f'{right_prefix}_Ch8_V',
            f'{left_prefix}_Acc_X', f'{left_prefix}_Acc_Y', f'{left_prefix}_Acc_Z', 'Blank_1',
            f'{left_prefix}_Gyro_X', f'{left_prefix}_Gyro_Y', f'{left_prefix}_Gyro_Z', 'Blank_2',
            f'{trunk_prefix}_Acc_X', f'{trunk_prefix}_Acc_Y', f'{trunk_prefix}_Acc_Z', 'Blank_3',
            f'{trunk_prefix}_Gyro_X', f'{trunk_prefix}_Gyro_Y', f'{trunk_prefix}_Gyro_Z', 'Blank_4']
        if len(df.columns) == len(new_column_names):
            df.columns = new_column_names
            blank_cols = [col for col in df.columns if 'Blank_' in col]; df = df.drop(columns=blank_cols)
        else: print(f"エラー: 列数不一致 ({len(df.columns)} vs {len(new_column_names)})"); return error_return
    except Exception as e: print(f"エラー: データ読み込み/列名設定中にエラー: {e}"); return error_return


    # --- 信号の抽出と左角速度の符号反転 ---
    # (変更なし)
    sync_left_short, sync_right_short = None, None
    try:
        sync_left_full = df[f'{left_prefix}{sync_col_suffix}'].fillna(0).values
        sync_right_full = df[f'{right_prefix}{sync_col_suffix}'].fillna(0).values
        align_left_raw = df[f'{left_prefix}{align_col_suffix}'].fillna(0).values
        align_right = df[f'{right_prefix}{align_col_suffix}'].fillna(0).values
        align_left = -align_left_raw
        # print(f"  情報: 左 {align_col_suffix} の符号を反転しました。") # ログは main 側で制御

        num_samples_for_sync = 1000 # 同期計算に使う最大サンプル数
        actual_sync_length = min(len(sync_left_full), len(sync_right_full), num_samples_for_sync)
        if actual_sync_length <= 0: raise ValueError("同期に使用できるデータがありません。")
        sync_left_short = sync_left_full[:actual_sync_length]
        sync_right_short = sync_right_full[:actual_sync_length]
    except Exception as e:
        print(f"エラー: 信号抽出中にエラー: {e}")
        return error_return


    # --- ピーク検出によるラグ計算 (find_peaks を使用) ---
    lag_samples = 0
    peak_index_left = -1
    peak_index_right = -1
    try:
        print(f"  情報: find_peaks (h={peak_height}, p={peak_prominence}, d={peak_distance}) @ 先頭 {actual_sync_length} samples")

        # ★★★ 引数で渡されたパラメータを使用 ★★★
        left_peaks, _ = find_peaks(np.abs(sync_left_short),
                                   height=peak_height,
                                   prominence=peak_prominence,
                                   distance=peak_distance)
        right_peaks, _ = find_peaks(np.abs(sync_right_short),
                                    height=peak_height,
                                    prominence=peak_prominence,
                                    distance=peak_distance)

        print(f"  find_peaks 結果 - 左: {left_peaks}, 右: {right_peaks}")

        if len(left_peaks) > 0 and len(right_peaks) > 0:
            peak_index_left = left_peaks[0]
            peak_index_right = right_peaks[0]
            lag_samples = peak_index_left - peak_index_right
            lag_ms = lag_samples * sampling_interval_ms
            print(f"  採用した左ピーク index: {peak_index_left}, 右ピーク index: {peak_index_right}")
            print(f"  計算されたラグ: {lag_samples} サンプル ({lag_ms:.2f} ms)")
        else:
            print("  警告: find_peaks で左右両方のピークを検出できませんでした。ラグは0とします。")
            lag_samples = 0 # ピークが見つからない場合はラグ0とする

    except Exception as e:
        print(f"エラー: find_peaks またはラグ計算中にエラーが発生しました: {e}")
        return error_return


    # --- ラグ適用 ---
    # (変更なし)
    try:
        # ... (ラグ適用ロジック - 省略) ...
        if lag_samples > 0: aligned_left_gyro, aligned_right_gyro = align_left[lag_samples:], align_right[:len(align_left[lag_samples:])]
        elif lag_samples < 0: aligned_right_gyro, aligned_left_gyro = align_right[abs(lag_samples):], align_left[:len(align_right[abs(lag_samples):])]
        else: min_len = min(len(align_left), len(align_right)); aligned_left_gyro, aligned_right_gyro = align_left[:min_len], align_right[:min_len]
        aligned_length = len(aligned_left_gyro)
        time_aligned = np.arange(aligned_length) * (sampling_interval_ms / 1000.0)
    except Exception as e: print(f"エラー: ラグ適用中にエラー: {e}"); return error_return

    # --- DataFrame作成 & return ---
    # (変更なし)
    try:
        sync_gyro_df = pd.DataFrame({'time_aligned_sec': time_aligned, f'L{align_col_suffix}_aligned': aligned_left_gyro, f'R{align_col_suffix}_aligned': aligned_right_gyro})
    except Exception as e: print(f"エラー: DataFrame作成中にエラー: {e}"); return error_return

    print("--- [Function@preprocessing] 1. 角速度データの前処理終了 ---")
    return sync_gyro_df, lag_samples, sampling_rate_hz, peak_index_left, peak_index_right, sync_left_short, sync_right_short