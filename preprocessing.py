# preprocessing.py
import pandas as pd
import numpy as np
from scipy.signal import find_peaks # find_peaks を使うので import
from pathlib import Path

def preprocess_angular_velocity(data_file, rows_to_skip=11, sampling_interval_ms=5,
                                right_prefix='R', left_prefix='L', trunk_prefix='T',
                                sync_col_suffix='_Acc_Y', align_col_suffix='_Gyro_Z',
                                peak_height=None, peak_prominence=None, peak_distance=None): # find_peaksパラメータを引数に追加
    """
    CSVデータファイルを読み込み、列名を整理し、指定された信号とfind_peaksパラメータで同期を行い、
    ターゲット信号（角速度）を同期して返す。左角速度の符号反転も行う。

    Args:
        data_file (str or Path): 入力CSVファイルパス
        rows_to_skip (int): スキップする先頭行数
        sampling_interval_ms (float): サンプリング間隔 (ミリ秒)
        right_prefix (str): 右センサーの列名プレフィックス
        left_prefix (str): 左センサーの列名プレフィックス
        trunk_prefix (str): 体幹センサーの列名プレフィックス
        sync_col_suffix (str): 同期基準にする信号の列名サフィックス
        align_col_suffix (str): 同期を適用する信号の列名サフィックス
        peak_height (float, optional): find_peaksのheightパラメータ. Defaults to None.
        peak_prominence (float, optional): find_peaksのprominenceパラメータ. Defaults to None.
        peak_distance (int, optional): find_peaksのdistanceパラメータ. Defaults to None.

    Returns:
        tuple: (pd.DataFrame, float, float, int, int, np.ndarray, np.ndarray)
            - sync_gyro_df: 同期済みの角速度データ DataFrame (Noneならエラー)
            - lag_samples: 計算されたラグ (サンプル数)
            - sampling_rate_hz: サンプリング周波数 (Hz)
            - peak_index_left: 検出された左ピークのインデックス (-1なら失敗/未検出)
            - peak_index_right: 検出された右ピークのインデックス (-1なら失敗/未検出)
            - sync_left_short: 同期に使った左信号の先頭部分 (プロット/デバッグ用)
            - sync_right_short: 同期に使った右信号の先頭部分 (プロット/デバッグ用)
            エラー時は (None, 0, 0, -1, -1, None, None) のような値を返す想定
    """
    print(f"--- [Function@preprocessing] 1. 角速度データの前処理開始 (find_peaks: h={peak_height}, p={peak_prominence}, d={peak_distance}) ---")
    sampling_rate_hz = 1000.0 / sampling_interval_ms
    # エラー発生時のデフォルト戻り値
    error_return = (None, 0, sampling_rate_hz, -1, -1, None, None)

    # --- データの読み込みと列名設定 ---
    try:
        data_file_path = Path(data_file) if not isinstance(data_file, Path) else data_file
        try:
            # encoding='cp932' (または 'shift_jis') を指定
            df = pd.read_csv(data_file_path, skiprows=rows_to_skip, encoding='cp932')
        except UnicodeDecodeError:
            print(f"  encoding='cp932' で失敗。encoding='shift_jis' を試します...")
            df = pd.read_csv(data_file_path, skiprows=rows_to_skip, encoding='shift_jis')

        # 期待される列名のリスト (run_gait_analysis.py と共通化が必要)
        expected_column_names = [
            'Time',
            f'{right_prefix}_Acc_X', f'{right_prefix}_Acc_Y', f'{right_prefix}_Acc_Z', f'{right_prefix}_Ch4_V',
            f'{right_prefix}_Gyro_X', f'{right_prefix}_Gyro_Y', f'{right_prefix}_Gyro_Z', f'{right_prefix}_Ch8_V',
            f'{left_prefix}_Acc_X', f'{left_prefix}_Acc_Y', f'{left_prefix}_Acc_Z', 'Blank_1',
            f'{left_prefix}_Gyro_X', f'{left_prefix}_Gyro_Y', f'{left_prefix}_Gyro_Z', 'Blank_2',
            f'{trunk_prefix}_Acc_X', f'{trunk_prefix}_Acc_Y', f'{trunk_prefix}_Acc_Z', 'Blank_3',
            f'{trunk_prefix}_Gyro_X', f'{trunk_prefix}_Gyro_Y', f'{trunk_prefix}_Gyro_Z', 'Blank_4',
            'Blank_5', 'Blank_6', 'Blank_7', 'Blank_8', 'Blank_9', 'Blank_10', 'Blank_11', 'Blank_12'
        ]

        if len(df.colu# preprocessing.py
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from pathlib import Path

# 列名リストを返すヘルパー関数 (run_gait_analysis.pyと共通化推奨)
def get_expected_column_names(right_prefix, left_prefix, trunk_prefix):
     return [
            'Time', f'{right_prefix}_Acc_X', f'{right_prefix}_Acc_Y', f'{right_prefix}_Acc_Z', f'{right_prefix}_Ch4_V',
            f'{right_prefix}_Gyro_X', f'{right_prefix}_Gyro_Y', f'{right_prefix}_Gyro_Z', f'{right_prefix}_Ch8_V',
            f'{left_prefix}_Acc_X', f'{left_prefix}_Acc_Y', f'{left_prefix}_Acc_Z', 'Blank_1',
            f'{left_prefix}_Gyro_X', f'{left_prefix}_Gyro_Y', f'{left_prefix}_Gyro_Z', 'Blank_2',
            f'{trunk_prefix}_Acc_X', f'{trunk_prefix}_Acc_Y', f'{trunk_prefix}_Acc_Z', 'Blank_3',
            f'{trunk_prefix}_Gyro_X', f'{trunk_prefix}_Gyro_Y', f'{trunk_prefix}_Gyro_Z', 'Blank_4',
            'Blank_5', 'Blank_6', 'Blank_7', 'Blank_8', 'Blank_9', 'Blank_10', 'Blank_11', 'Blank_12' ]

def preprocess_angular_velocity(data_file, rows_to_skip=11, sampling_interval_ms=5,
                                right_prefix='R', left_prefix='L', trunk_prefix='T',
                                sync_col_suffix='_Acc_Y', align_col_suffix='_Gyro_Z',
                                peak_height=None, peak_prominence=None, peak_distance=None):
    """
    CSV読込、同期、同期済み角速度(Gyro Z) DataFrameを返す。(加速度は返さない)
    Returns: tuple: (sync_gyro_df, lag_samples, sampling_rate_hz, peak_idx_l, peak_idx_r, sync_l_short, sync_r_short)
    """
    print(f"--- [Function@preprocessing] 1. 角速度データの前処理開始 (find_peaks: h={peak_height}, p={peak_prominence}, d={peak_distance}) ---")
    sampling_rate_hz = 1000.0 / sampling_interval_ms
    error_return = (None, 0, sampling_rate_hz, -1, -1, None, None)

    # --- データ読み込みと列名設定 ---
    try:
        data_file_path = Path(data_file) if not isinstance(data_file, Path) else data_file
        try: df = pd.read_csv(data_file_path, skiprows=rows_to_skip, encoding='cp932')
        except UnicodeDecodeError: df = pd.read_csv(data_file_path, skiprows=rows_to_skip, encoding='shift_jis')
        expected_column_names = get_expected_column_names(right_prefix, left_prefix, trunk_prefix)
        if len(df.columns) == len(expected_column_names):
            df.columns = expected_column_names; df = df.drop(columns=[col for col in df.columns if 'Blank_' in col])
            print(f"  ファイル '{data_file_path.name}' 読込・列名修正完了。")
        else: print(f"エラー: 列数不一致"); return error_return
    except Exception as e: print(f"エラー: データ読み込み/列名設定エラー: {e}"); return error_return

    # --- 同期用信号抽出とラグ計算 ---
    sync_left_short, sync_right_short = None, None; actual_sync_length = 0
    lag_samples = 0; peak_index_left = -1; peak_index_right = -1
    try:
        sync_left_full = df[f'{left_prefix}{sync_col_suffix}'].fillna(0).values
        sync_right_full = df[f'{right_prefix}{sync_col_suffix}'].fillna(0).values
        num_samples_for_sync = 1000
        actual_sync_length = min(len(sync_left_full), len(sync_right_full), num_samples_for_sync)
        if actual_sync_length <= 0: raise ValueError("同期用データなし")
        sync_left_short = sync_left_full[:actual_sync_length]; sync_right_short = sync_right_full[:actual_sync_length]
        left_peaks, _ = find_peaks(np.abs(sync_left_short), height=peak_height, prominence=peak_prominence, distance=peak_distance)
        right_peaks, _ = find_peaks(np.abs(sync_right_short), height=peak_height, prominence=peak_prominence, distance=peak_distance)
        if len(left_peaks) > 0 and len(right_peaks) > 0:
            peak_index_left = left_peaks[0]; peak_index_right = right_peaks[0]; lag_samples = peak_index_left - peak_index_right
            print(f"  計算されたラグ: {lag_samples} サンプル")
        else: print("  警告: 同期ピーク検出不可。ラグ0とします。"); lag_samples = 0
    except Exception as e: print(f"エラー: 同期ラグ計算エラー: {e}"); return error_return

    # --- ターゲット信号(GyroZ)抽出とラグ適用 ---
    sync_gyro_df = None # 初期化
    try:
        align_left_full = df[f'{left_prefix}{align_col_suffix}'].fillna(0).values
        align_right_full = df[f'{right_prefix}{align_col_suffix}'].fillna(0).values
        align_left = -align_left_full # 左角速度の符号を反転
        align_right = align_right_full

        if lag_samples > 0: aligned_left_gyro, aligned_right_gyro = align_left[lag_samples:], align_right[:len(align_left[lag_samples:])]
        elif lag_samples < 0: aligned_right_gyro, aligned_left_gyro = align_right[abs(lag_samples):], align_left[:len(align_right[abs(lag_samples):])]
        else: min_len = min(len(align_left), len(align_right)); aligned_left_gyro, aligned_right_gyro = align_left[:min_len], align_right[:min_len]
        aligned_length = len(aligned_left_gyro); time_aligned = np.arange(aligned_length) * (sampling_interval_ms / 1000.0)
        if aligned_length <= 0: raise ValueError("アライメント後のデータ長が0以下")

        sync_gyro_df = pd.DataFrame({ # ★ Gyro Z のみ含む DF を作成 ★
            'time_aligned_sec': time_aligned,
            f'L{align_col_suffix}_aligned': aligned_left_gyro,
            f'R{align_col_suffix}_aligned': aligned_right_gyro
        })
    except Exception as e: print(f"エラー: ラグ適用/DataFrame作成エラー: {e}"); return error_return

    print("--- [Function@preprocessing] 1. 角速度データの前処理終了 ---")
    # ★ 戻り値を元の7要素タプルに戻す ★
    return sync_gyro_df, lag_samples, sampling_rate_hz, peak_index_left, peak_index_right, sync_left_short, sync_right_short