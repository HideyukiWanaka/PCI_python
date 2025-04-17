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
            f'{trunk_prefix}_Gyro_X', f'{trunk_prefix}_Gyro_Y', f'{trunk_prefix}_Gyro_Z', 'Blank_4'
        ]

        if len(df.columns) == len(expected_column_names):
            df.columns = expected_column_names
            blank_cols = [col for col in df.columns if 'Blank_' in col]
            df = df.drop(columns=blank_cols)
            print(f"  ファイル '{data_file_path.name}' を読み込み、列名を修正しました。")
        else:
            print(f"エラー: 読み込んだ列数 ({len(df.columns)}) が期待される列数 ({len(expected_column_names)}) と一致しません。")
            return error_return

    except FileNotFoundError:
        print(f"エラー: データファイルが見つかりません: {data_file_path}")
        return error_return
    except UnicodeDecodeError as e:
        print(f"[エラー@preprocessing] ファイル '{data_file_path}' の読み込み中にエンコーディングエラーが発生しました。")
        print(f"  試したエンコーディング: 'cp932', 'shift_jis'")
        print(f"  エラー詳細: {e}")
        return error_return
    except Exception as e:
        print(f"エラー: データ読み込み・列名設定中に予期せぬエラーが発生しました: {e}")
        return error_return

    # --- 信号の抽出と左角速度の符号反転 ---
    sync_left_short, sync_right_short = None, None
    actual_sync_length = 0 # 初期化
    try:
        # 同期用信号（全体）
        sync_left_full = df[f'{left_prefix}{sync_col_suffix}'].fillna(0).values
        sync_right_full = df[f'{right_prefix}{sync_col_suffix}'].fillna(0).values
        # 同期適用対象信号（全体）
        align_left_raw = df[f'{left_prefix}{align_col_suffix}'].fillna(0).values
        align_right = df[f'{right_prefix}{align_col_suffix}'].fillna(0).values

        # 左角速度の符号を反転
        align_left = -align_left_raw
        print(f"  情報: 左 {align_col_suffix} の符号を反転しました。")

        # 同期計算用に最初の N サンプルを切り出す
        num_samples_for_sync = 1000 # この値は固定でOK (find_peaksで探す範囲)
        actual_sync_length = min(len(sync_left_full), len(sync_right_full), num_samples_for_sync)

        if actual_sync_length <= 0:
            raise ValueError("同期に使用できるデータがありません。")
        if actual_sync_length < num_samples_for_sync:
             print(f"  警告: 同期用信号の長さが {num_samples_for_sync} サンプル未満です。利用可能な先頭 {actual_sync_length} サンプルで同期します。")

        sync_left_short = sync_left_full[:actual_sync_length]
        sync_right_short = sync_right_full[:actual_sync_length]

    except KeyError as e:
        print(f"エラー: 必要な列名 '{e}' が見つかりません。列名設定を確認してください。")
        return error_return
    except ValueError as e:
        print(f"エラー: {e}")
        return error_return
    except Exception as e:
        print(f"エラー: 信号抽出中に予期せぬエラーが発生しました: {e}")
        return error_return

    # --- ピーク検出によるラグ計算 (find_peaks を使用) ---
    lag_samples = 0
    peak_index_left = -1
    peak_index_right = -1
    try:
        print(f"  情報: find_peaks (h={peak_height}, p={peak_prominence}, d={peak_distance}) を使用して先頭 {actual_sync_length} サンプルからピーク検出...")

        # find_peaks を絶対値信号に適用
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
            # 検出された「最初の」ピークを採用
            peak_index_left = left_peaks[0]
            peak_index_right = right_peaks[0]

            # ラグ計算: lag = left_peak_index - right_peak_index
            lag_samples = peak_index_left - peak_index_right
            lag_ms = lag_samples * sampling_interval_ms

            print(f"  採用した左ピーク index: {peak_index_left} (値: {sync_left_short[peak_index_left]:.2f})")
            print(f"  採用した右ピーク index: {peak_index_right} (値: {sync_right_short[peak_index_right]:.2f})")
            print(f"  計算されたラグ: {lag_samples} サンプル ({lag_ms:.2f} ms)")
        else:
            print("  警告: find_peaks で左右両方のピークを検出できませんでした。パラメータ調整が必要かもしれません。ラグは0とします。")
            lag_samples = 0 # ピークが見つからない場合はラグ0とする
            # peak_index_left, peak_index_right は -1 のまま

    except Exception as e:
        print(f"エラー: find_peaks またはラグ計算中にエラーが発生しました: {e}")
        return error_return # エラー時はNoneなどを返す

    # --- ラグ適用 (元の長い信号に適用) ---
    try:
        if lag_samples > 0:
            # 左が遅れている -> 左の先頭を削る
            aligned_left_gyro = align_left[lag_samples:]
            aligned_right_gyro = align_right[:len(aligned_left_gyro)] # 右の末尾を合わせる
        elif lag_samples < 0:
            # 右が遅れている -> 右の先頭を削る
            aligned_right_gyro = align_right[abs(lag_samples):]
            aligned_left_gyro = align_left[:len(aligned_right_gyro)] # 左の末尾を合わせる
        else: # lag_samples == 0
            # ラグがない場合は短い方に合わせる
            min_len = min(len(align_left), len(align_right))
            aligned_left_gyro = align_left[:min_len]
            aligned_right_gyro = align_right[:min_len]

        aligned_length = len(aligned_left_gyro)
        time_aligned = np.arange(aligned_length) * (sampling_interval_ms / 1000.0)

    except Exception as e:
        print(f"エラー: ラグの適用中にエラーが発生しました: {e}")
        return error_return

    # --- 同期済みデータをDataFrameにまとめる ---
    try:
        sync_gyro_df = pd.DataFrame({
            'time_aligned_sec': time_aligned,
            f'L{align_col_suffix}_aligned': aligned_left_gyro,
            f'R{align_col_suffix}_aligned': aligned_right_gyro
        })
    except Exception as e:
        print(f"エラー: 結果のDataFrame作成中にエラーが発生しました: {e}")
        return error_return

    print("--- [Function@preprocessing] 1. 角速度データの前処理終了 ---")
    # 戻り値の構成は同じ: df, lag, fs, peak_l, peak_r, sync_l_short, sync_r_short
    return sync_gyro_df, lag_samples, sampling_rate_hz, peak_index_left, peak_index_right, sync_left_short, sync_right_short