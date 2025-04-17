# preprocessing.py
import pandas as pd
import numpy as np
from scipy import signal
from pathlib import Path  # Pathオブジェクトを扱う可能性を考慮


def preprocess_angular_velocity(data_file, rows_to_skip=11, sampling_interval_ms=5,
                                right_prefix='R', left_prefix='L', trunk_prefix='T',
                                sync_col_suffix='_Acc_Y', align_col_suffix='_Gyro_Z'):
    """
    CSVデータファイルを読み込み、列名を整理し、指定された信号で同期を行い、
    ターゲット信号（角速度）を同期して返す。左角速度の符号反転も行う。

    (中略 - gait_analysis_functions.py 内の preprocess_angular_velocity と同じ実装)

    Returns:
        tuple: (pd.DataFrame, float, float)
            - sync_gyro_df: 同期済みの角速度データを含むDataFrame
            - lag_samples: 計算されたラグ (サンプル数)
            - sampling_rate_hz: サンプリング周波数 (Hz)
            エラー時は (None, None, None)
    """
    print("--- [Function@preprocessing] 1. 角速度データの前処理開始 ---")
    sampling_rate_hz = 1000.0 / sampling_interval_ms

    # --- データの読み込みと列名設定 ---
    try:
        data_file_path = Path(data_file) if not isinstance(
            data_file, Path) else data_file
        try:
            # まず 'cp932' (WindowsのShift_JIS系) を試す
            df = pd.read_csv(
                data_file_path, skiprows=rows_to_skip, encoding='cp932')
            print(f"  ファイル '{data_file_path}' を encoding='cp932' で読み込みました。")
        except UnicodeDecodeError:
            print(f"  encoding='cp932' で失敗。encoding='shift_jis' を試します...")
            # 'cp932' がダメなら 'shift_jis' を試す
            df = pd.read_csv(
                data_file_path, skiprows=rows_to_skip, encoding='shift_jis')
            print(
                f"  ファイル '{data_file_path}' を encoding='shift_jis' で読み込みました。")

        new_column_names = [
            'Time',
            f'{right_prefix}_Acc_X', f'{right_prefix}_Acc_Y', f'{right_prefix}_Acc_Z',
            f'{right_prefix}_Ch4_V',
            f'{right_prefix}_Gyro_X', f'{right_prefix}_Gyro_Y', f'{right_prefix}_Gyro_Z',
            f'{right_prefix}_Ch8_V',
            f'{left_prefix}_Acc_X', f'{left_prefix}_Acc_Y', f'{left_prefix}_Acc_Z',
            'Blank_1',
            f'{left_prefix}_Gyro_X', f'{left_prefix}_Gyro_Y', f'{left_prefix}_Gyro_Z',
            'Blank_2',
            f'{trunk_prefix}_Acc_X', f'{trunk_prefix}_Acc_Y', f'{trunk_prefix}_Acc_Z',
            'Blank_3',
            f'{trunk_prefix}_Gyro_X', f'{trunk_prefix}_Gyro_Y', f'{trunk_prefix}_Gyro_Z',
            'Blank_4',
            'Blank_5', 'Blank_6', 'Blank_7', 'Blank_8',
            'Blank_9', 'Blank_10', 'Blank_11', 'Blank_12'
        ]

        if len(df.columns) == len(new_column_names):
            df.columns = new_column_names
            blank_cols = [col for col in df.columns if 'Blank_' in col]
            df = df.drop(columns=blank_cols)
        else:
            print(
                f"エラー: 読み込んだ列数 ({len(df.columns)}) が期待される列数 ({len(new_column_names)}) と一致しません。")
            return None, None, None

    except FileNotFoundError:
        print(f"エラー: データファイルが見つかりません: {data_file_path}")
        return None, None, None
    except UnicodeDecodeError as e:
        # 上記の try/except でも読み込めなかった場合の最終的なエラー表示
        print(
            f"[エラー@preprocessing] ファイル '{data_file_path}' の読み込み中にエンコーディングエラーが発生しました。")
        print(f"  試したエンコーディング: 'cp932', 'shift_jis'")
        print(f"  エラー詳細: {e}")
        print(f"  ファイルのエンコーディングが上記以外（例: 'euc_jp'）であるか、ファイルが破損している可能性があります。")
        return None, None, None
    except Exception as e:
        print(f"エラー: データ読み込み・列名設定中にエラーが発生しました: {e}")
        return None, None, None

    # --- 信号の抽出と左角速度の符号反転 ---
    try:
        sync_left_full = df[f'{left_prefix}{sync_col_suffix}'].fillna(0).values
        sync_right_full = df[f'{right_prefix}{sync_col_suffix}'].fillna(0).values
        align_left_raw = df[f'{left_prefix}{align_col_suffix}'].fillna(0).values
        align_right = df[f'{right_prefix}{align_col_suffix}'].fillna(0).values
        align_left = -align_left_raw
        print(f"  情報: 左 {align_col_suffix} の符号を反転しました。")
        
                # ★★★ 変更点: 同期計算用に最初の1000サンプルを切り出す ★★★
        num_samples_for_sync = 1000
        # データが1000サンプルより短い場合の対応
        actual_sync_length = min(len(sync_left_full), len(sync_right_full), num_samples_for_sync)
        if actual_sync_length < num_samples_for_sync:
            print(f"  警告: 同期用信号の長さが {num_samples_for_sync} サンプル未満です。利用可能な先頭 {actual_sync_length} サンプルで同期します。")

        sync_left_short = sync_left_full[:actual_sync_length]
        sync_right_short = sync_right_full[:actual_sync_length]
        print(f"  情報: 同期計算には最初の {actual_sync_length} サンプルを使用します。")
        # ★★★ 変更ここまで ★★★
        
    except KeyError as e:
        print(f"エラー: 必要な列名 '{e}' が見つかりません。")
        return None, None, None
    except Exception as e:
        print(f"エラー: 信号抽出中にエラーが発生しました: {e}")
        return None, None, None

    # --- ★★★ ピーク検出によるラグ計算 ★★★ ---
    lag_samples = 0
    peak_index_left = -1
    peak_index_right = -1
    try:
        if actual_sync_length > 0:
            print(f"  情報: ピーク検出による同期には最初の {actual_sync_length} サンプルを使用します。")
            # 絶対値が最大となる点のインデックスを検出
            peak_index_left = np.argmax(np.abs(sync_left_short))
            peak_index_right = np.argmax(np.abs(sync_right_short))

            # ラグ計算: lag = left_peak_index - right_peak_index
            lag_samples = peak_index_left - peak_index_right
            lag_ms = lag_samples * sampling_interval_ms

            print(f"  同期基準: {sync_col_suffix} (先頭{actual_sync_length}サンプルのピーク検出)")
            print(f"  左ピーク index: {peak_index_left} (値: {sync_left_short[peak_index_left]:.2f})")
            print(f"  右ピーク index: {peak_index_right} (値: {sync_right_short[peak_index_right]:.2f})")
            print(f"  計算されたラグ: {lag_samples} サンプル ({lag_ms:.2f} ms)")
        else:
            print("  警告: 同期に使用できるデータがありません。ラグは0とします。")

    except Exception as e:
        print(f"エラー: ピーク検出またはラグ計算中にエラーが発生しました: {e}")
        # エラーが起きても、ラグ0で処理を試みるか、エラーを返すか選択できる
        # ここではエラーを返す
        return error_return
    # --- ★★★ ピーク検出 終了 ★★★ ---
    # --- ラグ適用 ---
    try:
         # ★★★ 注意: ラグ適用は元の長い信号 (align_left, align_right) に対して行う ★★★
        if lag_samples > 0:
            aligned_left_gyro = align_left[lag_samples:]
            aligned_right_gyro = align_right[:len(aligned_left_gyro)]
        elif lag_samples < 0:
            aligned_right_gyro = align_right[abs(lag_samples):]
            aligned_left_gyro = align_left[:len(aligned_right_gyro)]
        else: # lag_samples == 0
            min_len = min(len(align_left), len(align_right))
            aligned_left_gyro = align_left[:min_len]
            aligned_right_gyro = align_right[:min_len]

        aligned_length = len(aligned_left_gyro)
        time_aligned = np.arange(aligned_length) * (sampling_interval_ms / 1000.0)
        # ★★★ 変更ここまで (ラグ適用ロジック自体に変更はないが、適用するラグ値が変わった) ★★★
        
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
        return None, None, None

    print("--- [Function@preprocessing] 1. 角速度データの前処理終了 ---")
    return sync_gyro_df, lag_samples, sampling_rate_hz, peak_index_left, peak_index_right, sync_left_short, sync_right_short
