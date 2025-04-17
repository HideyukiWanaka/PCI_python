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
        sync_left = df[f'{left_prefix}{sync_col_suffix}'].fillna(0).values
        sync_right = df[f'{right_prefix}{sync_col_suffix}'].fillna(0).values
        align_left_raw = df[f'{left_prefix}{align_col_suffix}'].fillna(
            0).values
        align_right = df[f'{right_prefix}{align_col_suffix}'].fillna(0).values
        align_left = -align_left_raw
        print(f"  情報: 左 {align_col_suffix} の符号を反転しました。")
    except KeyError as e:
        print(f"エラー: 必要な列名 '{e}' が見つかりません。")
        return None, None, None
    except Exception as e:
        print(f"エラー: 信号抽出中にエラーが発生しました: {e}")
        return None, None, None

    # --- 相互相関とラグ計算 ---
    try:
        correlation = signal.correlate(sync_right, sync_left, mode='full')
        lags = signal.correlation_lags(
            len(sync_right), len(sync_left), mode='full')
        peak_index = np.argmax(correlation)
        lag_samples = lags[peak_index]
    except Exception as e:
        print(f"エラー: 相互相関の計算中にエラーが発生しました: {e}")
        return None, None, None

    # --- ラグ適用 ---
    try:
        if lag_samples > 0:
            aligned_left_gyro = align_left[lag_samples:]
            aligned_right_gyro = align_right[:len(aligned_left_gyro)]
        elif lag_samples < 0:
            aligned_right_gyro = align_right[abs(lag_samples):]
            aligned_left_gyro = align_left[:len(aligned_right_gyro)]
        else:
            min_len = min(len(align_left), len(align_right))
            aligned_left_gyro = align_left[:min_len]
            aligned_right_gyro = align_right[:min_len]
        aligned_length = len(aligned_left_gyro)
        time_aligned = np.arange(aligned_length) * \
            (sampling_interval_ms / 1000.0)
    except Exception as e:
        print(f"エラー: ラグの適用中にエラーが発生しました: {e}")
        return None, None, None

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
    return sync_gyro_df, lag_samples, sampling_rate_hz
