# run_gait_analysis.py
# --- 各機能ファイルをインポート ---
from preprocessing import preprocess_angular_velocity
from gait_cycles import identify_gait_cycles # Placeholder
from pci import calculate_pci             # Placeholder
from file_utils import find_latest_csv_file, save_results

# --- 必要な標準ライブラリ・外部ライブラリをインポート ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path

# --- 解析実行のための設定 ---
DATA_FOLDER = Path("./")
OUTPUT_SUFFIX_SYNC = '_synchronized_gyro_z.csv'
OUTPUT_SUFFIX_GAIT = '_gait_events.csv'
OUTPUT_SUFFIX_PCI = '_pci_results.csv'
ROWS_TO_SKIP = 11
SAMPLING_INTERVAL_MS = 5
SYNC_SIGNAL_SUFFIX = '_Acc_Y' # 同期に使う信号
ALIGN_TARGET_SUFFIX = '_Gyro_Z'
RIGHT_PREFIX = 'R'
LEFT_PREFIX = 'L'
TRUNK_PREFIX = 'T'
# ★ デフォルトの find_peaks パラメータ ★
DEFAULT_PEAK_HEIGHT = 0.5
DEFAULT_PEAK_PROMINENCE = 0.3
DEFAULT_PEAK_DISTANCE = 50
NUM_SAMPLES_TO_PLOT = 1000 # 初期プロットのサンプル数

# --- 日本語フォント設定 ---
try:
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Hiragino Sans', 'Yu Gothic', 'Meiryo', 'TakaoPGothic', 'Noto Sans CJK JP']
except Exception as e: print(f"警告: 日本語フォント設定エラー: {e}")


# --- ★ ユーザーからパラメータ入力を受け取るヘルパー関数 ★ ---
def get_parameter_input(prompt, default_value):
    """指定されたプロンプトでユーザー入力を求め、数値に変換して返す。Enterのみならデフォルト値。"""
    while True:
        try:
            # input() は文字列を返す
            user_input_str = input(f"  {prompt} (デフォルト: {default_value}): ")
            if not user_input_str.strip(): # 空文字かチェック (Enterのみ)
                print(f"    -> デフォルト値 {default_value} を使用します。")
                # デフォルト値が None でないか、または float/int に変換可能か確認
                if default_value is None:
                    return None
                # distance は整数にする
                if "distance" in prompt.lower():
                    return int(default_value)
                else:
                    return float(default_value)

            value = float(user_input_str) # まずfloatに変換試行
            if value < 0:
                 print("  エラー: 値は0以上である必要があります。")
                 continue
            # distance の場合は整数にする
            if "distance" in prompt.lower():
                if value != int(value) or value <= 0: # 整数でない、または0以下
                    print("  エラー: distance は1以上の整数である必要があります。")
                    continue
                return int(value)
            return value # height, prominence は float でOK
        except ValueError:
            print("  エラー: 有効な数値を入力してください。")
        except Exception as e:
            print(f"  予期せぬ入力エラー: {e}")

# --- 列名定義 (preprocessing.py と共通化が必要だが、ここではコピー) ---
# 本来は別ファイルか設定で共通化すべき
def get_expected_column_names(right_prefix, left_prefix, trunk_prefix):
     return [
            'Time', f'{right_prefix}_Acc_X', f'{right_prefix}_Acc_Y', f'{right_prefix}_Acc_Z', f'{right_prefix}_Ch4_V',
            f'{right_prefix}_Gyro_X', f'{right_prefix}_Gyro_Y', f'{right_prefix}_Gyro_Z', f'{right_prefix}_Ch8_V',
            f'{left_prefix}_Acc_X', f'{left_prefix}_Acc_Y', f'{left_prefix}_Acc_Z', 'Blank_1',
            f'{left_prefix}_Gyro_X', f'{left_prefix}_Gyro_Y', f'{left_prefix}_Gyro_Z', 'Blank_2',
            f'{trunk_prefix}_Acc_X', f'{trunk_prefix}_Acc_Y', f'{trunk_prefix}_Acc_Z', 'Blank_3',
            f'{trunk_prefix}_Gyro_X', f'{trunk_prefix}_Gyro_Y', f'{trunk_prefix}_Gyro_Z', 'Blank_4']

# --- メイン実行ブロック ---
if __name__ == "__main__":
    print("========================================")
    print("=== 歩行データ解析処理開始 (インタラクティブ) ===")
    print(f"データ検索フォルダ: {DATA_FOLDER.resolve()}")
    print("========================================")

    print("\n[準備] 最新のCSVデータを検索中...")
    input_data_file_path = find_latest_csv_file(DATA_FOLDER)
    if input_data_file_path is None: exit()
    print(f"処理対象ファイル: '{input_data_file_path.name}'")

    base_filename = input_data_file_path.stem
    output_sync_file = DATA_FOLDER / (base_filename + OUTPUT_SUFFIX_SYNC)
    output_gait_file = DATA_FOLDER / (base_filename + OUTPUT_SUFFIX_GAIT)
    output_pci_file = DATA_FOLDER / (base_filename + OUTPUT_SUFFIX_PCI)

    # === ★ ステップ 0: 初期データの表示 ★ ===
    print(f"\n[ステップ0] 同期用信号 ({SYNC_SIGNAL_SUFFIX}) の先頭 {NUM_SAMPLES_TO_PLOT} サンプルをプロットします...")
    sync_l_short_plot, sync_r_short_plot = None, None
    plot_len = 0
    try:
        # --- 初期プロット用にデータを一時読み込み ---
        # (preprocessing.py と同じ読み込み・列名設定ロジックが必要)
        temp_df = None
        try: temp_df = pd.read_csv(input_data_file_path, skiprows=ROWS_TO_SKIP, encoding='cp932')
        except UnicodeDecodeError: temp_df = pd.read_csv(input_data_file_path, skiprows=ROWS_TO_SKIP, encoding='shift_jis')

        expected_cols = get_expected_column_names(RIGHT_PREFIX, LEFT_PREFIX, TRUNK_PREFIX)
        if len(temp_df.columns) == len(expected_cols):
            temp_df.columns = expected_cols
            blank_cols = [col for col in temp_df.columns if 'Blank_' in col]; temp_df = temp_df.drop(columns=blank_cols)
        else: raise ValueError(f"初期読み込みで列数が不一致: {len(temp_df.columns)} vs {len(expected_cols)}")
        # --- 読み込み・列名設定 完了 ---

        sync_l_full = temp_df[f'{LEFT_PREFIX}{SYNC_SIGNAL_SUFFIX}'].fillna(0).values
        sync_r_full = temp_df[f'{RIGHT_PREFIX}{SYNC_SIGNAL_SUFFIX}'].fillna(0).values
        plot_len = min(len(sync_l_full), len(sync_r_full), NUM_SAMPLES_TO_PLOT)
        if plot_len <= 0 : raise ValueError("プロットするデータがありません。")
        sync_l_short_plot = sync_l_full[:plot_len]
        sync_r_short_plot = sync_r_full[:plot_len]

        plt.figure(figsize=(14, 7))
        time_plot = np.arange(plot_len) * (SAMPLING_INTERVAL_MS / 1000.0)
        plt.plot(time_plot, sync_l_short_plot, label=f'同期用 左信号 ({LEFT_PREFIX}{SYNC_SIGNAL_SUFFIX})', alpha=0.8)
        plt.plot(time_plot, sync_r_short_plot, label=f'同期用 右信号 ({RIGHT_PREFIX}{SYNC_SIGNAL_SUFFIX})', linestyle='--', alpha=0.8)
        plt.title(f'同期用信号の先頭 {plot_len} サンプル - パラメータ入力のために確認してください')
        plt.xlabel('時間 (s)')
        plt.ylabel('信号値 (単位?)')
        plt.legend()
        plt.grid(True)
        print("  グラフウィンドウを表示します。内容を確認し、パラメータをメモして、ウィンドウを閉じてください...")
        plt.show() # ★★★ 実行はここで一時停止 ★★★

    except Exception as e:
        print(f"  [エラー] 初期データのプロット中にエラーが発生しました: {e}")
        exit()

    # === ★ ステップ 0.5: ユーザーからパラメータ入力 ★ ===
    print(f"\n[ステップ0.5] find_peaks のパラメータを入力してください (信号: {SYNC_SIGNAL_SUFFIX})")
    # None をデフォルト値として渡すことも可能 (find_peaks は None を無視する)
    # user_peak_height = get_parameter_input("  Minimum Peak Height (高さ)", None)
    user_peak_height = get_parameter_input("  Minimum Peak Height (高さ)", DEFAULT_PEAK_HEIGHT)
    user_peak_prominence = get_parameter_input("  Minimum Peak Prominence (突出度)", DEFAULT_PEAK_PROMINENCE)
    user_peak_distance = get_parameter_input("  Minimum Peak Distance (サンプル間隔)", DEFAULT_PEAK_DISTANCE)
    print(f"  使用するパラメータ: height={user_peak_height}, prominence={user_peak_prominence}, distance={user_peak_distance}")


    # === ステップ 1: 角速度データの前処理 ===
    print("\n[ステップ1] 角速度データの前処理 (入力パラメータ使用) を実行中...")
    result_tuple = preprocess_angular_velocity(
        data_file=input_data_file_path,
        rows_to_skip=ROWS_TO_SKIP,
        sampling_interval_ms=SAMPLING_INTERVAL_MS,
        right_prefix=RIGHT_PREFIX,
        left_prefix=LEFT_PREFIX,
        trunk_prefix=TRUNK_PREFIX,
        sync_col_suffix=SYNC_SIGNAL_SUFFIX,
        align_col_suffix=ALIGN_TARGET_SUFFIX,
        # ★★★ ユーザーが入力したパラメータを渡す ★★★
        peak_height=user_peak_height,
        peak_prominence=user_peak_prominence,
        peak_distance=user_peak_distance
    )

    # --- 結果の処理 ---
    if result_tuple is None or len(result_tuple) != 7:
         print("\n[エラー] 角速度の前処理に失敗しました (予期せぬ戻り値)。処理を中断します。")
         exit()
    # 診断プロット用の sync_short は不要なので捨てる
    sync_gyro_df, lag_samples, sampling_rate, peak_idx_l, peak_idx_r, _, _ = result_tuple

    if sync_gyro_df is None:
         print("\n[エラー] 角速度の前処理に失敗しました。処理を中断します。")
         exit()
    else:
         print(f"\n[成功] 前処理完了。")
         print(f"  左ピーク index: {peak_idx_l}, 右ピーク index: {peak_idx_r}")
         print(f"  同期ラグ: {lag_samples} サンプル")
         print(f"  サンプリング周波数: {sampling_rate:.2f} Hz")
         print(f"\n[ステップ1.1] 同期済みデータを '{output_sync_file.name}' に保存中...")
         save_results(output_sync_file, sync_gyro_df, "同期済み角速度データ")

    # === ステップ 2, 3 (Gait Cycles, PCI - Placeholders) ===
    # (変更なし)
    print("\n[ステップ2] 歩行周期の同定を実行中...")
    gait_events_df = identify_gait_cycles(sync_gyro_df, sampling_rate)
    if gait_events_df is None:
        print("  (歩行周期の同定は未実装か、データがありませんでした)")
        print("\n[ステップ3] PCI計算はスキップされます (歩行周期データなし)")
    else:
        # ... (将来の処理) ...
        pass

    # === ステップ 4: 最終結果プロット (同期済み角速度) ===
    # ★★★ 診断プロットは削除 (ステップ0で表示済み) ★★★
    if sync_gyro_df is not None:
         print("\n[ステップ4] 同期済みZ軸角速度のグラフを表示します...")
         try:
             plt.figure(figsize=(14, 7))
             left_col_name = f'L{ALIGN_TARGET_SUFFIX}_aligned'
             right_col_name = f'R{ALIGN_TARGET_SUFFIX}_aligned'
             if left_col_name in sync_gyro_df.columns and right_col_name in sync_gyro_df.columns:
                plt.plot(sync_gyro_df['time_aligned_sec'], sync_gyro_df[left_col_name], label=f'同期後 左 {ALIGN_TARGET_SUFFIX} (符号反転済)')
                plt.plot(sync_gyro_df['time_aligned_sec'], sync_gyro_df[right_col_name], label=f'同期後 右 {ALIGN_TARGET_SUFFIX}', linestyle='--', alpha=0.8)
                plt.title(f'同期済み角速度 ({ALIGN_TARGET_SUFFIX}) - {input_data_file_path.name}')
                plt.xlabel('同期後の時間 (秒)')
                plt.ylabel('角速度 (単位?)')
                plt.legend()
                plt.grid(True)
                plt.show() # 最後のグラフを表示
             else: print(f"  警告: プロットに必要な列が見つかりません。")
         except Exception as e: print(f"  エラー: グラフの表示中にエラー: {e}")

    print("\n========================================")
    print(f"=== 解析処理終了 ({input_data_file_path.name}) ===")
    print("========================================")