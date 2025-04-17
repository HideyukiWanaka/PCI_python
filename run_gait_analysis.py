from pathlib import Path
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd
from file_utils import find_latest_csv_file, save_results
from pci import calculate_pci             # Placeholder
from gait_cycles import identify_gait_cycles  # Placeholder
from preprocessing import preprocess_angular_velocity
# run_gait_analysis.py

# --- 各機能ファイルをインポート ---

# --- 必要な標準ライブラリ・外部ライブラリをインポート ---

# --- 解析実行のための設定 ---
# ★★★ これらの設定値は必要に応じて変更してください ★★★
DATA_FOLDER = Path("./")  # ★デフォルト: スクリプトと同じフォルダを検索
OUTPUT_SUFFIX_SYNC = '_synchronized_gyro_z.csv'
OUTPUT_SUFFIX_GAIT = '_gait_events.csv'     # (将来用)
OUTPUT_SUFFIX_PCI = '_pci_results.csv'   # (将来用)
ROWS_TO_SKIP = 11
SAMPLING_INTERVAL_MS = 5
SYNC_SIGNAL_SUFFIX = '_Acc_Y'
ALIGN_TARGET_SUFFIX = '_Gyro_Z'
RIGHT_PREFIX = 'R'
LEFT_PREFIX = 'L'
TRUNK_PREFIX = 'T'
# ★★★ ここまで設定値 ★★★

# --- 日本語フォント設定 ---
try:
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Hiragino Sans',
                                       'Yu Gothic', 'Meiryo', 'TakaoPGothic', 'Noto Sans CJK JP']
except Exception as e:
    print(f"警告: 日本語フォントの設定中にエラーが発生しました。エラー: {e}")

# --- メイン実行ブロック ---
if __name__ == "__main__":
    print("========================================")
    print("=== 歩行データ解析処理開始 (メイン) ===")
    print(f"データ検索フォルダ: {DATA_FOLDER.resolve()}")
    print("========================================")

    # === 準備: 最新の入力ファイルを検索 ===
    print("\n[準備] 最新のCSVデータを検索中...")
    # find_latest_csv_file を file_utils から呼び出す
    input_data_file_path = find_latest_csv_file(DATA_FOLDER)

    if input_data_file_path is None:
        print("\n[エラー] 処理可能な入力ファイルが見つからないため、処理を中断します。")
        exit()
    else:
        print(f"処理対象ファイル: '{input_data_file_path.name}'")

    # --- 入力ファイルに基づいて出力ファイル名を生成 ---
    base_filename = input_data_file_path.stem
    # 出力ファイルも DATA_FOLDER 内に保存する想定
    output_sync_file = DATA_FOLDER / (base_filename + OUTPUT_SUFFIX_SYNC)
    output_gait_file = DATA_FOLDER / (base_filename + OUTPUT_SUFFIX_GAIT)
    output_pci_file = DATA_FOLDER / (base_filename + OUTPUT_SUFFIX_PCI)

    # === ステップ 1: 角速度データの前処理 ===
    print("\n[ステップ1] 角速度データの前処理を実行中...")
    # ★★★ 変更点: 戻り値のアンパックを変更 (7つ受け取る) ★★★
    result_tuple = preprocess_angular_velocity(
        data_file=input_data_file_path,
        rows_to_skip=ROWS_TO_SKIP,
        sampling_interval_ms=SAMPLING_INTERVAL_MS,
        right_prefix=RIGHT_PREFIX,
        left_prefix=LEFT_PREFIX,
        trunk_prefix=TRUNK_PREFIX,
        sync_col_suffix=SYNC_SIGNAL_SUFFIX,
        align_col_suffix=ALIGN_TARGET_SUFFIX
    )

    # 戻り値の数を確認してアンパック
    if result_tuple is None or len(result_tuple) != 7:
         print("\n[エラー] 角速度の前処理に失敗しました (予期せぬ戻り値)。処理を中断します。")
         exit()
    # 変数名を変更
    sync_gyro_df, lag_samples, sampling_rate, peak_idx_l, peak_idx_r, sync_l_short, sync_r_short = result_tuple
    # ★★★ 変更ここまで ★★★

    if sync_gyro_df is None:
        print("\n[エラー] 角速度の前処理に失敗しました。処理を中断します。")
        exit()
    else:
        print(f"\n[成功] 前処理完了。")
        print(f"  同期ラグ: {lag_samples} サンプル")
        print(f"  サンプリング周波数: {sampling_rate:.2f} Hz")
        print(f"\n[ステップ1.1] 同期済みデータを '{output_sync_file.name}' に保存中...")
        # save_results を file_utils から呼び出す
        save_results(output_sync_file, sync_gyro_df, "同期済み角速度データ")

        # === ★★★ 診断プロット (ピーク検出用) ★★★ ===
        print("\n[診断] 同期用信号と検出されたピークをプロットします...")
        try:
            if sync_l_short is not None and sync_r_short is not None: # データがあるか確認
                plt.figure(figsize=(14, 7))
                actual_sync_len = len(sync_l_short) # 実際に使われた長さ
                time_short = np.arange(actual_sync_len) * (SAMPLING_INTERVAL_MS / 1000.0)
                plt.plot(time_short, sync_l_short, label=f'同期用 左信号 ({LEFT_PREFIX}{SYNC_SIGNAL_SUFFIX})', alpha=0.8)
                plt.plot(time_short, sync_r_short, label=f'同期用 右信号 ({RIGHT_PREFIX}{SYNC_SIGNAL_SUFFIX})', linestyle='--', alpha=0.8)

                # 検出されたピーク位置にマーカーをプロット (インデックスが有効範囲内か確認)
                if 0 <= peak_idx_l < actual_sync_len:
                     plt.plot(time_short[peak_idx_l], sync_l_short[peak_idx_l], 'ro', markersize=8, label=f'左ピーク (idx={peak_idx_l})')
                else:
                     print("  警告: 左ピークインデックスが無効です。")
                if 0 <= peak_idx_r < actual_sync_len:
                     plt.plot(time_short[peak_idx_r], sync_r_short[peak_idx_r], 'gx', markersize=10, markeredgewidth=2, label=f'右ピーク (idx={peak_idx_r})')
                else:
                     print("  警告: 右ピークインデックスが無効です。")

                plt.title(f'同期に使用した信号区間と検出ピーク ({SYNC_SIGNAL_SUFFIX})')
                plt.xlabel('時間 (s)')
                plt.ylabel('信号値 (単位?)') # '加速度'とは限らないので変更
                plt.legend()
                plt.grid(True)
                plt.show()
            else:
                print("  情報: プロット用の同期信号データがありません。")
        except Exception as e:
            print(f"  エラー: 診断グラフの表示中にエラーが発生しました: {e}")
        # === 診断プロットここまで ===

    # === ステップ 2: 歩行周期の同定 ===
    print("\n[ステップ2] 歩行周期の同定を実行中...")
    # identify_gait_cycles を gait_cycles から呼び出す
    gait_events_df = identify_gait_cycles(sync_gyro_df, sampling_rate)

    if gait_events_df is None:
        print("  (歩行周期の同定は未実装か、データがありませんでした)")
        print("\n[ステップ3] PCI計算はスキップされます (歩行周期データなし)")
        pci_results = None
    else:
        print(f"\n[ステップ2.1] 歩行周期データを '{output_gait_file.name}' に保存中...")
        # save_results を file_utils から呼び出す
        save_results(output_gait_file, gait_events_df, "歩行周期データ")

        # === ステップ 3: PCIの計算 ===
        print("\n[ステップ3] PCI計算を実行中...")
        # calculate_pci を pci から呼び出す
        pci_results = calculate_pci(gait_events_df)

        if pci_results is None:
            print("  (PCI計算は未実装か、データがありませんでした)")
        else:
            print(f"\n[ステップ3.1] PCI計算結果を '{output_pci_file.name}' に保存中...")
            try:
                pci_df = pd.DataFrame([pci_results])
                # save_results を file_utils から呼び出す
                save_results(output_pci_file, pci_df, "PCI計算結果")
                print("\n  PCI 計算結果 (コンソール表示):", pci_results)
            except Exception as e:
                print(f"  エラー: PCI結果のDataFrame変換または保存中にエラー: {e}")

    # === ステップ 4: 結果のプロット (同期済み角速度) ===
    # プロット処理はメインファイルに残すことが多い
    if sync_gyro_df is not None:  # 前処理が成功した場合のみプロット
        print("\n[ステップ4] 同期済みZ軸角速度のグラフを表示します...")
        try:
            plt.figure(figsize=(14, 7))
            left_col_name = f'L{ALIGN_TARGET_SUFFIX}_aligned'
            right_col_name = f'R{ALIGN_TARGET_SUFFIX}_aligned'
            if left_col_name in sync_gyro_df.columns and right_col_name in sync_gyro_df.columns:
                plt.plot(sync_gyro_df['time_aligned_sec'], sync_gyro_df[left_col_name],
                         label=f'同期後 左 {ALIGN_TARGET_SUFFIX} (符号反転済)')
                plt.plot(sync_gyro_df['time_aligned_sec'], sync_gyro_df[right_col_name],
                         label=f'同期後 右 {ALIGN_TARGET_SUFFIX}', linestyle='--', alpha=0.8)
                plt.title(
                    f'同期済み角速度 ({ALIGN_TARGET_SUFFIX}) - {input_data_file_path.name}')
                plt.xlabel('同期後の時間 (秒)')
                plt.ylabel('角速度 (単位?)')
                plt.legend()
                plt.grid(True)
                plt.show()
            else:
                print(
                    f"  警告: プロットに必要な列が見つかりません ({left_col_name} or {right_col_name})。")
        except Exception as e:
            print(f"  エラー: グラフの表示中にエラーが発生しました: {e}")

    print("\n========================================")
    print(
        f"=== 解析処理終了 ({input_data_file_path.name if input_data_file_path else 'エラー発生'}) ===")
    print("========================================")
    # input("エンターキーを押して終了します...") # 必要ならコメント解除
