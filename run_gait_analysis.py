# run_gait_analysis.py

# --- 各機能ファイルをインポート ---
from preprocessing import preprocess_angular_velocity
from gait_cycles import identify_gait_cycles
from pci import calculate_pci             # Placeholder
from file_utils import find_latest_csv_file, save_results

# --- 必要な標準ライブラリ・外部ライブラリをインポート ---
import pandas as pd
import numpy as np
from pathlib import Path

# --- GUIとプロットのためのインポート ---
import tkinter as tk
from tkinter import Frame, Label, Button, BOTH, W, LEFT, messagebox, HORIZONTAL, Scale, Toplevel, DISABLED, NORMAL
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.font_manager as fm

# --- 日本語フォント設定 (japanize-matplotlib推奨) ---
try:
    import japanize_matplotlib
    print("japanize_matplotlib をインポートしました。")
except ImportError:
    print("警告: japanize-matplotlib が見つかりません。'pip install japanize-matplotlib' でインストールしてください。")
    try:
        jp_font_name = 'Hiragino Sans' # 環境に合わせて変更
        plt.rcParams['font.family'] = jp_font_name
        print(f"フォールバック: 日本語フォントとして '{jp_font_name}' を試みます。")
    except Exception as e_font:
        print(f"警告: フォールバックの日本語フォント設定エラー: {e_font}")

# --- 解析実行のための設定 ---
DATA_FOLDER = Path("./")
# ★ 出力ファイル名変更 ★
OUTPUT_SUFFIX_SYNC = '_synchronized_gyro_z.csv' # 同期済み角速度データの接尾辞
OUTPUT_SUFFIX_GAIT_TRIMMED = '_gait_events_steady.csv'   # 前後除外後の定常部分
OUTPUT_SUFFIX_PCI = '_pci_results.csv'
# ... (ROWS_TO_SKIPなど - 変更なし) ...
ROWS_TO_SKIP = 11; SAMPLING_INTERVAL_MS = 5; SYNC_SIGNAL_SUFFIX = '_Acc_Y'; ALIGN_TARGET_SUFFIX = '_Gyro_Z'
RIGHT_PREFIX = 'R'; LEFT_PREFIX = 'L'; TRUNK_PREFIX = 'T'
DEFAULT_PEAK_HEIGHT = 10.0; DEFAULT_PEAK_PROMINENCE = 0.3; DEFAULT_PEAK_DISTANCE = 50
NUM_SAMPLES_TO_PLOT = 1000
MAX_IC_INTERVAL_SEC = 2.0; MIN_ICS_PER_TRIAL = 11
# ★★★ 前後除外する歩数を追加 ★★★
NUM_ICS_REMOVE_START = 3
NUM_ICS_REMOVE_END = 5

# --- 列名定義関数 ---
def get_expected_column_names(right_prefix, left_prefix, trunk_prefix):
     return [
            'Time', f'{right_prefix}_Acc_X', f'{right_prefix}_Acc_Y', f'{right_prefix}_Acc_Z', f'{right_prefix}_Ch4_V',
            f'{right_prefix}_Gyro_X', f'{right_prefix}_Gyro_Y', f'{right_prefix}_Gyro_Z', f'{right_prefix}_Ch8_V',
            f'{left_prefix}_Acc_X', f'{left_prefix}_Acc_Y', f'{left_prefix}_Acc_Z', 'Blank_1',
            f'{left_prefix}_Gyro_X', f'{left_prefix}_Gyro_Y', f'{left_prefix}_Gyro_Z', 'Blank_2',
            f'{trunk_prefix}_Acc_X', f'{trunk_prefix}_Acc_Y', f'{trunk_prefix}_Acc_Z', 'Blank_3',
            f'{trunk_prefix}_Gyro_X', f'{trunk_prefix}_Gyro_Y', f'{trunk_prefix}_Gyro_Z', 'Blank_4',
            'Blank_5', 'Blank_6', 'Blank_7', 'Blank_8', 'Blank_9', 'Blank_10', 'Blank_11', 'Blank_12' ]

# --- 歩行トライアル分割関数 ---
def segment_walking_trials(events_df, max_interval_sec, min_ics_per_trial):
    """IC間の時間差に基づき歩行トライアルを分割し、短すぎるトライアルを除外"""
    print(f"--- 歩行トライアルの自動分割開始 (最大IC間隔: {max_interval_sec}s, 最小IC数: {min_ics_per_trial}) ---")
    if events_df is None or events_df.empty: return pd.DataFrame()
    df_sorted = events_df.sort_values(by="IC_Time").copy()
    df_sorted['Time_Diff'] = df_sorted['IC_Time'].diff()
    df_sorted['Trial_ID_Raw'] = (df_sorted['Time_Diff'] > max_interval_sec).cumsum() + 1
    df_sorted['Trial_IC_Count'] = df_sorted.groupby('Trial_ID_Raw')['IC_Index'].transform('size')
    df_segmented = df_sorted[df_sorted['Trial_IC_Count'] >= min_ics_per_trial].copy()
    if df_segmented.empty: print("警告: 有効な歩行トライアル検出不可"); return pd.DataFrame()
    df_segmented['Trial_ID'] = df_segmented.groupby('Trial_ID_Raw').ngroup() + 1
    valid_trials = df_segmented['Trial_ID'].unique()
    print(f"  -> {len(valid_trials)} 個の有効な歩行トライアル (ID: {valid_trials.tolist()}) を検出")
    return df_segmented.drop(columns=['Time_Diff', 'Trial_ID_Raw', 'Trial_IC_Count'])

# --- ★★★ トライアルの最初と最後を除外する関数 ★★★ ---
def trim_trial_ends(df_segmented, n_start=3, n_end=5):
    """
    Trial_ID と Leg でグループ化し、各グループの先頭n_start個と末尾n_end個のICを除外する。
    """
    print(f"--- トライアルの前後除外開始 (先頭: {n_start}歩, 末尾: {n_end}歩) ---")
    if df_segmented is None or df_segmented.empty:
        print("  入力データがないため、処理をスキップします。")
        return pd.DataFrame()

    min_required_ics = n_start + n_end + 1 # 除外後に最低1歩残るための最小歩数

    # 各グループに対して処理を適用し、結果を結合する
    trimmed_groups = []
    # Trial_ID と Leg でグループ化し、IC_Indexでソートしておく
    grouped = df_segmented.sort_values(by='IC_Index').groupby(['Trial_ID', 'Leg'], sort=False)

    total_removed_count = 0
    original_count = len(df_segmented)

    for name, group in grouped:
        if len(group) >= min_required_ics:
            # iloc を使って先頭 n_start 個と末尾 n_end 個を除外
            trimmed_group = group.iloc[n_start:-n_end]
            trimmed_groups.append(trimmed_group)
            total_removed_count += len(group) - len(trimmed_group)
        else:
            # 条件を満たさないグループは完全に除外
            # print(f"  情報: Trial {name[0]}, Leg {name[1]} はIC数が足りないため除外 ({len(group)} < {min_required_ics})")
            total_removed_count += len(group)

    if not trimmed_groups:
        print("  警告: 前後除外の結果、有効な歩行周期が残りませんでした。")
        return pd.DataFrame()

    # 有効なグループを結合
    df_trimmed = pd.concat(trimmed_groups).reset_index(drop=True)
    print(f"  前後除外処理完了。 {original_count} -> {len(df_trimmed)} イベント ({total_removed_count} イベント除外)")

    return df_trimmed


# --- Tkinter GUI アプリケーションクラス ---
class GaitAnalysisApp:
    def __init__(self, master):
        self.master = master
        master.title("歩行データ同期・周期同定ツール")
        master.geometry("900x700")
        self.input_file_path = None
        self.sampling_rate = 1000.0 / SAMPLING_INTERVAL_MS

        top_frame = Frame(master, pady=5); top_frame.pack(side=tk.TOP, fill=tk.X)
        self.file_label = Label(top_frame, text="処理対象ファイル: 検索中..."); self.file_label.pack(side=LEFT, padx=10)
        self.status_label = Label(top_frame, text="初期化中..."); self.status_label.pack(side=LEFT, padx=10)

        self.fig = Figure(figsize=(8, 4), dpi=100); self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=master); self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=BOTH, expand=True)
        toolbar_frame = Frame(master); toolbar_frame.pack(side=tk.TOP, fill=tk.X)
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame); self.toolbar.update()

        param_frame = Frame(master, pady=10); param_frame.pack(side=tk.TOP, fill=tk.X)
        self.height_var = tk.DoubleVar(value=DEFAULT_PEAK_HEIGHT)
        self.prominence_var = tk.DoubleVar(value=DEFAULT_PEAK_PROMINENCE)
        self.distance_var = tk.IntVar(value=DEFAULT_PEAK_DISTANCE)

        # スライダーと値表示ラベル
        row_idx = 0
        Label(param_frame, text="Height:").grid(row=row_idx, column=0, padx=5, pady=2, sticky=W)
        self.height_scale = Scale(param_frame, from_=0, to=50, resolution=0.1, orient=HORIZONTAL, variable=self.height_var, length=150); self.height_scale.grid(row=row_idx, column=1, padx=5, pady=2)
        self.height_label_val = Label(param_frame, text=f"{self.height_var.get():.1f}", width=5); self.height_label_val.grid(row=row_idx, column=2, padx=5, pady=2)
        self.height_scale.config(command=lambda v: self.height_label_val.config(text=f"{float(v):.1f}"))
        Label(param_frame, text="Prominence:").grid(row=row_idx, column=3, padx=5, pady=2, sticky=W)
        self.prominence_scale = Scale(param_frame, from_=0, to=10, resolution=0.1, orient=HORIZONTAL, variable=self.prominence_var, length=150); self.prominence_scale.grid(row=row_idx, column=4, padx=5, pady=2)
        self.prominence_label_val = Label(param_frame, text=f"{self.prominence_var.get():.1f}", width=5); self.prominence_label_val.grid(row=row_idx, column=5, padx=5, pady=2)
        self.prominence_scale.config(command=lambda v: self.prominence_label_val.config(text=f"{float(v):.1f}"))
        row_idx += 1
        Label(param_frame, text="Distance:").grid(row=row_idx, column=0, padx=5, pady=2, sticky=W)
        self.distance_scale = Scale(param_frame, from_=1, to=200, resolution=1, orient=HORIZONTAL, variable=self.distance_var, length=150); self.distance_scale.grid(row=row_idx, column=1, padx=5, pady=2)
        self.distance_label_val = Label(param_frame, text=f"{self.distance_var.get():d}", width=5); self.distance_label_val.grid(row=row_idx, column=2, padx=5, pady=2)
        self.distance_scale.config(command=lambda v: self.distance_label_val.config(text=f"{int(float(v)):d}"))

        # 実行ボタン (最初は無効)
        self.run_button = Button(param_frame, text="同期実行 ＆ 歩行周期同定", command=self.run_analysis_pipeline, width=25, state=DISABLED)
        self.run_button.grid(row=row_idx-1, column=6, rowspan=2, padx=20, pady=5, sticky=tk.W + tk.E + tk.S) # 配置微調整

        # 初期データロード（GUI準備後に実行）
        self.status_label.config(text="最新ファイル検索中...")
        self.master.update_idletasks()
        self.master.after(100, self.load_and_plot_initial_data)

    # --- 初期データのロードとプロット ---
    def load_and_plot_initial_data(self):
        print("\n[準備] 最新のCSVデータを検索中...")
        self.input_file_path = find_latest_csv_file(DATA_FOLDER)
        if self.input_file_path is None: messagebox.showerror("エラー", f"CSVファイル未検出 ({DATA_FOLDER.resolve()})"); self.master.quit(); return
        self.file_label.config(text=f"処理対象: {self.input_file_path.name}")

        self.status_label.config(text="データ読込・プロット中...")
        self.master.update_idletasks()

        try:
            temp_df = None
            try: temp_df = pd.read_csv(self.input_file_path, skiprows=ROWS_TO_SKIP, encoding='cp932')
            except UnicodeDecodeError: temp_df = pd.read_csv(self.input_file_path, skiprows=ROWS_TO_SKIP, encoding='shift_jis')
            expected_cols = get_expected_column_names(RIGHT_PREFIX, LEFT_PREFIX, TRUNK_PREFIX)
            if len(temp_df.columns) == len(expected_cols):
                temp_df.columns = expected_cols
                blank_cols = [col for col in temp_df.columns if 'Blank_' in col]; temp_df = temp_df.drop(columns=blank_cols)
            else: raise ValueError(f"初期読み込みで列数が不一致 ({len(temp_df.columns)} vs {len(expected_cols)})")

            sync_l_full = temp_df[f'{LEFT_PREFIX}{SYNC_SIGNAL_SUFFIX}'].fillna(0).values
            sync_r_full = temp_df[f'{RIGHT_PREFIX}{SYNC_SIGNAL_SUFFIX}'].fillna(0).values
            plot_len = min(len(sync_l_full), len(sync_r_full), NUM_SAMPLES_TO_PLOT)
            if plot_len <= 0: raise ValueError("プロットデータなし")
            sync_l_short_plot = sync_l_full[:plot_len]; sync_r_short_plot = sync_r_full[:plot_len]
            time_plot = np.arange(plot_len) * (SAMPLING_INTERVAL_MS / 1000.0)

            self.ax.clear()
            self.ax.plot(time_plot, sync_l_short_plot, label=f'左 ({LEFT_PREFIX}{SYNC_SIGNAL_SUFFIX})', alpha=0.8)
            self.ax.plot(time_plot, sync_r_short_plot, label=f'右 ({RIGHT_PREFIX}{SYNC_SIGNAL_SUFFIX})', linestyle='--', alpha=0.8)
            self.ax.set_title(f'同期用信号 先頭 {plot_len} サンプル (パラメータ調整用)')
            self.ax.set_xlabel('時間 (s)'); self.ax.set_ylabel('信号値 (単位?)'); self.ax.legend(); self.ax.grid(True); self.canvas.draw()

            self.run_button.config(state=NORMAL)
            self.status_label.config(text="準備完了. パラメータ調整後、ボタン実行.")
            print(f"  プロット完了。パラメータを調整してボタンを押してください。")

        except Exception as e:
            self.status_label.config(text="エラー発生！")
            messagebox.showerror("エラー", f"初期データのプロット中にエラー:\n{e}", parent=self.master)
            self.run_button.config(state=DISABLED)

    # --- 「同期実行 ＆ 歩行周期同定」ボタンの処理 ---
    def run_analysis_pipeline(self):
        self.run_button.config(state=DISABLED); self.status_label.config(text="解析処理実行中..."); self.master.update_idletasks()
        sync_gyro_df = None # スコープのために初期化
        gait_events_df_segmented = pd.DataFrame() # 初期化
        try:
            print("\n========================================"); print("=== 解析パイプライン実行 ===")
            user_peak_height = self.height_var.get(); user_peak_prominence = self.prominence_var.get(); user_peak_distance = self.distance_var.get()
            print(f"  使用 find_peaks パラメータ: h={user_peak_height:.2f}, p={user_peak_prominence:.2f}, d={user_peak_distance}")

            # === ステップ 1: 前処理 ===
            print("\n[ステップ1] 角速度データの前処理...")
            result_tuple = preprocess_angular_velocity(
                 data_file=self.input_file_path, rows_to_skip=ROWS_TO_SKIP, sampling_interval_ms=SAMPLING_INTERVAL_MS,
                 right_prefix=RIGHT_PREFIX, left_prefix=LEFT_PREFIX, trunk_prefix=TRUNK_PREFIX,
                 sync_col_suffix=SYNC_SIGNAL_SUFFIX, align_col_suffix=ALIGN_TARGET_SUFFIX,
                 peak_height=user_peak_height, peak_prominence=user_peak_prominence, peak_distance=user_peak_distance)
            if result_tuple is None or len(result_tuple) != 7: messagebox.showerror("エラー", "前処理失敗(戻り値不正)"); return
            sync_gyro_df, lag_samples, self.sampling_rate, peak_idx_l, peak_idx_r, _, _ = result_tuple
            if sync_gyro_df is None: messagebox.showerror("エラー", "前処理失敗"); return

            result_message = f"前処理完了。\n左ピーク idx: {peak_idx_l}, 右ピーク idx: {peak_idx_r}\n同期ラグ: {lag_samples} サンプル"
            print(f"\n[成功] {result_message}"); messagebox.showinfo("前処理完了", result_message, parent=self.master)
            base_filename = self.input_file_path.stem; output_sync_file = DATA_FOLDER / (base_filename + OUTPUT_SUFFIX_SYNC)
            print(f"\n[ステップ1.1] 同期済みデータを保存中..."); save_results(output_sync_file, sync_gyro_df, "同期済みデータ")


            # === ステップ 2: 歩行周期同定 (全体データに対して) ===
                        # === ステップ 2: 歩行周期同定 (全体データに対して) ===
            gait_events_df_all = None; filtered_signals = {}; time_vector = None
            if sync_gyro_df is not None:
                print("\n[ステップ2] 歩行周期の同定 (全体) を実行中...")
                current_swing_threshold = 100 # ★必要なら調整★
                gait_events_result = identify_gait_cycles(sync_gyro_df=sync_gyro_df, sampling_rate_hz=self.sampling_rate, swing_threshold=current_swing_threshold)
                if gait_events_result is not None and isinstance(gait_events_result, dict):
                     gait_events_df_all = gait_events_result.get("events_df"); filtered_signals = gait_events_result.get("filtered_signals", {}); time_vector = gait_events_result.get("time_vector")

                if gait_events_df_all is None or gait_events_df_all.empty:
                     print("  歩行周期を同定できませんでした。"); messagebox.showwarning("歩行周期同定", "歩行周期同定失敗", parent=self.master)
                     print("\n[ステップ3] PCI計算スキップ")
                else:
                     print(f"  {len(gait_events_df_all)} 個の歩行周期候補を全体から検出。")
                     # === ★ ステップ 2.1: 歩行トライアル分割 ★ ===
                     print("\n[ステップ2.1] 歩行トライアルの自動分割を実行中...")
                     gait_events_df_segmented = segment_walking_trials( # 分割後のDFを受け取る
                         events_df=gait_events_df_all, max_interval_sec=MAX_IC_INTERVAL_SEC, min_ics_per_trial=MIN_ICS_PER_TRIAL )

                     if gait_events_df_segmented.empty:
                          print("  有効な歩行トライアルが見つかりませんでした。"); messagebox.showwarning("トライアル分割", "有効歩行トライアル検出不可", parent=self.master)
                          print("\n[ステップ3] PCI計算スキップ")
                     else:
                          # === ★ ステップ 2.2: トライアルの前後除外 ★ ===
                          print("\n[ステップ2.2] 定常歩行部分の抽出（前後除外）を実行中...")
                          gait_events_df_steady = trim_trial_ends( # ★新しい関数を呼び出し★
                              df_segmented=gait_events_df_segmented,
                              n_start=NUM_ICS_REMOVE_START,
                              n_end=NUM_ICS_REMOVE_END
                          )

                if gait_events_df_all is None or gait_events_df_all.empty:
                     print("  歩行周期を同定できませんでした。"); messagebox.showwarning("歩行周期同定", "歩行周期同定失敗", parent=self.master)
                     print("\n[ステップ3] PCI計算スキップ")
                     # gait_events_df_segmented は空のまま
                else:
                     print(f"  {len(gait_events_df_all)} 個の歩行周期候補を全体から検出。")
                     # === ステップ 2.1: 歩行トライアル分割 ===
                     print("\n[ステップ2.1] 歩行トライアルの自動分割を実行中...")
                     gait_events_df_segmented = segment_walking_trials(
                         events_df=gait_events_df_all,
                         max_interval_sec=MAX_IC_INTERVAL_SEC,
                         min_ics_per_trial=MIN_ICS_PER_TRIAL
                     )

                     if gait_events_df_segmented.empty:
                          print("  有効な歩行トライアルが見つかりませんでした。"); messagebox.showwarning("トライアル分割", "有効歩行トライアル検出不可", parent=self.master)
                          print("\n[ステップ3] PCI計算スキップ")
                     else:
                          # --- ステップ 2.2: 歩行周期データ(分割後)の保存 ---
                          output_gait_file = DATA_FOLDER / (base_filename + OUTPUT_SUFFIX_GAIT_TRIMMED)
                          print(f"\n[ステップ2.2] 有効な歩行周期データを保存中..."); save_results(output_gait_file, gait_events_df_segmented, "歩行周期データ(分割後)")
                          print("\n--- 同定された有効歩行周期 (最初の5件) ---"); print(gait_events_df_segmented[['Leg', 'Trial_ID', 'Cycle', 'IC_Time', 'FO_Time']].head().to_string()); print("---")
                          # --- ステップ 2.3: IC/FOイベントのプロット ---
                          self.plot_gait_events(gait_events_df_segmented, filtered_signals, time_vector) # 分割後データでプロット
                          # === ステップ 3: PCI計算 ===
                          print("\n[ステップ3] PCI計算を実行中...")
                          pci_results = calculate_pci(gait_events_df_steady)

                          if pci_results is None:
                                print("  PCI計算に失敗しました。")
                                messagebox.showwarning("PCI計算", "PCI計算失敗", parent=self.master)
                          else:
                                print(f"\n  PCI 計算結果:")
                                # ★★★ 表示項目を追加 ★★★
                                for key, value in pci_results.items():
                                     # 値が float なら小数点以下2桁で表示 (それ以外はそのまま)
                                     if isinstance(value, (float, np.floating)):
                                         print(f"    {key}: {value:.2f}")
                                     else:
                                          print(f"    {key}: {value}")
                                # ★★★ 表示ここまで ★★★

                                # 結果の保存
                                try:
                                    pci_df = pd.DataFrame([pci_results])
                                    output_pci_file = DATA_FOLDER / (base_filename + OUTPUT_SUFFIX_PCI)
                                    print(f"\n[ステップ3.1] PCI計算結果を '{output_pci_file.name}' に保存中...")
                                    save_results(output_pci_file, pci_df, "PCI計算結果")
                                    # ポップアップ表示も少し詳細に
                                    pci_val_str = f"{pci_results.get('PCI', 'N/A'):.2f}" if not pd.isna(pci_results.get('PCI')) else 'N/A'
                                    cv_l_str = f"{pci_results.get('StrideTime_CV_L_percent', 'N/A'):.2f}" if not pd.isna(pci_results.get('StrideTime_CV_L_percent')) else 'N/A'
                                    cv_r_str = f"{pci_results.get('StrideTime_CV_R_percent', 'N/A'):.2f}" if not pd.isna(pci_results.get('StrideTime_CV_R_percent')) else 'N/A'
                                    messagebox.showinfo("PCI計算完了", f"PCI計算完了。\nPCI: {pci_val_str}\nStride CV L: {cv_l_str}%\nStride CV R: {cv_r_str}%", parent=self.master)
                                except Exception as e_pci_save:
                                     print(f"  エラー: PCI結果の保存中にエラー: {e_pci_save}")
                                     messagebox.showerror("ファイル保存エラー", f"PCI結果の保存エラー:\n{e_pci_save}", parent=self.master)

            # === ステップ 4: 同期済み角速度の最終プロット ===
            self.plot_final_synchronized_data(sync_gyro_df)

            print("\n--- 解析パイプライン 完了 ---")
            self.status_label.config(text="解析完了")

        except Exception as e:
             print(f"\n[エラー] 解析パイプライン実行中にエラー: {e}"); messagebox.showerror("実行時エラー", f"解析中にエラー:\n{e}", parent=self.master)
             self.status_label.config(text="エラー発生")
        finally:
             self.run_button.config(state=NORMAL); self.master.update_idletasks()


    # --- IC/FO イベントプロット用メソッド ---
    def plot_gait_events(self, gait_events_df, filtered_signals, time_vector):
        print("\n[ステップ2.3] IC/FO イベント (自動分割後) をグラフにプロットします...")
        if gait_events_df is None or gait_events_df.empty: print("プロットするイベントデータなし"); return
        try:
            fig_events, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
            fig_events.suptitle(f'検出された歩行イベント (自動分割後) - {self.input_file_path.name}')
            plot_successful = False
            if time_vector is None or len(time_vector) == 0:
                 sig_len = len(filtered_signals.get('L', []));
                 if sig_len == 0: sig_len = len(filtered_signals.get('R',[]))
                 if sig_len > 0 : time_vector = np.arange(sig_len) * (1.0/self.sampling_rate)
                 else: raise ValueError("プロット用時間ベクトル不明")

            for i, leg in enumerate(['L', 'R']):
                ax = axes[i]; filt_signal = filtered_signals.get(leg)
                if filt_signal is None or len(filt_signal) != len(time_vector):
                     print(f"警告: {leg}脚 プロットデータ不整合"); ax.set_title(f'{leg} 脚 - データ不整合'); continue

                ax.plot(time_vector, filt_signal, label=f'{leg} Gyro Z (Filtered)', alpha=0.7)
                leg_events = gait_events_df[gait_events_df['Leg'] == leg]
                ic_times = leg_events['IC_Time'].dropna().values; fo_times = leg_events['FO_Time'].dropna().values
                # 時間に対応するインデックスを見つける
                ic_indices_plot = np.searchsorted(time_vector, ic_times, side='left'); fo_indices_plot = np.searchsorted(time_vector, fo_times, side='left')
                ic_indices_plot = ic_indices_plot[ic_indices_plot < len(time_vector)]; fo_indices_plot = fo_indices_plot[fo_indices_plot < len(time_vector)]

                if len(ic_indices_plot) > 0: ax.plot(time_vector[ic_indices_plot], filt_signal[ic_indices_plot], 'ro', markersize=6, label='IC (踵接地)', linestyle='None')
                if len(fo_indices_plot) > 0: ax.plot(time_vector[fo_indices_plot], filt_signal[fo_indices_plot], 'gx', markersize=8, markeredgewidth=2, label='FO (爪先離地)', linestyle='None')
                ax.set_title(f'{leg} 脚'); ax.set_ylabel('角速度 (単位?)'); ax.legend(loc='upper right'); ax.grid(True); plot_successful = True

            if plot_successful:
                 axes[1].set_xlabel('時間 (s)'); plt.tight_layout(rect=[0, 0.03, 1, 0.96]); plt.show(block=False)
            else: plt.close(fig_events); print("イベントグラフプロット不可")
        except Exception as e: print(f"  エラー: IC/FOプロットエラー: {e}")

    # --- 同期済み角速度の最終プロット用メソッド ---
    def plot_final_synchronized_data(self, sync_gyro_df):
         if sync_gyro_df is not None:
             print(f"\n[ステップ4] 同期済み{ALIGN_TARGET_SUFFIX}のグラフ (先頭部分) を表示します...")
             try:
                 if not sync_gyro_df.empty:
                      num_samples_final_plot = 1000; sync_gyro_df_short = sync_gyro_df.head(num_samples_final_plot); actual_plot_len = len(sync_gyro_df_short)
                      plt.figure(figsize=(12, 6))
                      left_col_name = f'L{ALIGN_TARGET_SUFFIX}_aligned'; right_col_name = f'R{ALIGN_TARGET_SUFFIX}_aligned'
                      if left_col_name in sync_gyro_df_short.columns and right_col_name in sync_gyro_df_short.columns:
                         plt.plot(sync_gyro_df_short['time_aligned_sec'], sync_gyro_df_short[left_col_name], label=f'同期後 左 {ALIGN_TARGET_SUFFIX}'); plt.plot(sync_gyro_df_short['time_aligned_sec'], sync_gyro_df_short[right_col_name], label=f'同期後 右 {ALIGN_TARGET_SUFFIX}', linestyle='--', alpha=0.8)
                         plt.title(f'同期済み角速度 - 先頭 {actual_plot_len} サンプル ({self.input_file_path.name})'); plt.xlabel('時間 (s)'); plt.ylabel('角速度 (単位?)'); plt.legend(); plt.grid(True)
                         plt.show(block=False)
                      else: print(f"  警告: プロット列なし")
                 else: print("  警告: 同期済みデータ空")
             except Exception as e: print(f"  エラー: 最終グラフ表示エラー: {e}")


# --- メイン実行ブロック ---
if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    app_window = Toplevel(root)
    app = GaitAnalysisApp(app_window)
    app_window.protocol("WM_DELETE_WINDOW", root.destroy)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("\nCtrl+C により中断されました。")
    finally:
         print("\n========================================")
         print("=== アプリケーション終了 ===")
         print("========================================")