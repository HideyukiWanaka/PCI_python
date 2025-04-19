# run_gait_analysis.py

# --- 各機能ファイルをインポート ---
from preprocessing import preprocess_angular_velocity
from gait_cycles import identify_gait_cycles # 変更した関数をインポート
from pci import calculate_pci             # Placeholder
from file_utils import find_latest_csv_file, save_results

# --- 必要な標準ライブラリ・外部ライブラリをインポート ---
import pandas as pd
import numpy as np
from pathlib import Path

# --- GUIとプロットのためのインポート ---
import tkinter as tk
from tkinter import Frame, Label, Entry, Button, BOTH, W, LEFT, messagebox, HORIZONTAL, Scale, Toplevel
import matplotlib
matplotlib.use('TkAgg') # Tkinter連携用バックエンドを明示的に指定
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.font_manager as fm # フォント設定用だが、japanizeが推奨

# ★★★ japanize-matplotlib をインポート ★★★
try:
    import japanize_matplotlib # これだけで日本語表示が有効になる
    print("japanize_matplotlib をインポートしました。")
except ImportError:
    print("警告: japanize-matplotlib が見つかりません。'pip install japanize-matplotlib' でインストールしてください。")
    # フォント設定のフォールバック (手動設定)
    try:
        jp_font_name = 'Hiragino Sans' # macOS の例, 環境に合わせて変更
        plt.rcParams['font.family'] = jp_font_name
        print(f"フォールバック: 日本語フォントとして '{jp_font_name}' を試みます。")
    except Exception as e_font:
        print(f"警告: フォールバックの日本語フォント設定エラー: {e_font}")
# ★★★ ここまで ★★★


# --- 解析実行のための設定 ---
DATA_FOLDER = Path("./") # データ検索フォルダ
OUTPUT_SUFFIX_SYNC = '_synchronized_gyro_z.csv'
OUTPUT_SUFFIX_GAIT = '_gait_events.csv'
OUTPUT_SUFFIX_PCI = '_pci_results.csv'
ROWS_TO_SKIP = 11
SAMPLING_INTERVAL_MS = 5
SYNC_SIGNAL_SUFFIX = '_Acc_Y'
ALIGN_TARGET_SUFFIX = '_Gyro_Z'
RIGHT_PREFIX = 'R'
LEFT_PREFIX = 'L'
TRUNK_PREFIX = 'T'
DEFAULT_PEAK_HEIGHT = 10.0 # find_peaks デフォルト Height
DEFAULT_PEAK_PROMINENCE = 0.3  # find_peaks デフォルト Prominence
DEFAULT_PEAK_DISTANCE = 50   # find_peaks デフォルト Distance
NUM_SAMPLES_TO_PLOT = 1000   # 初期プロットサンプル数

# --- 列名定義関数 ---
def get_expected_column_names(right_prefix, left_prefix, trunk_prefix):
     # 33列分のリストを返す
     return [
            'Time', f'{right_prefix}_Acc_X', f'{right_prefix}_Acc_Y', f'{right_prefix}_Acc_Z', f'{right_prefix}_Ch4_V',
            f'{right_prefix}_Gyro_X', f'{right_prefix}_Gyro_Y', f'{right_prefix}_Gyro_Z', f'{right_prefix}_Ch8_V',
            f'{left_prefix}_Acc_X', f'{left_prefix}_Acc_Y', f'{left_prefix}_Acc_Z', 'Blank_1',
            f'{left_prefix}_Gyro_X', f'{left_prefix}_Gyro_Y', f'{left_prefix}_Gyro_Z', 'Blank_2',
            f'{trunk_prefix}_Acc_X', f'{trunk_prefix}_Acc_Y', f'{trunk_prefix}_Acc_Z', 'Blank_3',
            f'{trunk_prefix}_Gyro_X', f'{trunk_prefix}_Gyro_Y', f'{trunk_prefix}_Gyro_Z', 'Blank_4',
            'Blank_5', 'Blank_6', 'Blank_7', 'Blank_8', 'Blank_9', 'Blank_10', 'Blank_11', 'Blank_12' ]


# --- Tkinter GUI アプリケーションクラス ---
class GaitAnalysisApp:
    def __init__(self, master):
        self.master = master
        master.title("歩行データ同期パラメータ調整")
        master.geometry("900x700")

        self.input_file_path = None
        self.sync_l_short_plot = None
        self.sync_r_short_plot = None
        self.plot_len = 0
        self.time_plot = None
        self.sampling_rate = 200.0 # デフォルト (前処理後に更新)

        # --- GUI要素の作成 ---
        top_frame = Frame(master, pady=5)
        top_frame.pack(side=tk.TOP, fill=tk.X)
        self.file_label = Label(top_frame, text="処理対象ファイル: 検索中...")
        self.file_label.pack(side=LEFT, padx=10)

        # Matplotlibグラフ描画エリア
        self.fig = Figure(figsize=(8, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=master)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=BOTH, expand=True)
        toolbar_frame = Frame(master); toolbar_frame.pack(side=tk.TOP, fill=tk.X)
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame); self.toolbar.update()

        # パラメータ入力エリア
        param_frame = Frame(master, pady=10); param_frame.pack(side=tk.TOP, fill=tk.X)
        self.height_var = tk.DoubleVar(value=DEFAULT_PEAK_HEIGHT)
        self.prominence_var = tk.DoubleVar(value=DEFAULT_PEAK_PROMINENCE)
        self.distance_var = tk.IntVar(value=DEFAULT_PEAK_DISTANCE)

        # Height Slider & Label
        Label(param_frame, text="Height:").grid(row=0, column=0, padx=5, sticky=W)
        self.height_scale = Scale(param_frame, from_=0, to=50, resolution=0.1, orient=HORIZONTAL, variable=self.height_var, length=150)
        self.height_scale.grid(row=0, column=1, padx=5)
        self.height_label_val = Label(param_frame, text=f"{self.height_var.get():.1f}", width=5)
        self.height_label_val.grid(row=0, column=2)
        self.height_scale.config(command=lambda v: self.height_label_val.config(text=f"{float(v):.1f}"))

        # Prominence Slider & Label
        Label(param_frame, text="Prominence:").grid(row=0, column=3, padx=5, sticky=W)
        self.prominence_scale = Scale(param_frame, from_=0, to=10, resolution=0.1, orient=HORIZONTAL, variable=self.prominence_var, length=150)
        self.prominence_scale.grid(row=0, column=4, padx=5)
        self.prominence_label_val = Label(param_frame, text=f"{self.prominence_var.get():.1f}", width=5)
        self.prominence_label_val.grid(row=0, column=5)
        self.prominence_scale.config(command=lambda v: self.prominence_label_val.config(text=f"{float(v):.1f}"))

        # Distance Slider & Label
        Label(param_frame, text="Distance:").grid(row=1, column=0, padx=5, sticky=W)
        self.distance_scale = Scale(param_frame, from_=1, to=200, resolution=1, orient=HORIZONTAL, variable=self.distance_var, length=150)
        self.distance_scale.grid(row=1, column=1, padx=5)
        self.distance_label_val = Label(param_frame, text=f"{self.distance_var.get():d}", width=5)
        self.distance_label_val.grid(row=1, column=2)
        self.distance_scale.config(command=lambda v: self.distance_label_val.config(text=f"{int(float(v)):d}"))

        # 実行ボタン
        self.run_button = Button(param_frame, text="同期実行 ＆ 歩行周期同定", command=self.run_analysis_pipeline, width=25) # ボタンテキスト変更
        self.run_button.grid(row=1, column=3, columnspan=3, padx=20, pady=5)

        # 初期データロード
        self.load_and_plot_initial_data()

    # --- 初期データのロードとプロット ---
    def load_and_plot_initial_data(self):
        print("\n[準備] 最新のCSVデータを検索中...")
        self.input_file_path = find_latest_csv_file(DATA_FOLDER)
        if self.input_file_path is None: messagebox.showerror("エラー", "CSVファイルが見つかりません。"); self.master.quit(); return
        self.file_label.config(text=f"処理対象: {self.input_file_path.name}")
        print(f"処理対象ファイル: '{self.input_file_path.name}'")

        print(f"\n[ステップ0] 同期用信号 ({SYNC_SIGNAL_SUFFIX}) の先頭 {NUM_SAMPLES_TO_PLOT} サンプルをプロット...")
        try:
            # --- データの一時読み込み & 列名設定 ---
            temp_df = None
            try: temp_df = pd.read_csv(self.input_file_path, skiprows=ROWS_TO_SKIP, encoding='cp932')
            except UnicodeDecodeError: temp_df = pd.read_csv(self.input_file_path, skiprows=ROWS_TO_SKIP, encoding='shift_jis')
            expected_cols = get_expected_column_names(RIGHT_PREFIX, LEFT_PREFIX, TRUNK_PREFIX)
            if len(temp_df.columns) == len(expected_cols):
                temp_df.columns = expected_cols; blank_cols = [col for col in temp_df.columns if 'Blank_' in col]; temp_df = temp_df.drop(columns=blank_cols)
            else: raise ValueError(f"初期読み込みで列数が不一致")

            sync_l_full = temp_df[f'{LEFT_PREFIX}{SYNC_SIGNAL_SUFFIX}'].fillna(0).values
            sync_r_full = temp_df[f'{RIGHT_PREFIX}{SYNC_SIGNAL_SUFFIX}'].fillna(0).values
            self.plot_len = min(len(sync_l_full), len(sync_r_full), NUM_SAMPLES_TO_PLOT)
            if self.plot_len <= 0: raise ValueError("プロットデータなし")
            self.sync_l_short_plot = sync_l_full[:self.plot_len]; self.sync_r_short_plot = sync_r_full[:self.plot_len]
            self.time_plot = np.arange(self.plot_len) * (SAMPLING_INTERVAL_MS / 1000.0)

            # --- Matplotlib Axes にプロット ---
            self.ax.clear(); self.ax.plot(self.time_plot, self.sync_l_short_plot, label=f'左 ({LEFT_PREFIX}{SYNC_SIGNAL_SUFFIX})', alpha=0.8)
            self.ax.plot(self.time_plot, self.sync_r_short_plot, label=f'右 ({RIGHT_PREFIX}{SYNC_SIGNAL_SUFFIX})', linestyle='--', alpha=0.8)
            self.ax.set_title(f'同期用信号 先頭 {self.plot_len} サンプル (パラメータ調整用)'); self.ax.set_xlabel('時間 (s)'); self.ax.set_ylabel('信号値 (単位?)')
            self.ax.legend(); self.ax.grid(True); self.canvas.draw()
            print(f"  プロット完了。パラメータを調整してボタンを押してください。")

        except Exception as e:
            messagebox.showerror("エラー", f"初期データのプロット中にエラー:\n{e}"); self.master.quit()

    # --- 「同期実行 ＆ 歩行周期同定」ボタンの処理 ---
    def run_analysis_pipeline(self):
        print("\n========================================")
        print("=== 解析パイプライン実行 ===")
        print("========================================")

        # --- ステップ 1: 前処理 ---
        print("\n[ステップ1] 角速度データの前処理 (GUIパラメータ使用) を実行中...")
        user_peak_height = self.height_var.get()
        user_peak_prominence = self.prominence_var.get()
        user_peak_distance = self.distance_var.get()
        print(f"  使用パラメータ: height={user_peak_height:.2f}, prominence={user_peak_prominence:.2f}, distance={user_peak_distance}")

        result_tuple = preprocess_angular_velocity(
            data_file=self.input_file_path, rows_to_skip=ROWS_TO_SKIP, sampling_interval_ms=SAMPLING_INTERVAL_MS,
            right_prefix=RIGHT_PREFIX, left_prefix=LEFT_PREFIX, trunk_prefix=TRUNK_PREFIX,
            sync_col_suffix=SYNC_SIGNAL_SUFFIX, align_col_suffix=ALIGN_TARGET_SUFFIX,
            peak_height=user_peak_height, peak_prominence=user_peak_prominence, peak_distance=user_peak_distance
        )

        if result_tuple is None or len(result_tuple) != 7: messagebox.showerror("エラー", "前処理失敗(戻り値不正)"); return
        sync_gyro_df, lag_samples, self.sampling_rate, peak_idx_l, peak_idx_r, _, _ = result_tuple # sampling_rate を更新
        if sync_gyro_df is None: messagebox.showerror("エラー", "前処理失敗"); return

        result_message = f"前処理完了。\n左ピーク idx: {peak_idx_l}, 右ピーク idx: {peak_idx_r}\n同期ラグ: {lag_samples} サンプル"
        print(f"\n[成功] {result_message}"); messagebox.showinfo("前処理完了", result_message, parent=self.master) # 親を指定
        base_filename = self.input_file_path.stem; output_sync_file = DATA_FOLDER / (base_filename + OUTPUT_SUFFIX_SYNC)
        print(f"\n[ステップ1.1] 同期済みデータを '{output_sync_file.name}' に保存中...")
        save_results(output_sync_file, sync_gyro_df, "同期済み角速度データ")

        # --- ステップ 2: 歩行周期同定 ---
        print("\n[ステップ2] 歩行周期の同定を実行中...")
        gait_events_result = identify_gait_cycles(
            sync_gyro_df=sync_gyro_df, sampling_rate_hz=self.sampling_rate, swing_threshold=200 # 他はデフォルト
        )

        gait_events_df = None # 初期化
        filtered_signals = {}
        time_vector = None
        if gait_events_result is not None and isinstance(gait_events_result, dict):
             gait_events_df = gait_events_result.get("events_df")
             filtered_signals = gait_events_result.get("filtered_signals", {})
             time_vector = gait_events_result.get("time_vector")

        if gait_events_df is None or gait_events_df.empty:
             print("  歩行周期を同定できませんでした。")
             messagebox.showwarning("歩行周期同定", "歩行周期を同定できませんでした。\nパラメータ調整やデータ内容を確認してください。", parent=self.master)
             print("\n[ステップ3] PCI計算はスキップされます")
        else:
             print(f"  {len(gait_events_df)} 個の歩行周期（IC/FOペア）を同定しました。")
             output_gait_file = DATA_FOLDER / (base_filename + OUTPUT_SUFFIX_GAIT)
             print(f"\n[ステップ2.1] 歩行周期データを '{output_gait_file.name}' に保存中...")
             save_results(output_gait_file, gait_events_df, "歩行周期データ")
             print("\n--- 同定された歩行周期データ (最初の5件) ---"); print(gait_events_df[['Leg', 'Cycle', 'IC_Time', 'FO_Time']].head().to_string()); print("---")

             # --- ステップ 2.2: IC/FOイベントのプロット ---
             self.plot_gait_events(gait_events_df, filtered_signals, time_vector)

             # --- ステップ 3: PCI計算 (Placeholder) ---
             print("\n[ステップ3] PCI計算を実行中...")
             pci_results = calculate_pci(gait_events_df)
             if pci_results is None: print("  (PCI計算 未実装/失敗)")
             else: print(f"\n  PCI 結果:", pci_results) # ...保存など...

        # --- ステップ 4: 同期済み角速度の最終プロット ---
        self.plot_final_synchronized_data(sync_gyro_df)

        print("\n--- 解析パイプライン 完了 ---")

    # --- IC/FO イベントプロット用メソッド ---
    def plot_gait_events(self, gait_events_df, filtered_signals, time_vector):
        print("\n[ステップ2.2] IC/FO イベントをグラフにプロットします...")
        try:
            fig_events, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
            fig_events.suptitle(f'検出された歩行イベント - {self.input_file_path.name}')
            plot_successful = False
            for i, leg in enumerate(['L', 'R']):
                ax = axes[i]
                filt_signal = filtered_signals.get(leg)
                if filt_signal is None or time_vector is None or len(filt_signal) != len(time_vector):
                     print(f"  警告: {leg}脚のプロットデータ不足。"); ax.set_title(f'{leg} Leg - データ不足'); continue

                ax.plot(time_vector, filt_signal, label=f'{leg} Gyro Z (Filtered)', alpha=0.7)
                leg_events = gait_events_df[gait_events_df['Leg'] == leg]
                ic_indices = leg_events['IC_Index'].dropna().astype(int).values
                fo_indices = leg_events['FO_Index'].dropna().astype(int).values
                valid_ic = ic_indices[(ic_indices >= 0) & (ic_indices < len(filt_signal))]
                valid_fo = fo_indices[(fo_indices >= 0) & (fo_indices < len(filt_signal))]
                if len(valid_ic) > 0: ax.plot(time_vector[valid_ic], filt_signal[valid_ic], 'ro', markersize=6, label='IC (踵接地)')
                if len(valid_fo) > 0: ax.plot(time_vector[valid_fo], filt_signal[valid_fo], 'gx', markersize=8, markeredgewidth=2, label='FO (爪先離地)')
                ax.set_title(f'{leg} 脚'); ax.set_ylabel('角速度 (単位?)'); ax.legend(loc='upper right'); ax.grid(True)
                plot_successful = True # 少なくとも片脚はプロットできた

            if plot_successful:
                 axes[1].set_xlabel('時間 (s)')
                 plt.tight_layout(rect=[0, 0.03, 1, 0.96])
                 plt.show(block=False)
            else:
                 plt.close(fig_events) # プロットできなかったFigureは閉じる
                 print("  イベントグラフはプロットされませんでした。")

        except Exception as e:
            print(f"  エラー: IC/FOイベントのプロット中にエラー: {e}")

    # --- 同期済み角速度の最終プロット用メソッド ---
    def plot_final_synchronized_data(self, sync_gyro_df):
         if sync_gyro_df is not None:
             print(f"\n[ステップ4] 同期済み{ALIGN_TARGET_SUFFIX}のグラフ (先頭部分) を表示します...")
             try:
                 if not sync_gyro_df.empty:
                      num_samples_final_plot = 1000; sync_gyro_df_short = sync_gyro_df.head(num_samples_final_plot); actual_plot_len = len(sync_gyro_df_short)
                      plt.figure(figsize=(12, 6)) # 新しいウィンドウ
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
    # Tkinter ルートウィンドウの作成と非表示
    root = tk.Tk()
    root.withdraw() # GUIアプリケーション本体が表示されるまで非表示

    # アプリケーションインスタンスの作成と実行
    app = GaitAnalysisApp(Toplevel(root)) # Toplevelをメインウィンドウとして使う
    app.master.protocol("WM_DELETE_WINDOW", root.destroy) # ウィンドウを閉じたら終了

    try:
        root.mainloop() # イベントループ開始
    except KeyboardInterrupt:
        print("\nCtrl+C により中断されました。")
    finally:
         # Mainloop終了後（ウィンドウが閉じられた後）にここに来る場合がある
         print("\n========================================")
         print("=== アプリケーション終了 ===")
         print("========================================")