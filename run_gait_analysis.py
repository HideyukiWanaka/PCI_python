# run_gait_analysis.py

# --- 各機能ファイルをインポート ---
from preprocessing import preprocess_angular_velocity
from gait_cycles import identify_gait_cycles # 実装済みのものをインポート
from pci import calculate_pci             # Placeholder
from file_utils import find_latest_csv_file, save_results

# --- 必要な標準ライブラリ・外部ライブラリをインポート ---
import pandas as pd
import numpy as np
from pathlib import Path

# --- GUIとプロットのためのインポート ---
import tkinter as tk
from tkinter import Frame, Label, Entry, Button, BOTH, W, LEFT, messagebox, HORIZONTAL, Scale, Toplevel, DISABLED, NORMAL # DISABLED, NORMAL を追加
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
DATA_FOLDER = Path("./") # データ検索フォルダ (デフォルト: スクリプトと同じ場所)
OUTPUT_SUFFIX_SYNC = '_synchronized_gyro_z.csv' # 同期済み角速度データの接尾辞
OUTPUT_SUFFIX_GAIT = '_gait_events.csv'     # 歩行周期データの接尾辞
OUTPUT_SUFFIX_PCI = '_pci_results.csv'   # (将来用) PCI結果の接尾辞

# 前処理パラメータ
ROWS_TO_SKIP = 11                 # スキップするヘッダー前の行数
SAMPLING_INTERVAL_MS = 5          # サンプリング周期 (ms)
SYNC_SIGNAL_SUFFIX = '_Acc_Y'     # 同期に使う信号の軸
ALIGN_TARGET_SUFFIX = '_Gyro_Z'   # 同期を適用する信号
RIGHT_PREFIX = 'R'                # 右センサーのプレフィックス
LEFT_PREFIX = 'L'                 # 左センサーのプレフィックス
TRUNK_PREFIX = 'T'                # 体幹センサーのプレフィックス (列名設定用)

# find_peaks のデフォルトパラメータ (ユーザー入力用)
DEFAULT_PEAK_HEIGHT = 10.0        # find_peaks デフォルト Height
DEFAULT_PEAK_PROMINENCE = 0.3     # find_peaks デフォルト Prominence
DEFAULT_PEAK_DISTANCE = 50        # find_peaks デフォルト Distance
NUM_SAMPLES_TO_PLOT = 1000        # 初期プロットで表示するサンプル数

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
        master.title("歩行データ同期・周期同定ツール") # タイトル変更
        master.geometry("900x750") # 少し縦長に

        self.input_file_path = None
        self.sampling_rate = 1000.0 / SAMPLING_INTERVAL_MS

        # --- GUI要素の作成 ---
        top_frame = Frame(master, pady=5)
        top_frame.pack(side=tk.TOP, fill=tk.X)
        self.file_label = Label(top_frame, text="処理対象ファイル: 検索中...")
        self.file_label.pack(side=LEFT, padx=10)
        self.status_label = Label(top_frame, text="初期化中...") # ステータス表示用
        self.status_label.pack(side=LEFT, padx=10)

        # Matplotlibグラフ描画エリア
        self.fig = Figure(figsize=(8, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=master)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=BOTH, expand=True)
        toolbar_frame = Frame(master); toolbar_frame.pack(side=tk.TOP, fill=tk.X)
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame); self.toolbar.update()

        # --- パラメータ入力エリア ---
        param_frame = Frame(master, pady=10)
        param_frame.pack(side=tk.TOP, fill=tk.X)

        # find_peaks パラメータ用変数
        self.height_var = tk.DoubleVar(value=DEFAULT_PEAK_HEIGHT)
        self.prominence_var = tk.DoubleVar(value=DEFAULT_PEAK_PROMINENCE)
        self.distance_var = tk.IntVar(value=DEFAULT_PEAK_DISTANCE)

        # スライダーと値表示ラベルの配置
        Label(param_frame, text="Height:").grid(row=0, column=0, padx=5, sticky=W)
        self.height_scale = Scale(param_frame, from_=0, to=50, resolution=0.1, orient=HORIZONTAL, variable=self.height_var, length=150)
        self.height_scale.grid(row=0, column=1, padx=5)
        self.height_label_val = Label(param_frame, text=f"{self.height_var.get():.1f}", width=5)
        self.height_label_val.grid(row=0, column=2)
        self.height_scale.config(command=lambda v: self.height_label_val.config(text=f"{float(v):.1f}"))

        Label(param_frame, text="Prominence:").grid(row=0, column=3, padx=5, sticky=W)
        self.prominence_scale = Scale(param_frame, from_=0, to=10, resolution=0.1, orient=HORIZONTAL, variable=self.prominence_var, length=150)
        self.prominence_scale.grid(row=0, column=4, padx=5)
        self.prominence_label_val = Label(param_frame, text=f"{self.prominence_var.get():.1f}", width=5)
        self.prominence_label_val.grid(row=0, column=5)
        self.prominence_scale.config(command=lambda v: self.prominence_label_val.config(text=f"{float(v):.1f}"))

        Label(param_frame, text="Distance:").grid(row=1, column=0, padx=5, sticky=W)
        self.distance_scale = Scale(param_frame, from_=1, to=200, resolution=1, orient=HORIZONTAL, variable=self.distance_var, length=150)
        self.distance_scale.grid(row=1, column=1, padx=5)
        self.distance_label_val = Label(param_frame, text=f"{self.distance_var.get():d}", width=5)
        self.distance_label_val.grid(row=1, column=2)
        self.distance_scale.config(command=lambda v: self.distance_label_val.config(text=f"{int(float(v)):d}"))

        # 分析範囲入力フィールド
        Label(param_frame, text="分析開始時間(s):").grid(row=2, column=0, padx=5, pady=5, sticky=W)
        self.start_time_entry = Entry(param_frame, width=10)
        self.start_time_entry.grid(row=2, column=1, padx=5, pady=5, sticky=W)
        self.start_time_entry.insert(0, "") # 初期値は空

        Label(param_frame, text="分析終了時間(s):").grid(row=2, column=3, padx=5, pady=5, sticky=W)
        self.end_time_entry = Entry(param_frame, width=10)
        self.end_time_entry.grid(row=2, column=4, padx=5, pady=5, sticky=W)
        self.end_time_entry.insert(0, "") # 初期値は空

        # 実行ボタン (最初は無効)
        self.run_button = Button(param_frame, text="同期実行 ＆ 歩行周期同定", command=self.run_analysis_pipeline, width=25, state=DISABLED)
        self.run_button.grid(row=2, column=5, rowspan=2, padx=20, pady=5, sticky=tk.S + tk.E) # 位置調整

        # アプリケーション開始時にデータをロードしてプロット
        self.status_label.config(text="最新ファイル検索中...")
        self.master.update_idletasks() # GUIの更新を強制
        # Tkinterのafterを使って、GUIの準備ができてから重い処理を開始
        self.master.after(100, self.load_and_plot_initial_data) # 100ms後に実行

    # --- 初期データのロードとプロット ---
    def load_and_plot_initial_data(self):
        print("\n[準備] 最新のCSVデータを検索中...")
        self.input_file_path = find_latest_csv_file(DATA_FOLDER)
        if self.input_file_path is None: messagebox.showerror("エラー", "CSVファイル未検出"); self.master.quit(); return
        self.file_label.config(text=f"処理対象: {self.input_file_path.name}")

        self.status_label.config(text="データ読込・プロット中...")
        self.master.update_idletasks()

        try:
            # データの一時読み込み & 列名設定
            temp_df = None
            try: temp_df = pd.read_csv(self.input_file_path, skiprows=ROWS_TO_SKIP, encoding='cp932')
            except UnicodeDecodeError: temp_df = pd.read_csv(self.input_file_path, skiprows=ROWS_TO_SKIP, encoding='shift_jis')
            expected_cols = get_expected_column_names(RIGHT_PREFIX, LEFT_PREFIX, TRUNK_PREFIX)
            if len(temp_df.columns) == len(expected_cols):
                temp_df.columns = expected_cols
                blank_cols = [col for col in temp_df.columns if 'Blank_' in col]; temp_df = temp_df.drop(columns=blank_cols)
            else: raise ValueError(f"初期読み込みで列数が不一致 ({len(temp_df.columns)} vs {len(expected_cols)})")

            # プロット用データの抽出と準備
            sync_l_full = temp_df[f'{LEFT_PREFIX}{SYNC_SIGNAL_SUFFIX}'].fillna(0).values
            sync_r_full = temp_df[f'{RIGHT_PREFIX}{SYNC_SIGNAL_SUFFIX}'].fillna(0).values
            plot_len = min(len(sync_l_full), len(sync_r_full), NUM_SAMPLES_TO_PLOT)
            if plot_len <= 0: raise ValueError("プロットデータなし")
            sync_l_short_plot = sync_l_full[:plot_len]; sync_r_short_plot = sync_r_full[:plot_len]
            time_plot = np.arange(plot_len) * (SAMPLING_INTERVAL_MS / 1000.0)

            # Matplotlib Axes にプロット
            self.ax.clear()
            self.ax.plot(time_plot, sync_l_short_plot, label=f'左 ({LEFT_PREFIX}{SYNC_SIGNAL_SUFFIX})', alpha=0.8)
            self.ax.plot(time_plot, sync_r_short_plot, label=f'右 ({RIGHT_PREFIX}{SYNC_SIGNAL_SUFFIX})', linestyle='--', alpha=0.8)
            self.ax.set_title(f'同期用信号 先頭 {plot_len} サンプル (パラメータ調整用)')
            self.ax.set_xlabel('時間 (s)')
            self.ax.set_ylabel('信号値 (単位?)')
            self.ax.legend()
            self.ax.grid(True)
            self.canvas.draw() # キャンバスに描画

            # 処理完了 -> ボタン有効化 & ステータス更新
            self.run_button.config(state=NORMAL)
            self.status_label.config(text="準備完了. パラメータ/範囲調整後、ボタン実行.")
            print(f"  プロット完了。パラメータ/範囲を調整してボタンを押してください。")

        except Exception as e:
            self.status_label.config(text="エラー発生！")
            messagebox.showerror("エラー", f"初期データのプロット中にエラー:\n{e}", parent=self.master)
            self.run_button.config(state=DISABLED) # エラー時はボタン無効のまま

    # --- 「同期実行 ＆ 歩行周期同定」ボタンの処理 ---
    def run_analysis_pipeline(self):
        # ボタン無効化 & ステータス更新
        self.run_button.config(state=DISABLED)
        self.status_label.config(text="解析処理実行中...")
        self.master.update_idletasks()

        try: # メインの処理
            print("\n========================================")
            print("=== 解析パイプライン実行 ===")

            # パラメータ取得
            user_peak_height = self.height_var.get()
            user_peak_prominence = self.prominence_var.get()
            user_peak_distance = self.distance_var.get()
            print(f"  使用 find_peaks パラメータ: height={user_peak_height:.2f}, prominence={user_peak_prominence:.2f}, distance={user_peak_distance}")

            # 分析範囲取得
            start_time_str = self.start_time_entry.get().strip()
            end_time_str = self.end_time_entry.get().strip()
            start_time_sec, end_time_sec = None, None
            try:
                if start_time_str: start_time_sec = float(start_time_str)
                if end_time_str: end_time_sec = float(end_time_str)
                print(f"  分析範囲: 開始={start_time_sec if start_time_sec is not None else '全体'}s, 終了={end_time_sec if end_time_sec is not None else '全体'}s")
            except ValueError:
                messagebox.showerror("入力エラー", "分析開始時間・終了時間には数値を入力してください。", parent=self.master)
                return # finallyブロックが実行される

            # === ステップ 1: 前処理 ===
            print("\n[ステップ1] 角速度データの前処理...")
            result_tuple = preprocess_angular_velocity(
                data_file=self.input_file_path, rows_to_skip=ROWS_TO_SKIP, sampling_interval_ms=SAMPLING_INTERVAL_MS,
                right_prefix=RIGHT_PREFIX, left_prefix=LEFT_PREFIX, trunk_prefix=TRUNK_PREFIX,
                sync_col_suffix=SYNC_SIGNAL_SUFFIX, align_col_suffix=ALIGN_TARGET_SUFFIX,
                peak_height=user_peak_height, peak_prominence=user_peak_prominence, peak_distance=user_peak_distance
            )
            if result_tuple is None or len(result_tuple) != 7: messagebox.showerror("エラー", "前処理失敗(戻り値不正)"); return
            sync_gyro_df, lag_samples, self.sampling_rate, peak_idx_l, peak_idx_r, _, _ = result_tuple
            if sync_gyro_df is None: messagebox.showerror("エラー", "前処理失敗"); return

            result_message = f"前処理完了。\n左ピーク idx: {peak_idx_l}, 右ピーク idx: {peak_idx_r}\n同期ラグ: {lag_samples} サンプル"
            print(f"\n[成功] {result_message}"); messagebox.showinfo("前処理完了", result_message, parent=self.master)
            base_filename = self.input_file_path.stem; output_sync_file = DATA_FOLDER / (base_filename + OUTPUT_SUFFIX_SYNC)
            print(f"\n[ステップ1.1] 同期済みデータを保存中..."); save_results(output_sync_file, sync_gyro_df, "同期済み角速度データ")


            # === ステップ 2: 歩行周期同定 ===
            gait_events_df = None; filtered_signals = {}; time_vector = None # 初期化
            if sync_gyro_df is not None:
                print("\n[ステップ2] 歩行周期の同定を実行中...")
                gait_events_result = identify_gait_cycles(
                    sync_gyro_df=sync_gyro_df, sampling_rate_hz=self.sampling_rate, swing_threshold=200,
                    start_time_sec=start_time_sec, end_time_sec=end_time_sec # 範囲指定
                )
                if gait_events_result is not None and isinstance(gait_events_result, dict):
                     gait_events_df = gait_events_result.get("events_df")
                     filtered_signals = gait_events_result.get("filtered_signals", {})
                     time_vector = gait_events_result.get("time_vector")

                if gait_events_df is None or gait_events_df.empty:
                     print("  歩行周期を同定できませんでした。"); messagebox.showwarning("歩行周期同定", "歩行周期同定失敗", parent=self.master)
                     print("\n[ステップ3] PCI計算スキップ")
                else:
                     print(f"  {len(gait_events_df)} 個の歩行周期同定"); output_gait_file = DATA_FOLDER / (base_filename + OUTPUT_SUFFIX_GAIT)
                     print(f"\n[ステップ2.1] 歩行周期データ保存中..."); save_results(output_gait_file, gait_events_df, "歩行周期データ")
                     print("\n--- 同定結果 (最初の5件) ---"); print(gait_events_df[['Leg', 'Cycle', 'IC_Time', 'FO_Time']].head().to_string()); print("---")
                     # --- ステップ 2.2: IC/FOイベントのプロット ---
                     self.plot_gait_events(gait_events_df, filtered_signals, time_vector)
                     # --- ステップ 3: PCI計算 (Placeholder) ---
                     print("\n[ステップ3] PCI計算実行中..."); pci_results = calculate_pci(gait_events_df)
                     if pci_results is None: print("  (PCI計算 未実装/失敗)")
                     else: print(f"\n  PCI 結果:", pci_results) # ...保存など...
            else: print("\n[情報] 前処理失敗のため歩行周期同定以降スキップ。")


            # === ステップ 4: 同期済み角速度の最終プロット ===
            self.plot_final_synchronized_data(sync_gyro_df)

            print("\n--- 解析パイプライン 完了 ---")
            self.status_label.config(text="解析完了")

        except Exception as e: # パイプライン全体のエラー
             print(f"\n[エラー] 解析パイプライン実行中にエラー: {e}")
             messagebox.showerror("実行時エラー", f"解析中にエラー:\n{e}", parent=self.master)
             self.status_label.config(text="エラー発生")
        finally:
             # 処理終了後 (成功・失敗問わず) ボタンを再度有効化
             self.run_button.config(state=NORMAL)
             self.master.update_idletasks()


    # --- IC/FO イベントプロット用メソッド ---
    def plot_gait_events(self, gait_events_df, filtered_signals, time_vector):
        print("\n[ステップ2.2] IC/FO イベントをグラフにプロットします...")
        if gait_events_df is None or gait_events_df.empty: return # データなければ何もしない
        try:
            fig_events, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
            fig_events.suptitle(f'検出された歩行イベント - {self.input_file_path.name}')
            plot_successful = False
            # プロット対象の時間ベクトルと信号長を決定 (スライスされている可能性があるため)
            if time_vector is None or len(time_vector) == 0: # time_vectorがgait_cyclesから返されなかった場合
                 # filtered_signalsから代表的な長さを取得(例: 左脚)
                 sig_len = len(filtered_signals.get('L', []))
                 if sig_len == 0: sig_len = len(filtered_signals.get('R',[]))
                 if sig_len > 0 : time_vector = np.arange(sig_len) * (1.0/self.sampling_rate) # 時間を再計算
                 else: raise ValueError("プロット用の時間ベクトルが不明")

            for i, leg in enumerate(['L', 'R']):
                ax = axes[i]
                filt_signal = filtered_signals.get(leg)
                # スライスされた長さと時間ベクトル長が一致するか確認
                if filt_signal is None or len(filt_signal) != len(time_vector):
                     print(f"  警告: {leg}脚のフィルタ信号/時間ベクトル不一致。プロットスキップ。")
                     ax.set_title(f'{leg} 脚 - データ不一致'); continue

                ax.plot(time_vector, filt_signal, label=f'{leg} Gyro Z (Filtered)', alpha=0.7)
                leg_events = gait_events_df[gait_events_df['Leg'] == leg]
                # インデックスは全体座標なので、プロットの時間ベクトル/信号に合わせるオフセットが必要
                # -> gait_cyclesで時間ベースにしたのでTimeを使うのが確実
                ic_times = leg_events['IC_Time'].dropna().values
                fo_times = leg_events['FO_Time'].dropna().values

                # 時間に対応する信号の値を見つける (補間がより正確だが、近いインデックスで代用)
                ic_indices_plot = np.searchsorted(time_vector, ic_times, side='left')
                fo_indices_plot = np.searchsorted(time_vector, fo_times, side='left')
                # 配列境界チェック
                ic_indices_plot = ic_indices_plot[ic_indices_plot < len(time_vector)]
                fo_indices_plot = fo_indices_plot[fo_indices_plot < len(time_vector)]

                if len(ic_indices_plot) > 0: ax.plot(time_vector[ic_indices_plot], filt_signal[ic_indices_plot], 'ro', markersize=6, label='IC (踵接地)', linestyle='None')
                if len(fo_indices_plot) > 0: ax.plot(time_vector[fo_indices_plot], filt_signal[fo_indices_plot], 'gx', markersize=8, markeredgewidth=2, label='FO (爪先離地)', linestyle='None')

                ax.set_title(f'{leg} 脚'); ax.set_ylabel('角速度 (単位?)'); ax.legend(loc='upper right'); ax.grid(True)
                plot_successful = True

            if plot_successful:
                 axes[1].set_xlabel('時間 (s)'); plt.tight_layout(rect=[0, 0.03, 1, 0.96]); plt.show(block=False)
            else: plt.close(fig_events)
        except Exception as e: print(f"  エラー: IC/FOイベントプロット中にエラー: {e}")

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

    # Toplevelをメインウィンドウとしてアプリケーションインスタンスを作成
    app_window = Toplevel(root)
    app = GaitAnalysisApp(app_window)
    # ウィンドウが閉じられたときにTkinterのメインループを終了させる
    app_window.protocol("WM_DELETE_WINDOW", root.destroy)

    try:
        root.mainloop() # イベントループ開始
    except KeyboardInterrupt:
        print("\nCtrl+C により中断されました。")
    finally:
         # Mainloop終了後
         print("\n========================================")
         print("=== アプリケーション終了 ===")
         print("========================================")