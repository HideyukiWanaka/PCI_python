# run_gait_analysis.py

# --- 各機能ファイルをインポート ---
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from preprocessing import preprocess_angular_velocity
from gait_cycles import identify_gait_cycles  # Placeholder
from pci import calculate_pci             # Placeholder
from file_utils import find_latest_csv_file, save_results

# --- 必要な標準ライブラリ・外部ライブラリをインポート ---
import pandas as pd
import numpy as np
from pathlib import Path

# --- GUIとプロットのためのインポート ---
import tkinter as tk
from tkinter import Frame, Label, Entry, Button, BOTH, W, LEFT, messagebox, HORIZONTAL, Scale  # Scale を追加
import matplotlib
matplotlib.use('TkAgg')  # Tkinter連携用バックエンドを明示的に指定

# ★★★ japanize-matplotlib をインポート ★★★
try:
    import japanize_matplotlib  # これだけで日本語表示が有効になる
    print("japanize_matplotlib をインポートしました。")
except ImportError:
    print("警告: japanize-matplotlib が見つかりません。'pip install japanize-matplotlib' でインストールしてください。")
# ★★★ 追加ここまで ★★★


# --- 解析実行のための設定 (変更なし) ---
DATA_FOLDER = Path("./")
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
DEFAULT_PEAK_HEIGHT = 15.0
DEFAULT_PEAK_PROMINENCE = 0.3
DEFAULT_PEAK_DISTANCE = 50
NUM_SAMPLES_TO_PLOT = 1000

# --- 列名定義関数 (変更なし) ---


def get_expected_column_names(right_prefix, left_prefix, trunk_prefix):
    return [  # ... (省略 - 33列分のリスト) ...
        'Time', f'{right_prefix}_Acc_X', f'{right_prefix}_Acc_Y', f'{right_prefix}_Acc_Z', f'{right_prefix}_Ch4_V',
        f'{right_prefix}_Gyro_X', f'{right_prefix}_Gyro_Y', f'{right_prefix}_Gyro_Z', f'{right_prefix}_Ch8_V',
        f'{left_prefix}_Acc_X', f'{left_prefix}_Acc_Y', f'{left_prefix}_Acc_Z', 'Blank_1',
        f'{left_prefix}_Gyro_X', f'{left_prefix}_Gyro_Y', f'{left_prefix}_Gyro_Z', 'Blank_2',
        f'{trunk_prefix}_Acc_X', f'{trunk_prefix}_Acc_Y', f'{trunk_prefix}_Acc_Z', 'Blank_3',
        f'{trunk_prefix}_Gyro_X', f'{trunk_prefix}_Gyro_Y', f'{trunk_prefix}_Gyro_Z', 'Blank_4',
        'Blank_5', 'Blank_6', 'Blank_7', 'Blank_8', 'Blank_9', 'Blank_10', 'Blank_11', 'Blank_12']

# --- ★★★ Tkinter GUI アプリケーションクラス ★★★ ---


class GaitAnalysisApp:
    def __init__(self, master):
        self.master = master
        master.title("歩行データ同期パラメータ調整")
        master.geometry("900x700")  # ウィンドウサイズ調整

        self.input_file_path = None
        self.sync_l_short_plot = None
        self.sync_r_short_plot = None
        self.plot_len = 0
        self.time_plot = None

        # --- GUI要素の作成 ---
        top_frame = Frame(master, pady=5)
        top_frame.pack(side=tk.TOP, fill=tk.X)

        # ファイル選択（今回は自動検索なのでラベル表示のみ）
        self.file_label = Label(top_frame, text="処理対象ファイル: 検索中...")
        self.file_label.pack(side=LEFT, padx=10)

        # Matplotlibグラフ描画エリア
        self.fig = Figure(figsize=(8, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=master)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=BOTH, expand=True)

        # Matplotlibツールバー
        toolbar_frame = Frame(master)
        toolbar_frame.pack(side=tk.TOP, fill=tk.X)
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()

        # パラメータ入力エリア
        param_frame = Frame(master, pady=10)
        param_frame.pack(side=tk.TOP, fill=tk.X)

        # スライダー用の変数
        self.height_var = tk.DoubleVar(value=DEFAULT_PEAK_HEIGHT)
        self.prominence_var = tk.DoubleVar(value=DEFAULT_PEAK_PROMINENCE)
        self.distance_var = tk.IntVar(value=DEFAULT_PEAK_DISTANCE)

        # Height
        Label(param_frame, text="Height:").grid(
            row=0, column=0, padx=5, sticky=W)
        self.height_scale = Scale(param_frame, from_=0, to=50, resolution=0.1,
                                  orient=HORIZONTAL, variable=self.height_var, length=150)  # 範囲は調整
        self.height_scale.grid(row=0, column=1, padx=5)
        self.height_label_val = Label(
            param_frame, text=f"{self.height_var.get():.1f}", width=5)
        self.height_label_val.grid(row=0, column=2)
        self.height_scale.config(
            command=lambda v: self.height_label_val.config(text=f"{float(v):.1f}"))

        # Prominence
        Label(param_frame, text="Prominence:").grid(
            row=0, column=3, padx=5, sticky=W)
        self.prominence_scale = Scale(param_frame, from_=0, to=10, resolution=0.1,
                                      orient=HORIZONTAL, variable=self.prominence_var, length=150)  # 範囲は調整
        self.prominence_scale.grid(row=0, column=4, padx=5)
        self.prominence_label_val = Label(
            param_frame, text=f"{self.prominence_var.get():.1f}", width=5)
        self.prominence_label_val.grid(row=0, column=5)
        self.prominence_scale.config(
            command=lambda v: self.prominence_label_val.config(text=f"{float(v):.1f}"))

        # Distance
        Label(param_frame, text="Distance:").grid(
            row=1, column=0, padx=5, sticky=W)
        self.distance_scale = Scale(param_frame, from_=1, to=200, resolution=1,
                                    orient=HORIZONTAL, variable=self.distance_var, length=150)  # 範囲は調整
        self.distance_scale.grid(row=1, column=1, padx=5)
        self.distance_label_val = Label(
            param_frame, text=f"{self.distance_var.get():d}", width=5)
        self.distance_label_val.grid(row=1, column=2)
        self.distance_scale.config(
            command=lambda v: self.distance_label_val.config(text=f"{int(float(v)):d}"))

        # 実行ボタン
        self.run_button = Button(
            param_frame, text="同期実行", command=self.run_preprocessing, width=15)
        self.run_button.grid(row=1, column=3, columnspan=3, padx=20, pady=5)

        # アプリケーション開始時にデータをロードしてプロット
        self.load_and_plot_initial_data()

    # --- 初期データのロードとプロット ---
    def load_and_plot_initial_data(self):
        print("\n[準備] 最新のCSVデータを検索中...")
        self.input_file_path = find_latest_csv_file(DATA_FOLDER)

        if self.input_file_path is None:
            messagebox.showerror(
                "エラー", f"データフォルダ '{DATA_FOLDER.resolve()}'\nにCSVファイルが見つかりません。")
            self.master.quit()
            return

        self.file_label.config(text=f"処理対象ファイル: {self.input_file_path.name}")
        print(f"処理対象ファイル: '{self.input_file_path.name}'")

        print(
            f"\n[ステップ0] 同期用信号 ({SYNC_SIGNAL_SUFFIX}) の先頭 {NUM_SAMPLES_TO_PLOT} サンプルをプロット...")
        try:
            # --- データの一時読み込み & 列名設定 ---
            temp_df = None
            try:
                temp_df = pd.read_csv(
                    self.input_file_path, skiprows=ROWS_TO_SKIP, encoding='cp932')
            except UnicodeDecodeError:
                temp_df = pd.read_csv(
                    self.input_file_path, skiprows=ROWS_TO_SKIP, encoding='shift_jis')

            expected_cols = get_expected_column_names(
                RIGHT_PREFIX, LEFT_PREFIX, TRUNK_PREFIX)
            if len(temp_df.columns) == len(expected_cols):
                temp_df.columns = expected_cols
                blank_cols = [
                    col for col in temp_df.columns if 'Blank_' in col]
                temp_df = temp_df.drop(columns=blank_cols)
            else:
                raise ValueError(
                    f"列数が不一致: {len(temp_df.columns)} vs {len(expected_cols)}")

            sync_l_full = temp_df[f'{LEFT_PREFIX}{SYNC_SIGNAL_SUFFIX}'].fillna(
                0).values
            sync_r_full = temp_df[f'{RIGHT_PREFIX}{SYNC_SIGNAL_SUFFIX}'].fillna(
                0).values
            self.plot_len = min(len(sync_l_full), len(
                sync_r_full), NUM_SAMPLES_TO_PLOT)
            if self.plot_len <= 0:
                raise ValueError("プロットするデータがありません。")

            self.sync_l_short_plot = sync_l_full[:self.plot_len]
            self.sync_r_short_plot = sync_r_full[:self.plot_len]
            self.time_plot = np.arange(
                self.plot_len) * (SAMPLING_INTERVAL_MS / 1000.0)

            # --- Matplotlib Axes にプロット ---
            self.ax.clear()  # 前回のプロットをクリア
            self.ax.plot(self.time_plot, self.sync_l_short_plot,
                         label=f'左 ({LEFT_PREFIX}{SYNC_SIGNAL_SUFFIX})', alpha=0.8)
            self.ax.plot(self.time_plot, self.sync_r_short_plot,
                         label=f'右 ({RIGHT_PREFIX}{SYNC_SIGNAL_SUFFIX})', linestyle='--', alpha=0.8)
            self.ax.set_title(f'同期用信号 先頭 {self.plot_len} サンプル')
            self.ax.set_xlabel('時間 (s)')
            self.ax.set_ylabel('信号値 (単位?)')
            self.ax.legend()
            self.ax.grid(True)
            self.canvas.draw()  # キャンバスに描画

        except Exception as e:
            messagebox.showerror("エラー", f"初期データのプロット中にエラー:\n{e}")
            self.master.quit()

    # --- 「同期実行」ボタンが押されたときの処理 ---
    def run_preprocessing(self):

        print("--- '同期実行' ボタンがクリックされました ---")

        print("\n[ステップ1] 角速度データの前処理 (GUIパラメータ使用) を実行中...")

        # 現在のスライダー/エントリーの値を取得
        user_peak_height = self.height_var.get()
        user_peak_prominence = self.prominence_var.get()
        user_peak_distance = self.distance_var.get()
        print(
            f"  使用パラメータ: height={user_peak_height:.2f}, prominence={user_peak_prominence:.2f}, distance={user_peak_distance}")

        # preprocessing 関数を呼び出し
        result_tuple = preprocess_angular_velocity(
            data_file=self.input_file_path,
            rows_to_skip=ROWS_TO_SKIP,
            sampling_interval_ms=SAMPLING_INTERVAL_MS,
            right_prefix=RIGHT_PREFIX,
            left_prefix=LEFT_PREFIX,
            trunk_prefix=TRUNK_PREFIX,
            sync_col_suffix=SYNC_SIGNAL_SUFFIX,
            align_col_suffix=ALIGN_TARGET_SUFFIX,
            peak_height=user_peak_height,
            peak_prominence=user_peak_prominence,
            peak_distance=user_peak_distance
        )

        if result_tuple is None or len(result_tuple) != 7:
            messagebox.showerror("エラー", "角速度の前処理に失敗しました (予期せぬ戻り値)。")
            return
        sync_gyro_df, lag_samples, sampling_rate, peak_idx_l, peak_idx_r, _, _ = result_tuple

        if sync_gyro_df is None:
            messagebox.showerror(
                "エラー", "角速度の前処理に失敗しました。\nコンソールのエラーメッセージを確認してください。")
            return
        else:
            # 成功メッセージと結果表示
            result_message = f"前処理完了。\n左ピーク index: {peak_idx_l}, 右ピーク index: {peak_idx_r}\n同期ラグ: {lag_samples} サンプル"
            print(f"\n[成功] {result_message}")
            messagebox.showinfo("前処理完了", result_message)  # ポップアップでも表示

            # --- 同期済みデータの保存 ---
            base_filename = self.input_file_path.stem
            output_sync_file = DATA_FOLDER / \
                (base_filename + OUTPUT_SUFFIX_SYNC)
            print(f"\n[ステップ1.1] 同期済みデータを '{output_sync_file.name}' に保存中...")
            save_results(output_sync_file, sync_gyro_df, "同期済み角速度データ")

            # --- ★★★ 最終結果のプロット (新しいウィンドウ) ★★★ ---
            print("\n[ステップ4] 同期済みZ軸角速度のグラフを新しいウィンドウで表示します...")
            try:
                plt.figure(figsize=(12, 6))  # 新しいFigureを作成
                left_col_name = f'L{ALIGN_TARGET_SUFFIX}_aligned'
                right_col_name = f'R{ALIGN_TARGET_SUFFIX}_aligned'
                if left_col_name in sync_gyro_df.columns and right_col_name in sync_gyro_df.columns:
                    plt.plot(sync_gyro_df['time_aligned_sec'], sync_gyro_df[left_col_name],
                             label=f'同期後 左 {ALIGN_TARGET_SUFFIX} (符号反転済)')
                    plt.plot(sync_gyro_df['time_aligned_sec'], sync_gyro_df[right_col_name],
                             label=f'同期後 右 {ALIGN_TARGET_SUFFIX}', linestyle='--', alpha=0.8)
                    plt.title(
                        f'同期済み角速度 ({ALIGN_TARGET_SUFFIX}) - {self.input_file_path.name}')
                    plt.xlabel('同期後の時間 (秒)')
                    plt.ylabel('角速度 (単位?)')
                    plt.legend()
                    plt.grid(True)
                    plt.show(block=False)  # ★ block=False で非同期表示（推奨）
                else:
                    print(f"  警告: プロットに必要な列が見つかりません。")
            except Exception as e:
                print(f"  エラー: 最終グラフの表示中にエラー: {e}")

            # --- ★★★ 最終結果のプロット (先頭1000サンプルのみ) ★★★ ---
            print(
                f"\n[ステップ4] 同期済み{ALIGN_TARGET_SUFFIX}のグラフ (先頭1000サンプル) を表示します...")
            try:
                if sync_gyro_df is not None and not sync_gyro_df.empty:  # DataFrameがNoneでなく、空でもないことを確認
                    # ★★★ 変更点: 先頭1000サンプルを切り出す ★★★
                    num_samples_final_plot = 1000
                    sync_gyro_df_short = sync_gyro_df.head(
                        num_samples_final_plot)
                    actual_plot_len = len(sync_gyro_df_short)  # 実際にプロットするサンプル数

                    # 新しいウィンドウでプロット
                    plt.figure(figsize=(12, 6))
                    left_col_name = f'L{ALIGN_TARGET_SUFFIX}_aligned'
                    right_col_name = f'R{ALIGN_TARGET_SUFFIX}_aligned'

                    if left_col_name in sync_gyro_df_short.columns and right_col_name in sync_gyro_df_short.columns:
                        # ★★★ 変更点: 短いDataFrame (sync_gyro_df_short) を使用 ★★★
                        plt.plot(sync_gyro_df_short['time_aligned_sec'],
                                 sync_gyro_df_short[left_col_name], label=f'同期後 左 {ALIGN_TARGET_SUFFIX} (符号反転済)')
                        plt.plot(sync_gyro_df_short['time_aligned_sec'], sync_gyro_df_short[right_col_name],
                                 label=f'同期後 右 {ALIGN_TARGET_SUFFIX}', linestyle='--', alpha=0.8)

                        # ★★★ 変更点: タイトルも変更 ★★★
                        plt.title(
                            f'同期済み角速度 ({ALIGN_TARGET_SUFFIX}) - 先頭 {actual_plot_len} サンプル ({self.input_file_path.name})')
                        plt.xlabel('同期後の時間 (秒)')
                        plt.ylabel('角速度 (単位?)')
                        plt.legend()
                        plt.grid(True)
                        plt.show(block=False)  # 非同期表示
                    else:
                        print(
                            f"  警告: プロットに必要な列が見つかりません ({left_col_name} or {right_col_name})。")
                else:
                    print("  警告: 同期済みデータが空または無効なため、最終グラフは表示できません。")

            except Exception as e:
                print(f"  エラー: 最終グラフの表示中にエラー: {e}")
            # --- (将来的に) 歩行周期・PCI計算の呼び出し ---
            # print("\n[ステップ2] 歩行周期の同定を実行中...")
            # gait_events_df = identify_gait_cycles(sync_gyro_df, sampling_rate)
            # ...


# --- メイン実行ブロック ---
if __name__ == "__main__":
    print("========================================")
    print("=== 歩行データ解析 GUI アプリケーション起動 ===")
    print("========================================")

    root = tk.Tk()
    app = GaitAnalysisApp(root)
    root.mainloop()  # Tkinterのイベントループを開始

    print("\n========================================")
    print("=== アプリケーション終了 ===")
    print("========================================")
