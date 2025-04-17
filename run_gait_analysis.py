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
import tkinter as tk
from tkinter import simpledialog, messagebox, Toplevel, Frame, Label, Entry, Button, BOTH, W, LEFT

# --- 解析実行のための設定 ---
# ★★★ これらの設定値は必要に応じて変更してください ★★★
DATA_FOLDER = Path("./") # データ検索フォルダ (デフォルト: スクリプトと同じ場所)
OUTPUT_SUFFIX_SYNC = '_synchronized_gyro_z.csv' # 同期済み角速度データの接尾辞
OUTPUT_SUFFIX_GAIT = '_gait_events.csv'     # (将来用) 歩行周期データの接尾辞
OUTPUT_SUFFIX_PCI = '_pci_results.csv'   # (将来用) PCI結果の接尾辞

# 前処理パラメータ
ROWS_TO_SKIP = 11                 # スキップするヘッダー前の行数
SAMPLING_INTERVAL_MS = 5          # サンプリング周期 (ms)
SYNC_SIGNAL_SUFFIX = '_Acc_Y'     # ★同期に使う信号の軸 (例: _Acc_X, _Acc_Z も試す)
ALIGN_TARGET_SUFFIX = '_Gyro_Z'   # 同期を適用する信号
RIGHT_PREFIX = 'R'                # 右センサーのプレフィックス
LEFT_PREFIX = 'L'                 # 左センサーのプレフィックス
TRUNK_PREFIX = 'T'                # 体幹センサーのプレフィックス (列名設定用)

# find_peaks のデフォルトパラメータ (ユーザー入力用)
DEFAULT_PEAK_HEIGHT = 10         # ★グラフを見て調整★
DEFAULT_PEAK_PROMINENCE = 0.3     # ★グラフを見て調整★
DEFAULT_PEAK_DISTANCE = 50        # ★グラフを見て調整★
NUM_SAMPLES_TO_PLOT = 1000        # 初期プロットで表示するサンプル数
# ★★★ ここまで設定値 ★★★


# --- 日本語フォント設定 ---
try:
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Hiragino Sans', 'Yu Gothic', 'Meiryo', 'TakaoPGothic', 'Noto Sans CJK JP', 'IPAexGothic'] # IPAexGothicも追加
except Exception as e: print(f"警告: 日本語フォント設定エラー: {e}")

# --- ★ ポップアップでパラメータ入力を受け取る関数 ★ ---
def get_parameters_via_popup(defaults):
    """Tkinterを使用してポップアップウィンドウでfind_peaksパラメータを取得する。"""
    result = {} # ユーザー入力を格納する辞書

    # --- OKボタンが押されたときの処理 ---
    def on_ok():
        nonlocal result # 外側の result 変数を参照
        try:
            # Entryウィジェットから文字列を取得
            h_str = height_entry.get()
            p_str = prominence_entry.get()
            d_str = distance_entry.get()

            # --- 入力値の検証とデフォルト値の使用 ---
            # 空欄ならデフォルト、そうでなければfloatに変換
            h = float(h_str) if h_str.strip() else defaults.get('height')
            p = float(p_str) if p_str.strip() else defaults.get('prominence')
            # Distanceは少し複雑: 空欄ならデフォルト、入力あればint変換
            d_val = d_str.strip()
            d = None # distanceはNone許容
            if d_val: # 何か入力されている場合
                 d_int = int(float(d_val)) # float経由で整数化
                 if d_int <= 0: raise ValueError("Distance は1以上の整数である必要があります")
                 d = d_int
            elif defaults.get('distance') is not None: # 空欄でデフォルト値がある場合
                 d = int(defaults.get('distance')) # デフォルト値を整数化

            # 簡単な範囲チェック (負の値でないか)
            if h is not None and h < 0: raise ValueError("Height は0以上である必要があります")
            if p is not None and p < 0: raise ValueError("Prominence は0以上である必要があります")

            # 結果を辞書に格納
            result['height'] = h
            result['prominence'] = p
            result['distance'] = d
            popup.destroy() # ポップアップウィンドウを閉じる

        except ValueError as ve:
            messagebox.showerror("入力エラー", f"無効な数値が入力されました:\n{ve}", parent=popup) # エラーをポップアップで表示
        except Exception as ex:
             messagebox.showerror("エラー", f"予期せぬエラーが発生しました:\n{ex}", parent=popup)

    # --- キャンセルボタンが押されたときの処理 ---
    def on_cancel():
         nonlocal result
         result = None # キャンセルされたことを示すためにNoneを格納
         popup.destroy()

    # --- ポップアップウィンドウの作成 ---
    # Toplevelはメインウィンドウ(今回は非表示)の上に表示される
    popup = Toplevel(root) # 非表示のrootを親とする
    popup.title("Find Peaks パラメータ入力")
    popup.geometry("380x200") # ウィンドウサイズ

    # 他のウィンドウ操作を禁止 (モーダル)
    popup.grab_set()
    popup.transient(root) # 親ウィンドウ(root)の上に表示

    # --- ウィジェットの配置 ---
    main_frame = Frame(popup, padx=15, pady=15)
    main_frame.pack(expand=True, fill=BOTH)

    Label(main_frame, text="グラフを確認し、パラメータを入力してください\n(空欄のままOKでデフォルト値を使用)").grid(row=0, column=0, columnspan=2, pady=(0, 15), sticky=W)

    # Height
    Label(main_frame, text=f"Height (高さ):").grid(row=1, column=0, sticky=W, padx=5, pady=3)
    height_entry = Entry(main_frame, width=18)
    height_entry.grid(row=1, column=1, sticky=W, padx=5, pady=3)
    height_entry.insert(0, str(defaults.get('height', ''))) # デフォルト値を表示

    # Prominence
    Label(main_frame, text=f"Prominence (突出度):").grid(row=2, column=0, sticky=W, padx=5, pady=3)
    prominence_entry = Entry(main_frame, width=18)
    prominence_entry.grid(row=2, column=1, sticky=W, padx=5, pady=3)
    prominence_entry.insert(0, str(defaults.get('prominence', '')))

    # Distance
    Label(main_frame, text=f"Distance (サンプル間隔):").grid(row=3, column=0, sticky=W, padx=5, pady=3)
    distance_entry = Entry(main_frame, width=18)
    distance_entry.grid(row=3, column=1, sticky=W, padx=5, pady=3)
    distance_entry.insert(0, str(defaults.get('distance', '')))

    # --- ボタン ---
    button_frame = Frame(main_frame)
    button_frame.grid(row=4, column=0, columnspan=2, pady=(15, 0))
    ok_button = Button(button_frame, text="OK", width=12, command=on_ok)
    ok_button.pack(side=LEFT, padx=10)
    cancel_button = Button(button_frame, text="キャンセル", width=12, command=on_cancel)
    cancel_button.pack(side=LEFT, padx=10)

    # ウィンドウを中央に表示（おまじない）
    popup.update_idletasks()
    x = root.winfo_screenwidth() // 2 - popup.winfo_width() // 2
    y = root.winfo_screenheight() // 2 - popup.winfo_height() // 2
    popup.geometry(f"+{x}+{y}")

    # ユーザーの操作を待つ
    popup.wait_window()

    return result # 結果の辞書 (またはキャンセル時は None) を返す

# --- ユーザーからパラメータ入力を受け取るヘルパー関数 ---
def get_parameter_input(prompt, default_value):
    """指定されたプロンプトでユーザー入力を求め、数値に変換して返す。Enterのみならデフォルト値。"""
    while True:
        try:
            # input() は文字列を返す
            default_display = f"(デフォルト: {default_value})" if default_value is not None else "(デフォルト: なし)"
            user_input_str = input(f"  {prompt} {default_display}: ")

            if not user_input_str.strip(): # 空文字かチェック (Enterのみ)
                print(f"    -> デフォルト値 {default_value} を使用します。")
                # distance のデフォルトが int でない可能性を考慮
                if default_value is not None and "distance" in prompt.lower():
                    try:
                        return int(default_value)
                    except (ValueError, TypeError):
                         print(f"  警告: distance のデフォルト値 ({default_value}) を整数に変換できません。None を使用します。")
                         return None # または 1 などの安全な値
                return default_value # float or None

            value_f = float(user_input_str) # まずfloatに変換試行
            if value_f < 0:
                 print("  エラー: 値は0以上である必要があります。")
                 continue

            # distance の場合は整数にする
            if "distance" in prompt.lower():
                value_i = int(value_f)
                if value_f != value_i or value_i <= 0: # 整数でない、または0以下
                    print("  エラー: distance は1以上の整数である必要があります。")
                    continue
                return value_i
            return value_f # height, prominence は float でOK
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
            f'{trunk_prefix}_Gyro_X', f'{trunk_prefix}_Gyro_Y', f'{trunk_prefix}_Gyro_Z', 'Blank_4',
            'Blank_5', 'Blank_6', 'Blank_7', 'Blank_8', 'Blank_9', 'Blank_10', 'Blank_11', 'Blank_12']

# --- メイン実行ブロック ---
if __name__ == "__main__":
    # ★★★ Tkinter を使うためのおまじない (非表示のルートウィンドウ) ★★★
    root = tk.Tk()
    root.withdraw()
    
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

    # === ステップ 0: 初期データの表示 ===
    print(f"\n[ステップ0] 同期用信号 ({SYNC_SIGNAL_SUFFIX}) の先頭 {NUM_SAMPLES_TO_PLOT} サンプルをプロットします...")
    sync_l_short_plot, sync_r_short_plot = None, None
    plot_len = 0
    try:
        # --- 初期プロット用にデータを一時読み込み & 列名設定 ---
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

    # === ★ ステップ 0.5: ポップアップでパラメータ入力 ★ ===
    print("\n[ステップ0.5] find_peaks のパラメータをポップアップで入力してください...")
    # デフォルト値を辞書で準備
    current_defaults = {
        'height': DEFAULT_PEAK_HEIGHT,
        'prominence': DEFAULT_PEAK_PROMINENCE,
        'distance': DEFAULT_PEAK_DISTANCE
    }
    # ポップアップ関数を呼び出し
    user_params = get_parameters_via_popup(current_defaults)

    # キャンセルされたかチェック
    if user_params is None:
        print("\n[情報] パラメータ入力をキャンセルしました。処理を中断します。")
        exit()

    user_peak_height = user_params.get('height')
    user_peak_prominence = user_params.get('prominence')
    user_peak_distance = user_params.get('distance')
    print(f"  入力されたパラメータ: height={user_peak_height}, prominence={user_peak_prominence}, distance={user_peak_distance}")

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
    # 診断プロット用の sync_short は不要なので捨てる意味で _ を使う
    sync_gyro_df, lag_samples, sampling_rate, peak_idx_l, peak_idx_r, _, _ = result_tuple

    if sync_gyro_df is None:
         print("\n[エラー] 角速度の前処理に失敗しました。処理を中断します。")
         exit()
    else:
         print(f"\n[成功] 前処理完了。")
         # 検出されたピークインデックスとラグを表示
         if peak_idx_l != -1 and peak_idx_r != -1:
            print(f"  採用された 左ピーク index: {peak_idx_l}, 右ピーク index: {peak_idx_r}")
            print(f"  同期ラグ: {lag_samples} サンプル")
         else:
            print(f"  警告: ピークが検出されなかったため、ラグは {lag_samples} となっています。")
         print(f"  サンプリング周波数: {sampling_rate:.2f} Hz")
         print(f"\n[ステップ1.1] 同期済みデータを '{output_sync_file.name}' に保存中...")
         save_results(output_sync_file, sync_gyro_df, "同期済み角速度データ")

    # === ステップ 2, 3 (Gait Cycles, PCI - Placeholders) ===
    print("\n[ステップ2] 歩行周期の同定を実行中...")
    gait_events_df = identify_gait_cycles(sync_gyro_df, sampling_rate)
    if gait_events_df is None:
        print("  (歩行周期の同定は未実装か、データがありませんでした)")
        print("\n[ステップ3] PCI計算はスキップされます (歩行周期データなし)")
    else:
        # (将来の処理)
        print(f"\n[ステップ2.1] 歩行周期データを '{output_gait_file.name}' に保存中...")
        save_results(output_gait_file, gait_events_df, "歩行周期データ")
        print("\n[ステップ3] PCI計算を実行中...")
        pci_results = calculate_pci(gait_events_df)
        if pci_results is None:
             print("  (PCI計算は未実装か、データがありませんでした)")
        else:
             print(f"\n[ステップ3.1] PCI計算結果を '{output_pci_file.name}' に保存中...")
             try:
                 pci_df = pd.DataFrame([pci_results])
                 save_results(output_pci_file, pci_df, "PCI計算結果")
                 print("\n  PCI 計算結果 (コンソール表示):", pci_results)
             except Exception as e: print(f"  エラー: PCI結果のDataFrame変換/保存中にエラー: {e}")

    # === ステップ 4: 最終結果プロット (同期済み角速度) ===
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
                plt.ylabel('角速度 (単位?)') # 単位が分かれば追記 (例: deg/s)
                plt.legend()
                plt.grid(True)
                plt.show() # 最後のグラフを表示
             else: print(f"  警告: プロットに必要な列が見つかりません ({left_col_name} or {right_col_name})。")
         except Exception as e: print(f"  エラー: 最終グラフの表示中にエラー: {e}")

    print("\n========================================")
    print(f"=== 解析処理終了 ({input_data_file_path.name if input_data_file_path else 'エラー発生'}) ===")
    print("========================================")
    # input("エンターキーを押して終了します...") # 必要ならコメント解除