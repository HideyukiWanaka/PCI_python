# gait_cycles.py
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

# (lowpass_filter 関数は変更なし)
def lowpass_filter(data, cutoff, fs, order=4):
    nyq = 0.5 * fs; normal_cutoff = cutoff / nyq
    if not (0 < normal_cutoff < 1): print(f"警告: 正規化カットオフ周波数不適切"); return data
    try: b, a = butter(order, normal_cutoff, btype='low', analog=False); y = filtfilt(b, a, data); return y
    except Exception as e: print(f"エラー: フィルター適用中: {e}"); return data


def identify_gait_cycles(sync_gyro_df, sampling_rate_hz,
                         swing_threshold=200, swing_duration_threshold_ms=40,
                         filter_cutoff=15.0,
                         # --- find_peaks パラメータ ---
                         # IC/FO (谷) 検出用
                         min_peak_prominence=10, # IC/FO共通の突出度 (谷の深さ)
                         min_peak_distance_ms=50, # IC/FO共通の最小距離
                         # Swing (山) 検出用 (ICの後の待機時間計算に使う)
                         swing_peak_height=50,  # Swingの正ピークの最小高さ
                         swing_peak_prominence=50, # Swingの正ピークの最小突出度
                         # --- 待機時間関連 ---
                         # fo_wait_time_ms=150 # 固定待機時間は使わない
                         min_fo_wait_time_ms=50 # 動的計算結果が短すぎる場合の最小待機時間
                         ):
    """
    同期済みの角速度データから歩行周期イベント（IC, FO）を同定する。
    アルゴリズム v4:
    - Swing区間検出
    - IC検出 (Swing後の最初の谷)
    - ★ FO待機時間を動的に計算 (IC - 前のSwing Peak) ★
    - FO検出 (待機時間後の最も深い谷)
    出力は IC/FO のインデックスと時間、及びフィルタリング済み信号。
    """
    print("--- [Function@gait_cycles] 2. 歩行周期の同定開始 (動的待機時間) ---")
    if sync_gyro_df is None or sync_gyro_df.empty: return None

    results = []
    dt = 1.0 / sampling_rate_hz
    swing_duration_samples = int(swing_duration_threshold_ms / 1000 * sampling_rate_hz)
    min_peak_distance_samples = int(min_peak_distance_ms / 1000 * sampling_rate_hz)
    min_fo_wait_samples = int(min_fo_wait_time_ms / 1000 * sampling_rate_hz) # 最小待機サンプル数

    filtered_signals = {}
    time_sec = sync_gyro_df['time_aligned_sec'].values

    for leg_prefix in ['L', 'R']:
        col_name = f'{leg_prefix}_Gyro_Z_aligned'
        if col_name not in sync_gyro_df.columns: continue
        signal_raw = sync_gyro_df[col_name].values
        signal_filt = lowpass_filter(signal_raw, cutoff=filter_cutoff, fs=sampling_rate_hz)
        filtered_signals[leg_prefix] = signal_filt
        print(f"  {leg_prefix}脚: 信号フィルタリング完了")

        # 2. Swing区間の検出 (変更なし)
        # ... (Swing区間検出ロジック - 省略) ...
        above_threshold = signal_filt > swing_threshold; swing_starts, swing_ends = [], []; in_swing, count, potential_start = False, 0, -1
        for i in range(len(signal_filt)):
            if above_threshold[i]:
                if not in_swing: potential_start = i; count = 1; in_swing = True
                else: count += 1
            else:
                if in_swing:
                    if count >= swing_duration_samples: swing_starts.append(potential_start); swing_ends.append(i - 1)
                    in_swing = False; count = 0
        if in_swing and count >= swing_duration_samples: swing_starts.append(potential_start); swing_ends.append(len(signal_filt) - 1)
        if not swing_starts: print(f"  {leg_prefix}脚: Swing区間検出不可"); continue
        print(f"  {leg_prefix}脚: {len(swing_starts)} 個のSwing区間検出")

        # ★★★ 3. Swing Peak (正のピーク) の検出 ★★★
        swing_peaks, _ = find_peaks(signal_filt, height=swing_peak_height, prominence=swing_peak_prominence, distance=min_peak_distance_samples)
        print(f"  {leg_prefix}脚: {len(swing_peaks)} 個のSwingピーク (山) 検出")

        # 4. IC候補の検出 (谷)
        ic_candidates, _ = find_peaks(-signal_filt, prominence=min_peak_prominence, distance=min_peak_distance_samples)
        print(f"  {leg_prefix}脚: {len(ic_candidates)} 個のIC候補 (谷) 検出")

        # 5. FO候補の検出 (谷) - ICと同じ条件で検出
        fo_candidates, _ = find_peaks(-signal_filt, prominence=min_peak_prominence, distance=min_peak_distance_samples)
        fo_candidate_values = {idx: signal_filt[idx] for idx in fo_candidates}
        print(f"  {leg_prefix}脚: {len(fo_candidates)} 個のFO候補 (谷) 検出")

        if len(ic_candidates) == 0 or len(fo_candidates) == 0 or len(swing_peaks) == 0:
             print(f"  {leg_prefix}脚: ピーク/谷が見つからないため周期同定不可")
             continue

        # 6. イベントの関連付け
        num_cycles = 0
        for j in range(len(swing_starts)):
            current_swing_start = swing_starts[j]
            current_swing_end = swing_ends[j] if j < len(swing_ends) else len(signal_filt) -1

            # --- IC 検出 ---
            possible_ics = ic_candidates[ic_candidates > current_swing_end]
            if len(possible_ics) == 0: continue
            ic_idx = possible_ics[0]

            # --- FO 検出のための準備 ---
            # このICに対応する前のSwing Peakを探す (待機時間計算用)
            # current_swing_start と current_swing_end の間にある swing_peaks を探す
            relevant_swing_peaks = swing_peaks[(swing_peaks >= current_swing_start) & (swing_peaks <= current_swing_end)]
            if len(relevant_swing_peaks) == 0:
                # Swing区間内にSwing Peakが見つからない場合、待機時間は固定値を使うなど代替策が必要
                print(f"  警告: Swing区間 [{current_swing_start}, {current_swing_end}] 内にSwingピークが見つかりません。最小待機時間を使用します。")
                fo_wait_samples_dynamic = min_fo_wait_samples
            else:
                # 通常は区間内にピークは1つのはずだが、複数ある場合は最初(or最後?)のものを取る
                # ここでは区間内の最後のピークを使うことにする
                last_swing_peak_in_interval = relevant_swing_peaks[-1]
                # ★★★ 待機時間を動的に計算: ICインデックス - Swingピークインデックス ★★★
                fo_wait_samples_dynamic = ic_idx - last_swing_peak_in_interval
                # ★★★ 計算結果が短すぎる場合の最小値を保証 ★★★
                fo_wait_samples_dynamic = max(fo_wait_samples_dynamic, min_fo_wait_samples)
                print(f"  Cycle {num_cycles+1} - Swing Peak: {last_swing_peak_in_interval}, IC: {ic_idx} => Wait(samples): {fo_wait_samples_dynamic}")

            # FO探索区間を決定
            fo_search_start_idx = ic_idx + fo_wait_samples_dynamic
            next_swing_start = len(signal_filt) if j + 1 >= len(swing_starts) else swing_starts[j+1]
            fo_search_end_idx = next_swing_start

            # 区間内のFO候補を抽出
            possible_fos_in_window_idx = [idx for idx in fo_candidates if fo_search_start_idx <= idx < fo_search_end_idx]

            if len(possible_fos_in_window_idx) == 0: continue

            # 最も深い谷をFOとして採用
            fo_idx = min(possible_fos_in_window_idx, key=lambda idx: fo_candidate_values.get(idx, 0))

            # イベント記録 (変更なし)
            ic_time_val = time_sec[ic_idx] if 0 <= ic_idx < len(time_sec) else np.nan
            fo_time_val = time_sec[fo_idx] if 0 <= fo_idx < len(time_sec) else np.nan
            cycle_data = { "Leg": leg_prefix, "Cycle": num_cycles + 1, "IC_Index": ic_idx, "FO_Index": fo_idx, "IC_Time": ic_time_val, "FO_Time": fo_time_val }
            results.append(cycle_data)
            num_cycles += 1

        print(f"  {leg_prefix}脚: {num_cycles} 歩行周期 (IC/FOペア) を同定")

    if not results: return None # 辞書ではなくNoneを返す

    events_df = pd.DataFrame(results).sort_values(by=['Leg', 'IC_Index']).reset_index(drop=True)
    print(f"--- [Function@gait_cycles] 2. 歩行周期の同定終了 ---")

    # 戻り値の辞書に格納
    return { "events_df": events_df, "filtered_signals": filtered_signals, "time_vector": time_sec }