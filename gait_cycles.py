# gait_cycles.py
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

def lowpass_filter(data, cutoff, fs, order=4):
    """バターワースローパスフィルターを適用する"""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    if not (0 < normal_cutoff < 1):
        print(f"警告: 正規化カットオフ周波数 ({normal_cutoff:.4f}) が (0, 1) の範囲外。フィルタリングスキップ。")
        return data
    try:
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        y = filtfilt(b, a, data)
        return y
    except Exception as e:
        print(f"エラー: ローパスフィルター適用中にエラー: {e}")
        return data

def identify_gait_cycles(sync_gyro_df, sampling_rate_hz,
                         swing_threshold=200, swing_duration_threshold_ms=40,
                         filter_cutoff=15.0,
                         min_peak_prominence=10, min_peak_distance_ms=50,
                         swing_peak_height=50, swing_peak_prominence=50,
                         min_fo_wait_time_ms=50):
    """
    同期済みの角速度データ全体から歩行周期イベント（IC, FO）を同定する。
    """
    print("--- [Function@gait_cycles] 2. 歩行周期の同定開始 (全体) ---")
    if sync_gyro_df is None or sync_gyro_df.empty:
        print("  エラー: 入力データが空です。")
        return None

    results = []
    dt = 1.0 / sampling_rate_hz
    swing_duration_samples = int(swing_duration_threshold_ms / 1000 * sampling_rate_hz)
    min_peak_distance_samples = int(min_peak_distance_ms / 1000 * sampling_rate_hz)
    min_fo_wait_samples = int(min_fo_wait_time_ms / 1000 * sampling_rate_hz)

    filtered_signals = {}
    time_sec = sync_gyro_df['time_aligned_sec'].values

    for leg_prefix in ['L', 'R']:
        col_name = f'{leg_prefix}_Gyro_Z_aligned'
        if col_name not in sync_gyro_df.columns:
            filtered_signals[leg_prefix] = None
            continue

        signal_raw = sync_gyro_df[col_name].values
        signal_filt = lowpass_filter(signal_raw, cutoff=filter_cutoff, fs=sampling_rate_hz)
        if signal_filt is None:
            filtered_signals[leg_prefix] = None
            continue
        filtered_signals[leg_prefix] = signal_filt
        # print(f"  {leg_prefix}脚: 信号フィルタリング完了") # ログ削減

        # Swing区間の検出
        above_threshold = signal_filt > swing_threshold
        swing_starts, swing_ends = [], []
        in_swing, count, potential_start = False, 0, -1
        for i in range(len(signal_filt)):
            if above_threshold[i]:
                if not in_swing: potential_start = i; count = 1; in_swing = True
                else: count += 1
            else:
                if in_swing:
                    if count >= swing_duration_samples: swing_starts.append(potential_start); swing_ends.append(i - 1)
                    in_swing = False; count = 0
        if in_swing and count >= swing_duration_samples: swing_starts.append(potential_start); swing_ends.append(len(signal_filt) - 1)

        if not swing_starts:
            print(f"  {leg_prefix}脚: Swing区間検出不可 (閾値={swing_threshold})")
            continue
        # print(f"  {leg_prefix}脚: {len(swing_starts)} 個のSwing区間検出 (全体)")

        # Swing Peak 検出
        swing_peaks, _ = find_peaks(signal_filt, height=swing_peak_height, prominence=swing_peak_prominence, distance=min_peak_distance_samples)
        # print(f"  {leg_prefix}脚: {len(swing_peaks)} 個のSwingピーク検出")

        # IC候補 (谷) 検出
        ic_candidates, _ = find_peaks(-signal_filt, prominence=min_peak_prominence, distance=min_peak_distance_samples)
        # print(f"  {leg_prefix}脚: {len(ic_candidates)} 個のIC候補検出")

        # FO候補 (谷) 検出
        fo_candidates, _ = find_peaks(-signal_filt, prominence=min_peak_prominence, distance=min_peak_distance_samples)
        fo_candidate_values = {idx: signal_filt[idx] for idx in fo_candidates}
        # print(f"  {leg_prefix}脚: {len(fo_candidates)} 個のFO候補検出")

        if len(ic_candidates) == 0 or len(fo_candidates) == 0 or len(swing_peaks) == 0:
             print(f"  {leg_prefix}脚: ピーク/谷が見つからないため周期同定不可")
             continue

        # イベントの関連付け
        num_cycles = 0
        for j in range(len(swing_starts)):
            current_swing_start = swing_starts[j]
            current_swing_end = swing_ends[j] if j < len(swing_ends) else len(signal_filt) -1

            possible_ics = ic_candidates[ic_candidates > current_swing_end]
            if len(possible_ics) == 0: continue
            ic_idx = possible_ics[0]

            relevant_swing_peaks = swing_peaks[(swing_peaks >= current_swing_start) & (swing_peaks <= current_swing_end)]
            if len(relevant_swing_peaks) == 0: fo_wait_samples_dynamic = min_fo_wait_samples
            else: last_swing_peak_in_interval = relevant_swing_peaks[-1]; fo_wait_samples_dynamic = max(ic_idx - last_swing_peak_in_interval, min_fo_wait_samples)

            next_swing_start = len(signal_filt) if j + 1 >= len(swing_starts) else swing_starts[j+1]
            fo_search_start_idx = ic_idx + fo_wait_samples_dynamic
            fo_search_end_idx = next_swing_start
            possible_fos_in_window_idx = [idx for idx in fo_candidates if fo_search_start_idx <= idx < fo_search_end_idx]
            if len(possible_fos_in_window_idx) == 0: continue
            fo_idx = min(possible_fos_in_window_idx, key=lambda idx: fo_candidate_values.get(idx, 0))

            ic_time_val = time_sec[ic_idx] if 0 <= ic_idx < len(time_sec) else np.nan
            fo_time_val = time_sec[fo_idx] if 0 <= fo_idx < len(time_sec) else np.nan
            cycle_data = { "Leg": leg_prefix, "Cycle": num_cycles + 1, "IC_Index": ic_idx, "FO_Index": fo_idx, "IC_Time": ic_time_val, "FO_Time": fo_time_val }
            results.append(cycle_data)
            num_cycles += 1
        # print(f"  {leg_prefix}脚: {num_cycles} 歩行周期 (IC/FOペア) を同定 (全体)")


    events_df = pd.DataFrame(results)
    if not events_df.empty:
        events_df = events_df.sort_values(by=['Leg', 'IC_Index']).reset_index(drop=True)

    print(f"--- [Function@gait_cycles] 2. 歩行周期の同定終了 ---")
    # 常に辞書を返す
    return {
        "events_df": events_df, # 空の場合もある
        "filtered_signals": filtered_signals,
        "time_vector": time_sec
    }