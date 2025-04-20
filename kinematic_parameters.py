# kinematic_parameters.py
import pandas as pd
import numpy as np
from scipy.signal import find_peaks

def calculate_kinematic_params(gait_events_df, filtered_signals, time_vector, sampling_rate_hz):
    """定常歩行区間の角速度データから運動学的パラメータを計算"""
    print("--- [Function@kinematic] 運動学パラメータ計算開始 ---")
    results = {}
    required_cols = {'Trial_ID', 'Leg', 'IC_Index', 'FO_Index'}; required_legs = {'L', 'R'}
    if gait_events_df is None or gait_events_df.empty or not filtered_signals or time_vector is None or not required_cols.issubset(gait_events_df.columns) or not required_legs.issubset(gait_events_df['Leg'].unique()):
        print("  エラー: 計算に必要なデータ不足"); return results
    df = gait_events_df.sort_values(by=['Trial_ID', 'Leg', 'IC_Time']).copy()
    df['Next_IC_Index'] = df.groupby(['Trial_ID', 'Leg'])['IC_Index'].shift(-1)
    all_max_swing_vel_L, all_max_swing_vel_R = [], []; all_peak_stance_vel_L, all_peak_stance_vel_R = [], []

    for leg in ['L', 'R']:
        leg_df = df[df['Leg'] == leg].copy(); signal_filt = filtered_signals.get(leg)
        if signal_filt is None or len(signal_filt) != len(time_vector): print(f"警告: {leg}脚 信号データ不整合"); continue
        max_swing_vels, peak_stance_vels = [], []; max_signal_idx = len(signal_filt) - 1
        for index, row in leg_df.iterrows():
            ic_idx, fo_idx, next_ic_idx = row['IC_Index'], row['FO_Index'], row['Next_IC_Index']
            if pd.notna(ic_idx) and pd.notna(fo_idx):
                ic_idx, fo_idx = int(ic_idx), int(fo_idx); start_s, end_s = max(0, ic_idx), min(fo_idx + 1, max_signal_idx + 1)
                if start_s < end_s: stance_signal = signal_filt[start_s : end_s]; peak_stance_vels.append(np.min(stance_signal))
            if pd.notna(fo_idx) and pd.notna(next_ic_idx):
                 fo_idx, next_ic_idx = int(fo_idx), int(next_ic_idx); start_sw, end_sw = max(0, fo_idx), min(next_ic_idx + 1, max_signal_idx + 1)
                 if start_sw < end_sw: swing_signal = signal_filt[start_sw : end_sw]; max_swing_vels.append(np.max(swing_signal))
        if leg == 'L': all_max_swing_vel_L, all_peak_stance_vel_L = max_swing_vels, peak_stance_vels
        else: all_max_swing_vel_R, all_peak_stance_vel_R = max_swing_vels, peak_stance_vels

    for leg, max_vels, stance_vels in [('L', all_max_swing_vel_L, all_peak_stance_vel_L), ('R', all_max_swing_vel_R, all_peak_stance_vel_R)]:
        count_sw = len(max_vels); results[f'Num_Cycles_SwingVel_{leg}'] = count_sw
        if count_sw >= 2: arr = np.array(max_vels); results[f'Mean_Max_Swing_Vel_{leg}'] = np.mean(arr); results[f'SD_Max_Swing_Vel_{leg}'] = np.std(arr, ddof=0); results[f'CV_Max_Swing_Vel_{leg}_percent'] = (np.std(arr, ddof=0) / np.mean(arr)) * 100.0 if np.mean(arr) != 0 else np.nan
        else: results[f'Mean_Max_Swing_Vel_{leg}'], results[f'SD_Max_Swing_Vel_{leg}'], results[f'CV_Max_Swing_Vel_{leg}_percent'] = np.nan, np.nan, np.nan
        count_st = len(stance_vels); results[f'Num_Cycles_StanceVel_{leg}'] = count_st
        if count_st >= 2: arr = np.array(stance_vels); results[f'Mean_Peak_Stance_Vel_{leg}'] = np.mean(arr); results[f'SD_Peak_Stance_Vel_{leg}'] = np.std(arr, ddof=0); results[f'CV_Peak_Stance_Vel_{leg}_percent'] = (np.std(arr, ddof=0) / np.mean(arr)) * 100.0 if np.mean(arr) != 0 else np.nan
        else: results[f'Mean_Peak_Stance_Vel_{leg}'], results[f'SD_Peak_Stance_Vel_{leg}'], results[f'CV_Peak_Stance_Vel_{leg}_percent'] = np.nan, np.nan, np.nan

    print(f"--- [Function@kinematic] 運動学パラメータ計算終了 ---"); return results