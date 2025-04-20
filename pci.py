# pci.py
import pandas as pd
import numpy as np

def calculate_pci(gait_events_df):
    """PCI関連指標のみを計算"""
    print("--- [Function@pci] 3. PCI の計算開始 ---")
    required_cols = {'Trial_ID', 'Leg', 'IC_Time', 'FO_Time'}; required_legs = {'L', 'R'}
    if gait_events_df is None or gait_events_df.empty or not required_cols.issubset(gait_events_df.columns) or not required_legs.issubset(gait_events_df['Leg'].unique()):
        print("  エラー: PCI計算に必要なデータ/列/脚が不足。"); return None

    all_phases = []
    for trial_id, trial_df in gait_events_df.groupby('Trial_ID'):
        left_df = trial_df[trial_df['Leg'] == 'L'].sort_values(by='IC_Time').reset_index()
        right_df = trial_df[trial_df['Leg'] == 'R'].sort_values(by='IC_Time').reset_index()
        if len(left_df) < 2 or len(right_df) < 2: continue
        left_df['Next_IC_Time'] = left_df['IC_Time'].shift(-1); left_df['Swing_Time'] = left_df['Next_IC_Time'] - left_df['FO_Time']
        right_df['Next_IC_Time'] = right_df['IC_Time'].shift(-1); right_df['Swing_Time'] = right_df['Next_IC_Time'] - right_df['FO_Time']
        mean_swing_L = left_df['Swing_Time'].mean(); mean_swing_R = right_df['Swing_Time'].mean()
        if pd.isna(mean_swing_L) or pd.isna(mean_swing_R): continue
        if mean_swing_L >= mean_swing_R: ref_leg_df, con_leg_df = left_df, right_df
        else: ref_leg_df, con_leg_df = right_df, left_df
        for i in range(len(ref_leg_df) - 1):
            t_l_i, t_l_i_plus_1 = ref_leg_df.loc[i, 'IC_Time'], ref_leg_df.loc[i+1, 'IC_Time']
            stride_time = t_l_i_plus_1 - t_l_i
            if pd.isna(stride_time) or stride_time <= 0: continue
            con_ic_times = con_leg_df.loc[(con_leg_df['IC_Time'] >= t_l_i) & (con_leg_df['IC_Time'] < t_l_i_plus_1), 'IC_Time']
            if not con_ic_times.empty: all_phases.append(((con_ic_times.iloc[0] - t_l_i) / stride_time) * 360.0)

    phi_ABS, phi_CV, P_phi_ABS, PCI, mean_phi, std_phi, num_strides_for_pci = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0
    if len(all_phases) >= 2:
        phi_array = np.array(all_phases); num_strides_for_pci = len(phi_array)
        mean_phi = np.mean(phi_array); std_phi = np.std(phi_array, ddof=0); phi_ABS = np.mean(np.abs(phi_array - 180.0))
        if not np.isclose(mean_phi, 0): phi_CV = (std_phi / mean_phi) * 100.0
        else: phi_CV = np.nan
        P_phi_ABS = 100.0 * (phi_ABS / 180.0)
        if not pd.isna(phi_CV): PCI = phi_CV + P_phi_ABS
        else: PCI = np.nan
    else: print(f"エラー: 位相データ不足({len(all_phases)}個)のためPCI計算不可")

    print(f"--- [Function@pci] 3. PCI の計算終了 ---")
    return { "PCI": PCI, "phi_ABS_deg": phi_ABS, "phi_CV_percent": phi_CV, "P_phi_ABS": P_phi_ABS,
             "mean_phase_deg": mean_phi, "std_phase_deg": std_phi, "num_strides_used_pci": num_strides_for_pci }