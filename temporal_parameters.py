# temporal_parameters.py
import pandas as pd
import numpy as np

def calculate_temporal_params(gait_events_df):
    """定常歩行区間のイベントデータから時間関連パラメータを計算"""
    print("--- [Function@temporal] 時間パラメータ計算開始 ---")
    results = {}
    required_cols = {'Trial_ID', 'Leg', 'IC_Time', 'FO_Time'}
    if gait_events_df is None or gait_events_df.empty or not required_cols.issubset(gait_events_df.columns): print("エラー:時間パラメータ計算に必要なデータ不足"); return results
    df = gait_events_df.sort_values(by=['Trial_ID', 'Leg', 'IC_Time']).copy()
    df['Next_IC_Time'] = df.groupby(['Trial_ID', 'Leg'])['IC_Time'].shift(-1)
    df['Stride_Time'] = df['Next_IC_Time'] - df['IC_Time']
    df['Stance_Time'] = df['FO_Time'] - df['IC_Time']
    df['Swing_Time'] = df['Stride_Time'] - df['Stance_Time']
    df_valid = df.dropna(subset=['Stride_Time', 'Stance_Time', 'Swing_Time'])
    df_valid = df_valid[(df_valid['Stride_Time'] > 0) & (df_valid['Stance_Time'] > 0) & (df_valid['Swing_Time'] > 0)]
    if df_valid.empty: print("警告:有効な時間パラメータ計算不可"); return results

    for leg in ['L', 'R']:
        leg_df = df_valid[df_valid['Leg'] == leg]; count = len(leg_df); results[f'Num_Strides_{leg}'] = count
        if count >= 2:
            st = leg_df['Stride_Time']; results[f'Mean_Stride_Time_{leg}_s'] = st.mean(); results[f'SD_Stride_Time_{leg}_s'] = st.std(ddof=0); results[f'CV_Stride_Time_{leg}_percent'] = (st.std(ddof=0) / st.mean()) * 100.0 if st.mean() != 0 else np.nan
            stat = leg_df['Stance_Time']; results[f'Mean_Stance_Time_{leg}_s'] = stat.mean(); results[f'SD_Stance_Time_{leg}_s'] = stat.std(ddof=0); results[f'CV_Stance_Time_{leg}_percent'] = (stat.std(ddof=0) / stat.mean()) * 100.0 if stat.mean() != 0 else np.nan; results[f'Stance_Time_Percent_{leg}'] = (stat.mean() / st.mean()) * 100.0 if st.mean() != 0 else np.nan
            swt = leg_df['Swing_Time']; results[f'Mean_Swing_Time_{leg}_s'] = swt.mean(); results[f'SD_Swing_Time_{leg}_s'] = swt.std(ddof=0); results[f'CV_Swing_Time_{leg}_percent'] = (swt.std(ddof=0) / swt.mean()) * 100.0 if swt.mean() != 0 else np.nan; results[f'Swing_Time_Percent_{leg}'] = (swt.mean() / st.mean()) * 100.0 if st.mean() != 0 else np.nan
        else: print(f"警告: {leg}脚 データ不足({count})"); [results.update({k: np.nan}) for k in [f'Mean_Stride_Time_{leg}_s', f'SD_Stride_Time_{leg}_s', f'CV_Stride_Time_{leg}_percent', f'Mean_Stance_Time_{leg}_s', f'SD_Stance_Time_{leg}_s', f'CV_Stance_Time_{leg}_percent', f'Stance_Time_Percent_{leg}', f'Mean_Swing_Time_{leg}_s', f'SD_Swing_Time_{leg}_s', f'CV_Swing_Time_{leg}_percent', f'Swing_Time_Percent_{leg}']]

    mean_stride_overall = df_valid['Stride_Time'].mean()
    if pd.notna(mean_stride_overall) and mean_stride_overall > 0: results['Cadence_steps_per_min'] = 120.0 / mean_stride_overall
    else: results['Cadence_steps_per_min'] = np.nan
    print(f"--- [Function@temporal] 時間パラメータ計算終了 ---"); return results