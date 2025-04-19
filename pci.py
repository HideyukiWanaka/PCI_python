# pci.py
import pandas as pd
import numpy as np


def calculate_pci(gait_events_df):
    """
    歩行周期イベントデータからPCI (Phase Coordination Index) を計算する。
    入力 DataFrame は Trial_ID, Leg, IC_Time, FO_Time を含む必要がある。

    Args:
        gait_events_df (pd.DataFrame): segment_walking_trials と trim_trial_ends
                                       で処理済みの歩行イベントデータ。

    Returns:
        dict or None: 計算されたPCI値や関連指標を含む辞書。
                      計算不可の場合は None。
    """
    print("--- [Function@pci] 3. PCIの計算開始 ---")
    if gait_events_df is None or gait_events_df.empty or \
       not {'Trial_ID', 'Leg', 'IC_Time', 'FO_Time'}.issubset(gait_events_df.columns):
        print("  エラー: PCI計算に必要なデータが不足しています。")
        return None

    required_legs = {'L', 'R'}
    if not required_legs.issubset(gait_events_df['Leg'].unique()):
        print("  エラー: PCI計算には左右両方の脚のデータが必要です。")
        return None

    all_phases = []  # 全トライアルの全有効ストライドの位相を格納
    all_stride_times_L = []
    all_stride_times_R = []

    # Trialごとに処理
    for trial_id, trial_df in gait_events_df.groupby('Trial_ID'):
        print(f"\n  トライアル {trial_id} の処理中...")
        left_df = trial_df[trial_df['Leg'] == 'L'].sort_values(
            by='IC_Time').reset_index()
        right_df = trial_df[trial_df['Leg'] == 'R'].sort_values(
            by='IC_Time').reset_index()

        if len(left_df) < 2 or len(right_df) < 2:  # ストライド計算に最低2つのICが必要
            print(f"  警告: トライアル {trial_id} は左右どちらかのICが2未満のためスキップします。")
            continue

        # スイング時間とストライド時間を計算 (最後のICを除く)
        left_df['Next_IC_Time'] = left_df['IC_Time'].shift(-1)
        left_df['Stride_Time'] = left_df['Next_IC_Time'] - left_df['IC_Time']
        # Swing = Next_IC - FO (現在のFOを使う)
        left_df['Swing_Time'] = left_df['Next_IC_Time'] - left_df['FO_Time']
        # Stride Timeも保存しておく (CV計算用ではないが参考値として)
        all_stride_times_L.extend(left_df['Stride_Time'].dropna().tolist())

        right_df['Next_IC_Time'] = right_df['IC_Time'].shift(-1)
        right_df['Stride_Time'] = right_df['Next_IC_Time'] - \
            right_df['IC_Time']
        right_df['Swing_Time'] = right_df['Next_IC_Time'] - right_df['FO_Time']
        all_stride_times_R.extend(right_df['Stride_Time'].dropna().tolist())

        # 平均スイング時間を計算 (NaNを除外)
        mean_swing_L = left_df['Swing_Time'].mean()
        mean_swing_R = right_df['Swing_Time'].mean()

        if pd.isna(mean_swing_L) or pd.isna(mean_swing_R):
            print(f"  警告: トライアル {trial_id} で平均スイング時間が計算できませんでした。スキップします。")
            continue

        print(f"    平均スイング時間 - L: {mean_swing_L:.4f}s, R: {mean_swing_R:.4f}s")

        # 基準脚を決定
        if mean_swing_L >= mean_swing_R:
            ref_leg_df = left_df
            con_leg_df = right_df
            ref_leg_char = 'L'
        else:
            ref_leg_df = right_df
            con_leg_df = left_df
            ref_leg_char = 'R'
        print(f"    基準脚: {ref_leg_char} (平均スイング時間が長い方)")

        # 位相を計算
        num_strides_in_trial = 0
        # 基準脚の各ストライドについてループ (最後から2番目のICまで)
        for i in range(len(ref_leg_df) - 1):
            t_l_i = ref_leg_df.loc[i, 'IC_Time']
            t_l_i_plus_1 = ref_leg_df.loc[i + 1, 'IC_Time']  # = Next_IC_Time
            stride_time = t_l_i_plus_1 - t_l_i

            if pd.isna(stride_time) or stride_time <= 0:  # 無効なストライド時間はスキップ
                continue

            # この基準脚ストライド内にある対側脚のICを探す
            # t_l_i <= t_s_i < t_l_i_plus_1 となる最初の t_s_i
            con_ic_times_in_stride = con_leg_df.loc[(con_leg_df['IC_Time'] >= t_l_i) & (
                con_leg_df['IC_Time'] < t_l_i_plus_1), 'IC_Time']

            if not con_ic_times_in_stride.empty:
                t_s_i = con_ic_times_in_stride.iloc[0]  # 最初のものを採用
                phase = ((t_s_i - t_l_i) / stride_time) * 360.0
                all_phases.append(phase)
                num_strides_in_trial += 1
            # else:
                # print(f"    警告: 基準脚ストライド {i+1} ({t_l_i:.2f}s - {t_l_i_plus_1:.2f}s) 内に対側脚ICが見つかりません。")

        print(f"    トライアル {trial_id}: {num_strides_in_trial} 個の有効な位相を計算しました。")

# --- 全トライアルの処理完了後 ---

    # ★★★ Stride Time CV の計算 ★★★
    stride_CV_L, stride_CV_R = np.nan, np.nan
    mean_stride_L, mean_stride_R = np.nan, np.nan
    std_stride_L, std_stride_R = np.nan, np.nan
    

    if len(all_stride_times_L) >= 2:  # CV計算には最低2サンプル必要
        stride_times_L_arr = np.array(all_stride_times_L)
        mean_stride_L = np.mean(stride_times_L_arr)
        std_stride_L = np.std(stride_times_L_arr)
        if mean_stride_L != 0:
            stride_CV_L = (std_stride_L / mean_stride_L) * 100.0
        else:
            print("警告: 左脚の平均ストライド時間が0のためCV計算不可")
    else:
        print(f"警告: 左脚のストライド時間データ不足 ({len(all_stride_times_L)}個) のためCV計算不可")

    if len(all_stride_times_R) >= 2:
        stride_times_R_arr = np.array(all_stride_times_R)
        mean_stride_R = np.mean(stride_times_R_arr)
        std_stride_R = np.std(stride_times_R_arr)
        if mean_stride_R != 0:
            stride_CV_R = (std_stride_R / mean_stride_R) * 100.0
        else:
            print("警告: 右脚の平均ストライド時間が0のためCV計算不可")
    else:
        print(f"警告: 右脚のストライド時間データ不足 ({len(all_stride_times_R)}個) のためCV計算不可")
    # ★★★ 計算ここまで ★★★
    

    # --- 全トライアルの位相が集まったらPCIを計算 ---
    #条件分岐と代入、計算ロジックの修正 ★★★
    if len(all_phases) >= 2: # ★ 条件を「2以上」に変更 ★
        phi_array = np.array(all_phases)
        num_strides_for_pci = len(phi_array) # ★★★ ストライド数をここで代入 ★★★

        # --- PCI計算は len(all_phases) >= 2 の場合のみ実行 ---
        mean_phi = np.mean(phi_array)
        std_phi = np.std(phi_array)
        phi_ABS = np.mean(np.abs(phi_array - 180.0))

        if mean_phi == 0:
            phi_CV = np.nan
            print("  警告: 平均位相が0のため、phi_CV は計算できません。")
        else:
            phi_CV = (std_phi / mean_phi) * 100.0

        P_phi_ABS = 100.0 * (phi_ABS / 180.0)

        if pd.isna(phi_CV):
            PCI = np.nan
        else:
            PCI = phi_CV + P_phi_ABS
        # --- PCI計算ここまで ---

    else: # len(all_phases) < 2 の場合
         print(f"  エラー: 位相データが少なすぎるため ({len(all_phases)} 個)、PCIを計算できません。")
         # 変数は初期化時の NaN または 0 のままになる

    print(f"\n  計算結果:")
    print(f"    使用したストライド数: {len(phi_array)}")
    print(f"    平均 位相 (mean_phi): {mean_phi:.2f} 度")
    print(f"    位相 標準偏差 (std_phi): {std_phi:.2f} 度")
    print(f"    平均 位相誤差 (phi_ABS): {phi_ABS:.2f} 度")
    print(f"    位相 変動係数 (phi_CV): {phi_CV:.2f} %")
    print(f"    正規化 位相誤差 (P_phi_ABS): {P_phi_ABS:.2f}")
    print(f"    PCI (phi_CV + P_phi_ABS): {PCI:.2f}")

    # 参考値としてストライド時間の平均なども返す
    mean_stride_L = np.mean(
        all_stride_times_L) if all_stride_times_L else np.nan
    mean_stride_R = np.mean(
        all_stride_times_R) if all_stride_times_R else np.nan

    print(f"--- [Function@pci] 3. PCIの計算終了 ---")

    return {
        "PCI": PCI,
        "phi_ABS_deg": phi_ABS,
        "phi_CV_percent": phi_CV,
        "P_phi_ABS": P_phi_ABS,
        "mean_phase_deg": mean_phi,
        "std_phase_deg": std_phi,
        "num_strides_used_pci": num_strides_for_pci,
        "StrideTime_CV_L_percent": stride_CV_L,  # 追加
        "StrideTime_CV_R_percent": stride_CV_R,  # 追加
        "mean_stride_time_L_sec": mean_stride_L,  # 追加 (参考)
        "std_stride_time_L_sec": std_stride_L,   # 追加 (参考)
        "mean_stride_time_R_sec": mean_stride_R,  # 追加 (参考)
        "std_stride_time_R_sec": std_stride_R    # 追加 (参考)
    }
