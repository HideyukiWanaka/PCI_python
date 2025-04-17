# file_utils.py
import pandas as pd
from pathlib import Path
import os # find_latest_csv_fileで使う場合

def find_latest_csv_file(folder_path: Path):
    """指定されたフォルダ内で最終更新日時が最新のCSVファイルを探す"""
    folder_path = Path(folder_path) # Pathオブジェクトであることを確認
    latest_file = None

    # print(f"データフォルダ '{folder_path.resolve()}' を検索中...") # メイン側で表示

    if not folder_path.is_dir():
        print(f"[エラー@file_utils] 指定されたデータフォルダが見つかりません: {folder_path}")
        return None

    found_csv_files = list(folder_path.glob('*.csv'))

    if not found_csv_files:
         print(f"[エラー@file_utils] データフォルダ内にCSVファイルが見つかりません: {folder_path}")
         return None

    try:
        # 最終更新日時で比較して最新のファイルを見つける
        latest_file = max(found_csv_files, key=lambda p: p.stat().st_mtime)
    except Exception as e:
        print(f"[エラー@file_utils] 最新ファイルの検索中にエラーが発生しました: {e}")
        return None

    # print(f"最新のCSVファイルが見つかりました: '{latest_file.name}'") # メイン側で表示
    return latest_file


def save_results(output_filename, data_to_save, description):
    """
    指定されたDataFrameを指定されたファイル名でCSVとして保存する。

    Args:
        output_filename (str or Path): 出力ファイル名
        data_to_save (pd.DataFrame): 保存するデータ
        description (str): 保存するデータの説明（ログ表示用）
    """
    output_path = Path(output_filename) if not isinstance(output_filename, Path) else output_filename

    if data_to_save is not None and isinstance(data_to_save, pd.DataFrame):
        if not data_to_save.empty:
             try:
                 # 出力先フォルダが存在しない場合は作成する (任意)
                 output_path.parent.mkdir(parents=True, exist_ok=True)
                 data_to_save.to_csv(output_path, index=False, encoding='utf-8-sig') # BOM付きUTF-8
                 # print(f"  結果 ({description}) を '{output_path}' に保存しました。") # メイン側で表示
             except Exception as e:
                 print(f"  [エラー@file_utils] ファイル '{output_path}' ({description}) の保存中にエラー: {e}")
        else:
            # print(f"  情報: 保存するデータ ({description}) が空のため、'{output_path}' は出力されません。") # メイン側で表示
            pass
    else:
        # print(f"  情報: 保存するデータ ({description}) が無効なため、ファイル出力はスキップされます。") # メイン側で制御
        pass