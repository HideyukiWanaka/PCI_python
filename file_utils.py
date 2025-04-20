# file_utils.py
import pandas as pd
from pathlib import Path
import os

def find_latest_csv_file(folder_path: Path):
    folder_path = Path(folder_path)
    latest_file = None
    if not folder_path.is_dir(): print(f"[エラー@file_utils] データフォルダなし: {folder_path}"); return None
    found_csv_files = list(folder_path.glob('*.csv'))
    if not found_csv_files: print(f"[エラー@file_utils] CSVファイルなし: {folder_path}"); return None
    try: latest_file = max(found_csv_files, key=lambda p: p.stat().st_mtime)
    except Exception as e: print(f"[エラー@file_utils] 最新ファイル検索エラー: {e}"); return None
    return latest_file

def save_results(output_filename, data_to_save, description):
    output_path = Path(output_filename) if not isinstance(output_filename, Path) else output_filename
    if data_to_save is not None and isinstance(data_to_save, pd.DataFrame):
        if not data_to_save.empty:
             try:
                 output_path.parent.mkdir(parents=True, exist_ok=True)
                 data_to_save.to_csv(output_path, index=False, encoding='utf-8-sig')
                 print(f"  結果 ({description}) を '{output_path.name}' に保存しました。")
             except Exception as e: print(f"  [エラー@file_utils] '{output_path.name}' ({description}) 保存エラー: {e}")
        else: print(f"  情報: 保存データ ({description}) が空のためスキップ。")
    else: print(f"  情報: 保存データ ({description}) が無効なためスキップ。")