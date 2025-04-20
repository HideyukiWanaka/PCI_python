@echo off
REM --- スクリプトがあるフォルダへ移動 ---
cd /d "%~dp0"

REM --- Python実行ファイルのフルパスを指定 ---
REM ★★★ 手順1で確認したフルパスに書き換えてください ★★★
set PYTHON_PATH="C:\path\to\your\miniforge3\python.exe"

REM --- 実行するPythonスクリプト名 ---
set SCRIPT_NAME="run_gait_analysis.py"

REM --- Pythonスクリプトを実行 ---
echo %SCRIPT_NAME% を実行します...
%PYTHON_PATH% %SCRIPT_NAME%

REM --- 終了待ち ---
echo.
echo >>> 解析が終了しました。何かキーを押すとウィンドウが閉じます。 <<<
pause > nul
