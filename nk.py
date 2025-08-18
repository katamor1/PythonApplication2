# ...existing code...
import subprocess
import sys
import os

# 作業ディレクトリをスクリプトのある場所に設定（必要なら）
os.chdir(os.path.dirname(__file__))

# 出力ディレクトリを短いパスにする（リンカの LNK1104 回避）
out_dir = r"C:\tmp\nuitka_build"
os.makedirs(out_dir, exist_ok=True)

# "headless" を引数に渡すとヘッドレス（no-qt）でビルドする
headless = len(sys.argv) > 1 and sys.argv[1].lower() == "headless"

if headless:
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"  # matplotlib をヘッドレス用に設定
    cmd = [
        sys.executable, "-m", "nuitka",
        "--output-dir=" + out_dir,
        "--enable-plugin=no-qt",
        "--include-module=PIL.ImageQt",
        "PythonApplication2.py",
    ]
else:
    # 実際に使う GUI プラグイン名に変更（例: "tk-inter" または "PySide6"）
    gui_plugin = "tk-inter"
    cmd = [
        sys.executable, "-m", "nuitka",
        "--output-dir=" + out_dir,
        f"--enable-plugin={gui_plugin}",
        "--include-module=PIL.ImageQt",
        "PythonApplication2.py",
    ]

subprocess.run(cmd, check=True, env=env if headless else None)
# ...existing code...