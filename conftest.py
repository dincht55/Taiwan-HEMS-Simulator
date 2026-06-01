"""conftest.py — pytest 設定。

放在專案根目錄（與引擎模組同層）；pytest 會把此目錄加入 sys.path，
讓 tests/ 子目錄的測試能 `import config / hems_simulation` 等引擎模組。
本檔可為空，存在即生效。
"""
