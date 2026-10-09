# -*- coding: utf-8 -*-
"""v2 pipeline 設定：所有門檻與路徑集中在這裡。"""
from pathlib import Path

# ---------------------------------------------------------------------------
# 路徑 (以 repo 根目錄為基準；GitHub Actions 與本機都適用)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
HOLDINGS_DIR = DATA_DIR / "holdings"          # 每檔 ETF 一個 CSV：data/holdings/00981A.csv
ETF_DAILY_FILE = DATA_DIR / "etf_daily.csv"    # ETF 本身的收盤價 / 成交量 / 規模
ETF_LIST_FILE = DATA_DIR / "etf_list.csv"      # 目前追蹤的主動式 ETF 清單
DERIVED_DIR = DATA_DIR / "derived"             # 計算結果 (給網頁用)
REPORT_DIR = DATA_DIR / "reports"              # 每日報告 markdown
STATUS_FILE = DATA_DIR / "status.json"         # 最近一次執行狀態


def set_data_dir(path) -> None:
    """改用別的資料夾 (dry-run / 測試用)。"""
    global DATA_DIR, HOLDINGS_DIR, ETF_DAILY_FILE, ETF_LIST_FILE, DERIVED_DIR, REPORT_DIR, STATUS_FILE
    DATA_DIR = Path(path).resolve()
    HOLDINGS_DIR = DATA_DIR / "holdings"
    ETF_DAILY_FILE = DATA_DIR / "etf_daily.csv"
    ETF_LIST_FILE = DATA_DIR / "etf_list.csv"
    DERIVED_DIR = DATA_DIR / "derived"
    REPORT_DIR = DATA_DIR / "reports"
    STATUS_FILE = DATA_DIR / "status.json"

# ---------------------------------------------------------------------------
# 追蹤範圍
# ---------------------------------------------------------------------------
# "auto"：每次執行自動掃描下列代號區間，把 pocket.tw 有資料的主動式 ETF 全部納入。
# 也可以改成明確清單，例如 ["00981A", "00403A", "00991A"]。
ETF_UNIVERSE = "auto"
AUTO_SCAN_CODES = [f"00{i}A" for i in range(980, 1000)] + [f"00{i}A" for i in range(400, 461)]
EXCLUDE_ETFS: list[str] = []                   # 想排除的代號

# ---------------------------------------------------------------------------
# 抓取
# ---------------------------------------------------------------------------
POCKET_BOOTSTRAP_URL = "https://www.pocket.tw/etf/tw/00981A/fundholding"
DTNO_HOLDINGS = "59449513"     # 持股明細 (MajorTable=M722)：日期/代號/名稱/權重/持有數/單位
DTNO_ETF_QUOTE = "60465380"    # ETF 收盤 / 成交量 / 資產規模(億)
DAILY_RANGE = 10               # 平日每次抓最近 N 個交易日 (重疊抓，冪等寫入)
BACKFILL_RANGE = 2000          # 回補時抓的天數 (超過成立天數也沒關係)

# ---------------------------------------------------------------------------
# 主動變化判定
# ---------------------------------------------------------------------------
MIN_NAMES_FOR_K = 5            # 估申贖倍率 k 至少要幾檔有效持股
K_ELIGIBLE_MIN_WEIGHT = 0.10   # 估 k 時只用前一日權重 >= 0.10% 的持股 (排除 1 張的佔位部位)
REL_TOL = 0.01                 # 主動股數變化 > 前一日股數 1% 才算主動
LOT_TOL_TW = 1000              # 台股另外容忍 1 張的整股進位誤差
MATERIAL_WEIGHT = 0.05         # 主動變化換算成 NAV% 至少 0.05% 才進訊號 (濾掉 1 張的佔位部位)
MAX_CALENDAR_GAP_DAYS = 12     # 前後兩筆差超過 12 個日曆天 (春節最長約 9 天) 就不當成單日變化

# ---------------------------------------------------------------------------
# 資料驗證 (只警告不擋)
# ---------------------------------------------------------------------------
WEIGHT_SUM_RANGE = (80.0, 115.0)
ROW_CHANGE_WARN = 0.30         # 持股列數單日變化超過 30% 就警告

# ---------------------------------------------------------------------------
# 報告
# ---------------------------------------------------------------------------
REPORT_TOP_N = 15
STREAK_MIN_DAYS = 3            # 連續主動加碼至少幾天才列入
ACCUM_WINDOW = 5               # 「近 N 日主動累積」看幾個交易日
FLOW_WINDOWS = (5, 20)         # 申贖溫度計看幾日累積
