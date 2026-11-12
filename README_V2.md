# 主動式 ETF 每日追蹤 v2

## 跟 v1 差在哪

| | v1 (V1~V13.py) | v2 (`etf_tracker/`) |
|---|---|---|
| 資料來源 | 爬 pocket.tw 網頁表格，只有當天 | 直接呼叫 pocket.tw 底層 API，**可回補成立以來每天的持股** |
| 日期 | 爬蟲執行日 | API 給的**持股日期** |
| 追蹤範圍 | 手動 5 檔 | 自動掃描全部主動式 ETF（目前 32 檔），可在 `config.py` 改 |
| 比對方式 | 所有 ETF 混在一起取最新兩天 | **每檔只跟自己的前一個交易日比**，缺天就不算單日訊號 |
| 申購贖回 | 沒處理，「加碼」有一半以上是被動放大 | 用 `k = 中位數(今日股數/昨日股數)` 扣掉等比例放大 |
| 金額 | 權重 % 直接相加 | 主動股數 × (權重/股數) × 規模 → **新台幣** |
| 排程 | Windows 工作排程 (6/16 後沒跑過) | **GitHub Actions**，電腦不用開 |

## 檔案

```
etf_tracker/
  config.py        所有門檻、路徑、追蹤範圍
  fetch_pocket.py  Playwright 開 pocket.tw → 用頁面的 axios 打 API
  store.py         資料層：data/holdings/{ETF}.csv、驗證
  core.py          主動變化、申贖估計、跨 ETF 彙總、連買天數
  report.py        每日 markdown 報告 + 衍生 CSV
  notify.py        Telegram (token 只從環境變數 / .env 讀)
  run.py           主程式
  tests/           離線測試 (含一份 2026/10 真實資料)
data/
  holdings/00981A.csv ...   每檔 ETF 每日完整持股
  etf_daily.csv             ETF 收盤價 / 成交量 / 規模
  etf_list.csv              目前追蹤清單
  reports/YYYY-MM-DD.md     每日報告 (latest.md 是最新)
  derived/                  給網頁用：changes_latest / cross_latest / etf_flows
  status.json               最近一次執行狀態
```

## 怎麼用

**本機**（雙擊 `Run_V2.bat`）
- 第一次會自動回補全部歷史（32 檔 × 一年多，約 1–2 分鐘）
- 之後每次只抓最近 10 天，重複抓不會重複寫
- `Run_V2.bat --notify`：跑完推播 Telegram（`.env` 已建好，token 是從 V10 搬過去的；`.env` 不會被上傳）

**GitHub Actions（建議）**
1. workflow 已在 repo 的 `.github/workflows/etf-daily.yml`（本機改了程式要推上去時用 `Publish_V2.bat`）
2. GitHub repo → Actions → **Active ETF daily** → Run workflow 先手動跑一次
3. 要推播：Settings → Secrets and variables → Actions → 新增 `TG_TOKEN`、`TG_CHAT_ID`
4. 之後每個交易日 20:00、22:00、隔天 08:30 自動跑，資料直接 commit 回 repo

**每天晚上自動跳記事本**（本機）
- 雙擊一次 `Setup_Daily_Report.bat`：登記 Windows 排程，週一到週五 21:00 執行 `Daily_Report.bat`
  (從 GitHub 拉最新資料 → 用記事本打開 `data/reports/latest.md`)，並刪掉舊的 `Alpha_Daily_Engine` 排程
- 電腦要開機並登入才會跳出來；錯過的那天不會補跳，雙擊 `Sync_From_GitHub.bat` 隨時看最新的

## 報告怎麼讀

1. **申贖溫度計**：每檔 ETF 的單位數變化（規模/收盤價估）。這是散戶資金流向，最該盯的部位指標
2. **主動共振買進 / 賣出**：同一天 ≥2 家 ETF 主動買（或賣），已扣申贖
3. **主動買超 / 賣超金額 Top**：跨 ETF 加總的主動金額
4. **近 5 日累積**：經理人分好幾天建倉 / 出清會在這裡浮出來
5. **新建倉 / 出清**：權重 ≥ 0.05% 才列，濾掉 1 張的佔位部位
6. **連續主動加碼**：同一檔 ETF 連續 ≥3 天主動買
7. **申贖造成的被動買賣**：等比例放大，不是選股，但會真的進市場

## 已知限制

- 單位數用「規模 / 收盤價」估，含折溢價誤差；持股估的 k 只抓得到「等比例」申贖，申贖先進出現金的部分抓不到
- 權重只到 0.01%，小部位換算的金額不準
- 依賴 pocket.tw 的 API 格式；改版時 `fetch_pocket.py` 會先退回「逐頁攔截」模式（只抓得到當天）
- GitHub Actions 的機器在美國，pocket.tw 如果擋海外 IP，就改回本機排程跑 `Run_V2.bat`
