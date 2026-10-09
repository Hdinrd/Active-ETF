# -*- coding: utf-8 -*-
"""
主程式

  python -m etf_tracker.run                 # 平日：抓最近 10 天 (重疊抓、冪等寫入) + 產報告
  python -m etf_tracker.run --backfill      # 回補全部歷史
  python -m etf_tracker.run --no-fetch      # 不連網，只用現有資料重算報告
  python -m etf_tracker.run --dry-run       # 寫到暫存資料夾，不動 data/ (CI 測試用)
  python -m etf_tracker.run --notify        # 跑完推播 Telegram (需要 TG_TOKEN / TG_CHAT_ID)
  python -m etf_tracker.run --etfs 00981A,00403A
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone

import pandas as pd

from . import config, core, report, store

TW = timezone(timedelta(hours=8))


def log(msg: str) -> None:
    print(msg, flush=True)


def _resolve_universe(pc, explicit: list[str] | None) -> list[str]:
    known = store.read_etf_list()
    if explicit:
        return explicit
    if config.ETF_UNIVERSE != "auto":
        return [e for e in config.ETF_UNIVERSE if e not in config.EXCLUDE_ETFS]
    disc = pc.discover(config.AUTO_SCAN_CODES) if pc is not None else pd.DataFrame()
    if disc.empty:
        log("   ⚠️ 自動掃描沒有結果，沿用既有清單")
        etfs = known["etf"].tolist() if not known.empty else sorted(p.stem for p in config.HOLDINGS_DIR.glob("*.csv"))
        return [e for e in etfs if e not in config.EXCLUDE_ETFS]
    today = datetime.now(TW).strftime("%Y-%m-%d")
    lst = disc[["etf", "name", "inception", "aum_100m"]].copy()
    lst["last_seen"] = today
    if not known.empty:  # 保留曾經出現、這次沒掃到的 (例如下市)
        gone = known[~known["etf"].isin(lst["etf"])]
        lst = pd.concat([lst, gone], ignore_index=True)
    store.write_etf_list(lst)
    etfs = disc.sort_values("aum_100m", ascending=False)["etf"].tolist()
    log(f"   掃到 {len(etfs)} 檔主動式 ETF：{', '.join(etfs)}")
    return [e for e in etfs if e not in config.EXCLUDE_ETFS]


def fetch(args) -> tuple[list[str], dict]:
    from .fetch_pocket import PocketClient
    warnings: list[str] = []
    stats = {"ok": [], "failed": [], "new_dates": {}, "revised": {}}
    with PocketClient(log=log) as pc:
        etfs = _resolve_universe(pc, args.etfs)
        quotes = []
        for etf in etfs:
            existing = store.read_holdings_file(etf)
            if args.backfill or existing.empty:
                days = config.BACKFILL_RANGE
            else:
                # 停機好幾天也會自動補齊：抓「距離上次資料的日曆天數 + 2」與預設值取大
                gap = (datetime.now(TW).date() - datetime.strptime(existing["date"].max(), "%Y-%m-%d").date()).days
                days = min(config.BACKFILL_RANGE, max(args.days, gap + 2))
            try:
                h = pc.holdings(etf, days)
                if h.empty:
                    raise RuntimeError("沒有持股資料")
                res = store.upsert_holdings(etf, h)
                q = pc.quotes(etf, days)
                if not q.empty:
                    quotes.append(q)
                stats["ok"].append(etf)
                stats["new_dates"][etf] = res["new_dates"]
                if res["revised_dates"]:
                    stats["revised"][etf] = res["revised_dates"]
                span = f"{h['date'].min()}~{h['date'].max()}"
                log(f"   ✅ {etf}: {h['date'].nunique()} 天 ({span})，新增 {len(res['new_dates'])} 天"
                    + (f"，修正 {len(res['revised_dates'])} 天" if res["revised_dates"] else ""))
                if res["new_dates"]:
                    full = store.add_asset_type(store.read_holdings_file(etf))
                    check = res["new_dates"][-5:]  # 回補時只檢查最後幾天，避免洗版
                    warnings += store.validate_snapshots(etf, full, check)
            except Exception as e:  # noqa: BLE001
                stats["failed"].append(etf)
                warnings.append(f"{etf}: 抓取失敗 ({e})")
                log(f"   ❌ {etf}: {e}")
            time.sleep(0.3)
        if quotes:
            store.upsert_etf_daily(pd.concat(quotes, ignore_index=True))
    return warnings, stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="主動式 ETF 每日追蹤 v2")
    ap.add_argument("--backfill", action="store_true", help="回補全部歷史")
    ap.add_argument("--days", type=int, default=config.DAILY_RANGE, help="平日抓最近幾個交易日")
    ap.add_argument("--no-fetch", action="store_true", help="不連網，只重算")
    ap.add_argument("--dry-run", action="store_true", help="寫到暫存資料夾 (CI 用)")
    ap.add_argument("--data-dir", help="指定資料夾")
    ap.add_argument("--notify", action="store_true", help="推播 Telegram")
    ap.add_argument("--etfs", type=lambda s: [x.strip().upper() for x in s.split(",") if x.strip()])
    args = ap.parse_args(argv)

    if args.dry_run:
        config.set_data_dir(args.data_dir or tempfile.mkdtemp(prefix="etf_dryrun_"))
    elif args.data_dir:
        config.set_data_dir(args.data_dir)

    run_at = datetime.now(TW).strftime("%Y-%m-%d %H:%M (台北)")
    log(f"🚀 主動式 ETF 追蹤 v2 — {run_at}")
    log(f"   資料夾: {config.DATA_DIR}")

    prev_report_date = None
    if config.STATUS_FILE.exists():
        try:
            prev_report_date = json.loads(config.STATUS_FILE.read_text(encoding="utf-8")).get("report_date")
        except Exception:  # noqa: BLE001
            pass

    warnings: list[str] = []
    stats: dict = {}
    if not args.no_fetch:
        log("📡 抓取 pocket.tw ...")
        warnings, stats = fetch(args)
        if stats and not stats["ok"]:
            log("❌ 全部 ETF 都抓取失敗")
            _write_status(run_at, stats, warnings, None)
            return 1

    log("🧮 計算主動變化 ...")
    holdings = store.load_holdings()
    if holdings.empty:
        log("❌ 沒有任何持股資料")
        return 1
    etf_daily = store.read_etf_daily()
    changes, flows = core.compute_changes(holdings, etf_daily)
    cross = core.cross_etf(changes)
    streaks = core.buy_streaks(changes)
    etf_list = store.read_etf_list()

    md, D = report.build_report(holdings, changes, flows, cross, streaks, etf_list, warnings, {"run_at": run_at})
    written = report.write_outputs(md, D, changes, flows, cross)
    _write_status(run_at, stats, warnings, D, holdings)
    log(f"📝 報告基準日 {D}，輸出：")
    for p in written:
        log(f"   - {p}")

    if args.notify and not args.dry_run and D and D == prev_report_date:
        log(f"   報告基準日 {D} 跟上次一樣，不重複推播")
    elif args.notify and not args.dry_run and D:
        from .notify import send_telegram
        send_telegram(report.telegram_summary(cross, flows, D, etf_list), config.REPORT_DIR / f"{D}.md", log=log)

    print("\n" + md[:6000])
    return 0


def _write_status(run_at, stats, warnings, D, holdings=None):
    status = {
        "run_at": run_at,
        "report_date": D,
        "fetched_ok": stats.get("ok", []) if stats else [],
        "fetch_failed": stats.get("failed", []) if stats else [],
        "latest_date_by_etf": holdings.groupby("etf")["date"].max().to_dict() if holdings is not None and not holdings.empty else {},
        "warnings": warnings[:200],
    }
    config.STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    config.STATUS_FILE.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
