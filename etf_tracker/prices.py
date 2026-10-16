# -*- coding: utf-8 -*-
"""
抓回測用的日K (需要外網，設計成在 GitHub Actions 上跑)：

  python -m etf_tracker.prices            # → data/prices/prices.csv.gz

範圍：所有曾出現在主動式 ETF 持股裡的台股 + 主動式 ETF 本身 + 0050 + 加權指數。
上市先試 .TW，抓不到再試 .TWO (上櫃)。欄位含原始價與還原收盤價 (adj_close)。
"""
from __future__ import annotations

import sys
import time
from datetime import date, timedelta

import numpy as np
import pandas as pd

from . import config, store

OUT = config.DATA_DIR / "prices" / "prices.csv.gz"
FIELDS = {"Open": "open", "High": "high", "Low": "low", "Close": "close", "Adj Close": "adj_close", "Volume": "volume"}


def universe() -> tuple[list[str], str]:
    h = store.load_holdings()
    codes = sorted(h.loc[h["asset"] == "tw_stock", "code"].unique())
    etfs = sorted(h["etf"].unique())
    start = (pd.Timestamp(h["date"].min()) - pd.Timedelta(days=40)).strftime("%Y-%m-%d")
    return sorted(set(codes) | set(etfs) | {"0050"}), start


def _download(tickers: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    import yfinance as yf
    out: dict[str, pd.DataFrame] = {}
    for i in range(0, len(tickers), 80):
        chunk = tickers[i:i + 80]
        df = None
        for attempt in range(3):
            try:
                df = yf.download(chunk, start=start, end=end, auto_adjust=False, actions=False,
                                 progress=False, threads=True, group_by="ticker")
                break
            except Exception as e:  # noqa: BLE001
                print(f"   retry {attempt + 1}: {e}", flush=True)
                time.sleep(10 * (attempt + 1))
        if df is None or df.empty:
            continue
        for t in chunk:
            if isinstance(df.columns, pd.MultiIndex):
                if t not in df.columns.get_level_values(0):
                    continue
                sub = df[t]
            else:
                sub = df
            sub = sub.dropna(subset=["Close"]) if "Close" in sub.columns else pd.DataFrame()
            if len(sub) > 5:
                out[t] = sub
        print(f"   {min(i + 80, len(tickers))}/{len(tickers)} 檔，累計成功 {len(out)}", flush=True)
        time.sleep(2)
    return out


def main() -> int:
    codes, start = universe()
    end = (date.today() + timedelta(days=1)).strftime("%Y-%m-%d")
    print(f"📈 抓 {len(codes)} 檔股價 {start} ~ {end}", flush=True)
    got = _download([f"{c}.TW" for c in codes], start, end)
    missing = [c for c in codes if f"{c}.TW" not in got]
    print(f"   上市 .TW 成功 {len(got)}，改試上櫃 .TWO：{len(missing)} 檔", flush=True)
    got.update(_download([f"{c}.TWO" for c in missing], start, end))
    got.update(_download(["^TWII"], start, end))

    frames = []
    for t, sub in got.items():
        code, _, mkt = t.partition(".")
        f = sub.rename(columns=FIELDS)[[c for c in FIELDS.values() if c in sub.rename(columns=FIELDS).columns]].copy()
        f.index = pd.to_datetime(f.index).strftime("%Y-%m-%d")
        f.index.name = "date"
        f = f.reset_index()
        f.insert(1, "code", "TAIEX" if t == "^TWII" else code)
        f.insert(2, "market", "IDX" if t == "^TWII" else mkt)
        frames.append(f)
    if not frames:
        print("❌ 一檔都沒抓到")
        return 1
    px = pd.concat(frames, ignore_index=True).sort_values(["code", "date"])
    for c in ("open", "high", "low", "close", "adj_close"):
        px[c] = px[c].astype(float).round(4)
    px["volume"] = px["volume"].fillna(0).astype(np.int64)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    px.to_csv(OUT, index=False, compression="gzip")
    still = sorted(set(codes) - set(px["code"]))
    msg = (f"prices rows={len(px)} codes={px['code'].nunique()}/{len(codes) + 1} "
           f"range={px['date'].min()}~{px['date'].max()} missing={','.join(still[:60])}")
    print(f"::notice title=prices::{msg}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
