# -*- coding: utf-8 -*-
"""
資料層：每檔 ETF 一個 CSV (data/holdings/{ETF}.csv)，按日期遞增、只追加，git diff 友善。

holdings 欄位
  date    持股日期 (pocket.tw API 給的，不是爬蟲執行日)
  code    標的代號 (台股 4 碼、海外 "NVDA US"、現金 "C_NTD"、期貨 "202610TX")
  name    名稱
  weight  權重 %，現金列為空
  shares  持有數 (股 / 元 / 口，看 unit)
  unit    股 / 元 / 口
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

HOLDING_COLS = ["date", "code", "name", "weight", "shares", "unit"]
ETF_DAILY_COLS = ["date", "etf", "name", "close", "volume", "aum_100m", "inception"]

_TW_CODE = re.compile(r"^[0-9]{4,6}[A-Z]?$")


# ---------------------------------------------------------------------------
# 資產分類
# ---------------------------------------------------------------------------
def classify_asset(code: str, unit: str) -> str:
    code = str(code).strip()
    unit = str(unit).strip()
    if unit == "元":
        return "cash"
    if unit == "口":
        return "future"
    if unit == "股":
        if _TW_CODE.match(code):
            return "tw_stock"
        if " " in code:
            return "foreign_stock"
    return "other"


def add_asset_type(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["asset"] = [classify_asset(c, u) for c, u in zip(df["code"], df["unit"])]
    return df


# ---------------------------------------------------------------------------
# 讀寫
# ---------------------------------------------------------------------------
def _holdings_path(etf: str) -> Path:
    return config.HOLDINGS_DIR / f"{etf}.csv"


def _normalize_holdings(df: pd.DataFrame) -> pd.DataFrame:
    df = df[HOLDING_COLS].copy()
    df["code"] = df["code"].astype(str).str.strip()
    df["name"] = df["name"].astype(str).str.strip()
    df["unit"] = df["unit"].astype(str).str.strip()
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce").round(4)
    df["shares"] = pd.to_numeric(df["shares"], errors="coerce")
    df = df.drop_duplicates(subset=["date", "code"], keep="last")
    df["_w"] = df["weight"].fillna(-1.0)
    df = df.sort_values(["date", "_w", "code"], ascending=[True, False, True]).drop(columns="_w")
    return df.reset_index(drop=True)


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    if "shares" in out.columns:
        s = pd.to_numeric(out["shares"], errors="coerce").astype(float)
        v = s.dropna().to_numpy()
        if np.all(np.equal(np.mod(v, 1), 0)):
            out["shares"] = s.round().astype("Int64")
    out.to_csv(path, index=False, encoding="utf-8-sig", lineterminator="\n")


def read_holdings_file(etf: str) -> pd.DataFrame:
    path = _holdings_path(etf)
    if not path.exists():
        return pd.DataFrame(columns=HOLDING_COLS)
    df = pd.read_csv(path, dtype={"code": str, "name": str, "unit": str, "date": str}, encoding="utf-8-sig")
    return _normalize_holdings(df)


def upsert_holdings(etf: str, new: pd.DataFrame) -> dict:
    """以「日期」為單位覆蓋：API 每天給的是完整持股快照。回傳 {'new_dates': [...], 'revised_dates': [...]}。"""
    if new is None or new.empty:
        return {"new_dates": [], "revised_dates": []}
    new = _normalize_holdings(new)
    old = read_holdings_file(etf)
    old_dates = set(old["date"])
    new_dates = sorted(set(new["date"]) - old_dates)
    revised = []
    for d in sorted(set(new["date"]) & old_dates):
        a = old[old["date"] == d][["code", "weight", "shares"]].set_index("code").sort_index()
        b = new[new["date"] == d][["code", "weight", "shares"]].set_index("code").sort_index()
        if not a.equals(b):
            revised.append(d)
    keep_old = old[~old["date"].isin(set(new["date"]))]
    merged = _normalize_holdings(pd.concat([f for f in (keep_old, new) if not f.empty], ignore_index=True))
    if new_dates or revised or not _holdings_path(etf).exists():
        _write_csv(merged, _holdings_path(etf))
    return {"new_dates": new_dates, "revised_dates": revised}


def load_holdings(etfs: list[str] | None = None) -> pd.DataFrame:
    """回傳長表：etf + HOLDING_COLS + asset。"""
    if etfs is None:
        etfs = sorted(p.stem for p in config.HOLDINGS_DIR.glob("*.csv"))
    frames = []
    for etf in etfs:
        df = read_holdings_file(etf)
        if df.empty:
            continue
        df.insert(0, "etf", etf)
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["etf"] + HOLDING_COLS + ["asset"])
    return add_asset_type(pd.concat(frames, ignore_index=True))


def read_etf_daily() -> pd.DataFrame:
    p = config.ETF_DAILY_FILE
    if not p.exists():
        return pd.DataFrame(columns=ETF_DAILY_COLS)
    return pd.read_csv(p, dtype={"etf": str, "date": str, "name": str, "inception": str}, encoding="utf-8-sig")


def upsert_etf_daily(new: pd.DataFrame) -> None:
    if new is None or new.empty:
        return
    old = read_etf_daily()
    merged = pd.concat([f for f in (old, new[ETF_DAILY_COLS]) if not f.empty], ignore_index=True)
    merged = merged.drop_duplicates(subset=["date", "etf"], keep="last").sort_values(["etf", "date"])
    _write_csv(merged.reset_index(drop=True), config.ETF_DAILY_FILE)


def read_etf_list() -> pd.DataFrame:
    p = config.ETF_LIST_FILE
    if not p.exists():
        return pd.DataFrame(columns=["etf", "name", "inception", "aum_100m", "last_seen"])
    return pd.read_csv(p, dtype={"etf": str, "name": str, "inception": str, "last_seen": str}, encoding="utf-8-sig")


def write_etf_list(df: pd.DataFrame) -> None:
    _write_csv(df.sort_values("aum_100m", ascending=False).reset_index(drop=True), config.ETF_LIST_FILE)


# ---------------------------------------------------------------------------
# 驗證 (只回傳警告，不擋寫入)
# ---------------------------------------------------------------------------
def validate_snapshots(etf: str, df: pd.DataFrame, dates: list[str]) -> list[str]:
    """對指定日期的快照做健檢。df 為該 ETF 全部歷史 (含 asset 欄)。"""
    warns = []
    if df.empty:
        return warns
    if "asset" not in df.columns:
        df = add_asset_type(df)
    all_dates = sorted(df["date"].unique())
    lo, hi = config.WEIGHT_SUM_RANGE
    for d in dates:
        snap = df[df["date"] == d]
        wsum = snap.loc[snap["asset"] != "cash", "weight"].sum()
        if not (lo <= wsum <= hi):
            warns.append(f"{etf} {d}: 非現金權重合計 {wsum:.1f}% 不在 {lo:.0f}–{hi:.0f}% 之間")
        i = all_dates.index(d)
        if i > 0:
            prev = df[df["date"] == all_dates[i - 1]]
            n0, n1 = len(prev), len(snap)
            if n0 and abs(n1 - n0) / n0 > config.ROW_CHANGE_WARN:
                warns.append(f"{etf} {d}: 持股列數 {n0}→{n1}，單日變化超過 {config.ROW_CHANGE_WARN:.0%}")
            a = prev.set_index("code")[["weight", "shares"]].sort_index()
            b = snap.set_index("code")[["weight", "shares"]].sort_index()
            if len(a) and a.equals(b):
                warns.append(f"{etf} {d}: 與前一筆 {all_dates[i - 1]} 權重與股數完全相同 (疑似未更新)")
    return warns
