# -*- coding: utf-8 -*-
"""
核心計算：持股快照 → 每日主動變化 → 跨 ETF 彙總。

設計原則 (修正 v1 的三個根本問題)
  1. 每檔 ETF 只跟「自己的前一筆」比，並用交易日曆檢查中間有沒有缺天 (span)。
     ETF 第一天沒有前一筆 → 不產生任何「建倉」。
  2. 扣掉申購贖回造成的等比例增減：
        k        = median( s_t / s_{t-1} )  (前一日權重 >= 0.10% 且兩天都有持有的股票)
        主動股數 = s_t - k * s_{t-1}
     只有「實際有買 (raw>0) 且買得比等比例多」才算 add；賣出同理。
  3. 金額不用「權重加總」，而是換算成 NAV% 與新台幣：
        每股佔 NAV% q = w_t / s_t (出清時用 w_{t-1}/s_{t-1})
        主動 NAV%     = 主動股數 * q
        主動金額      = 主動 NAV% / 100 * AUM
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

STOCK_ASSETS = ("tw_stock", "foreign_stock")
BUY_ACTIONS = ("new", "add")
SELL_ACTIONS = ("exit", "trim")
# 有真的在市場上朝該方向交易、且超過容忍度的動作 → 主動金額才計入彙總
COUNTED_ACTIONS = ("new", "new_tiny", "add", "add_small", "exit", "exit_tiny", "trim", "trim_small")


# ---------------------------------------------------------------------------
# 交易日曆與配對
# ---------------------------------------------------------------------------
def trading_calendar(holdings: pd.DataFrame, etf_daily: pd.DataFrame | None = None) -> list[str]:
    """交易日 = ETF 報價有資料的日期 ∪ 持股快照日期。"""
    dates = set(holdings["date"].unique())
    if etf_daily is not None and not etf_daily.empty:
        dates |= set(etf_daily["date"].dropna().unique())
    return sorted(dates)


def build_pairs(holdings: pd.DataFrame, calendar: list[str]) -> pd.DataFrame:
    """
    每檔 ETF 的 (prev_date → date) 配對。
      span     = 中間隔了幾個交易日 (1 = 相鄰)
      gap_days = 日曆天數
      valid    = span == 1 且 gap_days <= MAX_CALENDAR_GAP_DAYS → 才能拿來算「單日」訊號
    """
    pos = {d: i for i, d in enumerate(calendar)}
    rows = []
    for etf, g in holdings.groupby("etf"):
        ds = sorted(g["date"].unique())
        for a, b in zip(ds[:-1], ds[1:]):
            gap = (pd.Timestamp(b) - pd.Timestamp(a)).days
            span = pos[b] - pos[a]
            rows.append((etf, b, a, span, gap, span == 1 and gap <= config.MAX_CALENDAR_GAP_DAYS))
    return pd.DataFrame(rows, columns=["etf", "date", "prev_date", "span", "gap_days", "valid"])


# ---------------------------------------------------------------------------
# AUM
# ---------------------------------------------------------------------------
def aum_lookup(etf_daily: pd.DataFrame) -> pd.DataFrame:
    """回傳 (etf, date, aum_ntd, units, units_chg)。units 以 規模/收盤價 估 (含折溢價誤差)。"""
    if etf_daily is None or etf_daily.empty:
        return pd.DataFrame(columns=["etf", "date", "aum_ntd", "units", "units_chg"])
    d = etf_daily[["etf", "date", "close", "aum_100m"]].dropna(subset=["aum_100m"]).copy()
    d = d.sort_values(["etf", "date"])
    d["aum_ntd"] = d["aum_100m"] * 1e8
    d["units"] = np.where(d["close"] > 0, d["aum_ntd"] / d["close"], np.nan)
    d["units_chg"] = d.groupby("etf")["units"].pct_change()
    return d[["etf", "date", "aum_ntd", "units", "units_chg"]]


# ---------------------------------------------------------------------------
# 主動變化
# ---------------------------------------------------------------------------
def compute_changes(holdings: pd.DataFrame, etf_daily: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    回傳
      changes : 每檔 ETF × 每日 × 每個標的 的變化 (只含股票與期貨)
      flows   : 每檔 ETF × 每日 的申贖估計
    """
    cal = trading_calendar(holdings, etf_daily)
    pairs = build_pairs(holdings, cal)
    if pairs.empty:
        return pd.DataFrame(), pd.DataFrame()

    h = holdings[holdings["asset"].isin(STOCK_ASSETS + ("future",))]
    cur = h[["etf", "date", "code", "name", "asset", "shares", "weight"]].rename(
        columns={"shares": "s_cur", "weight": "w_cur", "name": "name_cur"})
    cur = cur.merge(pairs[["etf", "date", "prev_date"]], on=["etf", "date"], how="inner")
    prev = h[["etf", "date", "code", "name", "asset", "shares", "weight"]].rename(
        columns={"date": "prev_date", "shares": "s_prev", "weight": "w_prev", "name": "name_prev", "asset": "asset_prev"})
    prev = prev.merge(pairs[["etf", "date", "prev_date"]], on=["etf", "prev_date"], how="inner")

    m = cur.merge(prev, on=["etf", "date", "prev_date", "code"], how="outer")
    m = m.merge(pairs[["etf", "date", "span", "gap_days", "valid"]], on=["etf", "date"], how="left")
    m["name"] = m["name_cur"].fillna(m["name_prev"])
    m["asset"] = m["asset"].fillna(m["asset_prev"])
    m = m.drop(columns=["name_cur", "name_prev", "asset_prev"])
    for c in ("s_cur", "s_prev", "w_cur", "w_prev"):
        m[c] = m[c].fillna(0.0)

    # ---- 申贖倍率 k (每檔 ETF 每天一個) ----
    elig = m[(m["asset"].isin(STOCK_ASSETS)) & (m["s_prev"] > 0) & (m["s_cur"] > 0)
             & (m["w_prev"] >= config.K_ELIGIBLE_MIN_WEIGHT)].copy()
    elig["ratio"] = elig["s_cur"] / elig["s_prev"]
    kdf = elig.groupby(["etf", "date"]).agg(k=("ratio", "median"), k_names=("ratio", "size")).reset_index()
    kdf.loc[kdf["k_names"] < config.MIN_NAMES_FOR_K, "k"] = 1.0
    m = m.merge(kdf, on=["etf", "date"], how="left")
    m["k"] = m["k"].fillna(1.0)
    m["k_names"] = m["k_names"].fillna(0).astype(int)

    # ---- 主動股數 ----
    m["raw_delta"] = m["s_cur"] - m["s_prev"]
    m["active_delta"] = m["s_cur"] - m["k"] * m["s_prev"]
    lot = np.where(m["asset"] == "tw_stock", config.LOT_TOL_TW, 0.0)
    tol = np.maximum(config.REL_TOL * m["s_prev"], lot)

    # 每股佔 NAV% (權重四捨五入到 0.01%，小部位會不準)
    q_cur = np.where((m["s_cur"] > 0) & (m["w_cur"] > 0), m["w_cur"] / m["s_cur"].where(m["s_cur"] > 0), np.nan)
    q_prev = np.where((m["s_prev"] > 0) & (m["w_prev"] > 0), m["w_prev"] / m["s_prev"].where(m["s_prev"] > 0), np.nan)
    q = np.where(np.isnan(q_cur), q_prev, q_cur)
    m["active_w"] = m["active_delta"] * q            # 主動變化，單位 NAV%
    m["raw_w"] = m["raw_delta"] * q
    m["flow_w"] = (m["k"] - 1.0) * m["s_prev"] * q   # 申贖帶來的被動變化

    is_new = (m["s_prev"] == 0) & (m["s_cur"] > 0)
    is_exit = (m["s_prev"] > 0) & (m["s_cur"] == 0)
    material = m["active_w"].abs().fillna(0) >= config.MATERIAL_WEIGHT
    conds = [
        is_new & (m["w_cur"] >= config.MATERIAL_WEIGHT),
        is_new,
        is_exit & (m["w_prev"] >= config.MATERIAL_WEIGHT),
        is_exit,
        (m["raw_delta"] > 0) & (m["active_delta"] > tol) & material,
        (m["raw_delta"] < 0) & (m["active_delta"] < -tol) & material,
        (m["raw_delta"] > 0) & (m["active_delta"] > tol),
        (m["raw_delta"] < 0) & (m["active_delta"] < -tol),
        (m["raw_delta"] <= 0) & (m["active_delta"] > tol),
        (m["raw_delta"] >= 0) & (m["active_delta"] < -tol),
        m["raw_delta"] != 0,
    ]
    choices = ["new", "new_tiny", "exit", "exit_tiny", "add", "trim", "add_small", "trim_small",
               "rel_hold", "rel_under", "flow"]
    m["action"] = np.select(conds, choices, default="none")

    # ---- 換算新台幣 ----
    aum = aum_lookup(etf_daily)
    if not aum.empty:
        m = m.merge(aum[["etf", "date", "aum_ntd"]], on=["etf", "date"], how="left")
    else:
        m["aum_ntd"] = np.nan
    for c in ("active", "raw", "flow"):
        m[f"{c}_ntd"] = m[f"{c}_w"] / 100.0 * m["aum_ntd"]

    m = m.sort_values(["date", "etf", "active_w"], ascending=[True, True, False]).reset_index(drop=True)

    # ---- ETF 層級申贖 ----
    flows = pairs.merge(kdf, on=["etf", "date"], how="left")
    flows["k"] = flows["k"].fillna(1.0)
    flows["k_names"] = flows["k_names"].fillna(0).astype(int)
    if not aum.empty:
        flows = flows.merge(aum, on=["etf", "date"], how="left")
    else:
        for c in ("aum_ntd", "units", "units_chg"):
            flows[c] = np.nan
    flows["flow_pct_holdings"] = flows["k"] - 1.0            # 持股估：等比例放大多少
    flows["flow_pct_units"] = flows["units_chg"]             # 規模/收盤價 估單位數變化
    flows = flows.sort_values(["etf", "date"]).reset_index(drop=True)
    return m, flows


# ---------------------------------------------------------------------------
# 跨 ETF 彙總 (只看台股、只看相鄰交易日的配對)
# ---------------------------------------------------------------------------
def cross_etf(changes: pd.DataFrame) -> pd.DataFrame:
    if changes.empty:
        return pd.DataFrame()
    c = changes[(changes["asset"] == "tw_stock") & changes["valid"]].copy()
    counted = c["action"].isin(COUNTED_ACTIONS)
    c["active_ntd"] = c["active_ntd"].where(counted, c["active_ntd"] * 0.0)   # 沒 AUM 時保持 NaN
    c["active_delta"] = c["active_delta"].where(counted, 0.0)
    c["is_buy"] = c["action"].isin(BUY_ACTIONS)
    c["is_sell"] = c["action"].isin(SELL_ACTIONS)
    c["holds"] = c["s_cur"] > 0
    c["held_prev"] = c["s_prev"] > 0
    g = c.groupby(["date", "code"], sort=False).agg(
        name=("name", "first"),
        n_etf=("etf", "nunique"),
        n_holders=("holds", "sum"),
        n_holders_prev=("held_prev", "sum"),
        n_buy=("is_buy", "sum"),
        n_sell=("is_sell", "sum"),
        active_ntd=("active_ntd", "sum"),
        raw_ntd=("raw_ntd", "sum"),
        flow_ntd=("flow_ntd", "sum"),
        _n_aum=("active_ntd", "count"),
        active_shares=("active_delta", "sum"),
        raw_shares=("raw_delta", "sum"),
    ).reset_index()
    no_aum = g["_n_aum"] == 0                 # 完全沒有 AUM 時金額顯示為空，不是 0
    g.loc[no_aum, ["active_ntd", "raw_ntd", "flow_ntd"]] = np.nan
    g = g.drop(columns="_n_aum")
    # ETF 名單字串只對有動作的列做 (全歷史幾十萬列時快很多)
    for col, mask in (("buyers", c["is_buy"]), ("sellers", c["is_sell"]), ("new_by", c["action"] == "new")):
        sub = c.loc[mask, ["date", "code", "etf"]].sort_values("etf")
        lst = sub.groupby(["date", "code"], sort=False)["etf"].agg(",".join).rename(col).reset_index()
        g = g.merge(lst, on=["date", "code"], how="left")
        g[col] = g[col].fillna("")
    for col in ("n_holders", "n_holders_prev", "n_buy", "n_sell"):
        g[col] = g[col].astype(int)
    return g.sort_values(["date", "active_ntd"], ascending=[True, False]).reset_index(drop=True)


def buy_streaks(changes: pd.DataFrame) -> pd.DataFrame:
    """每檔 ETF × 標的，截至該 ETF 最新日期的連續主動買進天數 (只算相鄰交易日)。"""
    if changes.empty:
        return pd.DataFrame(columns=["etf", "code", "name", "streak", "streak_active_w", "last_date"])
    c = changes[changes["asset"] == "tw_stock"].sort_values(["etf", "code", "date"])
    latest = changes.groupby("etf")["date"].max()
    rows = []
    for (etf, code), g in c.groupby(["etf", "code"], sort=False):
        g = g[g["date"] <= latest[etf]]
        if g.empty or g["date"].iloc[-1] != latest[etf]:
            continue
        n, w = 0, 0.0
        for act, ok, aw in zip(g["action"].values[::-1], g["valid"].values[::-1], g["active_w"].values[::-1]):
            if act in BUY_ACTIONS and ok:
                n += 1
                w += 0.0 if np.isnan(aw) else aw
            else:
                break
        if n:
            rows.append((etf, code, g["name"].iloc[-1], n, w, latest[etf]))
    return pd.DataFrame(rows, columns=["etf", "code", "name", "streak", "streak_active_w", "last_date"])


def report_date(changes: pd.DataFrame, min_share: float = 0.5) -> str | None:
    """報告基準日：最近一個『至少一半 ETF 都已更新』的日期。"""
    if changes.empty:
        return None
    v = changes[changes["valid"]] if changes["valid"].any() else changes
    per_date = v.groupby("date")["etf"].nunique().sort_index()
    n_total = changes["etf"].nunique()
    ok = per_date[per_date >= max(1, min_share * n_total)]
    return ok.index[-1] if len(ok) else per_date.index[-1]
