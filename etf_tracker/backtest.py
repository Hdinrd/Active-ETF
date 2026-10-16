# -*- coding: utf-8 -*-
"""
回測：主動式 ETF 持股變化訊號有沒有 alpha。

  python -m etf_tracker.backtest     (需要先有 data/prices/prices.csv.gz → python -m etf_tracker.prices)

時點 (不偷看未來)
  T 日收盤後才知道 T 日持股 → 最快 T+1 開盤進場 (lag=1)；另測 T+2 開盤 (lag=2)，涵蓋晚一天公布的 ETF。
  持有 H 個交易日，第 H 天收盤出場。價格用還原權值。

三種比較基準 (由寬到嚴)
  ex_uni   : 同一天「所有主動 ETF 有持有 (權重 ≥0.05%) 的台股」等權
  ex_beta  : 市場模型，扣掉 beta × 加權指數 (beta 用過去 120 天估)
  ex_match : 同一天、同一格「beta 三分位 × 20 日動能五分位 × 20 日成交值三分位」的其他持股 (不含訊號股本身)
             → 這是最嚴格的：把「高 beta、追強勢股、大型股」三個已知因子都扣掉，剩下的才算訊號自己的資訊

統計
  每個訊號日組一個等權籃子 → 每天一個超額報酬 → 跨天平均，Newey-West t (lag = H，處理持有期重疊)。
  成本：來回 0.45% (證交稅 0.3% + 手續費雙邊約 0.15%)。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, core, store

PRICES = config.DATA_DIR / "prices" / "prices.csv.gz"
HORIZONS = (1, 5, 10, 20)
LAGS = (1, 2)
COST = 0.0045
TOP_N = 10
MIN_HOLD_W = 0.05
MAIN_SIGNALS = ["共振買≥2家", "共振賣≥2家", "單家主動買", "單家主動賣", "主動買超金額Top10", "主動賣超金額Top10",
                "新建倉", "連續主動買≥3天", "5日累積主動買Top10", "主動買量/均量Top10", "共振買≥3家"]
PAIRS = [("共振買≥2家", "共振賣≥2家"), ("單家主動買", "單家主動賣"), ("主動買超金額Top10", "主動賣超金額Top10")]


def out_dir():
    return config.DATA_DIR / "backtest"


# ---------------------------------------------------------------------------
# 價格與因子
# ---------------------------------------------------------------------------
def load_prices(path=None) -> dict:
    px = pd.read_csv(path or (config.DATA_DIR / "prices" / "prices.csv.gz"), dtype={"code": str, "date": str, "market": str})
    px = px[px["close"] > 0].copy()
    px["aopen"] = px["open"] * px["adj_close"] / px["close"]
    px["value"] = px["close"] * px["volume"]
    piv = lambda col: px.pivot_table(index="date", columns="code", values=col, aggfunc="last").sort_index()
    ac = piv("adj_close")
    cal = ac["TAIEX"].dropna().index if "TAIEX" in ac.columns else ac.index
    p = {"ac": ac.reindex(cal), "ao": piv("aopen").reindex(cal), "value": piv("value").reindex(cal)}
    dr = p["ac"] / p["ac"].shift(1) - 1
    m = dr["TAIEX"]
    p["dr"] = dr
    p["beta"] = dr.rolling(120, min_periods=60).cov(m).div(m.rolling(120, min_periods=60).var(), axis=0)
    p["mom20"] = p["ac"] / p["ac"].shift(20) - 1
    p["adv20"] = p["value"].rolling(20, min_periods=10).mean()
    return p


def forward_returns(p: dict, H: int) -> pd.DataFrame:
    """R[e, i] = 第 e 天開盤買、第 e+H-1 天收盤賣的報酬。"""
    return p["ac"].shift(-(H - 1)) / p["ao"] - 1.0


def t_index(cal: pd.Index, dates) -> np.ndarray:
    """持股日 T 對應的交易日位置 (≤T 的最後一個交易日)。"""
    return np.searchsorted(cal.values, np.asarray(dates), side="right") - 1


# ---------------------------------------------------------------------------
# 訊號
# ---------------------------------------------------------------------------
def build_signals(changes: pd.DataFrame, cross: pd.DataFrame, p: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """回傳 events (date, code, signal)、universe (date, code)。"""
    tw = changes[(changes["asset"] == "tw_stock") & changes["valid"]]
    uni = tw[(tw["s_cur"] > 0) & (tw["w_cur"] >= MIN_HOLD_W)][["date", "code"]].drop_duplicates()
    cx = cross
    ev = []

    def add(df, name):
        if len(df):
            ev.append(pd.DataFrame({"date": df["date"].values, "code": df["code"].values, "signal": name}))

    add(cx[cx["n_buy"] >= 1], "單家主動買")
    add(cx[cx["n_buy"] >= 2], "共振買≥2家")
    add(cx[cx["n_buy"] >= 3], "共振買≥3家")
    add(cx[cx["n_sell"] >= 1], "單家主動賣")
    add(cx[cx["n_sell"] >= 2], "共振賣≥2家")
    add(cx[cx["new_by"].str.len() > 0], "新建倉")
    pos, neg = cx[cx["active_ntd"] > 0], cx[cx["active_ntd"] < 0]
    add(pos.sort_values("active_ntd", ascending=False).groupby("date").head(TOP_N), "主動買超金額Top10")
    add(neg.sort_values("active_ntd").groupby("date").head(TOP_N), "主動賣超金額Top10")

    piv = cx.pivot_table(index="date", columns="code", values="active_ntd", aggfunc="sum").fillna(0.0)
    acc = piv.rolling(5, min_periods=3).sum().stack().rename("acc").reset_index()
    acc.columns = ["date", "code", "acc"]
    add(acc[acc["acc"] > 0].sort_values("acc", ascending=False).groupby("date").head(TOP_N), "5日累積主動買Top10")

    s = tw.sort_values(["etf", "code", "date"])
    b = s["action"].isin(core.BUY_ACTIONS)
    run_id = (~b).groupby([s["etf"], s["code"]]).cumsum()
    streak = b.astype(int).groupby([s["etf"], s["code"], run_id]).cumsum()
    add(s.loc[(streak >= 3) & b, ["date", "code"]].drop_duplicates(), "連續主動買≥3天")

    adv = p["adv20"].stack().rename("adv").reset_index()
    adv.columns = ["date", "code", "adv"]
    pr = pos.merge(adv, on=["date", "code"], how="inner")
    pr["part"] = pr["active_ntd"] / pr["adv"]
    add(pr.sort_values("part", ascending=False).groupby("date").head(TOP_N), "主動買量/均量Top10")
    return pd.concat(ev, ignore_index=True).drop_duplicates(), uni


# ---------------------------------------------------------------------------
# 統計工具
# ---------------------------------------------------------------------------
def newey_west_t(x, lag: int) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 10:
        return np.nan
    e = x - x.mean()
    s = (e @ e) / n
    for k in range(1, min(lag, n - 1) + 1):
        s += 2 * (1 - k / (lag + 1)) * (e[k:] @ e[:-k]) / n
    return float(x.mean() / np.sqrt(s / n)) if s > 0 else np.nan


def _attach(df: pd.DataFrame, p: dict, R: pd.DataFrame, RM: pd.Series, lag: int) -> pd.DataFrame:
    cal = p["ac"].index
    df = df.copy()
    df["t"] = t_index(cal, df["date"].values)
    df["e"] = df["t"] + lag
    df = df[(df["t"] >= 0) & (df["e"] < len(cal))]
    ci = p["ac"].columns.get_indexer(df["code"])
    df = df[ci >= 0]
    ci = ci[ci >= 0]
    t, e = df["t"].values, df["e"].values
    df["r"] = R.values[e, ci]
    df["rm"] = RM.values[e]
    df["beta"] = p["beta"].values[t, ci]
    df["mom"] = p["mom20"].values[t, ci]
    df["adv"] = p["adv20"].values[t, ci]
    return df.dropna(subset=["r", "beta", "mom", "adv"])


def _cells(U: pd.DataFrame) -> pd.DataFrame:
    """每天依持股池分位數切格：beta 3 × 動能 5 × 成交值 3。"""
    U = U.copy()
    for c, q in (("beta", 3), ("mom", 5), ("adv", 3)):
        U["q_" + c] = U.groupby("t")[c].transform(lambda s, q=q: pd.qcut(s.rank(method="first"), q, labels=False))
    return U


def evaluate(events: pd.DataFrame, uni: pd.DataFrame, p: dict, lags=LAGS, horizons=HORIZONS):
    rows, daily = [], []
    for lag in lags:
        for H in horizons:
            R = forward_returns(p, H)
            RM = R["TAIEX"]
            U = _cells(_attach(uni, p, R, RM, lag))
            U["ar"] = U["r"] - U["beta"] * U["rm"]
            ub = U.groupby("t").agg(r_uni=("r", "mean"), ar_uni=("ar", "mean"))
            keys = ["t", "q_beta", "q_mom", "q_adv"]
            for sig in MAIN_SIGNALS:
                s = events.loc[events["signal"] == sig, ["date", "code"]].drop_duplicates()
                s["t"] = t_index(p["ac"].index, s["date"].values)
                X = U.merge(s[["t", "code"]].assign(is_sig=1), on=["t", "code"], how="left")
                X["is_sig"] = X["is_sig"].fillna(0)
                bench = X[X["is_sig"] == 0].groupby(keys)["r"].mean().rename("r_cell")
                Y = X[X["is_sig"] == 1].merge(bench, left_on=keys, right_index=True, how="left").merge(ub, left_on="t", right_index=True)
                Y["ex_uni"] = Y["r"] - Y["r_uni"]
                Y["ex_beta"] = (Y["r"] - Y["beta"] * Y["rm"]) - Y["ar_uni"]
                Y["ex_match"] = Y["r"] - Y["r_cell"]
                Y["ex_mkt"] = Y["r"] - Y["rm"]
                d = Y.groupby("t").agg(ex_uni=("ex_uni", "mean"), ex_beta=("ex_beta", "mean"), ex_match=("ex_match", "mean"),
                                       ex_mkt=("ex_mkt", "mean"), r=("r", "mean"), n=("code", "size"),
                                       beta=("beta", "mean"), mom=("mom", "mean")).sort_index()
                d["signal"], d["lag"], d["H"] = sig, lag, H
                daily.append(d.reset_index())
                half = len(d) // 2
                rows.append({"signal": sig, "lag": lag, "H": H, "days": len(d), "names_per_day": d["n"].mean(),
                             "beta": d["beta"].mean(), "mom20%": d["mom"].mean() * 100,
                             "raw%": d["r"].mean() * 100, "ex_mkt%": d["ex_mkt"].mean() * 100,
                             "ex_uni%": d["ex_uni"].mean() * 100, "t_uni": newey_west_t(d["ex_uni"], H),
                             "ex_beta%": d["ex_beta"].mean() * 100, "t_beta": newey_west_t(d["ex_beta"], H),
                             "ex_match%": d["ex_match"].mean() * 100, "t_match": newey_west_t(d["ex_match"], H),
                             "hit_match%": (d["ex_match"].dropna() > 0).mean() * 100,
                             "match_1st_half%": d["ex_match"].iloc[:half].mean() * 100,
                             "match_2nd_half%": d["ex_match"].iloc[half:].mean() * 100})
    daily = pd.concat(daily, ignore_index=True)
    # 買 - 賣 (同一天兩邊都有訊號才算)
    for a, b in PAIRS:
        for lag in lags:
            for H in horizons:
                da = daily[(daily["signal"] == a) & (daily["lag"] == lag) & (daily["H"] == H)].set_index("t")
                db = daily[(daily["signal"] == b) & (daily["lag"] == lag) & (daily["H"] == H)].set_index("t")
                for col in ("ex_uni", "ex_match"):
                    sp = (da[col] - db[col]).dropna()
                    rows.append({"signal": f"買減賣：{a}−{b}", "lag": lag, "H": H, "days": len(sp),
                                 f"{col}%": sp.mean() * 100, f"t_{col.split('_')[1]}": newey_west_t(sp, H)})
    table = pd.DataFrame(rows).groupby(["signal", "lag", "H"], sort=False).first().reset_index()
    return table, daily


def car_path(events: pd.DataFrame, uni: pd.DataFrame, p: dict, signals, K=range(-10, 21)) -> pd.DataFrame:
    """事件時間：第 k 天 (0 = ETF 交易當天) 收盤對收盤的超額報酬 (對持股池等權)，累積。"""
    dr, cal = p["dr"], p["ac"].index
    u = uni.copy()
    u["t"] = t_index(cal, u["date"].values)
    u["ci"] = dr.columns.get_indexer(u["code"])
    u = u[u["ci"] >= 0]
    out = {}
    for sig in signals:
        ev = events[events["signal"] == sig].copy()
        ev["t"] = t_index(cal, ev["date"].values)
        ev["ci"] = dr.columns.get_indexer(ev["code"])
        ev = ev[ev["ci"] >= 0]
        path = {}
        for k in K:
            d = ev["t"].values + k
            ok = (d >= 1) & (d < len(cal))
            r = dr.values[d[ok], ev["ci"].values[ok]]
            du = u["t"].values + k
            oku = (du >= 1) & (du < len(cal))
            ru = pd.Series(dr.values[du[oku], u["ci"].values[oku]]).groupby(u["t"].values[oku]).mean()
            ex = pd.Series(r - ru.reindex(ev["t"].values[ok]).values).groupby(ev["t"].values[ok]).mean()
            path[k] = ex.mean()
        out[sig] = pd.Series(path)
    df = pd.DataFrame(out)
    return df.cumsum() * 100 - (df.cumsum() * 100).loc[-1]   # 以 k=-1 收盤為 0


def persistence(changes: pd.DataFrame) -> pd.DataFrame:
    """P(明天仍主動買 | 今天主動買) vs 無條件 → ETF 會不會連續好幾天買同一檔 (拆單)。"""
    tw = changes[(changes["asset"] == "tw_stock") & changes["valid"]].sort_values(["etf", "code", "date"])
    tw = tw[tw["s_prev"] > 0]
    b = tw["action"].isin(core.BUY_ACTIONS).astype(int)
    sl = tw["action"].isin(core.SELL_ACTIONS).astype(int)
    nb = b.groupby([tw["etf"], tw["code"]]).shift(-1)
    ns = sl.groupby([tw["etf"], tw["code"]]).shift(-1)
    m = nb.notna()
    rows = []
    for name, cond in (("無條件", m), ("今天主動買", m & (b == 1)), ("今天主動賣", m & (sl == 1))):
        rows.append({"條件": name, "明天主動買%": nb[cond].mean() * 100, "明天主動賣%": ns[cond].mean() * 100, "樣本": int(cond.sum())})
    return pd.DataFrame(rows)


def flow_timing(flows: pd.DataFrame, p: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """全體主動 ETF 淨申贖 (散戶資金溫度) 五分位 vs 之後加權指數報酬。申贖日 d 的資料 d+1 才公布 → lag=2。"""
    f = flows.dropna(subset=["units_chg", "aum_ntd"]).sort_values(["etf", "date"]).copy()
    f["prev_aum"] = f.groupby("etf")["aum_ntd"].shift(1)
    f = f.dropna(subset=["prev_aum"])
    f = f[f["units_chg"].abs() < 0.5]        # 排除新上市前幾天的極端值
    g = f.groupby("date").apply(lambda x: (x["units_chg"] * x["prev_aum"]).sum() / x["prev_aum"].sum())
    s = g.rename("flow").to_frame()
    cal = p["ac"].index
    rows, ts = [], []
    for H in (1, 5, 20):
        R = forward_returns(p, H)["TAIEX"]
        t = t_index(cal, s.index.values)
        e = t + 2
        ok = (t >= 0) & (e < len(cal))
        x = s[ok].copy()
        x["taiex"] = R.values[e[ok]]
        x = x.dropna()
        x["q"] = pd.qcut(x["flow"].rank(method="first"), 5, labels=["Q1 大贖回", "Q2", "Q3", "Q4", "Q5 大申購"])
        q = x.groupby("q", observed=True).agg(flow=("flow", "mean"), taiex=("taiex", "mean"), n=("taiex", "size")).reset_index()
        q["H"] = H
        rows.append(q)
        hi_lo = x[x["q"] == "Q5 大申購"]["taiex"].mean() - x[x["q"] == "Q1 大贖回"]["taiex"].mean()
        ts.append({"H": H, "corr": x["flow"].corr(x["taiex"]), "Q5-Q1%": hi_lo * 100, "days": len(x)})
    return pd.concat(rows, ignore_index=True), pd.DataFrame(ts)


def etf_alpha(p: dict, etfs, names: dict) -> pd.DataFrame:
    """主動 ETF 本身：上市以來對加權指數的 beta 與年化 alpha (日報酬回歸)。"""
    dr = p["dr"]
    rows = []
    for e in etfs:
        if e not in dr.columns:
            continue
        d = pd.concat([dr[e], dr["TAIEX"]], axis=1).dropna()
        d = d.iloc[5:]                                           # 掛牌前幾天跳過
        if len(d) < 60:
            continue
        y, x = d.iloc[:, 0].values, d.iloc[:, 1].values
        X = np.column_stack([np.ones_like(x), x])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ coef
        se = np.sqrt(np.sum(resid ** 2) / (len(y) - 2) * np.linalg.inv(X.T @ X)[0, 0])
        tot = np.prod(1 + y) - 1
        mkt = np.prod(1 + x) - 1
        rows.append({"ETF": e, "名稱": names.get(e, ""), "起始": d.index[0], "交易日": len(d), "報酬%": tot * 100,
                     "同期加權%": mkt * 100, "beta": coef[1], "年化alpha%": coef[0] * 245 * 100, "t_alpha": coef[0] / se})
    return pd.DataFrame(rows).sort_values("交易日", ascending=False)


def equity_curves(events: pd.DataFrame, uni: pd.DataFrame, p: dict, signal: str, H: int = 20, lag: int = 1) -> pd.DataFrame:
    """每天開一個子帳戶等權買當天訊號股，持有 H 天；資金分成 H 份輪動。同樣方法套在持股池上當對照。"""
    cal, ac, ao = p["ac"].index, p["ac"], p["ao"]
    day_ret = (ac / ac.shift(1) - 1).values
    first_ret = (ac / ao - 1).values
    n = len(cal)

    def run(df, cost):
        df = df.copy()
        df["e"] = t_index(cal, df["date"].values) + lag
        df["ci"] = ac.columns.get_indexer(df["code"])
        df = df[(df["e"] < n) & (df["ci"] >= 0)]
        r = np.zeros(n)
        for e, g in df.groupby("e"):
            ci = g["ci"].unique()
            for k in range(H):
                d = e + k
                if d >= n:
                    break
                v = np.nanmean((first_ret if k == 0 else day_ret)[d, ci])
                if not np.isnan(v):
                    r[d] += v / H
            r[e] -= cost / H
        return r

    sig = events[events["signal"] == signal]
    out = pd.DataFrame({"date": cal, "策略(扣成本)": run(sig, COST), "主動ETF持股池等權": run(uni, 0.0),
                        "加權指數": np.nan_to_num(day_ret[:, ac.columns.get_loc("TAIEX")])})
    start = int(t_index(cal, [sig["date"].min()])[0] + lag)
    out = out.iloc[start:].reset_index(drop=True)
    return out


def perf(r: pd.Series) -> dict:
    r = pd.Series(r).fillna(0)
    nav = (1 + r).cumprod()
    yrs = len(r) / 245
    return {"年化報酬%": (nav.iloc[-1] ** (1 / yrs) - 1) * 100, "年化波動%": r.std() * np.sqrt(245) * 100,
            "Sharpe": r.mean() / r.std() * np.sqrt(245) if r.std() > 0 else np.nan,
            "最大回撤%": (nav / nav.cummax() - 1).min() * 100, "總報酬%": (nav.iloc[-1] - 1) * 100}


def run_all() -> dict:
    holdings = store.load_holdings()
    etf_daily = store.read_etf_daily()
    etf_list = store.read_etf_list()
    names = dict(zip(etf_list["etf"], etf_list["name"]))
    changes, flows = core.compute_changes(holdings, etf_daily)
    cross = core.cross_etf(changes)
    p = load_prices()
    events, uni = build_signals(changes, cross, p)
    table, daily = evaluate(events, uni, p)
    ft_q, ft_s = flow_timing(flows, p)
    curves = {s: equity_curves(events, uni, p, s, H=20) for s in ("共振買≥2家", "共振賣≥2家", "5日累積主動買Top10")}
    return {"events": events, "uni": uni, "table": table, "daily": daily, "p": p,
            "car": car_path(events, uni, p, ["共振買≥2家", "共振賣≥2家", "單家主動買", "單家主動賣", "新建倉"]),
            "persistence": persistence(changes), "flow_q": ft_q, "flow_s": ft_s,
            "etf_alpha": etf_alpha(p, sorted(holdings["etf"].unique()), names), "curves": curves,
            "period": (holdings["date"].min(), holdings["date"].max())}


def main() -> int:
    res = run_all()
    od = out_dir()
    od.mkdir(parents=True, exist_ok=True)
    res["table"].round(4).to_csv(od / "signal_stats.csv", index=False, encoding="utf-8-sig")
    res["car"].round(4).to_csv(od / "event_path.csv", encoding="utf-8-sig")
    res["persistence"].round(2).to_csv(od / "persistence.csv", index=False, encoding="utf-8-sig")
    res["flow_q"].round(5).to_csv(od / "flow_timing.csv", index=False, encoding="utf-8-sig")
    res["etf_alpha"].round(3).to_csv(od / "etf_alpha.csv", index=False, encoding="utf-8-sig")
    fname = {"共振買≥2家": "curve_cobuy2.csv", "共振賣≥2家": "curve_cosell2.csv", "5日累積主動買Top10": "curve_acc5_top10.csv"}
    for k, c in res["curves"].items():
        c.round(6).to_csv(od / fname.get(k, f"curve_{abs(hash(k))}.csv"), index=False, encoding="utf-8-sig")
    t = res["table"]
    cols = ["signal", "lag", "H", "days", "beta", "ex_uni%", "t_uni", "ex_beta%", "t_beta", "ex_match%", "t_match"]
    print(t[(t["lag"] == 1)][cols].round(2).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
