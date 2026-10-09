# -*- coding: utf-8 -*-
"""每日 markdown 報告 + 給網頁用的衍生 CSV。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, core

YI = 1e8  # 億


def _yi(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    return f"{x / YI:+,.2f}"


def _pct(x, digits=2) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    return f"{x * 100:+.{digits}f}%"


def _md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_（無）_\n"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def flow_table(flows: pd.DataFrame, date: str, names: dict) -> pd.DataFrame:
    """
    申贖溫度計：每檔 ETF 最近一個「有單位數資料」的交易日，與近 N 日累積。
    (規模是落後一天公布的，所以單位數最新只到報告日的前一個交易日。)
    """
    if flows.empty:
        return pd.DataFrame()
    f = flows.sort_values(["etf", "date"]).copy()
    rows = []
    for etf, g in f.groupby("etf"):
        g = g[g["date"] <= date]
        if g.empty:
            continue
        gu = g.dropna(subset=["flow_pct_units"])
        last = g.iloc[-1]
        r = {"ETF": etf, "名稱": names.get(etf, ""),
             "規模(億)": f"{last['aum_ntd'] / YI:,.0f}" if pd.notna(last.get("aum_ntd")) else "–",
             "持股估(報告日)": _pct(last["flow_pct_holdings"])}
        if len(gu):
            lu = gu.iloc[-1]
            prev_aum = gu["aum_ntd"].shift(1).iloc[-1] if len(gu) > 1 else np.nan
            r["申贖日"] = lu["date"]
            r["申贖(單位數)"] = _pct(lu["flow_pct_units"])
            r["_net_ntd"] = lu["flow_pct_units"] * prev_aum if pd.notna(prev_aum) else np.nan
            for w in config.FLOW_WINDOWS:
                tail = gu["flow_pct_units"].tail(w)
                r[f"近{w}日累積"] = _pct(float(np.prod(1 + tail) - 1))
        else:
            r["申贖日"], r["申贖(單位數)"], r["_net_ntd"] = "–", "–", np.nan
            for w in config.FLOW_WINDOWS:
                r[f"近{w}日累積"] = "–"
        r["_aum"] = last.get("aum_ntd", np.nan)
        rows.append(r)
    t = pd.DataFrame(rows).sort_values("_aum", ascending=False, na_position="last")
    t["淨申贖(億)"] = t["_net_ntd"].map(_yi)
    return t


def _flow_total(ft: pd.DataFrame) -> tuple[str, float]:
    """最近一個多數 ETF 都有單位數資料的日期，及當天合計淨申贖。"""
    d = ft.loc[ft["申贖日"] != "–", "申贖日"]
    if d.empty:
        return "–", np.nan
    day = d.mode().sort_values().iloc[-1]
    return day, ft.loc[ft["申贖日"] == day, "_net_ntd"].sum(min_count=1)


def build_report(holdings: pd.DataFrame, changes: pd.DataFrame, flows: pd.DataFrame,
                 cross: pd.DataFrame, streaks: pd.DataFrame, etf_list: pd.DataFrame,
                 warnings: list[str], run_info: dict) -> tuple[str, str | None]:
    D = core.report_date(changes)
    if D is None:
        return "# 主動式 ETF 每日追蹤\n\n資料不足 (每檔 ETF 至少需要 2 天)。\n", None
    names = dict(zip(etf_list["etf"], etf_list["name"])) if not etf_list.empty else {}
    latest_by_etf = holdings.groupby("etf")["date"].max()
    lagging = latest_by_etf[latest_by_etf < D]
    today = changes[changes["date"] == D]
    n_upd = today["etf"].nunique()
    top = config.REPORT_TOP_N
    out = []
    out.append(f"# 主動式 ETF 每日追蹤 — {D}\n")
    out.append(f"- 已更新到 {D} 的 ETF：**{n_upd} / {latest_by_etf.size}** 檔")
    if len(lagging):
        out.append("- 尚未更新：" + "、".join(f"{e}({d[5:]})" for e, d in lagging.items()))
    span_bad = today[~today["valid"]]["etf"].unique()
    if len(span_bad):
        out.append("- 中間缺資料、未納入共振計算：" + "、".join(span_bad))
    out.append(f"- 產生時間：{run_info.get('run_at', '')}\n")

    # ---- 申贖溫度計 ----
    ft = flow_table(flows, D, names)
    out.append("## 1. 申贖溫度計（散戶資金流向）\n")
    if not ft.empty:
        day, total = _flow_total(ft)
        out.append(f"{day} 合計淨申贖 ≈ **{_yi(total)} 億**（單位數 = 隔天公布的規模 / 當天收盤價，含折溢價誤差；"
                   f"規模晚一天公布，所以申贖日會比報告日早一天）\n")
        cols = ["ETF", "名稱", "申贖日", "規模(億)", "淨申贖(億)", "申贖(單位數)", "持股估(報告日)"] + \
               [f"近{w}日累積" for w in config.FLOW_WINDOWS]
        out.append(_md_table(ft[cols].head(40)))
    else:
        out.append("_（無資料）_\n")

    cx = cross[cross["date"] == D] if not cross.empty else pd.DataFrame()

    def fmt_cross(df: pd.DataFrame, who: str) -> pd.DataFrame:
        if df.empty:
            return df
        return pd.DataFrame({
            "代號": df["code"], "名稱": df["name"],
            "買方家數" if who == "buy" else "賣方家數": df["n_buy"] if who == "buy" else df["n_sell"],
            "ETF": df["buyers"] if who == "buy" else df["sellers"],
            "主動金額(億)": df["active_ntd"].map(_yi),
            "實際買賣(億)": df["raw_ntd"].map(_yi),
            "申贖被動(億)": df["flow_ntd"].map(_yi),
            "持有家數": df["n_holders_prev"].astype(str) + "→" + df["n_holders"].astype(str),
        })

    out.append("\n## 2. 主動共振買進（≥2 家 ETF 同日主動買進，已扣申贖）\n")
    co_buy = cx[cx["n_buy"] >= 2].sort_values(["n_buy", "active_ntd"], ascending=False) if not cx.empty else cx
    out.append(_md_table(fmt_cross(co_buy.head(top), "buy")))

    out.append("\n## 3. 主動共振賣出（≥2 家 ETF 同日主動賣出）\n")
    co_sell = cx[cx["n_sell"] >= 2].sort_values(["n_sell", "active_ntd"], ascending=[False, True]) if not cx.empty else cx
    out.append(_md_table(fmt_cross(co_sell.head(top), "sell")))

    out.append("\n## 4. 主動買超金額 Top（跨 ETF 加總）\n")
    if not cx.empty:
        out.append(_md_table(fmt_cross(cx[cx["active_ntd"] > 0].nlargest(top, "active_ntd"), "buy")))
    out.append("\n## 5. 主動賣超金額 Top\n")
    if not cx.empty:
        out.append(_md_table(fmt_cross(cx[cx["active_ntd"] < 0].nsmallest(top, "active_ntd"), "sell")))

    # ---- 近 N 日累積 (多日拆單會在這裡浮出來) ----
    W = config.ACCUM_WINDOW
    if not cross.empty:
        recent_dates = sorted(d for d in cross["date"].unique() if d <= D)[-W:]
        rc = cross[cross["date"].isin(recent_dates)]
        acc = rc.groupby("code").agg(
            name=("name", "first"),
            active_ntd=("active_ntd", lambda s: s.sum(min_count=1)),
            buy_days=("n_buy", lambda s: int((s > 0).sum())),
            sell_days=("n_sell", lambda s: int((s > 0).sum())),
            buyers=("buyers", lambda s: ",".join(sorted({x for v in s for x in v.split(",") if x}))),
            sellers=("sellers", lambda s: ",".join(sorted({x for v in s for x in v.split(",") if x}))),
        ).reset_index()
        span_txt = f"{recent_dates[0]} ~ {recent_dates[-1]}" if recent_dates else ""
        fmt_acc = lambda df, who: pd.DataFrame({
            "代號": df["code"], "名稱": df["name"], "主動金額(億)": df["active_ntd"].map(_yi),
            "有買的天數" if who == "buy" else "有賣的天數": df["buy_days"] if who == "buy" else df["sell_days"],
            "參與 ETF": df["buyers"] if who == "buy" else df["sellers"]})
        out.append(f"\n## 5b. 近 {W} 個交易日主動淨買 Top（{span_txt}）\n")
        out.append(_md_table(fmt_acc(acc[acc["active_ntd"] > 0].nlargest(top, "active_ntd"), "buy")))
        out.append(f"\n## 5c. 近 {W} 個交易日主動淨賣 Top\n")
        out.append(_md_table(fmt_acc(acc[acc["active_ntd"] < 0].nsmallest(top, "active_ntd"), "sell")))

    t = today[(today["asset"] == "tw_stock") & today["valid"]]
    out.append("\n## 6. 新建倉（權重 ≥ %.2f%%）\n" % config.MATERIAL_WEIGHT)
    nb = t[t["action"] == "new"].sort_values("w_cur", ascending=False)
    out.append(_md_table(pd.DataFrame({"ETF": nb["etf"], "代號": nb["code"], "名稱": nb["name"],
                                       "權重%": nb["w_cur"].round(2), "金額(億)": nb["active_ntd"].map(_yi)})))
    out.append("\n## 7. 出清\n")
    ex = t[t["action"] == "exit"].sort_values("w_prev", ascending=False)
    out.append(_md_table(pd.DataFrame({"ETF": ex["etf"], "代號": ex["code"], "名稱": ex["name"],
                                       "原權重%": ex["w_prev"].round(2), "金額(億)": ex["active_ntd"].map(_yi)})))

    out.append(f"\n## 8. 連續主動加碼（≥{config.STREAK_MIN_DAYS} 個交易日）\n")
    st = streaks[(streaks["streak"] >= config.STREAK_MIN_DAYS) & (streaks["last_date"] == D)] if not streaks.empty else streaks
    st = st.sort_values(["streak", "streak_active_w"], ascending=False).head(top) if not st.empty else st
    out.append(_md_table(pd.DataFrame({"ETF": st["etf"], "代號": st["code"], "名稱": st["name"],
                                       "連買天數": st["streak"], "累積主動 NAV%": st["streak_active_w"].round(2)})
                         if not st.empty else pd.DataFrame()))

    out.append("\n## 9. 申贖造成的被動買賣盤 Top（不是經理人選股，但會真的進市場）\n")
    fl = cx[cx["flow_ntd"].abs() >= 0.01 * YI] if not cx.empty else cx
    if fl.empty:
        out.append("_（今天沒有等比例申贖造成的被動買賣，申贖多半先進出現金）_\n")
    else:
        fl = fl.reindex(fl["flow_ntd"].abs().sort_values(ascending=False).index).head(10)
        out.append(_md_table(pd.DataFrame({"代號": fl["code"], "名稱": fl["name"], "被動金額(億)": fl["flow_ntd"].map(_yi),
                                           "主動金額(億)": fl["active_ntd"].map(_yi)})))

    if warnings:
        out.append("\n## 資料警示\n")
        out.extend(f"- {w}" for w in warnings[:50])
        out.append("")

    out.append("""
---
**判定方式**：每檔 ETF 只跟自己前一個交易日比；`k = 中位數(今日股數/昨日股數)` 估申購贖回造成的等比例放大，
主動股數 = 今日股數 − k × 昨日股數。只有實際有買且買得比等比例多才算「主動買進」(賣出同理)；
金額 = 主動股數 × (權重/股數) × 規模。中間缺天的配對不納入共振。
""")
    return "\n".join(out), D


def write_outputs(report_md: str, D: str | None, changes: pd.DataFrame, flows: pd.DataFrame,
                  cross: pd.DataFrame) -> list:
    written = []
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    config.DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    latest = config.REPORT_DIR / "latest.md"
    latest.write_text(report_md, encoding="utf-8")
    written.append(latest)
    if D:
        p = config.REPORT_DIR / f"{D}.md"
        p.write_text(report_md, encoding="utf-8")
        written.append(p)
        keep = ["etf", "date", "prev_date", "span", "valid", "code", "name", "asset", "s_prev", "s_cur", "w_prev", "w_cur",
                "k", "raw_delta", "active_delta", "active_w", "action", "active_ntd", "raw_ntd", "flow_ntd"]
        ch = changes[(changes["date"] == D) & (changes["action"] != "none")][keep]
        p = config.DERIVED_DIR / "changes_latest.csv"
        ch.to_csv(p, index=False, encoding="utf-8-sig", lineterminator="\n")
        written.append(p)
        p = config.DERIVED_DIR / "cross_latest.csv"
        cross[cross["date"] == D].to_csv(p, index=False, encoding="utf-8-sig", lineterminator="\n")
        written.append(p)
    p = config.DERIVED_DIR / "etf_flows.csv"
    cols = ["etf", "date", "prev_date", "span", "gap_days", "valid", "k", "k_names", "aum_ntd", "units", "flow_pct_holdings", "flow_pct_units"]
    flows[[c for c in cols if c in flows.columns]].to_csv(p, index=False, encoding="utf-8-sig", lineterminator="\n")
    written.append(p)
    return written


def telegram_summary(cross: pd.DataFrame, flows: pd.DataFrame, D: str, etf_list: pd.DataFrame) -> str:
    names = dict(zip(etf_list["etf"], etf_list["name"])) if not etf_list.empty else {}
    lines = [f"<b>主動式 ETF 追蹤 {D}</b>"]
    ft = flow_table(flows, D, names)
    if not ft.empty:
        day, total = _flow_total(ft)
        lines.append(f"{day} 合計淨申贖 ≈ {_yi(total)} 億")
    cx = cross[cross["date"] == D] if not cross.empty else pd.DataFrame()
    if not cx.empty:
        cb = cx[cx["n_buy"] >= 2].sort_values(["n_buy", "active_ntd"], ascending=False).head(8)
        if len(cb):
            lines.append("\n<b>主動共振買進</b>")
            lines += [f"{r.name_}({r.code}) {r.n_buy}家 {_yi(r.active_ntd)}億" for r in cb.rename(columns={"name": "name_"}).itertuples()]
        cs = cx[cx["n_sell"] >= 2].sort_values(["n_sell", "active_ntd"], ascending=[False, True]).head(8)
        if len(cs):
            lines.append("\n<b>主動共振賣出</b>")
            lines += [f"{r.name_}({r.code}) {r.n_sell}家 {_yi(r.active_ntd)}億" for r in cs.rename(columns={"name": "name_"}).itertuples()]
    return "\n".join(lines)[:3900]
