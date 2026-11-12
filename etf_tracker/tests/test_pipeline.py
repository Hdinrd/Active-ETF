# -*- coding: utf-8 -*-
"""離線測試：python -m pytest etf_tracker/tests -q"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from etf_tracker import config, core, fetch_pocket, store

HERE = Path(__file__).parent


def _h(etf, date, rows):
    """rows: [(code, weight, shares)]，現金用 unit 元。"""
    out = []
    for code, w, s in rows:
        unit = "元" if code.endswith("_NTD") else "股"
        out.append(dict(etf=etf, date=date, code=code, name=code, weight=w, shares=s, unit=unit))
    return out


def test_parse_holdings_and_assets():
    payload = json.loads((HERE / "fixture_holdings_00990A.json").read_text(encoding="utf-8"))
    df = fetch_pocket.parse_holdings(payload)
    assert len(df) == 9
    assert set(df["date"]) == {"2026-10-07"}
    cash = df[df["code"] == "PFUR_NTD"].iloc[0]
    assert np.isnan(cash["weight"]) and cash["shares"] == -1029546726
    assert fetch_pocket._clean_name("信  驊") == "信驊"
    assert fetch_pocket._clean_name("NVIDIA  CORP ") == "NVIDIA CORP"
    df = store.add_asset_type(df)
    a = dict(zip(df["code"], df["asset"]))
    assert a["2308"] == "tw_stock" and a["LITE US"] == "foreign_stock"
    assert a["009150 KP"] == "foreign_stock" and a["C_NTD"] == "cash"


def test_parse_quotes():
    payload = json.loads((HERE / "fixture_quote_00981A.json").read_text(encoding="utf-8"))
    q = fetch_pocket.parse_quotes(payload)
    assert list(q["date"]) == ["2026-10-08", "2026-10-07"]
    assert q["aum_100m"].iloc[0] == pytest.approx(2924.64)
    assert q["etf"].iloc[0] == "00981A"


def _synthetic():
    rows = []
    # ETF A：d1 → d2 申購 +5%，全部持股等比例放大，只有 2330 多買
    base = [("2330", 10.0, 1_000_000), ("2454", 8.0, 200_000), ("2317", 6.0, 900_000),
            ("2308", 5.0, 300_000), ("3017", 4.0, 150_000), ("2383", 3.0, 120_000), ("C_NTD", np.nan, 1e8)]
    rows += _h("A", "2026-01-02", base)
    d2 = [(c, w, s * 1.05) for c, w, s in base]
    d2 = [(c, (w + 1.0) if c == "2330" else w, s * (1.10 / 1.05) if c == "2330" else s) for c, w, s in d2]
    rows += _h("A", "2026-01-05", d2)
    # d3：沒申贖，出清 2383、新建 6669
    d3 = [(c, w, s) for c, w, s in d2 if c != "2383"] + [("6669", 2.0, 50_000)]
    rows += _h("A", "2026-01-06", d3)
    # ETF B：d2 才成立 (第一天不能算建倉)，d3 新建 2330
    b2 = [("2454", 9.0, 50_000), ("2317", 9.0, 200_000), ("2308", 9.0, 80_000),
          ("3017", 9.0, 40_000), ("2383", 9.0, 30_000)]
    rows += _h("B", "2026-01-05", b2)
    rows += _h("B", "2026-01-06", b2 + [("2330", 3.0, 100_000), ("6669", 1.0, 20_000)])
    h = store.add_asset_type(pd.DataFrame(rows))
    quotes = pd.DataFrame([
        dict(date=d, etf=e, name=e, close=10.0, volume=1, aum_100m=a, inception="")
        for e, a in (("A", 100.0), ("B", 10.0)) for d in ("2026-01-02", "2026-01-05", "2026-01-06")])
    return h, quotes


def test_flow_adjusted_actions():
    h, q = _synthetic()
    ch, flows = core.compute_changes(h, q)
    a2 = ch[(ch["etf"] == "A") & (ch["date"] == "2026-01-05")].set_index("code")
    # 申購 5% → k ≈ 1.05，被動放大的不能算主動買
    assert a2["k"].iloc[0] == pytest.approx(1.05)
    assert a2.loc["2454", "action"] == "flow"
    assert a2.loc["2330", "action"] == "add"
    # B 成立第一天沒有前一筆 → 不會出現在 changes
    assert ch[(ch["etf"] == "B") & (ch["date"] == "2026-01-05")].empty
    a3 = ch[(ch["etf"] == "A") & (ch["date"] == "2026-01-06")].set_index("code")
    assert a3.loc["2383", "action"] == "exit" and a3.loc["6669", "action"] == "new"
    cx = core.cross_etf(ch)
    c3 = cx[cx["date"] == "2026-01-06"].set_index("code")
    assert c3.loc["6669", "n_buy"] == 2          # A、B 同日新建倉
    assert c3.loc["6669", "buyers"] == "A,B"
    # 金額：A 的 6669 = 2% × 100 億 = 2 億
    assert c3.loc["6669", "active_ntd"] == pytest.approx(0.02 * 100e8 + 0.01 * 10e8)


def test_gap_excluded_from_cross():
    h, q = _synthetic()
    # 把 A 的 d2 拿掉 → A 的 d1→d3 配對 span=2，不能進共振
    h = h[~((h["etf"] == "A") & (h["date"] == "2026-01-05"))]
    ch, _ = core.compute_changes(h, q)
    assert set(ch[ch["etf"] == "A"]["span"]) == {2}
    cx = core.cross_etf(ch)
    c3 = cx[cx["date"] == "2026-01-06"].set_index("code")
    assert c3.loc["6669", "n_buy"] == 1          # 只剩 B


def test_store_roundtrip(tmp_path):
    config.set_data_dir(tmp_path)
    payload = json.loads((HERE / "fixture_holdings_00990A.json").read_text(encoding="utf-8"))
    df = fetch_pocket.parse_holdings(payload)
    r1 = store.upsert_holdings("00990A", df)
    r2 = store.upsert_holdings("00990A", df)
    assert r1["new_dates"] == ["2026-10-07"] and r2["new_dates"] == [] and r2["revised_dates"] == []
    back = store.read_holdings_file("00990A")
    assert len(back) == 9 and back["code"].iloc[0] == "LITE US"   # 依權重排序
    raw = (tmp_path / "holdings" / "00990A.csv").read_text(encoding="utf-8-sig")
    assert "-1029546726" in raw and ".0" not in raw.split("\n")[1]


# ---------------------------------------------------------------------------
# 真實資料 (pocket.tw 2026/10/01–10/08，4 檔 ETF 的台股持股，部分佔位部位已刪減)
# ---------------------------------------------------------------------------
def load_real_fixture():
    hold, quotes, etf, dates = [], [], None, []
    for line in (HERE / "fixture_real_202610.txt").read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            etf, ds = line[1:].split()
            dates = [f"2026-{d[:2]}-{d[2:]}" for d in ds.split(",")]
        elif line.startswith("Q|"):
            for item in line[2:].split(","):
                d, close, aum = item.split(":")
                quotes.append(dict(date=f"2026-{d[:2]}-{d[2:]}", etf=etf, name=etf, close=float(close),
                                   volume=0, aum_100m=float(aum), inception=""))
        elif line.strip():
            code, shares, weights = line.split("|")
            for d, s, w in zip(dates, shares.split(","), weights.split(",")):
                if s:
                    hold.append(dict(etf=etf, date=d, code=code, name=code, weight=float(w), shares=float(s), unit="股"))
    return store.add_asset_type(pd.DataFrame(hold)), pd.DataFrame(quotes)


def test_real_data_signals():
    h, q = load_real_fixture()
    ch, flows = core.compute_changes(h, q)
    assert (flows["k"] == 1.0).all()                       # 這週沒有等比例申贖放大
    cx = core.cross_etf(ch).set_index(["date", "code"])
    # 聯電：3 家同步主動賣出兩天
    assert cx.loc[("2026-10-05", "2303"), "n_sell"] == 3
    assert cx.loc[("2026-10-06", "2303"), "n_sell"] == 3
    assert -45e8 < cx.loc[("2026-10-06", "2303"), "active_ntd"] < -35e8   # ≈ 7.5 + 21.1 + 12.2 億
    # 金像電：00981A 加碼 + 00403A 新建倉，連三天
    for d in ("2026-10-06", "2026-10-07", "2026-10-08"):
        assert cx.loc[(d, "2368"), "n_buy"] == 2, d
    assert cx.loc[("2026-10-06", "2368"), "new_by"] == "00403A"
    # 創意：10/06、10/07 兩家同買，10/08 只剩 00981A
    assert cx.loc[("2026-10-06", "3443"), "n_buy"] == 2
    assert cx.loc[("2026-10-08", "3443"), "n_buy"] == 1
    # 佔位部位 (1 張、權重 0.00%) 不能變成訊號
    a = ch.set_index(["etf", "date", "code"])
    assert a.loc[("00981A", "2026-10-06", "4958"), "action"] == "trim"   # 348 萬股 → 1 張
    st = core.buy_streaks(ch).set_index(["etf", "code"])
    assert st.loc[("00981A", "2368"), "streak"] == 3
    assert st.loc[("00981A", "3443"), "streak"] == 3


def test_backtest_helpers():
    from etf_tracker import backtest as bt
    cal = pd.Index(["2026-01-02", "2026-01-05", "2026-01-06"])
    # 週末的持股日 (01-03) 對應到前一個交易日 01-02
    assert list(bt.t_index(cal, ["2026-01-02", "2026-01-03", "2026-01-06"])) == [0, 0, 2]
    rng = np.random.default_rng(0)
    x = rng.normal(0.01, 0.01, 400)
    assert bt.newey_west_t(x, 5) > 5
    assert abs(bt.newey_west_t(rng.normal(0, 0.01, 400), 5)) < 3
