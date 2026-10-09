# -*- coding: utf-8 -*-
"""
pocket.tw 資料擷取。

pocket.tw 前端用 Nuxt + axios 打 /api/cm/MobileService/ashx/GetDtnoData.ashx，
需要頁面載入時拿到的訪客 Bearer token。所以作法是：
  1. Playwright 開一頁 pocket.tw，等 window.$nuxt.$axios 帶好 token
  2. 直接在頁面裡用它的 axios 呼叫 API (DTRange=N 可以一次拿 N 個交易日的歷史)
抓不到 $nuxt 時，退回「逐檔開持股頁、攔截 API 回應」(只能拿最新 1 天)。
"""
from __future__ import annotations

import time
from typing import Iterable

import pandas as pd

from . import config

API_PATH = "/cm/MobileService/ashx/GetDtnoData.ashx"

_JS_CALL = """
async ({dtno, param}) => {
  const params = {action: 'getdtnodata', DtNo: dtno, ParamStr: param, FilterNo: '0'};
  try {
    const r = await window.$nuxt.$axios.get('%s', {params});
    return r.data;
  } catch (e) {
    return {Error: {Code: -1, Message: String(e)}};
  }
}
""" % API_PATH

_JS_CALL_MANY = """
async ({dtno, params}) => {
  const one = async (p) => {
    try {
      const r = await window.$nuxt.$axios.get('%s', {params: {action: 'getdtnodata', DtNo: dtno, ParamStr: p, FilterNo: '0'}});
      return r.data;
    } catch (e) { return {Error: {Code: -1, Message: String(e)}}; }
  };
  const out = [];
  for (let i = 0; i < params.length; i += 10) {
    const chunk = params.slice(i, i + 10);
    out.push(...await Promise.all(chunk.map(one)));
  }
  return out;
}
""" % API_PATH

_JS_READY = "() => !!(window.$nuxt && window.$nuxt.$axios && window.$nuxt.$axios.defaults.headers.common.Authorization)"


def holdings_param(etf: str, days: int) -> str:
    return f"AssignID={etf};MTPeriod=0;DTMode=0;DTRange={int(days)};DTOrder=1;MajorTable=M722;"


def quote_param(etf: str, days: int) -> str:
    return f"AssignID={etf};DTRange={int(days)}"


# ---------------------------------------------------------------------------
# 純函式：把 API 回傳 (Title + Data) 轉成 DataFrame，方便離線測試
# ---------------------------------------------------------------------------
def _to_frame(payload: dict) -> pd.DataFrame:
    if not isinstance(payload, dict) or payload.get("Error") or not payload.get("Data"):
        return pd.DataFrame()
    title = payload.get("Title") or []
    rows = payload["Data"]
    width = max(len(r) for r in rows)
    cols = list(title) + [f"col{i}" for i in range(len(title), width)]
    return pd.DataFrame(rows, columns=cols[:width])


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(",", "", regex=False).str.strip().replace({"": None}), errors="coerce")


def _date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s.astype(str).str.strip(), format="%Y%m%d", errors="coerce").dt.strftime("%Y-%m-%d")


def parse_holdings(payload: dict) -> pd.DataFrame:
    """回傳欄位：date, code, name, weight, shares, unit (weight 為 %，現金列為 NaN)。"""
    df = _to_frame(payload)
    if df.empty:
        return pd.DataFrame(columns=["date", "code", "name", "weight", "shares", "unit"])
    by_name = {"日期": "date", "標的代號": "code", "標的名稱": "name", "權重(%)": "weight", "持有數": "shares", "單位": "unit"}
    if set(by_name).issubset(df.columns):
        df = df.rename(columns=by_name)
    else:  # 欄名變了就照位置取
        df = df.iloc[:, :6]
        df.columns = ["date", "code", "name", "weight", "shares", "unit"]
    out = pd.DataFrame({
        "date": _date(df["date"]),
        "code": df["code"].astype(str).str.strip(),
        "name": df["name"].astype(str).str.strip(),
        "weight": _num(df["weight"]),
        "shares": _num(df["shares"]),
        "unit": df["unit"].astype(str).str.strip(),
    })
    return out.dropna(subset=["date"]).reset_index(drop=True)


def parse_quotes(payload: dict) -> pd.DataFrame:
    """回傳欄位：date, etf, name, close, volume, aum_100m, inception。"""
    df = _to_frame(payload)
    cols = ["date", "etf", "name", "close", "volume", "aum_100m", "inception"]
    if df.empty:
        return pd.DataFrame(columns=cols)
    get = lambda key, idx: df[key] if key in df.columns else df.iloc[:, idx]
    out = pd.DataFrame({
        "date": _date(get("日期", 0)),
        "etf": get("股票代號", 2).astype(str).str.strip(),
        "name": get("股票名稱", 1).astype(str).str.strip(),
        "close": _num(get("收盤價", 3)),
        "volume": _num(get("成交量", 6)),
        "aum_100m": _num(get("資產規模(億)", 7)),
        "inception": _date(get("成立時間", 8)),
    })
    return out.dropna(subset=["date"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Playwright client
# ---------------------------------------------------------------------------
class PocketClient:
    def __init__(self, headless: bool = True, timeout_ms: int = 45000, log=print):
        self.headless = headless
        self.timeout_ms = timeout_ms
        self.log = log
        self._pw = self._browser = self._page = None
        self.mode = None  # "axios" 或 "capture"

    # -- lifecycle ----------------------------------------------------------
    def __enter__(self):
        from playwright.sync_api import sync_playwright
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=self.headless)
        ctx = self._browser.new_context(
            locale="zh-TW",
            timezone_id="Asia/Taipei",
            user_agent=("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"),
        )
        self._page = ctx.new_page()
        self._bootstrap()
        return self

    def __exit__(self, *exc):
        try:
            if self._browser:
                self._browser.close()
        finally:
            if self._pw:
                self._pw.stop()

    def _bootstrap(self):
        page = self._page
        for attempt in range(3):
            try:
                page.goto(config.POCKET_BOOTSTRAP_URL, wait_until="domcontentloaded", timeout=self.timeout_ms)
                page.wait_for_function(_JS_READY, timeout=self.timeout_ms)
                self.mode = "axios"
                self.log("   pocket.tw 連線成功 (axios 模式)")
                return
            except Exception as e:  # noqa: BLE001
                self.log(f"   ⚠️ bootstrap 第 {attempt + 1} 次失敗: {e}")
                time.sleep(3 * (attempt + 1))
        self.mode = "capture"
        self.log("   ⚠️ 拿不到 pocket.tw 的 axios，改用逐頁攔截模式 (只能抓最新 1 天)")

    # -- raw calls ------------------------------------------------------------
    def _call(self, dtno: str, param: str) -> dict:
        data = self._page.evaluate(_JS_CALL, {"dtno": dtno, "param": param})
        if isinstance(data, dict) and data.get("Error"):
            # token 過期之類：重新載入一次再試
            self.log(f"   ⚠️ API 回傳錯誤 {data['Error']}，重新載入頁面再試一次")
            self._bootstrap()
            if self.mode == "axios":
                data = self._page.evaluate(_JS_CALL, {"dtno": dtno, "param": param})
        return data

    def _call_many(self, dtno: str, params: list[str]) -> list[dict]:
        return self._page.evaluate(_JS_CALL_MANY, {"dtno": dtno, "params": params})

    def _capture_holdings(self, etf: str) -> dict:
        url = f"https://www.pocket.tw/etf/tw/{etf}/fundholding"
        pred = lambda r: config.DTNO_HOLDINGS in r.url and f"AssignID%3D{etf}" in r.url
        with self._page.expect_response(pred, timeout=self.timeout_ms) as info:
            self._page.goto(url, wait_until="domcontentloaded", timeout=self.timeout_ms)
        return info.value.json()

    # -- public ---------------------------------------------------------------
    def discover(self, codes: Iterable[str]) -> pd.DataFrame:
        """掃描代號，回傳 pocket.tw 有報價資料的 ETF (最新一天的 quote)。"""
        codes = list(codes)
        if self.mode != "axios":
            return pd.DataFrame(columns=["date", "etf", "name", "close", "volume", "aum_100m", "inception"])
        payloads = self._call_many(config.DTNO_ETF_QUOTE, [quote_param(c, 1) for c in codes])
        frames = [parse_quotes(p) for p in payloads]
        frames = [f for f in frames if not f.empty]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def holdings(self, etf: str, days: int) -> pd.DataFrame:
        if self.mode == "axios":
            payload = self._call(config.DTNO_HOLDINGS, holdings_param(etf, days))
        else:
            payload = self._capture_holdings(etf)
        return parse_holdings(payload)

    def quotes(self, etf: str, days: int) -> pd.DataFrame:
        if self.mode != "axios":
            return pd.DataFrame()
        return parse_quotes(self._call(config.DTNO_ETF_QUOTE, quote_param(etf, days)))
