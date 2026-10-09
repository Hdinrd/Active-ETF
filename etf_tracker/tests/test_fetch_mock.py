# -*- coding: utf-8 -*-
"""
用本機假頁面 (模擬 pocket.tw 的 window.$nuxt.$axios) 測 Playwright 擷取流程，不連外網。
沒有安裝 Chromium 時自動略過。
"""
import functools
import http.server
import threading

import pytest

from etf_tracker import config, run, store

MOCK_HTML = r"""<!doctype html><html><head><meta charset="utf-8"></head><body>mock
<script>
function weekdays(n) {
  const out = []; const d = new Date(Date.UTC(2026, 9, 8));
  while (out.length < n) { const w = d.getUTCDay(); if (w !== 0 && w !== 6) out.push(d.toISOString().slice(0,10).replace(/-/g,'')); d.setUTCDate(d.getUTCDate()-1); }
  return out;
}
const ETFS = {'00981A': ['主動統一台股增長', 2924.64], '00990A': ['主動元大AI新經濟', 391.12]};
setTimeout(() => {
  window.$nuxt = {$axios: {
    defaults: {headers: {common: {Authorization: 'Bearer mock'}}},
    get: async (path, {params}) => {
      const p = Object.fromEntries(params.ParamStr.split(';').filter(Boolean).map(kv => kv.split('=')));
      const etf = p.AssignID, n = Math.min(parseInt(p.DTRange), 40);
      if (!ETFS[etf]) return {data: {Title: [], Data: []}};
      const days = weekdays(n);
      if (params.DtNo === '60465380') {
        return {data: {Title: ['日期','股票名稱','股票代號','收盤價','漲跌','漲幅(%)','成交量','資產規模(億)','成立時間'],
          Data: days.map((d, i) => [d, ETFS[etf][0], etf, '30.00', '0', '0', '1000', String(ETFS[etf][1] - i), '20250527'])}};
      }
      const rows = [];
      days.forEach((d, i) => {
        const g = 1 + 0.01 * (n - i);   // 越近的日子股數越多
        rows.push([d, '2330', '台積電', '10.00', String(Math.round(1000000 * g / 1000) * 1000), '股']);
        rows.push([d, '2454', '聯發科', '8.00', String(Math.round(300000 * g / 1000) * 1000), '股']);
        rows.push([d, '2317', '鴻海', '6.00', '500000', '股']);
        rows.push([d, 'NVDA US', 'NVIDIA', '5.00', '1234', '股']);
        rows.push([d, 'C_NTD', 'CASH', '', '123456789', '元']);
      });
      return {data: {Title: ['日期','標的代號','標的名稱','權重(%)','持有數','單位'], Data: rows}};
    }}};
}, 200);
</script></body></html>"""


@pytest.fixture()
def mock_server(tmp_path):
    (tmp_path / "mock.html").write_text(MOCK_HTML, encoding="utf-8")
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(tmp_path))
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/mock.html"
    srv.shutdown()


def _chromium_ok():
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            p.chromium.launch().close()
        return True
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(not _chromium_ok(), reason="Chromium 未安裝")
def test_end_to_end_with_mock(mock_server, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "POCKET_BOOTSTRAP_URL", mock_server)
    monkeypatch.setattr(config, "AUTO_SCAN_CODES", ["00981A", "00990A", "00999A"])
    data = tmp_path / "data"
    assert run.main(["--data-dir", str(data)]) == 0           # 第一次：沒檔案 → 自動回補
    h = store.load_holdings()
    assert set(h["etf"]) == {"00981A", "00990A"}
    assert h["date"].nunique() == 40 and h["date"].max() == "2026-10-08"
    lst = store.read_etf_list()
    assert list(lst["etf"]) == ["00981A", "00990A"]             # 依規模排序，00999A 沒資料不納入
    assert (data / "reports" / "latest.md").exists()
    assert run.main(["--data-dir", str(data)]) == 0           # 第二次：冪等，不會重複
    assert len(store.load_holdings()) == len(h)
