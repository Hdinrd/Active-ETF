# -*- coding: utf-8 -*-
"""Telegram 推播。Token 只從環境變數或 repo 根目錄的 .env 讀，不寫在程式裡。"""
from __future__ import annotations

import os
from pathlib import Path

from . import config


def _load_dotenv() -> None:
    p = config.ROOT / ".env"
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def telegram_credentials() -> tuple[str | None, str | None]:
    _load_dotenv()
    return os.environ.get("TG_TOKEN") or None, os.environ.get("TG_CHAT_ID") or None


def send_telegram(text: str, document: Path | None = None, log=print) -> bool:
    token, chat = telegram_credentials()
    if not token or not chat:
        log("   (未設定 TG_TOKEN / TG_CHAT_ID，略過推播)")
        return False
    import requests
    base = f"https://api.telegram.org/bot{token}"
    ok = True
    try:
        r = requests.post(f"{base}/sendMessage", json={"chat_id": chat, "text": text, "parse_mode": "HTML"}, timeout=20)
        ok &= r.ok
        if not r.ok:
            log(f"   ⚠️ Telegram 訊息失敗: {r.status_code} {r.text[:200]}")
        if document and Path(document).exists():
            with open(document, "rb") as f:
                r = requests.post(f"{base}/sendDocument", data={"chat_id": chat}, files={"document": f}, timeout=60)
            ok &= r.ok
            if not r.ok:
                log(f"   ⚠️ Telegram 附件失敗: {r.status_code} {r.text[:200]}")
    except Exception as e:  # noqa: BLE001
        log(f"   ⚠️ Telegram 連線失敗: {e}")
        return False
    if ok:
        log("   📲 已推播到 Telegram")
    return ok
