"""Hugging Face custom-Python Space entrypoint."""

from __future__ import annotations

import uvicorn


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=7860,
        workers=1,
        ws_max_size=16_000,
        proxy_headers=True,
        forwarded_allow_ips="*",
    )

