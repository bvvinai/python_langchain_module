from __future__ import annotations

import os

import uvicorn


def main() -> None:
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload_enabled = os.getenv("RELOAD", "true").lower() in {"1", "true", "yes"}

    uvicorn.run(
        "app.api.app:app",
        host=host,
        port=port,
        reload=reload_enabled,
        app_dir="src",
    )


if __name__ == "__main__":
    main()
