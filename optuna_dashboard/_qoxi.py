"""Generic, project-agnostic launcher for this optuna-dashboard fork.

Reusable in any project — point it at any Optuna storage. Two ways,
checked in order:

1. ``OPTUNA_DASHBOARD_STORAGE`` (alias ``STORAGE_URL``): a full storage
   URL/path. Anything optuna-dashboard understands — ``postgresql://``,
   ``mysql+pymysql://``, ``sqlite:///abs/path.db``, ``redis://``, a
   JournalFile path, etc. Simplest for most projects.

2. Discrete Postgres env vars ``PG_HOST`` / ``PG_PORT`` / ``PG_USER`` /
   ``PG_PASSWORD`` / ``OPTUNA_DB``: the URL is assembled with
   ``sqlalchemy.engine.URL.create()`` so passwords containing
   ``@ : / #`` (common from a Postgres operator) don't break parsing.
   Use this when you can't pre-escape the password into a URL.

Bind address: ``BIND_HOST`` (default ``0.0.0.0``) / ``BIND_PORT``
(default ``8080``).

Run under gunicorn (this is the image's default entrypoint)::

    gunicorn --bind 0.0.0.0:8080 --workers 1 --threads 4 \
        optuna_dashboard._qoxi:application

A single gunicorn worker is intentional: the dashboard's incremental
trial cache is per-process, so multiple workers would each re-read the
whole study from the backend. Threads give I/O concurrency without
splitting the cache.
"""

from __future__ import annotations

import logging
import os

from optuna.storages import BaseStorage
from optuna.storages import RDBStorage
from sqlalchemy.engine import URL

from optuna_dashboard import run_server
from optuna_dashboard import wsgi
from optuna_dashboard._storage_url import get_storage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
_log = logging.getLogger("optuna_dashboard.launcher")

# Headroom so the one long cold get_all_trials (~100s at startup) can't
# starve the many small concurrent queries (get_study's 4 lookups,
# incremental fetch). Single dashboard process → a handful of extra PG
# connections is well within budget.
_RDB_ENGINE_KWARGS = {
    "pool_size": 4,
    "max_overflow": 8,
    "pool_timeout": 10,
    "pool_pre_ping": True,
    "pool_recycle": 3600,
}


def _build_storage() -> BaseStorage:
    url = os.environ.get("OPTUNA_DASHBOARD_STORAGE") or os.environ.get(
        "STORAGE_URL"
    )
    if url:
        url = url.strip()
        _log.info("optuna-dashboard storage from URL env")
        # get_storage handles RDB / Journal / Redis URL guessing. For RDB
        # URLs, rebuild with the pool tuned for the per-process incremental
        # cache (single gunicorn worker); other backends pass through.
        is_rdb = "://" in url and not url.startswith(
            ("redis", "sqlite", "mysql+aiomysql", "postgresql+asyncpg")
        )
        if is_rdb:
            try:
                return RDBStorage(
                    url=url,
                    skip_compatibility_check=True,
                    skip_table_creation=True,
                    engine_kwargs=_RDB_ENGINE_KWARGS,
                )
            except Exception:
                pass  # fall back to optuna-dashboard's own guesser
        return get_storage(url)

    host = os.environ.get("PG_HOST", "").strip()
    port = int(os.environ.get("PG_PORT", "5432"))
    user = os.environ.get("PG_USER", "").strip()
    pwd = os.environ.get("PG_PASSWORD", "")
    db = os.environ.get("OPTUNA_DB", "").strip()
    if not all((host, user, pwd, db)):
        raise SystemExit(
            "No storage configured. Set OPTUNA_DASHBOARD_STORAGE=<url> "
            "or all of PG_HOST/PG_USER/PG_PASSWORD/OPTUNA_DB."
        )
    pg_url = URL.create(
        "postgresql+psycopg2",
        username=user,
        password=pwd,
        host=host,
        port=port,
        database=db,
    )
    _log.info("optuna-dashboard → %s@%s:%d/%s", user, host, port, db)
    return RDBStorage(
        # str(URL) masks the password as '***' in SQLAlchemy 2.x — must use
        # render_as_string(hide_password=False) to keep the credentials.
        url=pg_url.render_as_string(hide_password=False),
        skip_compatibility_check=True,
        skip_table_creation=True,
        engine_kwargs=_RDB_ENGINE_KWARGS,
    )


# gunicorn entrypoint.
application = wsgi(_build_storage())


def main() -> int:
    """Fallback wsgiref launcher (no gunicorn dependency)."""
    bind_host = os.environ.get("BIND_HOST", "0.0.0.0")
    bind_port = int(os.environ.get("BIND_PORT", "8080"))
    _log.info("listen %s:%d", bind_host, bind_port)
    run_server(_build_storage(), host=bind_host, port=bind_port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
