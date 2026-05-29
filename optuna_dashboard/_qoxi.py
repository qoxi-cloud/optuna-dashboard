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
import threading
import time

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

# Headroom so the warmer's one long cold get_all_trials (~100s) can't
# starve get_study's small lookups. Default pool_timeout (30s) — a short
# 10s timeout made queries fail/retry under transient contention. Single
# dashboard process → a handful of extra PG connections is well within
# budget.
_RDB_ENGINE_KWARGS = {
    "pool_size": 4,
    "max_overflow": 8,
    "pool_pre_ping": True,
    "pool_recycle": 3600,
}


class _FollowActiveRedisStorage:
    """Single-study Redis JournalStorage that AUTO-FOLLOWS the sweep's currently
    running study.

    The no-ML sweep writes each Optuna study under its OWN Redis key prefix
    (``{study}:``) and GCs finished ones, so a plain ``JournalStorage(prefix=…)``
    is pinned to one study and goes empty the moment the sweep advances to the
    next (bar,expiry) cell / pair. pod-0 records the live study name in the key
    ``janus_dashboard:current_study``; this wrapper re-reads it on a short TTL and
    swaps the delegate ``JournalStorage`` when it changes, so the dashboard always
    tracks the running study with no hardcoded prefix and no restart. All optuna
    ``BaseStorage`` calls are forwarded to the live delegate via ``__getattr__``.
    When the delegate swaps, study_id stays 1 with a new trial count → the fork's
    COUNT-anchor incremental-cache self-heal reloads (same path as a wipe)."""

    _MARKER = "janus_dashboard:current_study"

    def __init__(self, redis_url: str, ttl: float = 10.0) -> None:
        import redis  # noqa: PLC0415

        self._url = redis_url
        self._ttl = ttl
        self._prefix: str | None = None
        self._delegate: BaseStorage | None = None
        self._last = 0.0
        self._lock = threading.Lock()
        self._redis = redis.from_url(redis_url, socket_connect_timeout=5)

    def _resolve_prefix(self) -> str:
        """Prefix of the currently-running study, from pod-0's marker key.
        Falls back to the most-recently-written journal prefix (highest log
        index) if the marker is absent, then to '' (empty → no studies)."""
        try:
            v = self._redis.get(self._MARKER)
            if v:
                name = v.decode() if isinstance(v, (bytes, bytearray)) else v
                return f"{name}:"
        except Exception as exc:  # noqa: BLE001
            _log.warning("current-study marker read failed (%s); keeping current", exc)
            return self._prefix or ""
        # marker absent → best-effort scan for the highest journal log index.
        best_prefix, best_n = "", -1
        try:
            for raw in self._redis.scan_iter(match="*:log:*", count=1000):
                k = raw.decode() if isinstance(raw, (bytes, bytearray)) else raw
                pre, sep, num = k.rpartition(":log:")
                if not sep:
                    continue
                try:
                    n = int(num)
                except ValueError:
                    continue
                if n > best_n:
                    best_n, best_prefix = n, pre
        except Exception as exc:  # noqa: BLE001
            _log.warning("active-prefix scan failed (%s); keeping current", exc)
            return self._prefix or ""
        return best_prefix

    def _live(self) -> BaseStorage:
        now = time.monotonic()
        with self._lock:
            if self._delegate is None or now - self._last > self._ttl:
                self._last = now
                pre = self._resolve_prefix()
                if self._delegate is None or pre != self._prefix:
                    from optuna.storages import JournalStorage  # noqa: PLC0415
                    from optuna.storages.journal import JournalRedisBackend  # noqa: PLC0415

                    if pre != self._prefix:
                        _log.info("dashboard now following study prefix %r", pre)
                    self._prefix = pre
                    self._delegate = JournalStorage(JournalRedisBackend(self._url, prefix=pre))
            return self._delegate

    def __getattr__(self, name: str):
        # Reached only for attributes not on the instance (the BaseStorage API).
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return getattr(self._live(), name)


def _build_storage() -> BaseStorage:
    # Redis JournalStorage (the no-ML sweep). The sweep writes each study under a
    # per-study key prefix (to bound each worker's in-RAM journal replay), so the
    # dashboard must be pointed at ONE study's prefix via OPTUNA_REDIS_PREFIX —
    # with an empty prefix it sees the (empty) root namespace. Set the prefix to
    # the study you want to inspect, e.g. "EURUSD_bs300s_1m_SELL_noml:".
    redis_url = os.environ.get("OPTUNA_REDIS", "").strip()
    if redis_url:
        from optuna.storages import JournalStorage
        from optuna.storages.journal import JournalRedisBackend

        prefix = os.environ.get("OPTUNA_REDIS_PREFIX", "").strip()
        # AUTO / empty → follow the sweep's currently-running per-study prefix at
        # runtime (never goes stale when the sweep advances cell/pair). A literal
        # prefix pins the dashboard to exactly that one study.
        if prefix in ("", "AUTO", "auto"):
            _log.info("optuna-dashboard → Redis %s (AUTO: follow active study)", redis_url)
            return _FollowActiveRedisStorage(redis_url)  # type: ignore[return-value]
        _log.info(
            "optuna-dashboard → Redis JournalStorage %s (prefix=%r)",
            redis_url,
            prefix,
        )
        return JournalStorage(JournalRedisBackend(redis_url, prefix=prefix))

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
