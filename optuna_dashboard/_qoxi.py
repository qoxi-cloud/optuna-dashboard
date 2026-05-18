"""qoxi PG launcher for the janus-sweep Optuna dashboard.

This fork is deployed as the ``janus-sweep-viz`` Deployment. It mirrors
the logic that previously lived in ``src/janus_sweep/viz.py`` of the
janus-sweep image: assemble the Postgres URL from discrete env vars with
``sqlalchemy.engine.URL.create()`` because passwords from the Postgres
operator routinely contain ``@ : / #`` etc., which would break a
string-interpolated ``postgresql://user:pwd@host`` URL.

Run under gunicorn:

    gunicorn --bind 0.0.0.0:8080 --workers 1 --threads 4 \
        optuna_dashboard._qoxi:application

A single worker is intentional: the dashboard's incremental trial cache
is per-process, so multiple workers would each re-read the whole study
from Postgres independently. Threads give I/O concurrency without
splitting the cache.
"""

from __future__ import annotations

import logging
import os

from optuna.storages import RDBStorage
from sqlalchemy.engine import URL

from optuna_dashboard import run_server
from optuna_dashboard import wsgi

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
_log = logging.getLogger("optuna_dashboard.qoxi")


def _build_storage() -> RDBStorage:
    host = os.environ.get("PG_HOST", "").strip()
    port = int(os.environ.get("PG_PORT", "5432"))
    user = os.environ.get("PG_USER", "").strip()
    pwd = os.environ.get("PG_PASSWORD", "")
    db = os.environ.get("OPTUNA_DB", "").strip()
    if not all((host, user, pwd, db)):
        raise SystemExit(
            "PG envs missing — need PG_HOST/PG_USER/PG_PASSWORD/OPTUNA_DB"
        )
    url = URL.create(
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
        url=url.render_as_string(hide_password=False),
        engine_kwargs={
            "pool_size": 2,
            "max_overflow": 2,
            "pool_pre_ping": True,
            "pool_recycle": 3600,
        },
    )


# gunicorn entrypoint.
application = wsgi(_build_storage())


def main() -> int:
    """Fallback wsgiref launcher (parity with the old janus_sweep.viz)."""
    bind_host = os.environ.get("BIND_HOST", "0.0.0.0")
    bind_port = int(os.environ.get("BIND_PORT", "8080"))
    _log.info("listen %s:%d", bind_host, bind_port)
    run_server(_build_storage(), host=bind_host, port=bind_port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
