FROM node:22 AS front-builder

# pnpm prompts "recreate node_modules?" before wiping the dir. Docker build
# has no TTY, so the prompt aborts with ERR_PNPM_ABORTED_REMOVE_MODULES_DIR_NO_TTY;
# CI=true tells pnpm to skip the prompt and proceed.
ENV CI=true
RUN corepack enable

WORKDIR /usr/src/tslib/types
ADD ./tslib/types/ /usr/src/tslib/types/
RUN pnpm install --frozen-lockfile && pnpm run build

WORKDIR /usr/src/tslib/storage
ADD ./tslib/storage/ /usr/src/tslib/storage/
RUN pnpm install --frozen-lockfile && pnpm run build

WORKDIR /usr/src/tslib/react
ADD ./tslib/react/ /usr/src/tslib/react/
RUN pnpm install --frozen-lockfile && pnpm run build

WORKDIR /usr/src/optuna_dashboard
ADD ./optuna_dashboard /usr/src/optuna_dashboard
RUN pnpm install --frozen-lockfile
RUN NODE_OPTIONS="--max-old-space-size=4096" pnpm run build:prd

FROM python:3.12-bookworm AS python-builder

WORKDIR /usr/src
RUN pip install --upgrade pip setuptools
RUN pip install --progress-bar off PyMySQL[rsa] psycopg2-binary gunicorn

ADD ./pyproject.toml /usr/src/pyproject.toml
ADD ./optuna_dashboard /usr/src/optuna_dashboard
COPY --from=front-builder /usr/src/optuna_dashboard/public/ /usr/src/optuna_dashboard/public/
RUN pip install --progress-bar off .

FROM python:3.12-slim-bookworm AS runner

COPY --from=python-builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=python-builder /usr/local/bin/optuna-dashboard /usr/local/bin/optuna-dashboard
# gunicorn console script — needed by the qoxi launcher
# (optuna_dashboard._qoxi); without this only `python -m gunicorn` works.
COPY --from=python-builder /usr/local/bin/gunicorn /usr/local/bin/gunicorn

RUN mkdir /app
WORKDIR /app

# gunicorn 26's control server needs a writable $HOME/.gunicorn.
ENV HOME=/tmp

EXPOSE 8080
# Project-agnostic default: point any project at its Optuna storage via
#   docker run -e OPTUNA_DASHBOARD_STORAGE=<url> ghcr.io/qoxi-cloud/optuna-dashboard
# (or discrete PG_* env vars). Single gunicorn worker on purpose — the
# incremental trial cache is per-process. Override `command` to change.
ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", \
            "--threads", "4", "--timeout", "180", \
            "--graceful-timeout", "30", "optuna_dashboard._qoxi:application"]
