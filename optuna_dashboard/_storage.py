from __future__ import annotations

from datetime import datetime
from datetime import timedelta
import threading

from optuna.storages import BaseStorage
from optuna.storages import RDBStorage
from optuna.study import StudyDirection
from optuna.study._frozen import FrozenStudy
from optuna.trial import FrozenTrial
from sqlalchemy import create_engine
from sqlalchemy import text
from sqlalchemy.engine import Engine

from ._inmemory_cache import InMemoryCache


_count_engine_lock = threading.Lock()
_count_engines: dict[str, "Engine"] = {}


def _count_engine(storage: BaseStorage) -> "Engine | None":
    """A dedicated, tightly-bounded engine just for the COUNT anchor.

    MUST NOT share the main RDBStorage pool: the COUNT runs on every
    refresh and, if it waited on a pool exhausted by big get_all_trials
    reads (or a slow pgbouncer pooler), it would stall. A separate
    pool_size=1 engine with short connect/pool timeouts isolates and
    bounds it — worst case it raises fast and we keep the fast path.
    """
    engine = getattr(storage, "engine", None)
    if engine is None:
        return None
    url = str(engine.url)
    with _count_engine_lock:
        ce = _count_engines.get(url)
        if ce is None:
            connect_args = {}
            if engine.dialect.name == "postgresql":
                connect_args = {"connect_timeout": 3}
            ce = create_engine(
                engine.url,
                pool_size=1,
                max_overflow=1,
                pool_timeout=3,
                pool_recycle=300,
                pool_pre_ping=True,
                connect_args=connect_args,
            )
            _count_engines[url] = ce
        return ce


def _authoritative_trial_count(storage: BaseStorage, study_id: int) -> int | None:
    """Cheap, bounded COUNT(*) of the study's trials straight from the RDB.

    The append-only incremental cache cannot see a study reset when the
    new trials get trial_ids <= the cached watermark (e.g. the sweep
    recreates the Optuna schema → the trial_id sequence restarts). The
    dense-number invariant also passes then, because the stale cached set
    is internally contiguous. An authoritative count is the only reliable
    anchor. Returns None if it can't be obtained quickly (non-RDB / error
    / timeout) so the caller just keeps the fast path. NEVER call this
    while holding the trials-cache lock — it does network I/O.
    """
    ce = _count_engine(storage)
    if ce is None:
        return None
    try:
        with ce.connect() as conn:
            if ce.dialect.name == "postgresql":
                # Cap the query so it can never hang the dashboard.
                conn.exec_driver_sql("SET statement_timeout = 4000")
            return int(
                conn.execute(
                    text("SELECT count(*) FROM trials WHERE study_id = :sid"),
                    {"sid": study_id},
                ).scalar()
            )
    except Exception:
        return None


def get_trials(
    in_memory_cache: InMemoryCache, storage: BaseStorage, study_id: int
) -> list[FrozenTrial]:
    with in_memory_cache._trials_cache_lock:
        trials = in_memory_cache._trials_cache.get(study_id, None)

        # Not a big fan of the heuristic, but I can't think of anything better.
        if trials is None or len(trials) < 100:
            ttl_seconds = 2
        elif len(trials) < 500:
            ttl_seconds = 5
        else:
            ttl_seconds = 10

        last_fetched_at = in_memory_cache._trials_last_fetched_at.get(study_id, None)
        if (
            trials is not None
            and last_fetched_at is not None
            and datetime.now() - last_fetched_at < timedelta(seconds=ttl_seconds)
        ):
            return trials

    # Incremental refresh: finished trials are immutable in Optuna, so once
    # cached we never re-read them. We only ask the backend for trials newer
    # than the last finished one plus the still-unfinished cached trials.
    # This turns a full O(all-trials) reload (the killer at 25k+ trials on an
    # RDB backend) into an O(new + running) fetch on every poll.
    # See optuna.storages._CachedStorage._read_trials_from_remote_storage.
    if trials is not None and hasattr(storage, "_get_trials"):
        stale = False
        try:
            with in_memory_cache._trials_cache_lock:
                included_trial_ids = set(
                    in_memory_cache._trials_unfinished_ids.get(study_id, set())
                )
                last_finished_id = in_memory_cache._trials_last_finished_id.get(study_id, -1)

            updated = storage._get_trials(  # type: ignore[attr-defined]
                study_id,
                states=None,
                included_trial_ids=included_trial_ids,
                trial_id_greater_than=last_finished_id,
            )

            # Authoritative count is network I/O — compute it OUTSIDE the
            # trials-cache lock. Doing it under the lock (the sha-32be77c
            # regression) serialised every request behind one DB round-trip
            # and, when the pooler stalled, wedged the whole worker.
            actual_count = _authoritative_trial_count(storage, study_id)

            with in_memory_cache._trials_cache_lock:
                cached = in_memory_cache._trials_cache.get(study_id) or []
                if updated:
                    by_id: dict[int, FrozenTrial] = {t._trial_id: t for t in cached}
                    for t in updated:
                        by_id[t._trial_id] = t
                    merged = sorted(by_id.values(), key=lambda t: t.number)
                else:
                    merged = cached

                # A single Optuna study numbers its trials exactly
                # {0..N-1} (contiguous, unique). If that breaks, trials
                # were deleted / the study was reset under this
                # long-lived process — the append-only incremental path
                # can't see deletions. Self-heal with a full reload.
                n = len(merged)
                if n:
                    nums = [t.number for t in merged]
                    if max(nums) != n - 1 or len(set(nums)) != n:
                        stale = True

                # Authoritative anchor: if the cache holds MORE trials than
                # the study actually has, it is stale (study reset/trimmed,
                # incl. the schema-recreate / trial_id-sequence-reset case
                # the dense-number invariant cannot detect). Only the
                # "cache > DB" direction is treated as stale — being a few
                # behind during a fast sweep is normal incremental lag, not
                # phantom, and must not thrash the full-reload path.
                if not stale and actual_count is not None and n > actual_count:
                    stale = True

                if not stale:
                    unfinished = in_memory_cache._trials_unfinished_ids.setdefault(
                        study_id, set()
                    )
                    last_finished = in_memory_cache._trials_last_finished_id.get(
                        study_id, -1
                    )
                    for t in updated:
                        if not t.state.is_finished():
                            unfinished.add(t._trial_id)
                            continue
                        last_finished = max(last_finished, t._trial_id)
                        unfinished.discard(t._trial_id)
                    in_memory_cache._trials_last_finished_id[study_id] = last_finished
                    in_memory_cache._trials_cache[study_id] = merged
                    in_memory_cache._trials_last_fetched_at[study_id] = datetime.now()
                    return merged
        except Exception:
            # Any backend/private-API surprise → fall back to a full reload
            # so the dashboard stays correct even if incremental fetch fails.
            pass

        if stale:
            # Wipe every per-study cache, then full-reload below.
            in_memory_cache.invalidate_study(study_id)

    # Cold full reload: storage.get_all_trials over the WHOLE study. At
    # 60k+ trials this is ~100s (6 queries + ORM hydration; host-/pooler-
    # independent). Single-flight it: only one thread (typically the
    # background warmer) does the reload; everyone else returns the
    # current cache immediately so a user request never blocks ~100s
    # (which would hit the cloudflared ~100s edge timeout → 524).
    with in_memory_cache._loading_lock:
        already_loading = study_id in in_memory_cache._loading
        if already_loading:
            current = in_memory_cache._trials_cache.get(study_id)
        else:
            in_memory_cache._loading.add(study_id)
    if already_loading:
        return current if current is not None else []

    try:
        trials = storage.get_all_trials(study_id, deepcopy=False)
        with in_memory_cache._trials_cache_lock:
            unfinished_ids = {t._trial_id for t in trials if not t.state.is_finished()}
            finished_ids = [t._trial_id for t in trials if t.state.is_finished()]
            in_memory_cache._trials_unfinished_ids[study_id] = unfinished_ids
            in_memory_cache._trials_last_finished_id[study_id] = (
                max(finished_ids) if finished_ids else -1
            )
            in_memory_cache._trials_last_fetched_at[study_id] = datetime.now()
            in_memory_cache._trials_cache[study_id] = trials
        return trials
    finally:
        with in_memory_cache._loading_lock:
            in_memory_cache._loading.discard(study_id)


def get_studies(storage: BaseStorage) -> list[FrozenStudy]:
    frozen_studies = storage.get_all_studies()
    if isinstance(storage, RDBStorage):
        frozen_studies = sorted(frozen_studies, key=lambda s: s._study_id)
    return frozen_studies


def get_study(storage: BaseStorage, study_id: int) -> FrozenStudy | None:
    # Build a single FrozenStudy from per-study lookups instead of scanning
    # every study + every study attr (get_all_studies) on every detail poll.
    try:
        study_name = storage.get_study_name_from_id(study_id)
    except KeyError:
        return None
    return FrozenStudy(
        study_name=study_name,
        direction=None,
        directions=storage.get_study_directions(study_id),
        user_attrs=storage.get_study_user_attrs(study_id),
        system_attrs=storage.get_study_system_attrs(study_id),
        study_id=study_id,
    )


def create_new_study(
    storage: BaseStorage, study_name: str, directions: list[StudyDirection]
) -> int:
    study_id = storage.create_new_study(directions, study_name=study_name)
    return study_id
