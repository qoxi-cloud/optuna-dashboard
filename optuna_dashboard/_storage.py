from __future__ import annotations

from datetime import datetime
from datetime import timedelta

from optuna.storages import BaseStorage
from optuna.storages import RDBStorage
from optuna.study import StudyDirection
from optuna.study._frozen import FrozenStudy
from optuna.trial import FrozenTrial

from ._inmemory_cache import InMemoryCache


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
