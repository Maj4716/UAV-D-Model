"""Small immutable document snapshots used at executable model boundaries."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping


class FrozenSnapshotDict(dict):
    """A JSON-like mapping that preserves ``dict`` read compatibility."""

    @staticmethod
    def _immutable(*_args, **_kwargs):
        raise TypeError("executable model snapshots are immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable

    def __deepcopy__(self, memo):
        copied = {key: deepcopy(value, memo) for key, value in self.items()}
        memo[id(self)] = copied
        return copied

    def __reduce__(self):
        # Reconstruct through ``dict.__init__`` instead of pickle's incremental
        # ``__setitem__`` protocol, which is intentionally disabled here.
        return (FrozenSnapshotDict, (dict(self),))


def freeze_snapshot(value: Any) -> Any:
    """Recursively freeze a detached JSON-compatible document."""

    if isinstance(value, Mapping):
        return FrozenSnapshotDict(
            {str(key): freeze_snapshot(child) for key, child in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(freeze_snapshot(child) for child in value)
    return value


def detached_snapshot(value: Mapping[str, Any]) -> FrozenSnapshotDict:
    """Copy then freeze a mapping so caller-owned aliases cannot survive."""

    return freeze_snapshot(deepcopy(dict(value)))


__all__ = ["FrozenSnapshotDict", "detached_snapshot", "freeze_snapshot"]
