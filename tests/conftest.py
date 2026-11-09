"""Shared pytest setup.

scikit-surprise is a compiled dependency that is installed in the container but
not in every local environment. recommender/collaborative.py imports it at
module scope and recommender/hybrid.py imports collaborative at module scope,
so without the real package `import recommender.hybrid` raises
ModuleNotFoundError and no test can reach the fallback chain at all.

When the real package is importable it is used untouched - the stub below is
registered only when `surprise` is genuinely absent, and it is test-only: the
recommender itself is not changed, which is where the regression risk lives.
"""

from __future__ import annotations

import importlib.util
import sys
import types


def _register_surprise_stub():
    """Put minimal placeholders for the names collaborative.py imports in sys.modules."""

    class _Placeholder:
        """Stands in for a surprise class; instantiating it is not supported."""

        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "scikit-surprise is not installed; this is a test-only placeholder"
            )

    def _train_test_split(*args, **kwargs):
        raise RuntimeError(
            "scikit-surprise is not installed; this is a test-only placeholder"
        )

    surprise = types.ModuleType("surprise")
    surprise.SVD = _Placeholder
    surprise.Dataset = _Placeholder
    surprise.Reader = _Placeholder

    model_selection = types.ModuleType("surprise.model_selection")
    model_selection.train_test_split = _train_test_split

    accuracy = types.ModuleType("surprise.accuracy")
    accuracy.rmse = _train_test_split
    accuracy.mae = _train_test_split

    surprise.model_selection = model_selection
    surprise.accuracy = accuracy

    sys.modules["surprise"] = surprise
    sys.modules["surprise.model_selection"] = model_selection
    sys.modules["surprise.accuracy"] = accuracy


if importlib.util.find_spec("surprise") is None:
    _register_surprise_stub()
