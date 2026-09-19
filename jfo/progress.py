"""Shared live progress reporting built on ``tqdm``."""

from typing import Any, Iterable, Optional

from tqdm import tqdm as _tqdm


def progress(
    iterable: Optional[Iterable] = None,
    total: Optional[int] = None,
    desc: Optional[str] = None,
    unit: Optional[str] = None,
    **kwargs: Any,
):
    """Create a live progress bar with consistent formatting."""
    return _tqdm(
        iterable,
        total=total,
        desc=desc,
        unit=unit,
        dynamic_ncols=True,
        leave=True,
        **kwargs,
    )


def note(message: str) -> None:
    """Write a line above the active progress bar without breaking it."""
    _tqdm.write(message)
