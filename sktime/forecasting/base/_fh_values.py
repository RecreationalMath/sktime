# !/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Numpy-backed internal types and utilities for ForecastingHorizon.

This module is pandas-free.
Contains the FHValueType enum, frequency validation, the _InternalFH
transfer type used by PandasFHConverter, and helper functions.
"""

__all__ = ["FHValueType", "_InternalFH", "VALID_FREQ_BASES", "validate_freq"]

import re
from enum import Enum, auto
from typing import NamedTuple

# core dependency
import numpy as np

# sentinel for distinguishing "not provided" from None in clone()
_UNSET = object()

# Standard time series frequency base mnemonics.
# These are well-established, domain-standard frequencies used in
# time series forecasting. Pandas-specific variants (e.g. "BM", "MS", "QS")
# are handled by PandasFHConverter when extracting from pandas objects.
VALID_FREQ_BASES = frozenset(
    {
        "Y",  # yearly
        "Q",  # quarterly
        "M",  # monthly
        "W",  # weekly
        "D",  # daily
        "h",  # hourly
        "min",  # minutely
        "s",  # secondly
        "ms",  # millisecond
        "us",  # microsecond
        "ns",  # nanosecond
    }
)

# Regex: optional integer multiplier followed by a frequency base
_FREQ_PATTERN = re.compile(
    r"^(\d+)?(" + "|".join(sorted(VALID_FREQ_BASES, key=len, reverse=True)) + r")$"
)


def validate_freq(freq_str):
    """Validate a frequency string against accepted standard values.

    Accepted format is an optional integer multiplier followed by a base
    frequency mnemonic, e.g. ``"M"``, ``"2D"``, ``"4h"``, ``"15min"``.

    Parameters
    ----------
    freq_str : str
        Frequency string to validate.

    Returns
    -------
    str
        The validated frequency string (unchanged).

    Raises
    ------
    ValueError
        If ``freq_str`` does not match any accepted frequency pattern.

    Examples
    --------
    >>> validate_freq("M")
    'M'
    >>> validate_freq("2D")
    '2D'
    >>> validate_freq("15min")
    '15min'
    """
    if _FREQ_PATTERN.match(freq_str):
        return freq_str
    raise ValueError(
        f"Invalid frequency string: {freq_str!r}. "
        f"Expected an optional integer multiplier followed by one of "
        f"{sorted(VALID_FREQ_BASES)}, e.g. 'M', '2D', '4h', '15min'."
    )


class FHValueType(Enum):
    """Enum describing the semantic type of forecasting horizon values.

    Attributes
    ----------
    INT : integer steps
        Used for both relative integer horizons and absolute integer indices.
        Stored as int64 values directly.
    TIMEDELTA : durations stored as int64 nanoseconds
        Used for relative time-based horizons.
    PERIOD : integer ordinals that represent pandas Period values
        Used for absolute period-based horizons. Requires freq.
    DATETIME : timestamps stored as int64 nanoseconds.
        Used for absolute datetime-based horizons.
    """

    INT = auto()
    TIMEDELTA = auto()
    PERIOD = auto()
    DATETIME = auto()


# Which value types can represent relative forecasting horizons
_RELATIVE_VALUE_TYPES = frozenset({FHValueType.INT, FHValueType.TIMEDELTA})

# Which value types can represent absolute forecasting horizons
_ABSOLUTE_VALUE_TYPES = frozenset(
    {FHValueType.INT, FHValueType.PERIOD, FHValueType.DATETIME}
)


class _InternalFH(NamedTuple):
    """Transfer type from PandasFHConverter to ForecastingHorizon.

    This is a lightweight, immutable container returned by
    PandasFHConverter.to_internal() and unpacked into ForecastingHorizon
    attributes during construction.

    Parameters
    ----------
    values : np.ndarray
        1-D int64 numpy array of horizon values.
    value_type : FHValueType
        Semantic type of the stored values.
    freq : str or None
        Frequency string (e.g. "M", "D", "h"), or None.
    timezone : str or None
        Timezone string for DATETIME values, or None.
    """

    values: np.ndarray
    value_type: FHValueType
    freq: str | None = None
    timezone: str | None = None


def is_contiguous(values, value_type):
    """Check if values form a contiguous sequence.

    Checking logic depends on value type:
    - For INT and PERIOD: checks consecutive integers.
    - For TIMEDELTA and DATETIME: infers step from min diff, checks coverage.

    Parameters
    ----------
    values : np.ndarray
        Sorted, deduplicated int64 array.
    value_type : FHValueType
        Semantic type of the values.

    Returns
    -------
    bool
        True if values form a contiguous sequence.
    """
    if len(values) <= 1:
        return True

    if value_type in (FHValueType.INT, FHValueType.PERIOD):
        # contiguous means every integer between min and max is present
        expected_len = int(values[-1] - values[0]) + 1
        # the above check is complete because values is sorted and unique
        return len(values) == expected_len

    # TIMEDELTA or DATETIME: check uniform spacing
    diffs = np.diff(values)
    if diffs.min() <= 0:
        return False
    min_diff = diffs.min()
    # all diffs should equal the minimum diff for uniform spacing
    return bool(np.all(diffs == min_diff))
