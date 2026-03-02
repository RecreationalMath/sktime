# !/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""
ForecastingHorizon: pandas-agnostic forecasting horizon implementation.

All pandas-specific logic (type conversions, frequency handling, version detection)
is delegated to the _fh_utils module.
"""

__all__ = ["ForecastingHorizon"]

import numpy as np

from sktime.forecasting.base._fh_utils import PandasFHConverter
from sktime.forecasting.base._fh_values import (
    _ABSOLUTE_VALUE_TYPES,
    _RELATIVE_VALUE_TYPES,
    _UNSET,
    FHValueType,
    is_contiguous,
    validate_freq,
)


class ForecastingHorizon:
    """Forecasting horizon with pandas-decoupled internals.

    Internally stores values as a sorted, deduplicated int64 numpy array
    together with metadata (value type, frequency, timezone). This follows
    the Arrow pattern of Schema (metadata) + Buffer (raw data).

    Parameters
    ----------
    values : int, list, np.ndarray, range, pd.Index, pd.Timedelta,
        pd.offsets.BaseOffset
        Values of forecasting horizon.
        Supported types without pandas dependency:
        - ``int`` or ``np.integer`` : single integer step
        - ``list[int]`` : list of integer steps
        - ``np.ndarray`` : integer, timedelta64, or datetime64 array
        - ``range`` : Python range object
        Supported pandas types (delegated to PandasFHConverter):
        - ``pd.PeriodIndex``, ``pd.DatetimeIndex``, ``pd.TimedeltaIndex``
        - ``pd.RangeIndex``, ``pd.Index`` (integer or timedelta dtype)
        - ``pd.Timedelta``, ``pd.offsets.BaseOffset``
        - ``list[pd.Period]``, ``list[pd.Timestamp]``, ``list[pd.Timedelta]``
        - ``list[pd.offsets.BaseOffset]``, ``list[np.timedelta64]``
    is_relative : bool, optional (default=None)
        Whether the forecasting horizon is relative to the training cutoff.
        If True, values are relative to end of training series.
        If False, values are absolute.
        If None, inferred from value type:
        - int values default to relative (is_relative=True)
        - timedelta values are always relative
        - Period and Timestamp values are always absolute
        Note: integer values are compatible with both relative and absolute
        interpretations. For integers, pass ``is_relative=False`` explicitly
        if absolute is intended, as the default inference interprets it as relative.
    freq : str, pd.Index, pd.Period, pandas offset, or sktime forecaster,
        optional (default=None)
        Frequency information for the horizon values.
        When values already carry frequency (e.g., pd.PeriodIndex,
        pd.DatetimeIndex, or pd.TimedeltaIndex), provided ``freq`` must match the
        values' frequency, otherwise a ValueError is raised.
        When values do not carry frequency (e.g. int, list, np.ndarray), ``freq``
        is used directly if provided.

    Examples
    --------
    >>> from sktime.forecasting.base._fh_v2 import ForecastingHorizon
    >>> fh = ForecastingHorizon([1, 2, 3])
    >>> fh.is_relative
    True
    >>> fh.to_numpy()
    numpy.ndarray([1, 2, 3])
    """

    def __init__(
        self,
        values=None,
        is_relative: bool | None = None,
        freq=None,
    ):
        # --- convert values to internal representation ---
        # canonical path: plain Python/numpy types — no pandas needed
        if isinstance(values, (int, np.integer)):
            arr = np.array([int(values)], dtype=np.int64)
            self._values = arr
            self._value_type = FHValueType.INT
            self._freq = None
            self._timezone = None
        elif isinstance(values, range):
            arr = np.array(list(values), dtype=np.int64)
            self._values = arr
            self._value_type = FHValueType.INT
            self._freq = None
            self._timezone = None
        elif isinstance(values, np.ndarray):
            result = self._ndarray_to_internal(values)
            self._values = result.values
            self._value_type = result.value_type
            self._freq = result.freq
            self._timezone = result.timezone
        elif (
            isinstance(values, list)
            and len(values) > 0
            and isinstance(values[0], (int, np.integer))
        ):
            for i, v in enumerate(values[1:], start=1):
                if not isinstance(v, (int, np.integer)):
                    raise TypeError(
                        f"Element at index 0 is of type "
                        f"{type(values[0]).__name__}, but element at "
                        f"index {i} is {type(v).__name__}. "
                        "All list elements must be of the same type."
                    )
            arr = np.array(values, dtype=np.int64)
            self._values = arr
            self._value_type = FHValueType.INT
            self._freq = None
            self._timezone = None
        # coerced path: pandas types and non-int lists — delegate to converter
        else:
            result = PandasFHConverter.to_internal(values)
            self._values = result.values
            self._value_type = result.value_type
            self._freq = result.freq
            self._timezone = result.timezone

        # sort and deduplicate values
        self._values = np.unique(self._values)

        # handle empty arrays
        if len(self._values) == 0:
            self._value_type = FHValueType.INT
            self._freq = None
            self._timezone = None

        # --- set freq via setter (single gate for validation) ---
        if freq is not None:
            self.freq = freq

        # --- determine is_relative ---
        if is_relative is not None:
            if not isinstance(is_relative, bool):
                raise TypeError("`is_relative` must be a boolean or None")
            if is_relative and self._value_type not in _RELATIVE_VALUE_TYPES:
                raise TypeError(
                    f"`values` type {self._value_type.name} is "
                    f"not compatible with `is_relative=True`."
                )
            if not is_relative and self._value_type not in _ABSOLUTE_VALUE_TYPES:
                raise TypeError(
                    f"`values` type {self._value_type.name} is "
                    f"not compatible with `is_relative=False`."
                )
            self._is_relative = is_relative
        else:
            self._is_relative = self._infer_is_relative(self._value_type)

    @staticmethod
    def _infer_is_relative(value_type: FHValueType) -> bool:
        """Infer is_relative from value type.

        Parameters
        ----------
        value_type : FHValueType
            The semantic type of the stored values.

        Returns
        -------
        bool
            Inferred is_relative flag.

        Raises
        ------
        TypeError
            If is_relative cannot be inferred for the given value type.
        """
        if value_type == FHValueType.TIMEDELTA:
            return True
        elif value_type in (FHValueType.PERIOD, FHValueType.DATETIME):
            return False
        elif value_type == FHValueType.INT:
            # INT can be either relative or absolute;
            # default to relative for backwards compatibility with _fh.py
            return True
        else:
            raise TypeError(
                f"Cannot infer is_relative for value type {value_type.name}"
            )

    @staticmethod
    def _ndarray_to_internal(values: np.ndarray):
        """Convert 1-D numpy array to _InternalFH, inferring type from dtype.

        Parameters
        ----------
        values : np.ndarray
            1-D numpy array with integer, timedelta64, or datetime64 dtype.

        Returns
        -------
        _InternalFH
            Internal representation.

        Raises
        ------
        ValueError
            If array is not 1-D or is empty.
        TypeError
            If array dtype is not supported.
        """
        from sktime.forecasting.base._fh_values import _InternalFH

        if values.ndim != 1:
            raise ValueError(f"Expected 1-D array, got {values.ndim}-D array")
        if len(values) == 0:
            raise ValueError("Forecasting horizon values must not be empty.")

        # timedelta64 and datetime64 checked before integer as a defensive
        # measure — some numpy versions consider datetime64 a subtype of
        # integer, which would cause incorrect classification.
        if np.issubdtype(values.dtype, np.timedelta64):
            arr = values.astype("timedelta64[ns]").view(np.int64).copy()
            return _InternalFH(arr, FHValueType.TIMEDELTA)

        if np.issubdtype(values.dtype, np.datetime64):
            arr = values.astype("datetime64[ns]").view(np.int64).copy()
            return _InternalFH(arr, FHValueType.DATETIME)

        if np.issubdtype(values.dtype, np.integer):
            arr = values.astype(np.int64).copy()
            return _InternalFH(arr, FHValueType.INT)

        raise TypeError(
            f"np.ndarray with dtype {values.dtype} is not supported. "
            f"Expected integer, timedelta64, or datetime64 dtype."
        )

    def clone(
        self,
        values=None,
        value_type=None,
        is_relative=None,
        freq=_UNSET,
        timezone=_UNSET,
    ):
        """Create a new ForecastingHorizon with selectively replaced attributes.

        Bypasses ``__init__`` conversion logic. Values are sorted and
        deduplicated when provided.

        Parameters
        ----------
        values : np.ndarray, optional
            New values array. If None, copies current values.
        value_type : FHValueType, optional
            New value type. If None, uses current value type.
        is_relative : bool, optional
            New is_relative flag. If None, uses current.
        freq : str or None, optional
            New freq. If not provided (sentinel), uses current freq.
        timezone : str or None, optional
            New timezone. If not provided (sentinel), uses current timezone.

        Returns
        -------
        ForecastingHorizon
            New instance with replaced attributes.
        """
        new = object.__new__(ForecastingHorizon)
        new._values = np.unique(values) if values is not None else self._values.copy()
        new._value_type = value_type if value_type is not None else self._value_type
        new._is_relative = is_relative if is_relative is not None else self._is_relative
        new._freq = self._freq if freq is _UNSET else freq
        new._timezone = self._timezone if timezone is _UNSET else timezone
        return new

    @classmethod
    def _from_internal(cls, values, value_type, is_relative, freq=None, timezone=None):
        """Construct a ForecastingHorizon without coercion.

        This is a fast-path constructor that bypasses ``__init__`` entirely.
        Values must already be sorted and deduplicated int64.

        Parameters
        ----------
        values : np.ndarray
            Sorted, deduplicated int64 array.
        value_type : FHValueType
            Semantic type of the values.
        is_relative : bool
            Whether the horizon is relative.
        freq : str or None, optional
            Frequency string.
        timezone : str or None, optional
            Timezone string.

        Returns
        -------
        ForecastingHorizon
            New instance.
        """
        obj = object.__new__(cls)
        obj._values = values
        obj._value_type = value_type
        obj._is_relative = is_relative
        obj._freq = freq
        obj._timezone = timezone
        return obj

    @property
    def is_relative(self) -> bool:
        """Whether forecasting horizon is relative to the end of the training series.

        Returns
        -------
        is_relative : bool
        """
        return self._is_relative

    @is_relative.setter
    def is_relative(self, value: bool) -> None:
        """Set is_relative flag."""
        self._is_relative = value

    @property
    def freq(self) -> str | None:
        """Frequency string, or None."""
        return self._freq

    @freq.setter
    def freq(self, obj) -> None:
        """Set frequency from string, pd.Index, pd.offset, or forecaster.

        For string inputs, the frequency is validated against accepted
        standard time series frequencies (e.g. ``"M"``, ``"D"``, ``"2h"``).
        If validation fails, a fallback check via pandas ``to_offset`` is
        attempted before raising an error.

        For non-string inputs (pd.Index, pd.offsets.BaseOffset, forecaster),
        frequency is extracted and normalized via PandasFHConverter.

        If the ForecastingHorizon already carries a frequency (inferred from
        values), the new frequency must match, otherwise a ValueError is raised.

        Parameters
        ----------
        obj : str, pd.Index, pd.offsets.BaseOffset, or forecaster
            Object carrying frequency information.
            When str, must be a valid frequency mnemonic, optionally
            with an integer multiplier prefix.
            Accepted base frequencies: Y, Q, M, W, D, h, min, s, ms, us, ns.
            Examples: ``"M"``, ``"2D"``, ``"4h"``, ``"15min"``.

        Raises
        ------
        ValueError
            If freq is already set and conflicts with new value, or
            if a string freq is not a recognized frequency mnemonic.
        """
        if isinstance(obj, str):
            try:
                new_freq = validate_freq(obj)
            except ValueError:
                # fallback: try pandas to_offset for exotic but valid freqs
                new_freq = PandasFHConverter.extract_freq(obj)
                if new_freq is None:
                    raise
        elif obj is None:
            return
        else:
            new_freq = PandasFHConverter.extract_freq(obj)

        old_freq = self._freq
        if old_freq is not None and new_freq is not None and old_freq != new_freq:
            raise ValueError(
                f"Frequencies do not match: current={old_freq!r}, new={new_freq!r}"
            )
        if new_freq is not None:
            self._freq = new_freq

    # core conversion methods

    def to_relative(self, cutoff=None):
        """Return relative version of forecasting horizon.

        Parameters
        ----------
        cutoff : pd.Period, pd.Timestamp, int, or pd.Index, optional
            Cutoff value required for conversion.

        Returns
        -------
        ForecastingHorizon
            Relative representation of forecasting horizon.
        """
        if self._is_relative:
            return self.clone()

        if cutoff is None:
            raise ValueError(
                "`cutoff` must be provided to convert absolute FH to relative."
            )

        cutoff_val, cutoff_type, cutoff_freq, cutoff_tz = (
            PandasFHConverter.cutoff_to_internal(cutoff, freq=self.freq)
        )

        # mismatch between the FH frequency and cutoff frequency
        # can happen and should be flagged
        if (
            self.freq is not None
            and cutoff_freq is not None
            and self.freq != cutoff_freq
        ):
            raise ValueError(
                f"Frequency mismatch between FH and cutoff: "
                f"FH freq={self.freq}, cutoff freq={cutoff_freq}"
            )
        freq = self.freq or cutoff_freq

        # vtype can only be absolute types (PERIOD, DATETIME, or INT) at this point,
        # because if it were a relative type,
        # to_relative would return at the start of the method
        vtype = self._value_type
        vals = self._values

        if vtype == FHValueType.PERIOD:
            # ordinal difference -> integer steps
            # divide by freq multiplier to get step count
            # e.g., "2D" has multiplier 2, so ordinal diff of 4 = 2 steps
            relative_vals = vals - cutoff_val
            if freq is not None:
                mult = PandasFHConverter.freq_multiplier(freq)
                if mult != 1:
                    relative_vals = relative_vals // mult
            return self.clone(
                values=relative_vals.astype(np.int64),
                value_type=FHValueType.INT,
                freq=freq,
                is_relative=True,
            )

        if vtype == FHValueType.DATETIME:
            # nanosecond difference
            relative_nanos = (vals - cutoff_val).astype(np.int64)
            if freq is not None:
                # convert nanosecond diffs to integer steps using freq
                relative_vals = PandasFHConverter.nanos_to_steps(
                    relative_nanos, freq, ref_nanos=cutoff_val
                )
                return self.clone(
                    values=relative_vals,
                    value_type=FHValueType.INT,
                    freq=freq,
                    is_relative=True,
                )
            else:
                # no freq: return as TIMEDELTA nanoseconds
                return self.clone(
                    values=relative_nanos,
                    value_type=FHValueType.TIMEDELTA,
                    freq=freq,
                    is_relative=True,
                )

        if vtype == FHValueType.INT:
            # absolute int - cutoff int -> relative int
            relative_vals = vals - cutoff_val
            return self.clone(
                values=relative_vals.astype(np.int64),
                value_type=FHValueType.INT,
                freq=freq,
                is_relative=True,
            )

        # if we reach this point,
        # it means the value type is not compatible with relative representation
        raise TypeError(f"Cannot convert {vtype.name} to relative.")

    def to_absolute(self, cutoff):
        """Return absolute version of forecasting horizon.

        Parameters
        ----------
        cutoff : pd.Period, pd.Timestamp, int, or pd.Index
            Cutoff value is required to convert a relative forecasting
            horizon to an absolute one (and vice versa).
            If pd.Index, last/latest value is considered the cutoff

        Returns
        -------
        ForecastingHorizon
            Absolute representation of forecasting horizon.
        """
        if not self._is_relative:
            return self.clone()

        cutoff_val, cutoff_type, cutoff_freq, cutoff_tz = (
            PandasFHConverter.cutoff_to_internal(cutoff, freq=self.freq)
        )

        # mismatch between the FH frequency and cutoff frequency
        # can happen and should be flagged
        if (
            self.freq is not None
            and cutoff_freq is not None
            and self.freq != cutoff_freq
        ):
            raise ValueError(
                f"Frequency mismatch between FH and cutoff: "
                f"FH freq={self.freq}, cutoff freq={cutoff_freq}"
            )
        freq = self.freq or cutoff_freq

        # vtype can only be relative types (INT or TIMEDELTA) at this point,
        # because if it were an absolute type,
        # to_absolute would return at the start of the method
        vtype = self._value_type
        vals = self._values

        if vtype == FHValueType.INT:
            if cutoff_type == FHValueType.PERIOD:
                # int steps + period ordinal -> period ordinals
                # multiply by freq multiplier for multi-step freqs
                # e.g., "2D" has multiplier 2, so step 1 = 2 ordinals
                step_vals = vals
                if freq is not None:
                    mult = PandasFHConverter.freq_multiplier(freq)
                    if mult != 1:
                        step_vals = vals * mult
                absolute_vals = cutoff_val + step_vals
                return self.clone(
                    values=absolute_vals.astype(np.int64),
                    value_type=FHValueType.PERIOD,
                    freq=freq,
                    is_relative=False,
                )
            if cutoff_type == FHValueType.DATETIME:
                if freq is None:
                    raise ValueError(
                        "freq is required to convert integer relative FH "
                        "to absolute datetime. Set freq on the FH or provide "
                        "a cutoff with frequency information."
                    )
                nanos = PandasFHConverter.steps_to_nanos(
                    vals, freq, ref_nanos=cutoff_val
                )
                absolute_vals = cutoff_val + nanos
                return self.clone(
                    values=absolute_vals.astype(np.int64),
                    value_type=FHValueType.DATETIME,
                    freq=freq,
                    timezone=cutoff_tz,
                    is_relative=False,
                )

            if cutoff_type == FHValueType.INT:
                # int + int -> int (absolute)
                absolute_vals = cutoff_val + vals
                return self.clone(
                    values=absolute_vals.astype(np.int64),
                    value_type=FHValueType.INT,
                    freq=freq,
                    is_relative=False,
                )
        if vtype == FHValueType.TIMEDELTA:
            if cutoff_type == FHValueType.DATETIME:
                # nanos + nanos -> absolute datetime nanos
                absolute_vals = cutoff_val + vals
                return self.clone(
                    values=absolute_vals.astype(np.int64),
                    value_type=FHValueType.DATETIME,
                    freq=freq,
                    timezone=cutoff_tz,
                    is_relative=False,
                )
        # if we reach this point,
        # it means the value type is not compatible with absolute representation
        raise TypeError(
            f"Cannot convert {vtype.name} (relative) to absolute "
            f"with cutoff type {cutoff_type.name}."
        )

    def to_pandas(self):
        """Return forecasting horizon values as pd.Index.

        Returns
        -------
        pd.Index
            Pandas Index containing the forecasting horizon values.
        """
        return PandasFHConverter.to_pandas_index(
            self._values, self._value_type, self._freq, self._timezone
        )

    def to_numpy(self, **kwargs) -> np.ndarray:
        """Return forecasting horizon values as numpy array.

        Returns
        -------
        np.ndarray
            Numpy array of int64 values.
        """
        return self._values.copy()

    def to_absolute_index(self, cutoff=None):
        """Return absolute values as pandas Index.

        Parameters
        ----------
        cutoff : pd.Period, pd.Timestamp, int, or pd.Index, optional
            Cutoff value for conversion.

        Returns
        -------
        pd.Index
            Absolute forecasting horizon as pandas Index.
        """
        return self.to_absolute(cutoff).to_pandas()

    def to_absolute_int(self, start, cutoff=None):
        """Return absolute values as zero-based integer index from ``start``.

        Parameters
        ----------
        start : pd.Period, pd.Timestamp, int
            Start value returned as zero.
        cutoff : pd.Period, pd.Timestamp, int, or pd.Index, optional
            Cutoff value for conversion.

        Returns
        -------
        ForecastingHorizon
            Absolute representation as zero-based integer index.
        """
        # get absolute representation
        absolute = self.to_absolute(cutoff)
        abs_vals = absolute._values
        abs_type = absolute._value_type
        abs_freq = absolute._freq

        # convert start to internal
        start_val, start_type, start_freq, _ = PandasFHConverter.cutoff_to_internal(
            start, freq=self.freq
        )

        # compute zero-based integers
        if abs_type == FHValueType.PERIOD:
            integers = abs_vals - start_val
            # check for frequency mismatch between FH freq, cutoff freq, and start freq
            freq = None
            for candidate in (abs_freq, self.freq, start_freq):
                if candidate is not None:
                    if freq is None:
                        freq = candidate
                    elif candidate != freq:
                        raise ValueError(
                            f"Frequency mismatch in to_absolute_int: "
                            f"abs_freq={abs_freq}, self.freq={self.freq}, "
                            f"start_freq={start_freq}. All must agree."
                        )
            # divide by freq multiplier for multi-step freqs
            if freq is not None:
                mult = PandasFHConverter.freq_multiplier(freq)
                if mult != 1:
                    integers = integers // mult
        elif abs_type == FHValueType.DATETIME:
            nanos_diff = abs_vals - start_val
            # check for frequency mismatch between FH freq, cutoff freq, and start freq
            freq = None
            for candidate in (abs_freq, self.freq, start_freq):
                if candidate is not None:
                    if freq is None:
                        freq = candidate
                    elif candidate != freq:
                        raise ValueError(
                            f"Frequency mismatch in to_absolute_int: "
                            f"abs_freq={abs_freq}, self.freq={self.freq}, "
                            f"start_freq={start_freq}. All must agree."
                        )
            if freq is not None:
                integers = PandasFHConverter.nanos_to_steps(
                    nanos_diff, freq, ref_nanos=start_val
                )
            else:
                # fall back to raw nanos difference
                integers = nanos_diff
        else:
            integers = abs_vals - start_val

        return self.clone(
            values=integers.astype(np.int64),
            value_type=FHValueType.INT,
            freq=self.freq,
            is_relative=False,
        )

    # In-sample and out-of-sample methods

    def _is_in_sample(self, cutoff=None) -> np.ndarray:
        """Return boolean array indicating in-sample values.

        In-sample values have relative representation <= 0.
        """
        relative = self.to_relative(cutoff)
        return relative._values <= 0

    def _is_out_of_sample(self, cutoff=None) -> np.ndarray:
        """Return boolean array indicating out-of-sample values."""
        return np.logical_not(self._is_in_sample(cutoff))

    def is_all_in_sample(self, cutoff=None) -> bool:
        """Whether the forecasting horizon is purely in-sample.

        Parameters
        ----------
        cutoff : pd.Period, pd.Timestamp, int, optional
            Cutoff value.

        Returns
        -------
        bool
        """
        return bool(self._is_in_sample(cutoff).all())

    def is_all_out_of_sample(self, cutoff=None) -> bool:
        """Whether the forecasting horizon is purely out-of-sample.

        Parameters
        ----------
        cutoff : pd.Period, pd.Timestamp, int, optional
            Cutoff value.

        Returns
        -------
        bool
        """
        return bool(self._is_out_of_sample(cutoff).all())

    def to_in_sample(self, cutoff=None):
        """Return in-sample values of fh.

        Parameters
        ----------
        cutoff : pd.Period, pd.Timestamp, int, optional
            Cutoff value for conversion.

        Returns
        -------
        ForecastingHorizon
            In-sample values of forecasting horizon.
        """
        mask = self._is_in_sample(cutoff)
        filtered_vals = self._values[mask]
        return self.clone(values=filtered_vals)

    def to_out_of_sample(self, cutoff=None):
        """Return out-of-sample values of fh.

        Parameters
        ----------
        cutoff : pd.Period, pd.Timestamp, int, optional
            Cutoff value for conversion.

        Returns
        -------
        ForecastingHorizon
            Out-of-sample values of forecasting horizon.
        """
        mask = self._is_out_of_sample(cutoff)
        filtered_vals = self._values[mask]
        return self.clone(values=filtered_vals)

    # indexer method
    def to_indexer(self, cutoff=None, from_cutoff=True):
        """Return zero-based indexer for array access.

        Parameters
        ----------
        cutoff : pd.Period, pd.Timestamp, int, optional
            Cutoff value for conversion.
        from_cutoff : bool, optional (default=True)
            If True, zero-based relative to cutoff.
            If False, zero-based relative to first value in fh.

        Returns
        -------
        pd.Index
            Zero-based integer indexer.
        """
        if from_cutoff:
            relative = self.to_relative(cutoff)
            vtype = relative._value_type
            if vtype == FHValueType.INT:
                indexer_vals = relative._values - 1
            elif vtype == FHValueType.TIMEDELTA:
                freq = self.freq
                if freq is None and cutoff is not None:
                    _, _, cutoff_freq, _ = PandasFHConverter.cutoff_to_internal(
                        cutoff, freq=self.freq
                    )
                    freq = cutoff_freq
                if freq is None:
                    raise ValueError(
                        "freq is required to compute an integer indexer "
                        "from timedelta-based forecasting horizon. "
                        "Set freq on the FH or provide a cutoff with "
                        "frequency information."
                    )
                # get cutoff nanos for calendar-aware conversion
                if cutoff is not None:
                    cutoff_val, _, _, _ = PandasFHConverter.cutoff_to_internal(
                        cutoff, freq=self.freq
                    )
                    ref_nanos = cutoff_val
                else:
                    ref_nanos = np.int64(0)
                # convert timedelta nanos to integer steps, then zero-base
                indexer_vals = (
                    PandasFHConverter.nanos_to_steps(
                        relative._values, freq, ref_nanos=ref_nanos
                    )
                    - 1
                )
            else:
                raise TypeError(
                    f"Cannot compute indexer for relative FH with "
                    f"value type {vtype.name}."
                )
        else:
            relative = self.to_relative(cutoff)
            vals = relative._values
            indexer_vals = vals - vals[0]

        return PandasFHConverter.to_pandas_index(
            indexer_vals.astype(np.int64), FHValueType.INT
        )

    def _is_contiguous(self) -> bool:
        """Check if forecasting horizon values form a contiguous sequence.

        Returns
        -------
        bool
        """
        return is_contiguous(self._values, self._value_type)

    def get_expected_pred_idx(self, y=None, cutoff=None, sort_by_time=False):
        """Construct expected prediction output index.

        Parameters
        ----------
        y : pd.DataFrame, pd.Series, pd.Index, or None (default=None)
            Data to compute fh relative to.
        cutoff : pd.Period, pd.Timestamp, int, or pd.Index, optional (default=None)
            Cutoff value. If None, inferred from ``y``.
        sort_by_time : bool, optional (default=False)
            For MultiIndex returns, whether to sort by time index.

        Returns
        -------
        pd.Index
            Expected index of y_pred returned by predict.
        """
        return PandasFHConverter.build_pred_index(
            fh=self,
            y=y,
            cutoff=cutoff,
            sort_by_time=sort_by_time,
        )

    # Dunders -> Arithmetic operators

    def __add__(self, other):
        if isinstance(other, ForecastingHorizon):
            result = self._values + other._values
        else:
            result = self._values + np.int64(other)
        return self.clone(values=result)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, ForecastingHorizon):
            result = self._values - other._values
        else:
            result = self._values - np.int64(other)
        return self.clone(values=result)

    def __rsub__(self, other):
        result = np.int64(other) - self._values
        return self.clone(values=result)

    def __mul__(self, other):
        if isinstance(other, ForecastingHorizon):
            result = self._values * other._values
        else:
            result = self._values * np.int64(other)
        return self.clone(values=result)

    def __rmul__(self, other):
        return self.__mul__(other)

    # Dunders -> comparison operators
    # Note:
    # Current implementation uses element-wise comparison (numpy-style):
    # fh == 3 → array([False, False, True])

    def __eq__(self, other):
        if isinstance(other, ForecastingHorizon):
            return self._values == other._values
        return self._values == np.int64(other)

    def __ne__(self, other):
        if isinstance(other, ForecastingHorizon):
            return self._values != other._values
        return self._values != np.int64(other)

    def __lt__(self, other):
        if isinstance(other, ForecastingHorizon):
            return self._values < other._values
        return self._values < np.int64(other)

    def __le__(self, other):
        if isinstance(other, ForecastingHorizon):
            return self._values <= other._values
        return self._values <= np.int64(other)

    def __gt__(self, other):
        if isinstance(other, ForecastingHorizon):
            return self._values > other._values
        return self._values > np.int64(other)

    def __ge__(self, other):
        if isinstance(other, ForecastingHorizon):
            return self._values >= other._values
        return self._values >= np.int64(other)

    # Dunders -> container methods len, getitem, max, min
    def __len__(self):
        return len(self._values)

    def __getitem__(self, key):
        result = self._values[key]
        if isinstance(result, np.ndarray):
            return self.clone(values=result)
        # scalar — return as-is
        return result

    def max(self):
        """Return the maximum value."""
        return self._values.max() if len(self._values) > 0 else None

    def min(self):
        """Return the minimum value."""
        return self._values.min() if len(self._values) > 0 else None

    def __hash__(self):
        return hash(
            (
                self._values.tobytes(),
                self._value_type,
                self._is_relative,
                self._freq,
                self._timezone,
            )
        )

    def __repr__(self):
        class_name = type(self).__name__
        vtype = self._value_type.name
        n = len(self._values)
        parts = [f"n={n}", f"type={vtype}", f"is_relative={self._is_relative}"]
        if self._freq is not None:
            parts.append(f"freq={self._freq!r}")
        # if less than 6 values, show all values in repr,
        # otherwise show 1st and last 3 only
        if n <= 6:
            parts.append(f"values={self._values.tolist()}")
        else:
            head = self._values[:3].tolist()
            tail = self._values[-3:].tolist()
            parts.append(
                f"values=[{head[0]}, {head[1]}, {head[2]}, ..., "
                f"{tail[0]}, {tail[1]}, {tail[2]}]"
            )
        return f"{class_name}({', '.join(parts)})"
