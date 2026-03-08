# !/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""
ForecastingHorizon: pandas-agnostic forecasting horizon implementation.

Architecture: ForecastingHorizon stores only {_values, _is_relative, _freq,
_values_are_nanos}. All temporal inputs are normalized to integer steps
(period ordinals) at construction.
This means that all internal arithmetic is pure integer math,
and the only place where pandas logic is needed is in the conversion of inputs to
this internal representation (PandasFHConverter).
This design allows ForecastingHorizon to be pandas-free,
while still supporting all the same input types and frequencies as before.
All pandas-specific logic is delegated to the _fh_utils module.

Internal state of ForecastingHorizon consists of the following attributes:

``_values``: int64 numpy array — integer steps (period ordinals for
  absolute, step counts for relative), or raw nanoseconds when
  ``_values_are_nanos`` is True. Read-only after construction.

``_is_relative``: bool — whether values are relative to training cutoff.

``_freq``: str or None — frequency mnemonic (e.g. ``"M"``, ``"D"``).
  None for plain integer horizons or when freq has not yet been assigned.
  The ``freq`` setter is the only deliberate mutation point on the object.

``_values_are_nanos``: bool — True when values are raw nanoseconds
  pending conversion to integer steps (e.g. freq-less TimedeltaIndex input).
  Set to False once freq is assigned via the ``freq`` setter.
"""

__all__ = ["ForecastingHorizon"]

import numpy as np

from sktime.forecasting.base._fh_utils import PandasFHConverter
from sktime.forecasting.base._freq_mnemonic import validate_freq

# types whose is_relative is compatible with both True and False
_RELATIVE_NEUTRAL_TYPES = (int, np.integer, list, range, np.ndarray)


class ForecastingHorizon:
    """Represents the time points to forecast, relative or absolute.

    A forecasting horizon specifies which future (or past) time points a
    forecaster should predict. It accepts a wide range of input types:
    plain integers, pandas PeriodIndex, DatetimeIndex, TimedeltaIndex, etc.
    and normalizes them internally to a sorted, deduplicated int64 numpy
    array of integer steps (period ordinals for absolute, step counts for
    relative). Temporal inputs that cannot be immediately converted to
    integer steps (e.g. freq-less TimedeltaIndex) are stored as raw
    nanoseconds and converted when frequency information becomes available.

    Parameters
    ----------
    values : int, list, np.ndarray, range, pd.Index, pd.Timedelta,
        pd.offsets.BaseOffset
        Values of forecasting horizon.
        Supported types without pandas dependency:
        - ``int`` or ``np.integer`` : single integer step
        - ``list[int]`` : list of integer steps
        - ``np.ndarray`` : integer or timedelta64 array
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
        if absolute is intended, as the default inference interprets it as
        relative.
    freq : str, pd.Index, pd.Period, pandas offset, or sktime forecaster,
        optional (default=None)
        Frequency information for the horizon values.
        When values already carry frequency (e.g. ``pd.PeriodIndex``,
        ``pd.DatetimeIndex`` with freq, or ``pd.TimedeltaIndex`` with freq),
        provided ``freq`` must match the values' frequency, otherwise a
        ValueError is raised.
        When values do not carry frequency (e.g. int, list, np.ndarray,
        or freq-less ``pd.TimedeltaIndex``), ``freq`` is used directly if
        provided. For freq-less ``pd.TimedeltaIndex``, values are stored as
        raw nanoseconds until freq is assigned (via this parameter or later
        through the ``freq`` setter).

    Examples
    --------
    >>> from sktime.forecasting.base._fh_v2 import ForecastingHorizon
    >>> fh = ForecastingHorizon([1, 2, 3])
    >>> fh.is_relative
    True
    >>> fh.to_numpy()
    array([1, 2, 3])
    """

    def __init__(
        self,
        values=None,
        is_relative: bool | None = None,
        freq=None,
    ):
        # convert values to internal representation
        # both paths return (values, is_relative, freq, values_are_nanos)
        if isinstance(values, (int, np.integer, range, np.ndarray)) or (
            isinstance(values, list)
            and len(values) > 0
            and isinstance(values[0], (int, np.integer))
        ):
            vals, inferred_is_relative, freq_val, nanos_flag = self._coerce_canonical(
                values
            )
        else:
            vals, inferred_is_relative, freq_val, nanos_flag = (
                PandasFHConverter.to_internal(values)
            )

        # sort, deduplicate, and store
        self._values = np.unique(vals)
        self._freq = freq_val
        self._values_are_nanos = nanos_flag

        # handle empty arrays
        if len(self._values) == 0:
            self._freq = None
            self._values_are_nanos = False

        # set freq via setter (single gate for validation)
        if freq is not None:
            self.freq = freq

        self._is_relative = self._resolve_is_relative(
            is_relative, inferred_is_relative, values
        )

        # lock values array against accidental mutation
        self._values.flags.writeable = False

    @staticmethod
    def _coerce_canonical(values):
        """Coerce canonical (non-pandas) values to internal representation.

        Handles int, np.integer, range, np.ndarray, and list[int].

        Parameters
        ----------
        values : int, np.integer, range, np.ndarray, or list[int]
            Input values.

        Returns
        -------
        tuple of (np.ndarray, bool, str or None, bool)
            (values_array, inferred_is_relative, freq, values_are_nanos)
            Same order as PandasFHConverter.to_internal.
        """
        inferred_is_relative = True
        freq = None
        values_are_nanos = False

        if isinstance(values, (int, np.integer)):
            arr = np.array([int(values)], dtype=np.int64)
            return arr, inferred_is_relative, freq, values_are_nanos

        if isinstance(values, range):
            arr = np.array(list(values), dtype=np.int64)
            return arr, inferred_is_relative, freq, values_are_nanos

        if isinstance(values, np.ndarray):
            if values.ndim != 1:
                raise ValueError(f"Expected 1-D array, got {values.ndim}-D array")
            if len(values) == 0:
                raise ValueError("Forecasting horizon values must not be empty.")
            if np.issubdtype(values.dtype, np.timedelta64):
                arr = values.astype("timedelta64[ns]").view(np.int64).copy()
                values_are_nanos = True
                return arr, inferred_is_relative, freq, values_are_nanos
            if np.issubdtype(values.dtype, np.integer):
                arr = values.astype(np.int64).copy()
                return arr, inferred_is_relative, freq, values_are_nanos
            raise TypeError(
                f"np.ndarray with dtype {values.dtype} is not supported. "
                f"Expected integer or timedelta64 dtype."
            )

        # list[int]
        for i, v in enumerate(values[1:], start=1):
            if not isinstance(v, (int, np.integer)):
                raise TypeError(
                    f"Element at index 0 is of type "
                    f"{type(values[0]).__name__}, but element at "
                    f"index {i} is {type(v).__name__}. "
                    "All list elements must be of the same type."
                )
        arr = np.array(values, dtype=np.int64)
        return arr, inferred_is_relative, freq, values_are_nanos

    @staticmethod
    def _resolve_is_relative(is_relative, inferred_is_relative, values):
        """Resolve is_relative from user-provided and inferred values.

        Parameters
        ----------
        is_relative : bool or None
            User-provided is_relative flag.
        inferred_is_relative : bool
            is_relative inferred from the type of values.
        values : object
            Original values passed to ForecastingHorizon.__init__.

        Returns
        -------
        bool
            Resolved is_relative value.

        Raises
        ------
        TypeError
            If is_relative is not a boolean or None.
        ValueError
            If is_relative conflicts with the inferred value and the input
            type strictly implies one interpretation (e.g. PeriodIndex is
            always absolute, TimedeltaIndex is always relative).
        """
        if is_relative is None:
            return inferred_is_relative

        if not isinstance(is_relative, bool):
            raise TypeError("`is_relative` must be a boolean or None")

        if inferred_is_relative is not None and is_relative != inferred_is_relative:
            if not isinstance(values, _RELATIVE_NEUTRAL_TYPES):
                raise ValueError(
                    f"Conflict between inferred "
                    f"is_relative={inferred_is_relative} "
                    f"and provided is_relative={is_relative}. Please resolve "
                    "the conflict by providing a consistent `is_relative` "
                    "value or adjusting the input `values`."
                )

        return is_relative

    @classmethod
    def _create(cls, values, is_relative, freq=None, values_are_nanos=False):
        """Construct a ForecastingHorizon without coercion or validation.

        Fast-path constructor for internal use. Creates a new instance
        directly from pre-computed attributes, bypassing ``__init__``.
        Values must already be sorted and deduplicated.

        Parameters
        ----------
        values : np.ndarray
            Sorted, deduplicated int64 array of values.
        is_relative : bool
            Whether the horizon is relative.
        freq : str or None, optional
            Frequency string.
        values_are_nanos : bool, optional (default=False)
            Whether values are raw nanoseconds.

        Returns
        -------
        ForecastingHorizon
            New instance.
        """
        obj = object.__new__(cls)
        assert len(values) == 0 or np.all(np.diff(values) > 0), (
            "_create expects sorted, unique values"
        )
        obj._values = values
        obj._values.flags.writeable = False
        obj._is_relative = is_relative
        obj._freq = freq
        obj._values_are_nanos = values_are_nanos
        return obj

    @property
    def is_relative(self) -> bool:
        """Whether forecasting horizon is relative to the end of training.

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

        If the FH has _values_are_nanos=True (freq-less TimedeltaIndex),
        setting freq triggers conversion of nanosecond values to integer
        steps.

        If the FH already carries a frequency, the new frequency must match,
        otherwise a ValueError is raised.

        Parameters
        ----------
        obj : str, pd.Index, pd.offsets.BaseOffset, or forecaster
            Object carrying frequency information.

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

        if new_freq is None:
            return

        old_freq = self._freq

        # if values are nanos, convert to steps using the new freq
        if self._values_are_nanos:
            new_values = PandasFHConverter.nanos_to_steps(self._values, new_freq)
            new_values.flags.writeable = False
            self._values = new_values
            self._values_are_nanos = False
            self._freq = new_freq
            return

        # normal path: first assignment or confirmation
        if old_freq is not None and old_freq != new_freq:
            raise ValueError(
                f"Frequencies do not match: current={old_freq!r}, new={new_freq!r}"
            )
        self._freq = new_freq

    # ---- core conversion methods ----

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
            return self._create(
                self._values.copy(),
                self._is_relative,
                self._freq,
                self._values_are_nanos,
            )

        if cutoff is None:
            raise ValueError(
                "`cutoff` must be provided to convert absolute FH to relative."
            )

        if self._values_are_nanos:
            raise ValueError(
                "Cannot convert to relative: values are raw nanoseconds "
                "pending freq assignment. Set freq first."
            )

        cutoff_step = PandasFHConverter.cutoff_to_steps(cutoff, freq=self._freq)
        relative_vals = self._values - cutoff_step

        return self._create(
            values=relative_vals.astype(np.int64),
            is_relative=True,
            freq=self._freq,
        )

    def to_absolute(self, cutoff):
        """Return absolute version of forecasting horizon.

        Parameters
        ----------
        cutoff : pd.Period, pd.Timestamp, int, or pd.Index
            Cutoff value is required to convert a relative forecasting
            horizon to an absolute one.

        Returns
        -------
        ForecastingHorizon
            Absolute representation of forecasting horizon.
        """
        if not self._is_relative:
            return self._create(
                self._values.copy(),
                self._is_relative,
                self._freq,
                self._values_are_nanos,
            )

        if self._values_are_nanos:
            # attempt to extract freq from cutoff for deferred conversion
            cutoff_freq = PandasFHConverter.extract_freq(cutoff)
            if cutoff_freq is not None:
                values = PandasFHConverter.nanos_to_steps(self._values, cutoff_freq)
                freq = cutoff_freq
            else:
                raise ValueError(
                    "Cannot convert to absolute: values are raw nanoseconds "
                    "and no freq is available. Set freq on the FH or provide "
                    "a cutoff with frequency information."
                )
        else:
            values = self._values
            freq = self._freq

        cutoff_step = PandasFHConverter.cutoff_to_steps(cutoff, freq=freq)
        absolute_vals = cutoff_step + values

        return self._create(
            values=absolute_vals.astype(np.int64),
            is_relative=False,
            freq=freq,
        )

    def to_pandas(self):
        """Return forecasting horizon values as pd.Index.

        Output type depends on state:
        - values_are_nanos=True: TimedeltaIndex
        - is_relative=False and freq is not None: PeriodIndex
        - otherwise: plain integer Index

        Returns
        -------
        pd.Index
            Pandas Index containing the forecasting horizon values.
        """
        return PandasFHConverter.to_pandas_index(
            self._values, self._is_relative, self._freq, self._values_are_nanos
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

        Output type is cutoff-driven:
        - DatetimeIndex cutoff -> DatetimeIndex output (with tz from cutoff)
        - PeriodIndex/int cutoff -> PeriodIndex output (via to_pandas())

        Parameters
        ----------
        cutoff : pd.Period, pd.Timestamp, int, or pd.Index, optional
            Cutoff value for conversion.

        Returns
        -------
        pd.Index
            Absolute forecasting horizon as pandas Index.
        """
        abs_fh = self.to_absolute(cutoff)

        # if cutoff is DatetimeIndex, produce DatetimeIndex output
        if cutoff is not None and PandasFHConverter.cutoff_is_datetime_index(cutoff):
            tz = PandasFHConverter.cutoff_tz(cutoff)
            return PandasFHConverter.steps_to_datetime(
                abs_fh._values, abs_fh._freq, tz=tz
            )

        return abs_fh.to_pandas()

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
        absolute = self.to_absolute(cutoff)
        start_step = PandasFHConverter.cutoff_to_steps(start, freq=self._freq)
        integers = absolute._values - start_step

        return self._create(
            values=integers.astype(np.int64),
            is_relative=False,
            freq=self._freq,
        )

    # ---- in-sample and out-of-sample methods ----

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
        return self._create(
            self._values[mask],
            self._is_relative,
            self._freq,
            self._values_are_nanos,
        )

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
        return self._create(
            self._values[mask],
            self._is_relative,
            self._freq,
            self._values_are_nanos,
        )

    # ---- indexer method ----

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
            indexer_vals = relative._values - 1
        else:
            relative = self.to_relative(cutoff)
            vals = relative._values
            indexer_vals = vals - vals[0]

        return PandasFHConverter.to_pandas_index(
            indexer_vals.astype(np.int64), is_relative=True
        )

    def _is_contiguous(self) -> bool:
        """Check if forecasting horizon values form a contiguous sequence.

        Returns
        -------
        bool
        """
        if len(self._values) <= 1:
            return True
        if self._values_are_nanos:
            # for nanos, check uniform spacing
            diffs = np.diff(self._values)
            return bool(np.all(diffs == diffs[0]))
        # integer steps: contiguous means every int between min and max present
        expected_len = int(self._values[-1] - self._values[0]) + 1
        return len(self._values) == expected_len

    def get_expected_pred_idx(self, y=None, cutoff=None, sort_by_time=False):
        """Construct expected prediction output index.

        Parameters
        ----------
        y : pd.DataFrame, pd.Series, pd.Index, or None (default=None)
            Data to compute fh relative to.
        cutoff : pd.Period, pd.Timestamp, int, or pd.Index, optional
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

    # ---- Dunders: arithmetic operators (scalar only) ----

    @staticmethod
    def _check_scalar(other):
        if isinstance(other, ForecastingHorizon):
            raise TypeError(
                "Arithmetic between two ForecastingHorizon objects is not "
                "supported. Use scalar operands (int, np.integer)."
            )
        return np.int64(other)

    def __add__(self, other):
        scalar = self._check_scalar(other)
        result = self._values + scalar
        return self._create(
            result, self._is_relative, self._freq, self._values_are_nanos
        )

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        scalar = self._check_scalar(other)
        result = self._values - scalar
        return self._create(
            result, self._is_relative, self._freq, self._values_are_nanos
        )

    def __rsub__(self, other):
        scalar = np.int64(other)
        # reverses order, so re-sort via np.unique
        result = np.unique(scalar - self._values)
        return self._create(
            result, self._is_relative, self._freq, self._values_are_nanos
        )

    def __mul__(self, other):
        scalar = self._check_scalar(other)
        result = self._values * scalar
        # negative scalar reverses order
        if scalar < 0:
            result = np.unique(result)
        return self._create(
            result, self._is_relative, self._freq, self._values_are_nanos
        )

    def __rmul__(self, other):
        return self.__mul__(other)

    # ---- Dunders: comparison operators ----
    # __eq__ / __ne__: FH-to-FH returns single bool (whole-object equality),
    #                   scalar returns element-wise boolean array.
    # __lt__ / __le__ / __gt__ / __ge__: scalar only.

    def __eq__(self, other):
        if isinstance(other, ForecastingHorizon):
            return (
                np.array_equal(self._values, other._values)
                and self._is_relative == other._is_relative
                and self._freq == other._freq
                and self._values_are_nanos == other._values_are_nanos
            )
        return self._values == np.int64(other)

    def __ne__(self, other):
        if isinstance(other, ForecastingHorizon):
            return not self.__eq__(other)
        return self._values != np.int64(other)

    def __lt__(self, other):
        self._check_scalar(other)
        return self._values < np.int64(other)

    def __le__(self, other):
        self._check_scalar(other)
        return self._values <= np.int64(other)

    def __gt__(self, other):
        self._check_scalar(other)
        return self._values > np.int64(other)

    def __ge__(self, other):
        self._check_scalar(other)
        return self._values >= np.int64(other)

    # ---- Dunders: container methods ----

    def __len__(self):
        return len(self._values)

    def __getitem__(self, key):
        result = self._values[key]
        if isinstance(result, np.ndarray):
            return self._create(
                result, self._is_relative, self._freq, self._values_are_nanos
            )
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
                self._is_relative,
                self._freq,
                self._values_are_nanos,
            )
        )

    def __repr__(self):
        class_name = type(self).__name__
        n = len(self._values)
        parts = [f"n={n}", f"is_relative={self._is_relative}"]
        if self._freq is not None:
            parts.append(f"freq={self._freq!r}")
        if self._values_are_nanos:
            parts.append("values_are_nanos=True")
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
