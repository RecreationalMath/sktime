# FH Migration: `_fh.py` → `_fh_v2.py` — Change Log

**PR**: #9365 | **Branch**: `fh_rem_fhvalues`
**Base commit**: `66dca4e40` ("Rerouted init to use new fh")

This document lists every file changed after the base commit, with line-level
details and a brief note on *why* each change was made.

---

## Core FH Implementation

### `sktime/forecasting/base/_fh_v2.py`

| Lines | Change | Why |
|-------|--------|-----|
| 33 | Added `VALID_FORECASTING_HORIZON_TYPES` to `__all__` | Exported for use by `split/base/_common.py` |
| 36–39 | Import `_PANDAS_FH_INPUT_TYPES` from `_fh_utils` | Needed to build `VALID_FORECASTING_HORIZON_TYPES` without importing pandas directly (keeps `_fh_v2.py` pandas-free) |
| 41 | `VALID_FORECASTING_HORIZON_TYPES = int \| list \| np.ndarray \| _PANDAS_FH_INPUT_TYPES` | Replaces the old `VALID_FORECASTING_HORIZON_TYPES` from `_fh.py`; used as a type alias in the split module |
| 144–170 | Canonical path guard: exclude `np.timedelta64` and handle empty lists | `np.timedelta64` is a subclass of `np.integer` — without exclusion it enters the int path instead of the converter path. Empty lists `[]` also need special handling to avoid downstream errors |
| 162 | `PandasFHConverter.to_internal(values, freq=freq)` | Pass the user-supplied `freq` as fallback so freq-less `DatetimeIndex` can still be constructed when `freq=` is given explicitly |
| 176–196 | Freq init: two setter calls instead of explicit comparison | Old code compared `freq_val != freq` which broke for non-string freq types (pd.Index, pd.Period). The setter already normalizes and checks mismatches |
| 211–216 | Negative integer FH: `ForecastingHorizon(-3)` → `[-3]` not `arange(1,-2)` | `np.arange(1, n+1)` produces an empty array for negative `n`, causing "fh must not be empty" error |
| 238–241 | Early return for empty list input | `[]` went through the converter which rejected it; now returns `np.array([], dtype=np.int64)` directly |
| 288–300 | `_resolve_is_relative`: accept FH copy-constructor and int-dtype `pd.Index` | FH copy-constructor passes an FH instance (not in `_RELATIVE_NEUTRAL_TYPES`); `pd.RangeIndex` has integer dtype but isn't in neutral types either |
| 527–528 | `to_absolute`: extract freq from cutoff when `_freq` is None | Plain integer FH with `_freq=None` failed in `to_absolute` when cutoff carried freq info |

### `sktime/forecasting/base/_fh_utils.py`

| Lines | Change | Why |
|-------|--------|-----|
| 17 | Added `_PANDAS_FH_INPUT_TYPES` to `___all__` | Exported for `_fh_v2.py` to build `VALID_FORECASTING_HORIZON_TYPES` |
| 24 | `_PANDAS_FH_INPUT_TYPES = pd.Index` | Module-level constant; keeps the pandas import inside `_fh_utils.py` so `_fh_v2.py` stays pandas-free |
| 204 | `to_internal(values, freq=None)` — added `freq` parameter | Allows fallback freq for `DatetimeIndex` without `.freq` attribute |
| 225–227 | Docstring update for new `freq` parameter | Documents the new fallback behavior |
| 239 | Error message update: "and no fallback" | Reflects that the error only fires when both index freq and fallback are None |
| 275–276 | `if freq_str is None and freq is not None: freq_str = extract_freq(freq)` | Uses the caller-supplied `freq` when `DatetimeIndex.freq` is None |
| 971–996 | Added `_check_cutoff()` and `_index_range()` standalone functions | Moved from `_fh.py` — used by `_reduce.py` for converting relative integer indices to absolute dates using pandas period arithmetic |

### `sktime/forecasting/base/_freq_mnemonic.py`

| Lines | Change | Why |
|-------|--------|-----|
| 67 | `{"Y", "YE", "A", "AE"}` → `{"Y", "YE", "A", "AE", "YS", "AS"}` | Start-anchored offsets (`YS`, `AS`) map to same period freq as end-anchored (`Y`); without this, `to_period("YS")` fails |
| 68 | `{"Q", "QE"}` → `{"Q", "QE", "QS"}` | Same: start-anchored quarterly |
| 69 | `{"M", "ME"}` → `{"M", "ME", "MS"}` | Same: start-anchored monthly |
| 70–75 | Added `{"h","H"}`, `{"min","T"}`, `{"s","S"}`, `{"ms","L"}`, `{"us","U"}`, `{"ns","N"}` | Deprecated pandas 2.2+ aliases ("H","T","S","L","U","N") need to map to canonical ("h","min","s","ms","us","ns"); without these, `normalize_freq("H")` returned "H" instead of "h" |
| 76–78 | Added `{"SM","SME","SMS"}`, `{"BQ","BQE","BQS"}`, `{"BY","BYE","BYS"}` | Start-anchored variants of semi-monthly, business quarterly, business yearly |

---

## Import Changes (Phase 3)

All changes below are `from sktime.forecasting.base._fh import X` →
`from sktime.forecasting.base._fh_v2 import X` (or `_fh_utils` where noted).

| File | Line | Old Import | New Import | Why |
|------|------|-----------|------------|-----|
| `forecasting/base/_base.py` | 58 | `_fh.ForecastingHorizon` | `_fh_v2.ForecastingHorizon` | Switch to new FH |
| `forecasting/compose/_pipeline.py` | 13 | `_fh.ForecastingHorizon` | `_fh_v2.ForecastingHorizon` | Switch to new FH |
| `forecasting/compose/_reduce.py` | 38 | `_fh._index_range` | `_fh_utils._index_range` | `_index_range` moved to `_fh_utils.py` (pandas-specific util) |
| `transformations/series/detrend/_detrend.py` | 11 | `_fh.ForecastingHorizon` | `_fh_v2.ForecastingHorizon` | Switch to new FH |
| `split/base/_common.py` | 12 | `_fh.VALID_FORECASTING_HORIZON_TYPES` | `_fh_v2.VALID_FORECASTING_HORIZON_TYPES` | Switch to new FH |
| `forecasting/tests/test_all_forecasters.py` | 17 | `_fh.ForecastingHorizon` | `_fh_v2.ForecastingHorizon` | Switch to new FH |
| `forecasting/tests/test_mapa.py` | 5 | `_fh.ForecastingHorizon` | `_fh_v2.ForecastingHorizon` | Switch to new FH |
| `forecasting/tests/test_pytorchforecasting.py` | 10 | `_fh.ForecastingHorizon` | `_fh_v2.ForecastingHorizon` | Switch to new FH |

---

## `._values` Access Pattern Updates (Phase 4)

The old FH's `._values` was a `pd.Index` (PeriodIndex, DatetimeIndex, etc.).
The new FH's `._values` is an `np.ndarray` of int64 period ordinals. Files
accessing `._values` as a pandas Index need targeted fixes.

### Category B: `._values` → `.to_absolute_index(cutoff)`

These files used `fh.to_absolute(cutoff)._values` to get a pandas Index for
DataFrame construction or `.loc[]` indexing. The new FH's `._values` returns
raw ordinals, not meaningful dates. `to_absolute_index(cutoff)` is the correct
public API — it returns a proper `pd.PeriodIndex` or `pd.DatetimeIndex`.

| File | Lines | Old Pattern | New Pattern |
|------|-------|-------------|-------------|
| `forecasting/toto.py` | 299, 359 | `fh.to_absolute(self._cutoff)._values` | `fh.to_absolute_index(self._cutoff)` |
| `forecasting/chronos.py` | 596–599 | `ForecastingHorizon(...).to_absolute(...)._values` | `ForecastingHorizon(...).to_absolute_index(...)` |
| `forecasting/hf_transformers_forecaster.py` | 411, 417 | `...to_absolute(...)._values` and `.loc[...._values]` | `...to_absolute_index(...)` |
| `forecasting/hf_momentfm_forecaster.py` | 454, 466 | `...._values.tolist()` and `...._values` | `...to_absolute_index(...).tolist()` and `...to_absolute_index(...)` |
| `forecasting/patch_tst.py` | 484, 496 | Same pattern as momentfm | Same fix |
| `forecasting/ttm.py` | 621, 633 | Same pattern as momentfm | Same fix |
| `forecasting/timesfm_forecaster.py` | 342, 360 | `...._values.tolist()` and `...._values` | `...to_absolute_index(...).tolist()` and `...to_absolute_index(...)` |

### Category C: `._values.values` → `._values` (double `.values`)

Old FH: `._values` (pd.Index) `.values` (numpy). New FH: `._values` is already numpy.

| File | Lines | Old | New |
|------|-------|-----|-----|
| `forecasting/timesfm_forecaster.py` | 260, 263, 313 | `fh._values.values` | `fh._values` |
| `forecasting/base/adapters/_pytorch.py` | 140 | `fh._values.values - 1` | `fh._values - 1` |

### Category D: Type checks on `._values` → freq/is_relative checks

| File | Lines | Old | New | Why |
|------|-------|-----|-----|-----|
| `forecasting/autots.py` | 645–653 | `isinstance(self._fh._values, (pd.Period, pd.PeriodIndex))` and `isinstance(..., pd.DatetimeIndex)` | `self._fh._freq is not None` | New FH `._values` is always `np.ndarray`. Both Period and DateTime branches collapse because new FH normalizes both to period ordinals |

---

## `.to_absolute(cutoff).to_pandas()` → `.to_absolute_index(cutoff)`

These files used `fh.to_absolute(cutoff).to_pandas()` to get a pandas Index.
In the new FH, `to_pandas()` returns a `PeriodIndex` for absolute FH, which
may not match the cutoff's type (e.g., DatetimeIndex cutoff).
`to_absolute_index(cutoff)` returns an Index matching the cutoff type.

| File | Line | Old | New |
|------|------|-----|-----|
| `forecasting/naive/_naive.py` | 372 | `fh.to_absolute(cutoff).to_pandas()` | `fh.to_absolute_index(cutoff)` |
| `forecasting/base/adapters/_pmdarima.py` | 93 | `fh.to_absolute(self.cutoff).to_pandas()` | `fh.to_absolute_index(self.cutoff)` |
| `forecasting/time_llm.py` | 237 | `fh.to_absolute(self.cutoff).to_pandas()` | `fh.to_absolute_index(self.cutoff)` |
| `forecasting/mapa.py` | 559 | `fh.to_absolute(self.cutoff).to_pandas()` | `fh.to_absolute_index(self.cutoff)` |
| `forecasting/tirex.py` | 242 | `fh.to_absolute(self.cutoff).to_pandas()` | `fh.to_absolute_index(self.cutoff)` |

---

## Statsmodels Adapter Fix

### `sktime/forecasting/base/adapters/_statsmodels.py`

| Lines | Old | New | Why |
|-------|-----|-----|-----|
| 116–118 | `start, end = fh.to_absolute_int(...)[[0, -1]]` | `abs_int = fh.to_absolute_int(...)._values; start, end = abs_int[[0, -1]]` | New FH is not a `pd.Index` subclass; `[[0, -1]]` indexing on an FH object doesn't return scalar ints. Must extract the numpy array first |
| 117 | `fh_int = fh.to_absolute_int(...) - self._y_len` | `fh_int = abs_int - self._y_len` | Same: arithmetic on numpy array, not FH object |
| 210–213 | Same pattern in `_predict_interval` | Same fix | Same reason |

---

## Split Module Fixes

### `sktime/split/fh.py`

| Lines | Change | Why |
|-------|--------|-----|
| 61–65 | Added type alignment: convert `PeriodIndex` ↔ `DatetimeIndex` between `idx` and `y` | `to_pandas()` now returns `PeriodIndex` for absolute FH. When `y` is a `DatetimeIndex`, `y < min_step` fails comparing `DatetimeIndex` with `Period`. Convert idx to match y's type |

### `sktime/split/base/_common.py`

| Lines | Change | Why |
|-------|--------|-----|
| 12 | Import from `_fh_v2` instead of `_fh` | Switch to new FH |
| 103–109 | `fh_pd = fh.to_pandas()` then use `fh_pd` for `array_is_int`, `fh_pd[-1]` | Raw `fh[-1]` returns nanosecond integers for timedelta-backed FH (`_values_are_nanos=True`), causing `IndexError` with huge values. `to_pandas()` converts to proper int steps or Timedelta objects |

### `sktime/split/base/_base_windowsplitter.py`

| Lines | Change | Why |
|-------|--------|-----|
| 60–61 | `fh_pd = fh.to_pandas(); fh_max = fh_pd[-1]` | Same nanos issue: `fh[-1]` returns raw nanoseconds for timedelta FH |
| 238–243 | `fh_pd = fh.to_pandas()` in `_split_windows_generic` | `array_is_int(fh)` and `fh.to_numpy()` return nanosecond values for nanos-backed FH; use `fh_pd` for correct user-facing values |
| 289–290 | `fh_pd = fh.to_pandas(); fh_min = abs(fh_pd[0])` in `_get_start` | Same nanos issue |

---

## Test File Changes

### `sktime/forecasting/base/tests/test_fh.py`

| Lines | Change | Why (design change reflected) |
|-------|--------|------|
| 18–21 | Replaced imports: `DELEGATED_METHODS, _check_freq, _extract_freq_from_cutoff` → `PandasFHConverter` | Old symbols don't exist in new FH. `PandasFHConverter.extract_freq` replaces `_check_freq` and `_extract_freq_from_cutoff` |
| 159–168 | `test_fh`: timedelta FH → expect int `to_pandas()`, datetime FH → expect `PeriodIndex` | New FH normalizes timedelta to int steps and datetime to PeriodIndex internally |
| 173–193 | `test_fh`: build `fh_relative` as int index for timedelta case, `fh_indexer` always works | New FH stores timedelta as int steps, so `to_indexer()` always works (no longer raises `NotImplementedError`) |
| 189 | `null = 0` (was conditional on dtype) | Relative values are always ints in new FH |
| 207–212 | `to_indexer` always works, removed `NotImplementedError` branch | New FH normalizes timedelta to ints, so indexer is always computable |
| 232–442 | `test_fh_method_delegation`: check dunder methods directly | Old test checked `DELEGATED_METHODS` (pd.Index delegation). New FH implements these natively, not via delegation |
| 283–285 | `test_check_fh_values_duplicate_input_values`: expect dedup, not error | New FH deduplicates instead of raising ValueError (design choice: more permissive) |
| 427–479 | `_get_expected_freqstr`: use `PandasFHConverter.normalize_freq` | New FH uses canonical freq strings aligned with Period-context APIs. No longer depends on pandas version |
| 447, 465, 488 | `._values.freqstr` → `.to_pandas().freqstr` | `._values` is now `np.ndarray` (no `.freqstr`); `to_pandas()` returns `PeriodIndex` with `.freqstr` |
| 514–516 | `test_to_absolute_with_multiple_freq`: use `pd.testing.assert_index_equal` | `to_numpy()` returns ordinals in new FH; compare as PeriodIndex instead |
| 525–526 | `test_estimator_fh`: `to_absolute_index` instead of `to_absolute().to_numpy()` | `to_numpy()` returns ordinals; `to_absolute_index` returns proper pandas Index |
| 536–541 | `test_error_with_incorrect_string_frequency`: broader match `"Invalid frequency"` | New FH error message format differs slightly from old |
| 854–574 | `test_extract_freq_*`: `_check_freq` → `PandasFHConverter.extract_freq`, compare against normalized freq | Old private functions replaced by `PandasFHConverter` methods. New FH normalizes freq (e.g., "H" → "h") |
| 908–948 | Range tests: `(fh == other).all()` → `fh == other` | New FH `__eq__` returns a single bool (not element-wise pd.Index comparison) |
| 1011–1013 | `test_tz_preserved`: use `to_absolute_index` | `to_absolute` returns FH object; tz info only on the pandas Index from `to_absolute_index` |
| 1080–1082 | `test_timestamp_format_to_absolute`: check `isinstance(DatetimeIndex)` and length | Old test checked for "12:00:00" in string repr; new FH normalizes to period-based index, so check type and length instead |

### `sktime/split/tests/test_temporaltraintest.py`

| Line | Change | Why |
|------|--------|-----|
| 34 | `fh.to_absolute(cutoff).to_numpy()` → `fh.to_absolute_index(cutoff)` | `to_numpy()` returns period ordinals in new FH; `to_absolute_index` returns proper pandas Index comparable with `test.index` |

### `sktime/utils/_testing/forecasting.py`

| Lines | Change | Why |
|-------|--------|-----|
| 172 | Added `fh_freq = None` default | New variable for passing freq to FH constructor |
| 181 | `fh_freq = cutoff_freq` for datetime FH type | DatetimeIndex without freq raises in new FH; pass freq explicitly via constructor |
| 186–188 | `ForecastingHorizon(fh_class(values, **kwargs), is_relative, freq=fh_freq)` | Passes the freq fallback so DatetimeIndex-based FH can be constructed |

---

## Files Summary

31 files changed, 242 insertions, 160 deletions.

| Category | Count | Files |
|----------|-------|-------|
| Core FH implementation | 3 | `_fh_v2.py`, `_fh_utils.py`, `_freq_mnemonic.py` |
| Import switches | 8 | `_base.py`, `_pipeline.py`, `_reduce.py`, `_detrend.py`, `_common.py`, `test_all_forecasters.py`, `test_mapa.py`, `test_pytorchforecasting.py` |
| `._values` → `to_absolute_index` | 7 | `toto.py`, `chronos.py`, `hf_transformers_forecaster.py`, `hf_momentfm_forecaster.py`, `patch_tst.py`, `ttm.py`, `timesfm_forecaster.py` |
| `.to_absolute().to_pandas()` → `.to_absolute_index()` | 5 | `_naive.py`, `_pmdarima.py`, `time_llm.py`, `mapa.py`, `tirex.py` |
| `._values.values` → `._values` | 2 | `timesfm_forecaster.py`, `_pytorch.py` |
| Type checks on `._values` | 1 | `autots.py` |
| Adapter fixes (non-pd.Index FH) | 1 | `_statsmodels.py` |
| Split module (nanos/type alignment) | 3 | `fh.py`, `_common.py`, `_base_windowsplitter.py` |
| Test updates | 3 | `test_fh.py`, `test_temporaltraintest.py`, `_testing/forecasting.py` |
