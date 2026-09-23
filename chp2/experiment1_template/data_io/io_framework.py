"""CSV loading and writing tools for experiment 1.

Complete the sections marked TODO. Keep the public function names and their
parameters unchanged so that the supplied self-check can call them.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


DEFAULT_DATA_DIR = (
    Path(__file__).resolve().parents[3]
    / "chp3_data"
    / "sample_code"
    / "dataset"
)
MARKETS = ("sh", "sz")


def _parse_period(
    start_date: str,
    end_date: str,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Parse and validate the requested closed date interval."""
    start = pd.to_datetime(start_date, errors="raise")
    end = pd.to_datetime(end_date, errors="raise")

    # TODO 1:
    # Raise ValueError if start is later than end.

    return start, end


def _read_market(
    field: str,
    market: str,
    data_dir: str | Path,
) -> pd.DataFrame:
    """Read and clean one market CSV file.

    The expected file name is ``<field>_<market>.csv``. The first CSV column
    stores dates and must become a sorted, duplicate-free DatetimeIndex.
    """
    data_dir = Path(data_dir)
    csv_path = data_dir / f"{field}_{market}.csv"

    # TODO 2:
    # If csv_path does not exist, raise FileNotFoundError and include the path
    # in the error message.

    # TODO 3:
    # Read the CSV with its first column as the index.
    # data = ...

    # TODO 4:
    # Convert the index to datetime with errors="coerce".
    # Remove rows whose dates could not be parsed.

    # TODO 5:
    # Convert every data column to numeric with errors="coerce".
    # Sort the date index and keep the last row for duplicated dates.

    raise NotImplementedError("Complete _read_market().")


def _add_market_prefix_if_needed(
    sh: pd.DataFrame,
    sz: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prefix duplicated asset names with their market names.

    Columns appearing in only one market keep their original names. If a name
    appears in both markets, the two output names become ``sh_<name>`` and
    ``sz_<name>``.
    """
    duplicated_names = set(sh.columns).intersection(sz.columns)
    if not duplicated_names:
        return sh, sz

    sh = sh.rename(
        columns={name: f"sh_{name}" for name in duplicated_names}
    )
    sz = sz.rename(
        columns={name: f"sz_{name}" for name in duplicated_names}
    )
    return sh, sz


def data_load(
    field: str,
    start_date: str,
    end_date: str,
    data_dir: str | Path = DEFAULT_DATA_DIR,
) -> pd.DataFrame:
    """Load Shanghai and Shenzhen data in a closed date interval.

    Parameters
    ----------
    field
        Data field such as ``"close"`` or ``"open"``.
    start_date, end_date
        Inclusive date boundaries accepted by ``pandas.to_datetime``.
    data_dir
        Directory containing ``<field>_sh.csv`` and ``<field>_sz.csv``.

    Returns
    -------
    pandas.DataFrame
        A date-indexed matrix containing columns from both markets.
    """
    if not isinstance(field, str) or not field.strip():
        raise ValueError("field must be a non-empty string")

    start, end = _parse_period(start_date, end_date)

    # TODO 6:
    # Read the two markets by calling _read_market().
    # sh = ...
    # sz = ...

    # Keep this helper call. It prevents ambiguous duplicate column names.
    # sh, sz = _add_market_prefix_if_needed(sh, sz)

    # TODO 7:
    # Concatenate sh and sz along columns, sort the index, remove duplicate
    # dates if any, and select the inclusive interval [start, end].
    # result = ...

    # TODO 8:
    # Set result.index.name to "date" and return result.

    raise NotImplementedError("Complete data_load().")


def data_write(
    data: pd.DataFrame,
    output_file: str | Path,
) -> Path:
    """Write a data matrix to CSV and return the output path."""
    # TODO 9:
    # Raise TypeError if data is not a pandas DataFrame.

    output_path = Path(output_file)

    # TODO 10:
    # Create output_path.parent when it does not exist.

    # TODO 11:
    # Copy data so that this function does not modify the caller's object.
    # Set the copied index name to "date".
    # Write it as UTF-8 CSV with dates formatted as YYYY-MM-DD.

    # TODO 12:
    # Return output_path.

    raise NotImplementedError("Complete data_write().")
