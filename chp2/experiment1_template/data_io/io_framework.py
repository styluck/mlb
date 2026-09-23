"""Data I/O framework for experiment 1.

This template follows the structure and behavior of the original
``codes/data_io/io_framework.py``. Complete the sections marked TODO without
changing the public function names or parameters.
"""

from __future__ import annotations

import os
from typing import Optional

import pandas as pd


MARKETS = ("sh", "sz")


def _data_dir() -> str:
    """Return the fixed directory containing the course CSV files."""
    return "C:/Users/dataset"


def _read_one(field: str, market: Optional[str]) -> Optional[pd.DataFrame]:
    """Read and clean one CSV file.

    ``field="close", market="sh"`` reads ``close_sh.csv``. When ``market``
    is ``None``, the function reads ``<field>.csv``. A missing file returns
    ``None``, matching the original framework.
    """
    if market is None:
        filename = f"{field}.csv"
    else:
        filename = f"{field}_{market}.csv"
    csv_file = os.path.join(_data_dir(), filename)

    if not os.path.exists(csv_file):
        return None

    # TODO 1:
    # Read csv_file and use the first CSV column as the DataFrame index.
    # data = ...

    # TODO 2:
    # Convert the index to datetime with errors="coerce", then remove rows
    # whose dates could not be parsed.

    # TODO 3:
    # Convert all data columns to numeric with errors="coerce".
    # Sort the index and keep the last row for duplicated dates.

    raise NotImplementedError("Complete _read_one().")


def data_load(
    field: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Load one field from Shanghai and Shenzhen market CSV files.

    As in the original framework, columns are temporarily grouped by market.
    When the same asset occurs in both markets, Shanghai values have priority
    and missing observations are filled by Shenzhen values.
    """
    if not isinstance(field, str) or not field.strip():
        raise ValueError("field must be a non-empty string")

    start = pd.to_datetime(start_date, errors="raise")
    end = pd.to_datetime(end_date, errors="raise")
    if start > end:
        raise ValueError("start_date must not exceed end_date")

    parts = []

    # TODO 4:
    # For each market in MARKETS:
    #   1. Call _read_one(field, market).
    #   2. Ignore a missing or empty market file.
    #   3. Add a temporary market level to its columns with
    #      pd.MultiIndex.from_product().
    #   4. Append the result to parts.

    # TODO 5:
    # If parts is empty, raise FileNotFoundError. Mention field and _data_dir()
    # in the error message.

    # TODO 6:
    # Concatenate parts along columns, sort the date index, and select the
    # inclusive interval [start, end]. Store the result in both.

    # TODO 7:
    # Reproduce the original merge rule:
    #   1. Get unique asset names from column level 1.
    #   2. Create an output DataFrame with the same index as both.
    #   3. For every asset, inspect markets in MARKETS order.
    #   4. Use Series.combine_first() so SH has priority and SZ fills NaN.
    #   5. Store the merged Series under the plain asset name.

    # TODO 8:
    # Set the output index name to "date" and return it.

    raise NotImplementedError("Complete data_load().")


def load_benchmark(start_date: str, end_date: str) -> pd.DataFrame:
    """Load ``benchmark.csv`` in the requested closed date interval."""
    benchmark = _read_one("benchmark", None)
    if benchmark is None:
        benchmark_file = os.path.join(_data_dir(), "benchmark.csv")
        raise FileNotFoundError(f"Benchmark file not found: {benchmark_file}")

    start = pd.to_datetime(start_date, errors="raise")
    end = pd.to_datetime(end_date, errors="raise")
    if start > end:
        raise ValueError("start_date must not exceed end_date")
    return benchmark.loc[start:end]


def data_write(data: pd.DataFrame, output_file: str) -> str:
    """Write a DataFrame to the CSV path given by a string."""
    # TODO 9:
    # Check that data is a pandas DataFrame and output_file is a string.

    # TODO 10:
    # Get the output directory from output_file and create it when necessary.
    # Use os.path.dirname(), os.path.abspath(), and os.makedirs().

    # TODO 11:
    # Copy data so that the caller's DataFrame is not modified. Set the copied
    # index name to "date", then write UTF-8 CSV with YYYY-MM-DD dates.

    # TODO 12:
    # Return output_file unchanged.

    raise NotImplementedError("Complete data_write().")
