"""Explicit script entry point for experiment 1."""

import os

from data_io import data_load, data_write


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output")


if __name__ == "__main__":
    field = "close"
    start_date = "2020-01-01"
    end_date = "2020-12-31"

    data = data_load(
        field=field,
        start_date=start_date,
        end_date=end_date,
    )

    print(f"shape: {data.shape}")
    if data.empty:
        print("No observations in the requested date interval.")
    else:
        print(f"date range: {data.index.min()} -- {data.index.max()}")

    output_file = os.path.join(
        OUTPUT_DIR,
        f"{field}_{start_date}_{end_date}.csv",
    )
    saved_file = data_write(data, output_file)
    print(f"saved to: {saved_file}")
