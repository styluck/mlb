"""Example entry point for experiment 1."""

from pathlib import Path

from data_io import data_load, data_write


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR.parents[1] / "chp3_data" / "sample_code" / "dataset"
OUTPUT_DIR = PROJECT_DIR / "output"


def main() -> None:
    """Load one field, show a summary, and save the selected data."""
    # You may change the field and dates after the template is complete.
    field = "close"
    start_date = "2020-01-01"
    end_date = "2020-12-31"

    data = data_load(
        field=field,
        start_date=start_date,
        end_date=end_date,
        data_dir=DATA_DIR,
    )

    print(f"shape: {data.shape}")
    if data.empty:
        print("No observations in the requested date interval.")
    else:
        print(f"date range: {data.index.min()} -- {data.index.max()}")

    output_file = OUTPUT_DIR / f"{field}_{start_date}_{end_date}.csv"
    saved_path = data_write(data, output_file)
    print(f"saved to: {saved_path}")


if __name__ == "__main__":
    main()
