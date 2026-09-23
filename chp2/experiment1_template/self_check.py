"""Small self-check for experiment 1.

The script creates temporary CSV files, so it does not depend on the full
course dataset. Run it from the template directory with ``python self_check.py``.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from data_io import data_load, data_write


def _write_sample_files(data_dir: Path) -> None:
    sh = pd.DataFrame(
        {
            "date": ["2020-01-03", "bad-date", "2020-01-01", "2020-01-03"],
            "shared": [3.0, 99.0, 1.0, 30.0],
            "600000.SH": [13.0, 99.0, 11.0, 130.0],
        }
    )
    sz = pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-01-02", "2020-01-03"],
            "shared": [4.0, 5.0, 6.0],
            "000001.SZ": [21.0, 22.0, 23.0],
        }
    )
    sh.to_csv(data_dir / "close_sh.csv", index=False)
    sz.to_csv(data_dir / "close_sz.csv", index=False)


def run_checks() -> None:
    with TemporaryDirectory() as tmp:
        data_dir = Path(tmp) / "dataset"
        output_dir = Path(tmp) / "nested" / "output"
        data_dir.mkdir()
        _write_sample_files(data_dir)

        result = data_load(
            "close",
            "2020-01-01",
            "2020-01-03",
            data_dir=data_dir,
        )

        assert isinstance(result.index, pd.DatetimeIndex)
        assert result.index.is_monotonic_increasing
        assert result.index.is_unique
        assert result.index.min() == pd.Timestamp("2020-01-01")
        assert result.index.max() == pd.Timestamp("2020-01-03")
        assert list(result.index) == list(pd.date_range("2020-01-01", periods=3))
        assert "sh_shared" in result.columns
        assert "sz_shared" in result.columns
        assert "600000.SH" in result.columns
        assert "000001.SZ" in result.columns
        assert result.loc[pd.Timestamp("2020-01-03"), "600000.SH"] == 130.0

        one_day = data_load(
            "close",
            "2020-01-02",
            "2020-01-02",
            data_dir=data_dir,
        )
        assert list(one_day.index) == [pd.Timestamp("2020-01-02")]

        result_for_write = result.copy()
        result_for_write.index.name = "original_name"
        original_index_name = result_for_write.index.name
        output_file = output_dir / "close.csv"
        returned_path = data_write(result_for_write, output_file)
        assert returned_path == output_file
        assert output_file.exists()
        assert result_for_write.index.name == original_index_name

        try:
            data_load("close", "2020-01-03", "2020-01-01", data_dir=data_dir)
        except ValueError:
            pass
        else:
            raise AssertionError("Reversed dates must raise ValueError.")

        try:
            data_load("open", "2020-01-01", "2020-01-03", data_dir=data_dir)
        except FileNotFoundError as exc:
            assert "open_sh.csv" in str(exc)
        else:
            raise AssertionError("Missing input files must raise FileNotFoundError.")

    print("All self-checks passed.")


if __name__ == "__main__":
    run_checks()
