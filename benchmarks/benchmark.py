import timeit
import unicodedata
from pathlib import Path
from typing import Callable

import polars as pl
import polars_ds as pds

BASE_PATH = Path(__file__).resolve().parents[0]

TIMING_RUNS = 10


class Bench:
    def __init__(
        self,
        df: pl.DataFrame,
        sizes: list[int] = [100, 1_000, 10_000, 100_000, 1_000_000],
        timing_runs: int = 10,
    ):
        self.benchmark_data = {"Function": [], "Size": [], "Time": []}
        self.df = df
        self.sizes = sizes
        self.timing_runs = timing_runs

    def run(self, funcs: list[Callable]):
        for n_rows in self.sizes:
            df = self.df.sample(n_rows, seed=208)

            for func in funcs:
                func_name = func.__name__
                time = timeit.timeit(lambda: func(df), number=self.timing_runs)

                self.benchmark_data["Function"].append(func_name)
                self.benchmark_data["Size"].append(n_rows)
                self.benchmark_data["Time"].append(time)

        return self

    def save(self, file: Path):
        pl.DataFrame(self.benchmark_data).write_parquet(file)


def python_remove_non_ascii(df: pl.DataFrame):
    df.select(
        pl.col("RANDOM_STRING").map_elements(
            lambda s: s.encode("ascii", errors="ignore").decode(),
            return_dtype=pl.String,
        )
    )


def regex_remove_non_ascii(df: pl.DataFrame):
    df.select(pl.col("RANDOM_STRING").str.replace_all(r"[^\p{Ascii}]", ""))


def pds_remove_non_ascii(df: pl.DataFrame):
    df.select(pds.replace_non_ascii("RANDOM_STRING"))


def python_remove_diacritics(df: pl.DataFrame):
    df.select(
        pl.col("RANDOM_STRING").map_elements(
            lambda s: unicodedata.normalize("NFD", s).encode("ASCII", "ignore"),
            return_dtype=pl.String,
        )
    )


def pds_remove_diacritics(df: pl.DataFrame):
    df.select(pds.remove_diacritics("RANDOM_STRING"))


def python_normalize_string(df: pl.DataFrame):
    df.select(
        pl.col("RANDOM_STRING").map_elements(
            lambda s: unicodedata.normalize("NFD", s), return_dtype=pl.String
        )
    )


def pds_normalize_string(df: pl.DataFrame):
    df.select(pds.normalize_string("RANDOM_STRING", "NFD"))


def main():
    benchmark_df = pl.read_parquet(BASE_PATH / "benchmark_df.parquet")

    Bench(benchmark_df).run(
        [
            python_remove_non_ascii,
            regex_remove_non_ascii,
            pds_remove_non_ascii,
            python_remove_diacritics,
            pds_remove_diacritics,
            python_normalize_string,
            pds_normalize_string,
        ]
    ).save(BASE_PATH / "benchmark_data.parquet")


if __name__ == "__main__":
    main()
