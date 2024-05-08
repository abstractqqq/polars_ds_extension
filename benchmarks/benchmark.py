import functools
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

            for f in funcs:
                func_name = f.func.__name__ if isinstance(f, functools.partial) else f.__name__
                time = timeit.timeit(lambda: f(df), number=self.timing_runs)

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


def python_map_words(df: pl.DataFrame, mapping: dict[str, str]):
    df.select(
        pl.col("RANDOM_ADDRESS").map_elements(
            lambda s: " ".join(mapping.get(word, word) for word in s.split()),
            return_dtype=pl.String,
        )
    )


def regex_map_words(df: pl.DataFrame, mapping: dict[str, str]):
    expr = pl.col("RANDOM_ADDRESS")
    for k, v in mapping.items():
        expr = expr.str.replace_all(k, v)
    df.select(expr)


def pds_map_words(df: pl.DataFrame, mapping: dict[str, str]):
    df.select(pds.map_words("RANDOM_ADDRESS", mapping))


def python_normalize_whitespace(df: pl.DataFrame):
    df.select(
        pl.col("RANDOM_STRING").map_elements(lambda s: " ".join(s.split()), return_dtype=pl.String)
    )


def python_normalize_whitespace_only_spaces(df: pl.DataFrame):
    df.select(
        pl.col("RANDOM_STRING").map_elements(
            lambda s: " ".join(s.split(" ")), return_dtype=pl.String
        )
    )


def expr_normalize_whitespace_only_spaces(df: pl.DataFrame):
    df.select(pl.col("RANDOM_STRING").str.split(" ").list.join(" "))


def pds_normalize_whitespace(df: pl.DataFrame):
    df.select(pds.normalize_whitespace("RANDOM_STRING"))


def pds_normalize_whitespace_only_spaces(df: pl.DataFrame):
    df.select(pds.normalize_whitespace("RANDOM_STRING", only_spaces=True))


def main():
    benchmark_df = pl.read_parquet(BASE_PATH / "benchmark_df.parquet")

    map_words_mapping = {
        "Apt.": "Apartment",
        "NY": "New York",
        "CT": "Connecticut",
        "Street": "ST",
        "Bypass": "BYP",
        "GA": "Georgia",
        "Parkways": "Pkwy",
        "PA": "Pennsylvania",
    }

    Bench(benchmark_df).run(
        [
            python_remove_non_ascii,
            regex_remove_non_ascii,
            pds_remove_non_ascii,
            python_remove_diacritics,
            pds_remove_diacritics,
            python_normalize_string,
            pds_normalize_string,
            functools.partial(python_map_words, mapping=map_words_mapping),
            functools.partial(
                regex_map_words,
                mapping={f"\b{k}\b": v for k, v in map_words_mapping.items()},
            ),
            functools.partial(pds_map_words, mapping=map_words_mapping),
            python_normalize_whitespace,
            python_normalize_whitespace_only_spaces,
            expr_normalize_whitespace_only_spaces,
            pds_normalize_whitespace,
            pds_normalize_whitespace_only_spaces,
        ]
    ).save(BASE_PATH / "benchmark_data.parquet")


if __name__ == "__main__":
    main()
