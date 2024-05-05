import random
import string
from pathlib import Path

import polars as pl
from faker import Faker

random.seed(208)

BASE_PATH = Path(__file__).resolve().parents[0]

INT32_RANGE = (-(2**31), 2**31 - 1)


UNICODE_ALPHABET = [
    chr(code_point)
    for current_range in [
        (0x0021, 0x0021),
        (0x0023, 0x0026),
        (0x0028, 0x007E),
        (0x00A1, 0x00AC),
        (0x00AE, 0x00FF),
        (0x0100, 0x017F),
        (0x0180, 0x024F),
        (0x2C60, 0x2C7F),
        (0x16A0, 0x16F0),
        (0x0370, 0x0377),
        (0x037A, 0x037E),
        (0x0384, 0x038A),
        (0x038C, 0x038C),
    ]
    for code_point in range(current_range[0], current_range[1] + 1)
]


def random_unicode(length: int) -> str:
    # https://stackoverflow.com/questions/1477294/generate-random-utf-8-string-in-python
    return "".join(random.choices(UNICODE_ALPHABET, k=length))


def random_ascii(length: int) -> str:
    return "".join(random.choices(string.printable, k=length))


class DataGenerator:
    def __init__(self, height: int):
        self.height = height

    def random_unicode_column(self, min_length: int, max_length: int) -> list[str]:
        column = []
        for _ in range(self.height):
            length = random.randint(min_length, max_length)
            column.append(random_unicode(length))

        return column

    def random_ascii_column(self, min_length: int, max_length: int) -> list[str]:
        column = []
        for _ in range(self.height):
            length = random.randint(min_length, max_length)
            column.append(random_ascii(length))

        return column

    def random_address_column(self) -> list[str]:
        column = []
        fake = Faker()
        for i in range(self.height):
            Faker.seed(i)
            column.append(fake.address())

        return column

    def random_integer_column(self, min_value: int, max_value: int) -> list[int]:
        return [random.randint(min_value, max_value) for _ in range(self.height)]

    def random_float_column(self, min_value: float, max_value: float) -> list[float]:
        return [random.uniform(min_value, max_value) for _ in range(self.height)]

    def generate(self) -> pl.DataFrame:
        min_length = 1
        max_length = 50

        return pl.DataFrame(
            {
                "RANDOM_STRING": self.random_unicode_column(min_length, max_length),
                "RANDOM_ASCII": self.random_ascii_column(min_length, max_length),
                "RANDOM_ADDRESS": self.random_address_column(),
            }
        )


def main():
    DataGenerator(1_000_000).generate().write_parquet(BASE_PATH / "benchmark_df.parquet")


if __name__ == "__main__":
    main()
