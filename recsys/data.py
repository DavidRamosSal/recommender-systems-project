from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class MovieLens1M:
    ratings: pd.DataFrame
    movies: pd.DataFrame
    users: pd.DataFrame


def load_movielens(data_dir: str | Path = "data/ml-1m") -> MovieLens1M:
    """Read the three .dat files without filtering, splitting, or remapping IDs.

    Timestamps remain Unix seconds; genres remain pipe-separated strings.
    Movie titles retain their release year, and ZIP codes remain strings.
    """
    data_dir = Path(data_dir)
    for name in ("ratings.dat", "movies.dat", "users.dat"):
        if not (data_dir / name).is_file():
            raise FileNotFoundError(
                f"Missing {data_dir / name}. "
                "Run `uv run scripts/download_data.py` for the default location, "
                "or pass the directory containing the .dat files."
            )

    def read(name: str, columns: dict[str, str]) -> pd.DataFrame:
        return pd.read_csv(
            data_dir / name,
            sep="::",
            engine="python",
            encoding="latin-1",
            names=list(columns),
            dtype=columns,
            keep_default_na=False,
        )

    return MovieLens1M(
        ratings=read("ratings.dat", {
            "user_id": "int64", "item_id": "int64",
            "rating": "int64", "timestamp": "int64",
        }),
        movies=read("movies.dat", {
            "item_id": "int64", "title": "string", "genres": "string",
        }),
        users=read("users.dat", {
            "user_id": "int64", "gender": "string", "age": "int64",
            "occupation": "int64", "zip_code": "string",
        }),
    )
