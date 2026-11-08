import pandas as pd


def to_epoch(date: str) -> int:
    """Convert a UTC date string to MovieLens timestamp seconds."""
    return int(pd.Timestamp(date, tz="UTC").timestamp())


def temporal_split_by_date(
    ratings: pd.DataFrame,
    val_start: str | int,
    test_start: str | int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Global date cuts: train < val_start <= validation < test_start <= test."""
    val_start = to_epoch(val_start) if isinstance(val_start, str) else val_start
    test_start = to_epoch(test_start) if isinstance(test_start, str) else test_start
    if val_start >= test_start:
        raise ValueError("Validation must start before test.")
    timestamps = ratings["timestamp"]
    return (
        ratings[timestamps < val_start],
        ratings[(timestamps >= val_start) & (timestamps < test_start)],
        ratings[timestamps >= test_start],
    )


def temporal_split(
    ratings: pd.DataFrame,
    *,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Global temporal split using timestamp quantiles.

    Return train, validation and test, retaining all users, items and ratings.
    """
    timestamps = ratings["timestamp"]

    # Quantiles balance interaction counts despite burstiness in rating timestamps
    t1, t2 = timestamps.quantile(
        [train_fraction, train_fraction + validation_fraction]
    )
    # Convention: boundary timestamps belong to the later split, so [t1, t2) and [t2, ...)
    train = ratings[timestamps < t1]
    validation = ratings[(timestamps >= t1) & (timestamps < t2)]
    test = ratings[timestamps >= t2]

    return train, validation, test
