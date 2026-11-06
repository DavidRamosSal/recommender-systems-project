import pandas as pd


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
