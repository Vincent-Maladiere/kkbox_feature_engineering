from time import time
import polars as pl

from download import PATH_DIR


def sessionize(
    input_path_csv=PATH_DIR / "transactions.csv",
    output_path_parquet=PATH_DIR / "sessions.pq",
):
    tic = time()
    (
        pl.scan_csv(input_path_csv)
        .pipe(clean_dates)
        .pipe(is_churn)
        .pipe(split_sessions)
        .pipe(session_features)
        .collect()
        .write_parquet(output_path_parquet)
    )
    duration = time() - tic 
    print(f"Wrote {output_path_parquet} - Took {duration:.1f}s")


def clean_dates(tx):
    return (
        tx.with_columns(
            to_datetime("transaction_date"),
            to_datetime("membership_expire_date"),
        )
        .filter(
            (pl.col("membership_expire_date") > pl.datetime(2015, 1, 1))
            & (pl.col("transaction_date") <= pl.col("membership_expire_date"))
        )
    )


def is_churn(
    tx,
    end_of_study=pl.datetime(2017, 3, 1),
    churn_threshold_days=30,
):
    """
    Churn is defined as 'churn_threshold_days' days past expiration date without \
    transaction.
    """
    return (
        tx.sort("transaction_date")
        .select(
            pl.all(),
            pl.col("transaction_date").shift(-1).over(pl.col("msno"))
            .fill_null(end_of_study).alias("next")
        )
        .with_columns(
            (pl.col("next") - pl.col("membership_expire_date"))
            .dt.total_days().alias("days_without_membership"),
        )
        .with_columns(
            (pl.col("days_without_membership") > churn_threshold_days).alias("churn")
        )
    )


def split_sessions(
    tx,
    end_of_study=pl.datetime(2017, 3, 1),
    churn_threshold_days=30,
):
    offset = f"{churn_threshold_days}d"
    return (
        tx.with_columns(
            pl.col("churn").cast(pl.Int8).cum_sum().shift(1).fill_null(0).over("msno")
                .alias("session"),
        )
        .sort("transaction_date")
        .with_columns(
            pl.col("membership_expire_date").shift(1).over("msno")
                .alias("prev_expire_date"),
            pl.col("transaction_date").alias("session_start_date"),
            pl.col("membership_expire_date").max().dt.offset_by(offset)
                .over("msno", "session").alias("session_end_date"),
            # We want the first transaction to take the churn status.
            pl.col("churn").max().over("msno", "session"),
            pl.col("transaction_date").max().over("msno", "session")
                .alias("max_transaction_date"),
        )
        .with_columns(
            pl.min_horizontal("session_end_date", end_of_study)
        )
        # Only keep the first transaction (since this is sorted).
        .unique(subset=["msno", "session"], keep="first")
    )


def session_features(tx):
    return (
        tx.sort("msno", "transaction_date")
        .with_columns(
            (pl.col("session_end_date") - pl.col("session_start_date")).dt.total_days()
                .alias("duration"),
            (
                pl.col("session_start_date")
                - pl.col("session_end_date").shift(1).over("msno") 
            ).dt.total_days().fill_null(0).alias("days_between_subs"),
            (
                pl.col("session_start_date")
                - pl.col("session_start_date").min().over("msno")
            ).dt.total_days().alias("days_from_initial_start")
        )
    )


def to_datetime(column_name):
    return pl.col(column_name).cast(pl.String).str.to_datetime("%Y%m%d")


if __name__ == "__main__":
    sessionize()