# %%
import polars as pl
import polars.selectors as cs


msnos = [
    'Gm6nq94d1kn48YpqQzvWshpfQ2F218e+ISrtxjaA/EI=',
    'bIwD9DI9jlYg0MMZT5oAj7leSYWc0p9qKfA199s3+CE=',
]

session = (
    pl.scan_parquet("../data/sessions.pq")
    .filter(pl.col("msno").is_in(msnos))
    .with_columns(
        pl.col("session_end_date").alias("observation_date")
    )
    .collect()
)

log = (
    pl.scan_parquet("../data/user_logs")
    .filter(pl.col("msno").is_in(msnos))
    .collect()
)

# %%
from skrub import TableReport
from sklearn.utils import check_random_state


def sample_obs_date(
    df,
    start_date_col,
    end_date_col,
    max_repetition=3,
    repetition_period=30,
    unit="day",
    strategy="uniform",
    random_state=None
):
    """Sample observation date for each sample, proportional to their duration.

    The duration is defined as end_date_col - start_date_col.
    """
    df = (
        df.with_columns(
            pl.col(end_date_col).sub(pl.col(start_date_col))
                .dt.total_days().alias("duration")
        )
        .with_columns(
            pl.max_horizontal(
                (pl.col("duration") // repetition_period),
                max_repetition,
            ).alias("n_draw")
        )
    )
    rng = check_random_state(random_state)
    all_draws = []
    for n_draw in range(max_repetition):
        draw = (
            df.filter(pl.col("n_draw") >= n_draw)
        )
        sampled_duration = rng.randint(0, draw.select("duration")).ravel()
        draw = (
            draw.with_columns(
                sampled_duration=sampled_duration
            )
            .with_columns(
                (pl.col(start_date_col) + pl.duration(days="sampled_duration"))
                    .alias("observation_date")
            )
        )
        all_draws.append(draw)

    return pl.concat(all_draws, how="vertical")


cols = [
    "session_id",
    "session_start_date",
    "session_end_date",
    "duration",
    "observation_date",
    "n_draw",
    "sampled_duration"
]
session_obs = (
    sample_obs_date(
        session,
        start_date_col="session_start_date",
        end_date_col="session_end_date",
        max_repetition=2,
    )
    .select(cols)
    .sort("session_id", "observation_date")
)

TableReport(session_obs)

# %%
# Could be turned into an sklearn transformer object.
# A function version of the transformer object could also be implemented,
# e.g.
# def temp_agg_join(...):
#     return TempAggJoiner(...).fit_transform(X)

def temp_agg_join(
    main,
    aux,
    main_key,
    aux_key,
    main_date_col,
    aux_date_col,
    selector,
    ops,
):
    """Filter, aggregate and join an auxiliary table on a main table.

    The auxiliary table is filtered with aux_date_col < main_date_col.
    """
    # If main_key is unique, then grouping and joining on (main_key, main_date_col)
    # is the same as using (main_key) only.
    aux_agg = (
        aux.join(
            main.select(main_key, main_date_col),
            left_on=aux_key,
            right_on=main_key
        )
        .filter(pl.col(aux_date_col) < pl.col(main_date_col))
        .group_by(aux_key, main_date_col)
        .agg(
            getattr(selector, ops)()
        )
    )
    main = (
        main.join(
            aux_agg,
            left_on=(main_key, main_date_col),
            right_on=(aux_key, main_date_col),
        )
    )

    return main


session_obs_agg = (
    temp_agg_join(
        main=session_obs,
        aux=log,
        main_key="session_id",
        aux_key="session_id",
        main_date_col="observation_date",
        aux_date_col="date",
        selector=cs.contains("num_"),
        ops="sum",
    )
    .sort("session_id", "observation_date")
)

TableReport(session_obs_agg)


# %%
