import pandas as pd
import polars as pl
import polars.selectors as cs
from sklearn.utils import check_random_state
from sklearn.base import TransformerMixin, BaseEstimator, check_is_fitted


def sample_obs_date(
    df,
    start_date_col,
    end_date_col,
    max_repetition=3,
    repetition_period=30,
    unit="day",
    strategy="uniform",
    random_state=None,
):
    """Sample observation date for each sample, proportional to their duration.

    The duration is defined as end_date_col - start_date_col.
    """
    if strategy == "uniform":
        df = df.with_columns(
            pl.col(end_date_col)
            .sub(pl.col(start_date_col))
            .dt.total_days()
            .alias("duration")
        ).with_columns(
            pl.min_horizontal(
                (pl.col("duration") // repetition_period),
                max_repetition,
            ).alias("n_draw")
        )
        rng = check_random_state(random_state)
        all_draws = []
        for n_draw in range(max_repetition):
            draw = df.filter(pl.col("n_draw") >= n_draw)
            sampled_duration = rng.randint(0, draw.select("duration")).ravel()
            draw = draw.with_columns(sampled_duration=sampled_duration).with_columns(
                (pl.col(start_date_col) + pl.duration(days="sampled_duration")).alias(
                    "observation_date"
                )
            )
            all_draws.append(draw)
        df = pl.concat(all_draws, how="vertical")

    elif strategy == "first":
        df = df.with_columns(
            pl.col(end_date_col)
            .sub(pl.col(start_date_col))
            .dt.total_days()
            .alias("duration"),
            pl.col(start_date_col).alias("observation_date"),
        ).with_columns(pl.col("duration").alias("sampled_duration"))

    elif strategy == "grid":
        observation_dates = (
            df.select(pl.col(start_date_col, end_date_col).dt.truncate("1mo"))
            .select(
                pl.date_ranges(
                    pl.col(start_date_col).min(),
                    pl.col(end_date_col).max(),
                    "1mo",
                ).alias("date_range")
            )
            .explode("date_range")
            .to_series(0)
        )
        all_segments = []
        for observation_date in observation_dates:
            segment = df.filter(
                (pl.col("session_start_date") < observation_date)
                & (observation_date < pl.col("session_end_date"))
            ).with_columns(observation_date=observation_date)
            all_segments.append(segment)
        df = pl.concat(all_segments, how="vertical")

    else:
        raise ValueError(f"strategy options are 'uniform', 'first'. Got {strategy}.")

    return df


class TemporalAggJoiner(TransformerMixin, BaseEstimator):
    """Filter, aggregate and join an auxiliary table on a main table.

    The auxiliary table is filtered with aux_date_col < main_date_col.
    """

    def __init__(
        self,
        aux_table,
        operations,
        *,
        main_key=None,
        aux_key=None,
        main_date_col=None,
        aux_date_col=None,
        cols=None,
        half_life=None,
    ):
        self.aux_table = aux_table
        self.operations = operations
        self.main_key = main_key
        self.aux_key = aux_key
        self.main_date_col = main_date_col
        self.aux_date_col = aux_date_col
        self.cols = cols
        self.half_life = half_life

    def fit(self, X, y=None):
        del y
        _ = self.fit_transform(X)
        return self

    def fit_transform(self, X, y=None):
        del y
        X_out = self.transform(X)

        self.all_outputs_ = X_out.columns

        return X_out

    def transform(self, X):

        pd_index = None
        if isinstance(X, pd.DataFrame):
            pd_index = X.index
            X = pl.from_pandas(X)

        order = X.select(self.main_key, self.main_date_col)

        # If main_key is unique, then grouping and joining on (main_key, main_date_col)
        # is the same as using (main_key) only.
        exprs = []
        for ops in self.operations:
            if ops == "ewm_mean":
                expr = self.cols.ewm_mean_by(
                    by=self.aux_date_col, half_life=self.half_life,
                ).last().name.suffix(f"_{self.half_life}")
            else:
                expr = getattr(self.cols, ops)().name.suffix(f"_{ops}")
            exprs.append(expr)

        X_out = []
        for observation_time in self.get_time_grid(X):
            X_obs = X.filter(pl.col(self.main_date_col) == observation_time)
            aux_agg = (
                self.aux_table.filter(pl.col(self.aux_date_col) < observation_time)
                .group_by(self.aux_key)
                .agg(exprs)
            )

            if cols := cs.expand_selector(aux_agg, cs.by_dtype(pl.List(pl.Float64))):
                aux_agg = aux_agg.explode(cols)

            X_out.append(
                X_obs.join(
                    aux_agg,
                    left_on=self.main_key,
                    right_on=self.aux_key,
                    how="left",
                )
            )

        X_out = order.join(
            pl.concat(X_out, how="vertical"),
            on=(self.main_key, self.main_date_col),
            how="left",
        )

        if pd_index is not None:
            return X_out.to_pandas().set_index(pd_index)

        return X_out

    def get_time_grid(self, X):
        return X[self.main_date_col].unique().sort()

    def get_feature_names_out(self):
        check_is_fitted(self, "all_outputs_")
        return self.all_outputs_
