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
    random_state=None
):
    """Sample observation date for each sample, proportional to their duration.

    The duration is defined as end_date_col - start_date_col.
    """
    if strategy == "uniform":
        df = (
            df.with_columns(
                pl.col(end_date_col).sub(pl.col(start_date_col))
                    .dt.total_days().alias("duration")
            )
            .with_columns(
                pl.min_horizontal(
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
        df = pl.concat(all_draws, how="vertical")

    elif strategy == "first":
        df = (
            df.with_columns(
                pl.col(end_date_col).sub(pl.col(start_date_col))
                    .dt.total_days().alias("duration"),
                pl.col(start_date_col).alias("observation_date")
            )
            .with_columns(
                pl.col("duration").alias("sampled_duration")
            )
        )

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
    ):
        self.aux_table = aux_table
        self.operations = operations
        self.main_key = main_key
        self.aux_key = aux_key
        self.main_date_col = main_date_col
        self.aux_date_col = aux_date_col
        self.cols = cols

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
        if isinstance(X, pd.DataFrame):
            pd_index = X.index
            X = pl.from_pandas(X)
        
        # If main_key is unique, then grouping and joining on (main_key, main_date_col)
        # is the same as using (main_key) only.
        exprs = []
        for ops in self.operations:
            if ops == "ewm_mean":
                expr = self.cols.ewm_mean_by(
                    by=self.aux_date_col,
                    half_life="7d"
                ).last()
            else:
                expr = getattr(self.cols, ops)()
            exprs.append(expr)

        aux_agg = (
            self.aux_table.join(
                X.select(self.main_key, self.main_date_col),
                left_on=self.aux_key,
                right_on=self.main_key
            )
            .filter(pl.col(self.aux_date_col) < pl.col(self.main_date_col))
            .group_by(self.aux_key, self.main_date_col)
            .agg(exprs)
            .explode(cs.by_dtype(pl.List(pl.Float64)))
        )
        X_out = (
            X.join(
                aux_agg,
                left_on=(self.main_key, self.main_date_col),
                right_on=(self.aux_key, self.main_date_col),
                how="left",
            )
            .to_pandas()
            .set_index(pd_index)
        )
        
        return X_out
        
    def get_feature_names_out(self):
        check_is_fitted(self, "all_outputs_")
        return self.all_outputs_
