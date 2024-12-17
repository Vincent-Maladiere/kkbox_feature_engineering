# %%

import pandas as pd
import numpy as np
import polars as pl
import polars.selectors as cs
from skrub import TableReport
from matplotlib import pyplot as plt
import seaborn as sns

import feature_engineering as fe


def set_target(df):
    """Rename targets in 'event', 'duration'."""
    df = df.with_columns(
        pl.col("churn").alias("churn_session"),
        (
            (
                pl.col("session_end_date")
                < pl.col("observation_date").dt.offset_by("30d")
            )
            & pl.col("churn")
        ).alias("churn"),
        pl.col("observation_date")
        .sub("session_start_date")
        .dt.total_days()
        .alias("age_days_at_obs"),
    )
    return df


session = (
    pl.scan_parquet("../data/sessions.pq")
    .filter(pl.col("duration") > 0)
    .collect()
    .pipe(
        fe.sample_obs_date,
        start_date_col="session_start_date",
        end_date_col="session_end_date",
        strategy="grid",
    )
    .pipe(set_target)
    .sort("observation_date")
)

msnos = session["msno"].unique()[:50_000]
session = session.filter(pl.col("msno").is_in(msnos)).sort("observation_date")
session.shape

# %%
def to_datetime(column_name):
    return pl.col(column_name).cast(pl.String).str.to_datetime("%Y%m%d")


members = (
    pl.scan_parquet("../data/members_v3.pq")
    .filter(pl.col("msno").is_in(msnos))
    .with_columns(to_datetime("registration_init_time"))
    .collect()
)

session = session.join(members, on="msno", how="left")

target_col = ["churn"]
session_id_cols = ["msno", "session_id", "observation_date"]
session_feature_cols = [
    "session",
    "payment_method_id",
    "payment_plan_days",
    "plan_list_price",
    "actual_amount_paid",
    "is_auto_renew",
    "is_cancel",
    "days_between_subs",
    "days_from_initial_start",
]
members_cols = ["city", "bd", "gender", "registered_via", "registration_init_time"]

X = session.select(*session_id_cols, *session_feature_cols, *members_cols).to_pandas()
y = session.to_pandas()[target_col].to_numpy().ravel()

X.shape, y.shape

# %%
prevalence = y.mean().round(4)
print(prevalence)

# %%

log = (
    pl.scan_parquet("../data/user_logs")
    .filter(pl.col("msno").is_in(msnos))
    .sort("date")
    .collect()
)
log.shape

# %%

from collections import defaultdict
from scipy.stats import loguniform, randint
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
from sklearn.metrics import brier_score_loss, roc_auc_score, average_precision_score
from skrub import DropCols, TableVectorizer


def evaluate(model, X, y):

    metrics = defaultdict(list)

    observation_dates = sorted(X["observation_date"].unique())[-6:-1]
    for observation_date in observation_dates:
        X_train = X.loc[X["observation_date"] <= observation_date]
        X_test = X.loc[X["observation_date"] > observation_date]
        y_train = y[X_train.index]
        y_test = y[X_test.index]
        print(X_train.shape, X_test.shape)

        model.fit(X_train, y_train)

        y_proba = model.predict_proba(X_test)[:, 1]

        brier_score = round(brier_score_loss(y_test, y_proba, pos_label=1), 4)
        print(f"{brier_score=!r}")
        metrics["brier_score"].append(brier_score)

        auc = round(roc_auc_score(y_test, y_proba), 4)
        print(f"{auc=!r}")
        metrics["auc"].append(auc)

        avg_precision = round(average_precision_score(y_test, y_proba, pos_label=1), 4)
        print(f"{avg_precision=!r}")
        metrics["avg_precision"].append(avg_precision)

        # We need to obtain X_test_transform before passing it to permutation_importance,
        # because our pipeline contains an AggJoiner. Otherwise, we couldn't inspect
        # the columns of the auxiliary table since they are not in X_test.
        transformers, estimator = model[:-1], model[-1]
        X_test_trans = transformers.transform(X_test)

        fig = plot_feat_imps(estimator, X_test_trans, y_test)

    aggregated = {}
    for metric, values in metrics.items():
        aggregated[metric] = f"{np.mean(values):.4f} ± {np.std(values):.4f}"

    return aggregated


def plot_feat_imps(estimator, X, y):

    results = permutation_importance(
        estimator, X, y, scoring="neg_brier_score", max_samples=10_000
    )

    feat_imps = pd.DataFrame(results["importances"], index=X.columns).T.melt(
        var_name="feature", value_name="importance"
    )
    order = (
        feat_imps.groupby("feature")
        .mean()
        .sort_values("importance", ascending=False)
        .index
    )
    fig, ax = plt.subplots()
    sns.boxplot(feat_imps, y="feature", x="importance", orient="h", order=order, ax=ax)
    plt.show()

    return fig


# %%

param_dist = {
    "estimator__learning_rate": loguniform(1e-3, 1e-1),
    "estimator__max_depth": [4, 5, 6],
    "estimator__max_iter": randint(30, 200),
}

X_empty = X.copy()
X_empty["single"] = 1

model = Pipeline(
    [
        ("drop", DropCols(cols=session_id_cols + session_feature_cols + members_cols)),
        ("encoder", TableVectorizer(low_cardinality=OrdinalEncoder())),
        ("estimator", HistGradientBoostingClassifier()),
    ]
)

evaluate(model, param_dist, X_empty, y)


# %%

cols = ["num_100", "num_985", "num_75", "num_50", "num_25", "num_unq"]

taj = fe.TemporalAggJoiner(
    aux_table=log,
    operations=["sum", "mean", "ewm_mean"],
    main_key="session_id",
    aux_key="session_id",
    main_date_col="observation_date",
    aux_date_col="date",
    cols=pl.col(cols),
    half_life="3d",
)

model = Pipeline(
    [
        ("temp_agg_joiner", taj),
        ("drop", DropCols(cols=session_id_cols + session_feature_cols + members_cols)),
        ("encoder", TableVectorizer(low_cardinality=OrdinalEncoder())),
        ("estimator", HistGradientBoostingClassifier()),
    ]
)

evaluate(model, X, y)


# %%
