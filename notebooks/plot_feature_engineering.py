# %%
# - Finish pipeline asap, might subsample
# - Try different sampling scheme
# - If it breaks, see how to optimize with DuckDB

import polars as pl
import polars.selectors as cs

import feature_engineering as fe


n_samples = 100_000

session = (
    pl.scan_parquet("../data/sessions.pq")
    .with_columns(
        pl.col("session_end_date").alias("observation_date")
    )
    .collect()
    .sample(n_samples)
    .pipe(
        fe.sample_obs_date,
        start_date_col="session_start_date",
        end_date_col="session_end_date",
        max_repetition=2,
        strategy="first",
    )
)

msnos = session.select("msno").unique()

session.shape

# %%
def to_datetime(column_name):
    return pl.col(column_name).cast(pl.String).str.to_datetime("%Y%m%d")


def renaming(df):
    """Rename targets in 'event', 'duration'.
    """
    df = df.with_columns(
        pl.col("session").alias("n_previous_churn"),
        pl.col("churn").cast(pl.Int8).alias("event"),
        pl.col("duration").alias("session_duration"),
        pl.col("sampled_duration").alias("duration")
    )
    return df

members = (
    pl.scan_parquet("../data/members_v3.pq")
    .filter(pl.col("msno").is_in(msnos))
    .with_columns(
        to_datetime("registration_init_time")
    )
    .collect()
)

session = (
    session.join(
        members, on="msno", how="left"
    )
    .pipe(renaming)
)

target_cols = ["event", "duration"]
session_id_cols = ['msno', 'session_id', 'observation_date']
session_feature_cols = [
    'session',
    'payment_method_id',
    'payment_plan_days',
    'plan_list_price',
    'actual_amount_paid',
    'is_auto_renew',
    'is_cancel',
    'days_between_subs',
    'days_from_initial_start',
]
members_cols = [
    "city",
    "bd",
    "gender",
    "registered_via",
    "registration_init_time"
]

X = (
    session.select(*session_id_cols, *session_feature_cols, *members_cols)
    .to_pandas()
)
y = (
    session.select(target_cols)
    .to_pandas()
)

X.shape, y.shape

# %%
from matplotlib import pyplot as plt
import seaborn as sns

sns.histplot(y, x="duration", hue="event", multiple="stack")
plt.show()

# %%

log = (
    pl.scan_parquet("../data/user_logs")
    .filter(pl.col("msno").is_in(msnos))
    .collect()
)
log.shape

# %%
from skrub import TableVectorizer, DropCols
from sklearn.preprocessing import OrdinalEncoder
from hazardous import SurvivalBoost

import utils


temporal_agg_joiner = fe.TemporalAggJoiner(
    aux_table=log,
    operations="sum",
    main_key="session_id",
    aux_key="session_id",
    main_date_col="observation_date",
    aux_date_col="date",
    cols=cs.contains("num_"),
)

model = utils.CumulativeIncidencePipeline([
    #("agg_joiner", temporal_agg_joiner),
    ("drop", DropCols(cols=session_id_cols)),
    ("encoder", TableVectorizer(low_cardinality=OrdinalEncoder())),
    ("model", SurvivalBoost(n_iter=30, show_progressbar=True)),
])


# %%
from sklearn.model_selection import cross_val_score

cross_val_score(model, X, y, groups=X["msno"], cv=3)


# %%
from tqdm import tqdm
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupShuffleSplit
from hazardous.metrics import brier_score_incidence, integrated_brier_score_incidence


all_imps, all_brier_scores, all_ibs = [], [], []

iter_ = tqdm(
    GroupShuffleSplit(n_splits=3)
    .split(X, y, groups=X["msno"])
)
for train_indices, test_indices in iter_:

    X_train, y_train = X.iloc[train_indices], y.iloc[train_indices]
    X_test, y_test = X.iloc[test_indices], y.iloc[test_indices]
    model.fit(X_train, y_train)
    y_proba = model.predict_cumulative_incidence(X_test)
    
    brier_scores = brier_score_incidence(
        y_train,
        y_test,
        y_pred=y_proba[:, 1, :],
        times=model.time_grid,
        event_of_interest=1,
    )
    ibs = integrated_brier_score_incidence(
        y_train,
        y_test,
        y_pred=y_proba[:, 1, :],
        times=model.time_grid,
        event_of_interest=1,
    )
    print(f"ibs: {ibs:.4f}")

    # We need to obtain X_test_transform before passing it to permutation_importance,
    # because our pipeline contains an AggJoiner. Otherwise, we couldn't inspect
    # the columns of the auxiliary table since they are not in X_test.
    transformers, estimator = model[:-1], model[-1]
    X_train_trans = transformers.fit_transform(X_train)
    X_test_trans = transformers.transform(X_test)
    estimator.fit(X_train_trans, y_train)
    perm_results = permutation_importance(estimator, X_test_trans, y_test)

    all_brier_scores.append(brier_scores)
    all_ibs.append(ibs)
    all_imps.append(perm_results)

# %%
# Plot distributions of feature importance across permutation iterations and
# cross validation iterations.
import pandas as pd
import numpy as np


feat_imps = (
    pd.DataFrame(
        np.concatenate(
            [imps["importances"] for imps in all_imps], axis=1
        ).T,
        columns=X_test_trans.columns,
    )
    .melt(var_name="feature", value_name="importance")
)
order = (
    feat_imps.groupby("feature")
    .mean()
    .sort_values("importance", ascending=False)
    .index
)
sns.boxplot(
    feat_imps, y="feature", x="importance", orient="h", order=order
)
plt.show()

# %%
# Plot the Brier score of the marginal Kaplan Meier estimator against our model.
from lifelines import KaplanMeierFitter


km = KaplanMeierFitter().fit(
    durations=y["duration"],
    event_observed=y["event"],
    timeline=model.time_grid,
)
y_proba_km = (
    km.cumulative_density_at_times(model.time_grid)
    .to_numpy()
    .ravel()
)
y_proba_km = np.repeat(y_proba_km[None, :], y.shape[0], axis=0)
brier_scores_km = brier_score_incidence(
    y,
    y,
    y_proba_km,
    model.time_grid,
    event_of_interest=1,
)
ibs_km = integrated_brier_score_incidence(
    y,
    y,
    y_proba_km,
    model.time_grid,
    event_of_interest=1,
)

brier_scores = pd.DataFrame(
    dict(
        scores=np.concatenate(all_brier_scores),
        times=np.tile(model.time_grid, 3),
    )
)
mean_ibs, std_ibs = np.mean(all_ibs), np.std(all_ibs)
label_model = f"Model (IBS: {mean_ibs:.4f} ± {std_ibs:.4f})"
label_km = f"KM (IBS: {ibs_km:.4f})"

ax = sns.lineplot(brier_scores, x="times", y="scores", label=label_model)
sns.lineplot(x=model.time_grid, y=brier_scores_km, ax=ax, label=label_km)
plt.legend()
plt.show()
# %%
# Compute the IPCW C-index.
from hazardous.metrics import concordance_index_incidence


concordance_index_incidence(
    y_test,
    y_proba[:, 1, :],
    time_grid=model.time_grid,
    taus=np.quantile(model.time_grid, [.25, .5, .75, .95]),
    y_train=y_train,
    ipcw_estimator="km"
)
# %%
