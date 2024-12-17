# %%
import polars as pl
from skrub import TableReport

tx = pl.read_parquet("../data/sessions.pq")
TableReport(tx, n_rows=20)

# %%
from matplotlib import pyplot as plt
import seaborn as sns


def to_datetime(column_name):
    return pl.col(column_name).cast(pl.String).str.to_datetime("%Y%m%d")


def plot_sessions_logs(tx, log, msnos):

    msnos = set(msnos) & set(log["msno"].unique())

    fix, axes = plt.subplots(
        figsize=(8, 2 * len(msnos)), ncols=1, nrows=len(msnos), sharex=True
    )
    
    for msno, ax in zip(msnos, axes):
        log_ = log.filter(pl.col("msno") == msno)
        if log_.shape[0] < 2:
            continue
        sns.histplot(
            log_,
            x="date",
            weights="num_unq",
            ax=ax,
            binwidth=3,
        )
        tx_ = tx.filter(pl.col("msno") == msno)
        y_lim = ax.get_ylim()
        for (start, end, churn) in tx_[
            ["session_start_date", "session_end_date", "churn"]
        ].iter_rows():
            ax.axvline(start, color='green')
            ax.axvline(end, color='red' if churn else 'blue')
            ax.fill_between([start, end], *y_lim, color="green", alpha=.1)
    
    ax.set_xticklabels(
        ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor'
    )
    plt.show()


msnos = tx.select(pl.col("msno")).sample(20).to_series(0).to_list()

log = (
    pl.scan_parquet("../data/user_logs.pq")
    .with_columns(
        to_datetime("date")
    )
    .sort("date")
    .filter(pl.col("msno").is_in(msnos))
    .collect()
)

plot_sessions_logs(tx, log, msnos)

# %%
TableReport(log)
# %%
