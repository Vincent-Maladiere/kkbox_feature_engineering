# %%
import polars as pl
from skrub import TableReport

tx = pl.read_parquet("data/sessions.pq")
TableReport(tx)

# %%
from matplotlib import pyplot as plt
import seaborn as sns


def to_datetime(column_name):
    return pl.col(column_name).cast(pl.String).str.to_datetime("%Y%m%d")


def plot_sessions_logs(tx, log, msnos):

    fix, axes = plt.subplots(ncols=1, nrows=len(msnos), sharex=True)

    for idx, ax in enumerate(axes):
        sns.histplot(
            log.filter(pl.col("msno") == msnos[idx]),
            x="date",
            weights="num_unq",
            ax=ax,
            binwidth=3,
        )
        tx_ = tx.filter(pl.col("msno") == msnos[idx])
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


msnos = [
    'Gm6nq94d1kn48YpqQzvWshpfQ2F218e+ISrtxjaA/EI=',
    'bIwD9DI9jlYg0MMZT5oAj7leSYWc0p9qKfA199s3+CE=',
]
weird_msno = 'mxpRHNaHkFo7IfnKAqUaA8xrgpaDnnZDsjJvw8gH+FQ='

log = (
    pl.scan_csv("data/user_logs.csv")
    .with_columns(
        to_datetime("date")
    )
    .sort("date")
    .filter(pl.col("msno").is_in(msnos))
    .collect()
)

plot_sessions_logs(tx, log, msnos)

# %%
