import polars as pl
from pathlib import Path
import kaggle
from py7zr import SevenZipFile

COMPETITION = 'kkbox-churn-prediction-challenge'
PATH_DIR = Path("./data")


def download(
    filenames=("train", "transactions", "user_logs", "members_v3"),
):
    PATH_DIR.mkdir(exist_ok=True)

    print("Downloading from Kaggle...")
    for filename in filenames:
        filename_zip = f"{filename}.csv.7z"
        kaggle.api.competition_download_file(
            COMPETITION,
            file_name=filename_zip,
            path=PATH_DIR,
            force=False
        )
        with SevenZipFile(PATH_DIR / filename_zip, mode="r") as zip:
            zip.extractall(path=PATH_DIR)

        # Convert to parquet for more efficient processing.
        (
            pl.scan_csv(PATH_DIR / f"{filename}.csv")
            .sink_parquet(PATH_DIR / f"{filename}.pq")
        )
        # Remove intermediary files from disk.
        for ext in [".csv.7z", ".csv"]:
            (PATH_DIR / f"{filename}{ext}").unlink()


if __name__ == "__main__":
    download()