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


if __name__ == "__main__":
    download()