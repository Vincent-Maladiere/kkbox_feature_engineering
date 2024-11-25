# KKBox Feature Engineering

The KKBOX dataset is a multi-tables classification / survival analysis challenge,
with time-varying features.

This repo will help you download the dataset, preprocess and apply feature engineering
for machine learning.

## Set up

### Environment

Create an environment, for instance with uv, and install the dependencies:
```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.12
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

### Dataset

Join the [KKBox Kaggle competition](https://www.kaggle.com/c/kkbox-churn-prediction-challenge/data),
head to your [Kaggle account page](https://www.kaggle.com/settings/account) and create your
API token. Then, download your token and place it under (`.kaggle/kaggle.json`) in your
home directory.

Next, download the dataset with the following command:
```shell
make download
```
This operation should takes a few minutes.

### Preprocessing

Once the download is over, you are ready to apply some basic preprocessing to turn
this binary classification problem into a survival analysis one, using sessionization!

```shell
make sessionize
```
This operation should also takes a few minutes.