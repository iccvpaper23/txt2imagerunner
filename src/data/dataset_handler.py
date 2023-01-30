import pandas as pd


def read_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df
