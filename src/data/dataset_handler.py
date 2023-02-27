import pandas as pd


def read_dataset(path: str, restaurant_dataset: bool) -> pd.DataFrame:
    print(restaurant_dataset)
    if restaurant_dataset == True:
        df = pd.read_csv(path, delimiter='\t', quoting=3)
    else:
        df = pd.read_csv(path)
    return df
