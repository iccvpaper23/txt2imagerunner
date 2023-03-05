import pandas as pd


def read_dataset(path: str, restaurant_dataset: bool, imdb_dataset: bool) -> pd.DataFrame:
    if restaurant_dataset == True:
        print("restaurant dataset selected")
        df = pd.read_csv(path, delimiter='\t', quoting=3)
    elif imdb_dataset == True:
        print("imdb dataset selected")
        df = pd.read_csv(path)
    else:
        print("airline reviews dataset selected")
        df = pd.read_csv(path)
    return df
