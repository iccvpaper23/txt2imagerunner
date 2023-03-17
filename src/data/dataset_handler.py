import pandas as pd


def __read_fold_zero(restaurant_dataset):
    print('__read_fold_zero')
    with open(f'{restaurant_dataset}/restaurant_folds_train.json', 'r') as ftrain:
        tweet_ids_train = pd.read_json(ftrain)
        tweet_ids_train = tweet_ids_train.drop(columns=['name'])

    tweet_ids_train = tweet_ids_train.rename(columns={'data': 'train'})

    with open(f'{restaurant_dataset}/restaurant_folds_test.json', 'r') as ftest:
        tweet_ids_test = pd.read_json(ftest)
        tweet_ids_test = tweet_ids_test.drop(columns=['name'])

    tweet_ids_test = tweet_ids_test.rename(columns={'data': 'test'})

    tweet_ids = pd.merge(tweet_ids_train, tweet_ids_test)
    tweet_ids = tweet_ids.rename(columns={'index': 'fold'})
    return tweet_ids['train'][0], tweet_ids['test'][0]

def read_dataset(path: str, restaurant_dataset: bool, restaurant_dataset_embeddings: bool) -> pd.DataFrame:
    print('read_dataset')
    if restaurant_dataset == True:
        print("restaurant dataset selected")
        df = pd.read_csv(path, delimiter='\t', quoting=3)
    elif restaurant_dataset_embeddings:
        df_ids_train, df_ids_test = __read_fold_zero(path)
        return df_ids_train, df_ids_test
    else:
        print("airline reviews dataset selected")
        df = pd.read_csv(path)
    return df, None
