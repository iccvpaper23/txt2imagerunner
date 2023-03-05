import os

import pandas as pd

from data.dataset_handler import read_dataset
from services.txt2imagewrapper import TextToImageWrapper

IS_RESTAURANT_REVIEW = bool(os.getenv('IS_RESTAURANT_REVIEW', False))
IS_IMDB = bool(os.getenv('IS_IMDB', False))

if __name__ == '__main__':
    dataset_path = os.getenv('DATASET_PATH', './dataset/airline_reviews.csv')
    dataset : pd.DataFrame = read_dataset(
        path=dataset_path,
        restaurant_dataset=IS_RESTAURANT_REVIEW,
        imdb_dataset=IS_IMDB)
    TextToImageWrapper(
        dataset=dataset,
        restaurant_dataset=IS_RESTAURANT_REVIEW,
        imdb_dataset=IS_IMDB).generate()
