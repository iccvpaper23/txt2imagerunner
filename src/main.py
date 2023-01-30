import os

import pandas as pd

from data.dataset_handler import read_dataset
from services.txt2imagewrapper import TextToImageWrapper


if __name__ == '__main__':
    dataset_path = os.getenv('DATASET_PATH', './dataset/airline_reviews.csv')
    dataset : pd.DataFrame = read_dataset(path=dataset_path)
    TextToImageWrapper(
        dataset=dataset).generate()
