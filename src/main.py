import os

import pandas as pd

from data.dataset_handler import read_dataset
from services.txt2imagewrapper import TextToImageWrapper

IS_RESTAURANT_REVIEW = bool(os.getenv('IS_RESTAURANT_REVIEW', False))
IS_RESTAURANT_REVIEW_FROM_EMBEDDINGS = bool(os.getenv('IS_RESTAURANT_REVIEW_FROM_EMBEDDINGS', False))

if __name__ == '__main__':
    dataset_path = os.getenv('DATASET_PATH', './dataset/airline_reviews.csv')
    dataset_or_fold0_train, fold1_test = read_dataset(
        path=dataset_path,
        restaurant_dataset=IS_RESTAURANT_REVIEW,
        restaurant_dataset_embeddings=IS_RESTAURANT_REVIEW_FROM_EMBEDDINGS)
    
    TextToImageWrapper(
        dataset=dataset_or_fold0_train,
        restaurant_dataset=IS_RESTAURANT_REVIEW,
        restaurant_embedding=IS_RESTAURANT_REVIEW_FROM_EMBEDDINGS).generate()
    
    if fold1_test:
        TextToImageWrapper(
            dataset=fold1_test,
            restaurant_dataset=IS_RESTAURANT_REVIEW,
            restaurant_embedding=IS_RESTAURANT_REVIEW_FROM_EMBEDDINGS,
            embedding_is_test=True).generate()
