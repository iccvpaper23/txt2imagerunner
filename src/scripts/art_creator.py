'''
To use this script, export following variables environments:

$ export ART_DATASET=~/foo/bar/data.csv
'''
import os

import pandas as pd

from txt2imagewrapper_minimal import TextToImageWrapperMinimal


def read_dataset(dataset_path):
    return pd.read_csv(dataset_path)

def get_prompt_details(row, index):
    output_dir = f'{row["style"].lower()}_{row["artist"]}_{index}'
    return output_dir, row["prompt"]

if __name__ == '__main__':
    dataset_path = os.getenv('ART_DATASET')
    txt2wrappermin = TextToImageWrapperMinimal()

    df = read_dataset(dataset_path)
    for index, row in df.iterrows():
        output_dir, prompt = get_prompt_details(row, index)
        txt2wrappermin.generate_sample(
            prompt=prompt,
            sample_uuid=index,
            output_dir=output_dir)
        break