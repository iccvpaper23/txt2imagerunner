import subprocess, os

import pandas as pd


class TextToImageWrapper:

    def __init__(self, dataset: pd.DataFrame) -> None:
        self.__dataset = dataset.reset_index()
        self.__script_path = "scripts/txt2img.py"
        self.__model_path = "v2-1_768-ema-pruned.ckpt"
        self.__heigth = 256
        self.__width = 256
        self.__config = "configs/stable-diffusion/v2-inference-v.yaml"
        self.__n_samples = 5
        self.__upper_outdir = "outputs"

    def generate(self):
        for index, row in self.__dataset.iterrows():
            print(f"{index}/{row['tweet_id']} - {row['text']}")
            self.__call_diffusion_stable(
                row['text'],
                self.__create_sample_outdir(row['tweet_id']))

    def __create_sample_outdir(self, tweet_id):
        datadir = f"{self.__upper_outdir}/{tweet_id}"
        os.mkdir(datadir)
        return datadir

    def __call_diffusion_stable(self, text: str, outdir):
        os.system(f"python3 {self.__script_path} --prompt \"{text}\" --ckpt {self.__model_path} --H {self.__heigth} --W {self.__width} --config {self.__config} --n_samples {self.__n_samples} --outdir {outdir}")
