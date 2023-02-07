import subprocess, os

import pandas as pd


class TextToImageWrapper:

    def __init__(self, dataset: pd.DataFrame) -> None:
        self.__repo_dir = os.getenv("STABLE_DIFFUSION_DIR", "stablediffusion")
        self.__dataset = dataset.reset_index()
        self.__script_path = f"{self.__repo_dir}/scripts/txt2img.py"
        self.__model_path = f"{self.__repo_dir}/v2-1_768-ema-pruned.ckpt"
        self.__heigth = 768
        self.__width = 768
        self.__config = f"{self.__repo_dir}/configs/stable-diffusion/v2-inference-v.yaml"
        self.__n_samples = int(os.getenv("N_SAMPLES", 1))
        self.__upper_outdir = f"{self.__repo_dir}/outputs"

    def generate(self):
        for index, row in self.__dataset.iterrows():
            print(f"{index}/{row['tweet_id']} - {row['text']}")
            can_proceeed = self.__create_sample_outdir(row['tweet_id'])
            if can_proceeed is not False:
                self.__call_diffusion_stable(
                    row['text'], can_proceeed)
            else:
                print(f"skipping {row['tweet_id']} sample already exists")

    def __create_sample_outdir(self, tweet_id):
        try:
            datadir = f"{self.__upper_outdir}/{tweet_id}"
            if os.path.isdir(datadir):
                return False
            os.mkdir(datadir)
        except Exception as e:
            print(e)
        return datadir

    def __call_diffusion_stable(self, text: str, outdir):
        status = os.system(f"python3 {self.__script_path} --prompt \"{text}\" --ckpt {self.__model_path} --H {self.__heigth} --W {self.__width} --config {self.__config} --n_samples {self.__n_samples} --outdir {outdir}")
        if status != 0:
            raise Exception(f"error creating image to text {text} on dir {outdir}")