import subprocess, os

import pandas as pd


class TextToImageWrapper:

    def __init__(self, dataset: pd.DataFrame, restaurant_dataset: bool, restaurant_embedding: bool) -> None:
        self.__repo_dir = os.getenv("STABLE_DIFFUSION_DIR", "stablediffusion")
        if restaurant_embedding is not True:
            self.__dataset = dataset.reset_index()
        else:
            self.__dataset = dataset

        self.__script_path = f"{self.__repo_dir}/scripts/txt2img.py"
        self.__model_path = f"{self.__repo_dir}/v2-1_768-ema-pruned.ckpt"
        self.__heigth = 768
        self.__width = 768
        self.__config = f"{self.__repo_dir}/configs/stable-diffusion/v2-inference-v.yaml"
        self.__n_samples = int(os.getenv("N_SAMPLES", 1))
        self.__upper_outdir = f"{self.__repo_dir}/outputs"
        self.__max_tries_on_sampling = int(os.getenv("MAX_TRIES", 10))
        self.__restaurant_dataset = restaurant_dataset
        self.__restaurant_embedding = restaurant_embedding

    def __get__keys_for_dataset(self):
        if self.__restaurant_dataset == True:
            return 'index', 'Review'
        elif self.__restaurant_embedding == True:
            return 'index', None
        return 'tweet_id', 'text'
    
    def __generate_for_embeddings(self):
        for index in range(len(self.__dataset)):
            uuid = self.__dataset[index]
            print(f"restaurant review index {index}  uuid {uuid} generating embedding")
            can_proceeed = self.__create_sample_outdir(uuid)
            if can_proceeed is not False:
                self.__call_diffusion_stable(
                    uuid, index, can_proceeed)
            else:
                print(f"skipping {uuid} sample already exists")

    def generate(self):
        if self.__restaurant_embedding:
            self.__generate_for_embeddings()
        else:
            for index, row in self.__dataset.iterrows():
                uuid, text_key = self.__get__keys_for_dataset()
                print(f"{index}/{row[uuid]} - {row[text_key]}")
                can_proceeed = self.__create_sample_outdir(row[uuid])
                if can_proceeed is not False:
                    self.__call_diffusion_stable(
                        row[uuid], row[uuid], can_proceeed)
                else:
                    print(f"skipping {row[uuid]} sample already exists")

    def __create_sample_outdir(self, tweet_id):
        try:
            datadir = f"{self.__upper_outdir}/{tweet_id}"
            if os.path.isdir(datadir):
                return False
            os.mkdir(datadir)
        except Exception as e:
            print(e)
        return datadir

    def __call_diffusion_stable(self, tweet_id: str, text: str, outdir):
        exit_status_code = 1
        sampling_tries = 0
        while exit_status_code != 0 and sampling_tries < self.__max_tries_on_sampling:
            sampling_tries += 1
            command = f"python3 {self.__script_path} --prompt \"{text}\" --ckpt {self.__model_path} --H {self.__heigth} --W {self.__width} --config {self.__config} --n_samples {self.__n_samples} --outdir {outdir}"
            exit_status_code = os.system(command)
        
        if sampling_tries >= self.__max_tries_on_sampling:
            with open("sampling_errors.log", "a") as f:
                f.write(f"error creating image to id {tweet_id} text '{text}' on dir '{outdir}'\n")
                f.write(f"---------------------------------------------------------------------\n")