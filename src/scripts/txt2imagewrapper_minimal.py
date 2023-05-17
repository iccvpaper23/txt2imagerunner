'''
Same class as txt2imagewrapper but simpler, having only
a method to actually generate a image.

manage prompt is not handled from this class
'''
import os


class TextToImageWrapperMinimal:


    def __init__(self,
                 stable_diffusion_dir: str = "stablediffusion",
                 numer_samples: int = 1,
                 max_tries_on_error: int = 10,
                 n_iter: int = 1) -> None:
        self.__script_path = f"{stable_diffusion_dir}/scripts/txt2img.py"
        self.__model_path = f"{stable_diffusion_dir}/v2-1_512-ema-pruned.ckpt"
        self.__heigth = 512
        self.__width = 512
        self.__config = f"{stable_diffusion_dir}/configs/stable-diffusion/v2-inference.yaml"
        self.__n_samples = numer_samples
        self.__n_iter = n_iter
        self.__max_tries_on_sampling = max_tries_on_error

    def generate_sample(self, prompt: str, sample_uuid: str, output_dir: str, seed: int = 42):
        exit_status_code = 1
        sampling_tries = 0

        while exit_status_code != 0 and sampling_tries < self.__max_tries_on_sampling:
            sampling_tries += 1
            command = f"python3 {self.__script_path} --prompt \"{prompt}\" --ckpt {self.__model_path} " \
                        f"--H {self.__heigth} --W {self.__width} --config {self.__config} " \
                        f"--n_samples {self.__n_samples} --n_iter {self.__n_iter} --outdir {output_dir} " \
                        f"--seed {seed}"
            exit_status_code = os.system(command)
        
        if sampling_tries >= self.__max_tries_on_sampling:
            with open("sampling_errors.log", "a") as f:
                f.write(f"error creating image to id {sample_uuid} text '{prompt}' on dir '{output_dir}'\n")
                f.write(f"---------------------------------------------------------------------\n")