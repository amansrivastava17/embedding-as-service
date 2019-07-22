from utils import home_directory, get_hashed_name, download_from_url, extract_file
from models import MODELS_DIR

from typing import Union, Optional, List, Dict
import importlib
import os


class Encoder(object):
    def __init__(self, embedding: str, model: str, download: bool = False):
        self.embedding = embedding
        self.model = model
        self.embedding_model_dict = None
        self.model_path = None

        supported_embeddings = self.get_supported_embeddings()

        # check if embedding exits
        if embedding not in supported_embeddings:
            print(f"Given embedding \"{embedding}\" is not supported, use below available embeddings:\n"
                  f"{supported_embeddings}")
            return

        self.embedding_cls = importlib.import_module(
            f'models.{embedding}').Embeddings()

        # check if given model exits for embedding
        model_names = list(self.embedding_cls.EMBEDDING_MODELS.keys())
        if model not in model_names:
            print(f"Given embedding \"{embedding}\" does not have any model \"{model}\", here are the supported "
                  f"models: {model_names}")
            return

        self.model_path = self._get_or_download_model(download)
        if not self.model_path:
            print(f"Model does not exits, pass download param as True")
            return

        print('Loading Model (this might take few minutes).....')
        self._load_model()

    @staticmethod
    def get_supported_embeddings() -> List[str]:
        """
        Return list of supported languages
        Returns:
            (list): supported languages
        """
        supported_languages = []
        cwd = os.path.dirname(os.path.abspath(__file__))
        cwd_dirs = [x for x in os.listdir(cwd) if os.path.isdir(os.path.join(cwd, x)) if not x.startswith('__') and
                    not x.startswith('.')]
        for _dir in cwd_dirs:
            supported_languages.append(_dir)
        return supported_languages

    def _get_or_download_model(self, download: bool) -> Optional[str]:
        """
        Return downloaded model path, if model path does not exits and download is true, it will download
        and return the path
        Args:
            download: flag to decide whether to download model in case it not exists
        Returns:
            str: model path or None
        """
        home_dir = home_directory()
        downloaded_models_dir = os.path.join(home_dir, MODELS_DIR)

        if not os.path.exists(downloaded_models_dir):
            os.makedirs(downloaded_models_dir)

        model_hashed_name = get_hashed_name(self.embedding + self.model)
        model_path = os.path.join(downloaded_models_dir,  model_hashed_name)

        if not os.path.exists(model_path):
            if not download:
                return

            model_download_path = model_path + '.' + self.embedding_cls.EMBEDDING_MODELS[self.model].format
            model_download_url = self.embedding_cls.EMBEDDING_MODELS[self.model].download_url
            print(f"Model does not exists, Downloading model: {self.model}")
            download_from_url(model_download_url, model_download_path)
            extract_file(model_download_path, model_path)
            os.remove(model_download_path)
            print(f"Model downloaded successfully!")
        return model_path

    def _load_model(self):
        self.embedding_cls.load_model(self.model, self.model_path)
        return

    def encode(self, texts: Union[list, str], batch_size: int, pooling: str, **kwargs):
        if type(texts) == str:
            return self.embedding_cls.encode([texts], pooling, **kwargs)
        else:
            batch_opt = []
            batch_tem = []
            for i, text in enumerate(texts):
                batch_tem.append(self.embedding_cls.encode([text], pooling, **kwargs))
                if i % batch_size == 0:
                    batch_opt.append(batch_tem)
                    batch_tem = []
            batch_opt.append(batch_tem)
            return batch_opt
