from typing import Union, Optional, List
import numpy as np
import importlib
import os

from embedding_as_service.utils import home_directory, get_hashed_name, download_from_url, extract_file
from embedding_as_service.text import MODELS_DIR


class Encoder(object):
    def __init__(self, embedding: str, model: str, download: bool = False):
        self.embedding = embedding
        self.model = model
        self.embedding_model_dict = None
        self.model_path = None
        self.batch_size = 128

        supported_embeddings = self.get_supported_embeddings()

        # check if embedding exits
        if embedding not in supported_embeddings:
            raise ValueError(f"Given embedding \"{embedding}\" is not supported, use from available embeddings:\n"
                             f"{supported_embeddings}")

        self.embedding_cls = importlib.import_module(
            f'embedding_as_service.text.{embedding}').Embeddings()

        # check if given model exits for embedding
        model_names = list(self.embedding_cls.EMBEDDING_MODELS.keys())
        if model not in model_names:
            raise ValueError(f"Given embedding \"{embedding}\" does not have support for model \"{model}\", "
                             f"the supported models are: {model_names}")

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
        Return downloaded model path, if model path does not exist and download is true, it will download
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
        model_path = os.path.join(downloaded_models_dir, model_hashed_name)

        if not os.path.exists(model_path):
            if not download:
                return

            model_download_path = model_path + '.' + self.embedding_cls.EMBEDDING_MODELS[self.model].format
            model_download_url = self.embedding_cls.EMBEDDING_MODELS[self.model].download_url
            print(f"Model does not exists, Downloading model: {self.model}")
            download_from_url(model_download_url, model_download_path)
            extract_file(model_download_path, model_path)
            if os.path.exists(model_download_path):
                os.remove(model_download_path)
            print(f"Model downloaded successfully!")
        return model_path

    def _load_model(self):
        self.embedding_cls.load_model(self.model, self.model_path)
        return

    def tokenize(self, texts: Union[List[str], str]) -> np.array:
        if isinstance(texts, str):
            tokens = self.embedding_cls.tokenize(texts)
        elif isinstance(texts, list):
            tokens = []
            for i in range(0, len(texts)):
                tokens.append(self.embedding_cls.tokenize(texts[i]))
        else:
            raise ValueError('Wrong input format!')
        return tokens

    def encode(self,
               texts: Union[List[str], List[List[str]]],
               pooling: Optional[str] = None,
               max_seq_length: Optional[int] = 128,
               is_tokenized=False,
               max_batch_size=128,
               ** kwargs
        ) -> np.array:
        if not isinstance(texts, list):
            raise ValueError('input expected to be list of str')
        if is_tokenized:
            if False in [isinstance(text, list) for text in texts]:
                raise ValueError('Input expected to be list of tokens for `is_tokenized` = True')
        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            vectors = self.embedding_cls.encode(text=texts[i: i + self.batch_size],
                                                pooling=pooling,
                                                max_seq_length=max_seq_length,
                                                is_tokenized=is_tokenized)
            embeddings.append(vectors)
        embeddings = np.vstack(embeddings)

        return embeddings
