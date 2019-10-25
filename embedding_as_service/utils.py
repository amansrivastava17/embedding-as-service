from typing import List
import sys
import zipfile
import hashlib
import requests
from pathlib import Path
import tarfile
import numpy as np
import os
import shutil


def any2unicode(text, encoding='utf8', errors='strict'):
    """Convert `text` (bytestring in given encoding or unicode) to unicode.
    Parameters
    ----------
    text : str
        Input text.
    errors : str, optional
        Error handling behaviour if `text` is a bytestring.
    encoding : str, optional
        Encoding of `text` if it is a bytestring.
    Returns
    -------
    str
        Unicode version of `text`.
    """
    if isinstance(text, str):
        return text
    return text.decode('utf-8')


to_unicode = any2unicode


def extract_file(zip_path: str, target_path: str = '.') -> None:
    """
    Unzip file at zip_path to target_path
    Args:
        zip_path:
        target_path:
    Returns:

    """
    if zip_path.endswith('.gz') and not zip_path.endswith('.tar.gz'):
        os.mkdir(target_path)
        shutil.move(zip_path, os.path.join(target_path, zip_path.split('.gz')[0]))
        return

    if zip_path.endswith('.zip'):
        opener, mode = zipfile.ZipFile, 'r'
    elif zip_path.endswith('.tar.gz') or zip_path.endswith('.tgz'):
        opener, mode = tarfile.open, 'r:gz'
    elif zip_path.endswith('.tar.bz2') or zip_path.endswith('.tbz'):
        opener, mode = tarfile.open, 'r:bz2'
    else:
        raise(ValueError, f"Could not extract `{zip_path}` as no appropriate extractor is found")

    with opener(zip_path, mode) as zipObj:
        zipObj.extractall(target_path)


def tokenizer(text: str, language: str) -> List[str]:
    """
    Download file/data from given url to download path
    Args:
        text: text
        language: language
    Returns:
        (list)
    """
    if language == 'en':
        return [x.strip() for x in text.split() if x]
    print(f"{language} is not yet supported")
    return []


def home_directory() -> str:
    """
    Return home directory path
    Returns:
        str: home path
    """
    return str(Path.home())


def get_hashed_name(name: str) -> str:
    """
   Generate hashed name
   Args:
       name: string to be hashed
   Returns:
      str: hashed string
   """
    return hashlib.sha224(name.encode('utf-8')).hexdigest()


def download_from_url(url: str, download_path: str) -> None:
    """
    Download file/data from given url to download path
    Args:
        url:
        download_path:
    Returns:
        None
    """
    with open(download_path, 'wb') as f:
        response = requests.get(url, stream=True)
        total = response.headers.get('content-length')

        if total is None:
            f.write(response.content)
        else:
            downloaded = 0
            total = int(total)
            total_kb = int(total/1024)
            for data in response.iter_content(chunk_size=max(int(total/1000), 1024*1024)):
                downloaded += len(data)
                f.write(data)
                done = int(50*downloaded/total)
                sys.stdout.write('\r[{}{}] {} % [{}/{} kb]'.
                                 format('|' * done, '.' * (50-done), int(done * 2),
                                        int(downloaded/1024), total_kb))
                sys.stdout.flush()
    sys.stdout.write('\n')


def reduce_mean_max(vectors: np.ndarray):
    return np.hstack(np.mean(vectors, 0), np.max(vectors, 0))


def np_first(vectors: np.ndarray):
    return np.array(vectors)[0]


def np_last(vectors: np.ndarray):
    return np.array(vectors)[-1]


POOL_FUNC_MAP = {
    "reduce_mean": np.mean,
    "reduce_max": np.max,
    "reduce_min": np.min,
    "reduce_mean_max": reduce_mean_max,
    "first_token": np_first,
    "last_token": np_last
}


class ArgSingleton(type):
    """ This is a Singleton metaclass. All classes affected by this metaclass
    have the property that only one instance is created for each set of arguments
    passed to the class constructor."""

    def __init__(cls, name, bases, dict):
        super(ArgSingleton, cls).__init__(cls, bases, dict)
        cls._instanceDict = {}

    def __call__(cls, *args, **kwargs):
        argdict = {'args': args}
        argdict.update(kwargs)
        argset = frozenset(sorted(argdict.items()))
        if argset not in cls._instanceDict:
            cls._instanceDict[argset] = super(ArgSingleton, cls).__call__(*args, **kwargs)
        return cls._instanceDict[argset]