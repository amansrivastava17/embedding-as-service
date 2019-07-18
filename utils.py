from typing import List
import sys
import zipfile
import hashlib
import requests
from pathlib import Path


def unzip(zip_path: str, target_path: str = '.') -> None:
    """
    Unzip file at zip_path to target_path
    Args:
        zip_path:
        target_path:
    Returns:

    """
    zip_ref = zipfile.ZipFile(zip_path, 'r')
    zip_ref.extractall(target_path)
    zip_ref.close()


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
                                 format('█' * done, '.' * (50-done), int(done * 2),
                                        int(downloaded/1024), total_kb))
                sys.stdout.flush()
    sys.stdout.write('\n')
