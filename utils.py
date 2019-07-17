from typing import List
import urllib
import sys
import time
import zipfile


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


def reporthook(count: int, block_size: int, total_size: int) -> None:
    """
    ProgressBar
    Args:
        count:
        block_size:
        total_size:
    Returns:
        None
    """
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration))
    percent = int(count * block_size * 100 / total_size)
    sys.stdout.write("\r...%d%%, %d MB, %d KB/s, %d seconds passed" %
                     (percent, progress_size / (1024 * 1024), speed, duration))
    sys.stdout.flush()


def download_from_url(url: str, download_path: str) -> None:
    """
    Download file/data from given url to download path
    Args:
        url: url
        download_path: download path
    Returns:
        None
    """
    urllib.request.urlretrieve(url, download_path, reporthook)


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
