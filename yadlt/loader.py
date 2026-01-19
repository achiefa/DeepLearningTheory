import logging
import os
import pathlib
import shutil
import sys
import tarfile
import tempfile

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

log = logging.getLogger(__name__)

MODULE_DIR = pathlib.Path(__file__).parent
FIT_FOLDER = (MODULE_DIR / "../Results/fits").resolve()


class LoaderError(Exception):
    pass


class RemoteLoaderError(Exception):
    pass


class FitNotFound(LoaderError):
    pass


class RemoteLoader:
    """
    Loader class for remote resources.
    """

    def __init__(self):
        local_fit_path = FIT_FOLDER
        local_fit_path.mkdir(parents=True, exist_ok=True)

        self._local_fit_path = local_fit_path
        self._remote_fit_path = "https://data.nnpdf.science/AMEDEO/fits_ntk/"
        self._remote_index = "ntkdata.json"

        # Timeout and entry logic
        self.session = requests.Session()
        retry = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def _remote_files_from_url(self, url, index):
        index_url = url + index
        try:
            resp = self.session.get(index_url)
            resp.raise_for_status()
        except Exception as e:
            raise RemoteLoaderError(
                f"Failed to fetch remote file index {index_url}: {e}"
            ) from e

        try:
            info = resp.json()["files"]
        except Exception as e:
            raise RemoteLoaderError(
                f"Malformed index {index_url}. Expecting json with a key 'files': {e}"
            ) from e

        return {file.split(".")[0]: url + file for file in info}

    def remote_files(self, urls, index):
        d = {}
        for url in urls:
            try:
                d.update(self._remote_files_from_url(url, index))
            except RemoteLoaderError as e:
                log.error(e)
        return d

    def remote_fits(self):
        rt = self.remote_files([self._remote_fit_path], self._remote_index)
        return {k: v for k, v in rt.items()}

    def download_remote_fit(self, fitname: str, force: bool = False):
        target = self._local_fit_path / fitname
        if target.exists() and not force:
            log.info(f"Fit {fitname} already exists at {target}")
            return target

        remote = self.remote_fits()
        if fitname not in remote:
            raise FitNotFound(f"Fit {fitname} not found in remote repository.")

        target_path = self._local_fit_path
        download_and_extract(remote[fitname], target_path)
        return target


def download_and_extract(url, local_path, target_name=None):
    """Download a compressed archive and then extract it to the given path"""
    local_path = pathlib.Path(local_path)
    if not local_path.is_dir():
        raise NotADirectoryError(local_path)
    name = url.split("/")[-1]
    archive_dest = tempfile.NamedTemporaryFile(
        delete=False, suffix=name, dir=local_path
    )
    with archive_dest as t:
        log.debug("Saving data to %s", t.name)
        download_file(url, t)
    log.info("Extracting archive to %s", local_path)
    try:
        with tarfile.open(archive_dest.name) as res_tar:
            # Extract to a temporary directory
            with tempfile.TemporaryDirectory(
                dir=local_path, suffix=name
            ) as folder_dest:
                dest_path = pathlib.Path(folder_dest)
                try:
                    res_tar.extractall(path=dest_path, filter="data")
                except TypeError as e:
                    if sys.version_info > (3, 9, 16):
                        # Filter was added in 3.9.17, and raises TypeError before then
                        # mac's default is still in 3.9.6, so fallback to the unsafe behaviour
                        raise e
                    res_tar.extractall(path=dest_path)
                except tarfile.LinkOutsideDestinationError as e:
                    if sys.version_info > (3, 11):
                        raise e
                    # For older versions of python ``filter=data`` might be too restrictive
                    # for the links inside the ``postfit`` folder if you are using more than one disk
                    res_tar.extractall(path=dest_path, filter="tar")

                # Check there are no more than one item in the top level
                top_level_stuff = list(dest_path.glob("*"))
                print(top_level_stuff)

                # Mac metadata files can sneak in, ignore them
                top_level_stuff = [
                    p for p in top_level_stuff if not p.name.startswith("._")
                ]

                if len(top_level_stuff) > 1:
                    raise RemoteLoaderError(
                        f"More than one item in the top level directory of {url}"
                    )

                if target_name is None:
                    target_path = local_path
                else:
                    target_path = local_path / target_name
                shutil.move(top_level_stuff[0], target_path)

    except Exception as e:
        log.error(
            f"The original archive at {t.name} was only extracted partially at \n{local_path}"
        )
        raise e
    else:
        os.unlink(archive_dest.name)


def download_file(url, stream_or_path, make_parents=False, delete_on_failure=True):
    """
    Download a file from a URL to a local path or stream.

    Args:
      url: URL of the remote resource
      stream_or_path: file-like object or pathlib.Path to write the data to
      make_parents: if True and stream_or_path is a Path, create parent directories if needed
      delete_on_failure: if True and stream_or_path is a Path, delete the file on failure
    """
    response = requests.get(url, stream=True)
    response.raise_for_status()

    if isinstance(stream_or_path, (str, bytes, os.PathLike)):
        # The tmp file has not been created yet
        p = pathlib.Path(stream_or_path)
        if p.is_dir():
            raise IsADirectoryError(p)
        log.info("Downloading %s to %s", url, stream_or_path)
        if make_parents:
            p.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile(
            delete=delete_on_failure, dir=p.parent, prefix=p.name, suffix=".part"
        ) as f:
            _download_and_show(response, f)
            shutil.move(f.name, p)
    else:
        log.info("Downloading %s.", url)
        _download_and_show(response, stream_or_path)


def _download_and_show(response, stream):
    """
    Download a remote resource showing a status bar.

    Args:
      response: requests.Response object
      stream: file-like object to write the downloaded data to
    """
    total_length = response.headers.get("content-length")

    if total_length is None or not log.isEnabledFor(logging.INFO):
        stream.write(response.content)
    else:
        dl = 0
        prev_percent = -1
        total_length = int(total_length)

        for data in response.iter_content(chunk_size=4096):
            dl += len(data)
            stream.write(data)

            percent = int(100 * dl / total_length)

            if sys.stdout.isatty():
                # Progress bar for terminal
                done = int(50 * dl / total_length)
                if done != prev_percent:  # Avoid redrawing same state
                    sys.stdout.write(f"\r[{'=' * done}{' '*(50 - done)}] {percent}%")
                    sys.stdout.flush()
                    prev_percent = done
            elif percent != prev_percent and percent % 10 == 0:
                # Log only every 10% to avoid flooding
                log.info(f"Download progress: {percent}%")
                prev_percent = percent

        if sys.stdout.isatty():
            sys.stdout.write("\n")
