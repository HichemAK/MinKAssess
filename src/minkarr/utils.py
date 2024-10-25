from io import BytesIO
import json
import zipfile

import requests
from tqdm import tqdm
from urllib3 import Retry
from requests.adapters import HTTPAdapter


def load_json(path) -> dict:
    with open(path, 'r') as f:
        return json.load(f)
    
class DownloadError(Exception):
    """Custom exception for download errors."""
    pass
    
def download_and_unzip(url, extract_to, retries=3, backoff_factor=0.3):
    def is_zip_file(response, url):
        """
        Determine if the response is a zip file either by the Content-Disposition header
        or by the file extension in the URL.
        """
        content_disposition = response.headers.get('Content-Disposition', '')
        if 'zip' in content_disposition:
            return True
        if url.lower().endswith('.zip'):
            return True
        return False

    # Create a session object
    session = requests.Session()
    # Define the retry parameters
    retry_strategy = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=[500, 502, 503, 504],
        # method_whitelist=["HEAD", "GET", "OPTIONS"]
    )
    # Mount it for both http and https usage
    session.mount('http://', HTTPAdapter(max_retries=retry_strategy))
    session.mount('https://', HTTPAdapter(max_retries=retry_strategy))

    try:
        # Send a HTTP request to the URL
        print(f"Downloading file from {url}")
        with session.get(url, stream=True) as response:
            # Raise an exception if the download failed
            response.raise_for_status()
            
            # Check if the response content is a zip file
            if is_zip_file(response, url):
                # Create a BytesIO object to hold the chunks of data
                zip_file_bytes = BytesIO()
                total_size = int(response.headers.get('content-length', 0))
                with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as progress_bar:
                    for chunk in response.iter_content(chunk_size=8192):
                        # Filter out keep-alive new chunks
                        if chunk:
                            zip_file_bytes.write(chunk)
                            progress_bar.update(len(chunk))
                
                # Extract the zip file
                zip_file_bytes.seek(0)  # Move to the beginning of the BytesIO object
                with zipfile.ZipFile(zip_file_bytes, 'r') as zip_ref:
                    # Extract the zip file to the specified directory
                    print(f"Extracting files to '{extract_to}' folder")
                    zip_ref.extractall(extract_to)
                    print("Extraction complete.")
            else:
                raise DownloadError("The URL does not contain a zip file.")
    except requests.exceptions.HTTPError as http_err:
        raise DownloadError(f"HTTP error occurred: {http_err}") from http_err
    except Exception as err:
        raise DownloadError(f"An error occurred: {err}") from err
    finally:
        # Close the session
        session.close()