from tqdm.auto import tqdm
import urllib.request
import shutil

def download_file(url, save_path, chunk_size=1024):
    with urllib.request.urlopen(url) as response, open(save_path, 'wb') as out_file:
        file_size = int(response.getheader('Content-Length', 0))
        downloaded = 0

        if file_size > 0:
            with tqdm(total=file_size, unit='B', unit_scale=True, desc=save_path) as pbar:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        print(f"✅ File downloaded: {save_path}")
                        break
                    out_file.write(chunk)
                    downloaded += len(chunk)
                    pbar.update(len(chunk))
        else:
            # If file size is not known, download without progress bar
            shutil.copyfileobj(response, out_file)
            print(f"✅ File downloaded: {save_path}")
