from tqdm.auto import tqdm
import urllib.request
import shutil
import h5py
import os.path as op

def download_file(url, save_path, chunk_size=1024):
    try:
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
    except urllib.error.URLError as e:
        print(f"❌ Download failed, network error occurred: {e}")
def list_h5_data(fullpath='../data/stim/predictors.hdf5', max_depth=None):
    """
    Describe the content of the h5 data file recursively up to a specified depth.

    Parameters
    ----------
    fullpath : Path-like (str), optional
        Full path to the HDF5 file.
    max_depth : int or None, optional
        Maximum depth to list. If None, list all levels.

    """
    def show_hierarchy(member, obj):
        branch = '│   '
        tee = '├── '
        basename = op.basename(member)
        depth = len(member.split('/'))
        
        if max_depth is None or depth <= max_depth:
            if basename == member:
                print(member)
            else:
                if depth <= 2:
                    print(tee + basename)
                else:
                    if isinstance(obj, h5py.Dataset):
                        print(branch * (depth - 2) + tee + basename, obj.shape)
                    else:
                        print(branch * (depth - 2) + tee + basename)

    with h5py.File(fullpath, 'r') as f:
        f.visititems(show_hierarchy)
