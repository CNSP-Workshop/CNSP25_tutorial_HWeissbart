from tqdm.auto import tqdm
import urllib.request
import shutil
import h5py
import numpy as np
from scipy.stats import zscore
import os.path as op

STORY_IDs = {
    11: 'Anderson_S01_P01_normalized',
    12: 'Anderson_S01_P02_normalized',
    13: 'Anderson_S01_P03_normalized',
    14: 'Anderson_S01_P04_normalized',
    21: 'grimm_23_1_normalized',
    22: 'grimm_23_2_normalized',
    23: 'grimm_23_3_normalized',
    31: 'grimm_20_1_normalized',
    32: 'grimm_20_2_normalized',
    41: 'EAUV_part1_normalized',
    42: 'EAUV_part2_normalized',
    51: 'ANGE_part1_normalized',
    61: 'BALL_part1_normalized',
}
storynames = [s for s in STORY_IDs.values()]

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

def read_h5_data(feat_type='acoustic',
                 datadir='../data/stim', fname='predictors.hdf5',
                 stories=storynames):
    """
    Read, for a given sampling rate and if available, the stimulus representation:
    acoustic (continuous) or word-level data for all story parts.

    Parameters
    ----------
    feat_type : str, optional
        Type of stimulus representation. The default is 'acoustic'.
    datadir : str, optional
        Path to folder containing h5py file.
    fname : str, optional
        filename. The default is 'predictors.hdf5'.

    Returns
    -------
    out : list<ndarray>
        Numpy array of time aligned features for each story parts.

    """
    srate=100
    transcript_key='transcripts_v2'
    out = []
    with h5py.File(op.join(datadir, fname), 'r') as f:
        try:
            for s in stories:
                out.append(f[transcript_key][str(srate)][feat_type][s][()])
        except KeyError:
            raise KeyError(f"The required sampling rate ({srate}), feature type ({feat_type}) or {transcript_key} transcript family is not available")
    return out

def get_feature_signal(feats=['acoustic', 'wordonsets'],
                       stories=storynames, normalise='all', verbose=True):
    """
    Extract the final design matrix from features data (as stored in HDF5 file).

    Parameters
    ----------
    feats : list
        Can contain any of ['acoustic', 'wordonsets', 'wordfrequency', 'surprisal', 'entropy',
        'kl', 'PE', 'depth', 'close', 'open']
    stories : list of str
        Which stories to extract
    normalise : None | str
        None, 'all' or 'story'. Whether to normalise per story or across all stories jointly or not at all.

    Returns
    -------
    X : list of ndarray
    """
    import numpy as np
    wl_feats_id = {f:k for f,k in zip(
        ['wordonsets', 'surprisal', 'entropy',
        'kl', 'PE', 'wordfrequency', 'depth', 'close', 'open'], range(9)
        )
    }
    srate=100
    transcripts='transcripts_v2'
    assert normalise in ['all', 'story', None], "Normalisation must per story ('story') or across all stories ('all')"
    if verbose: print("Load stimulus features")
    envs = read_h5_data(feat_type='acoustic', stories=stories)
    wordlevels = read_h5_data(feat_type='wordlevel', stories=stories)
    if verbose: print(f"Loading feature signal for : {feats}")
    X = []
    for k, s in enumerate(stories):
        x_ = []
        for f in feats:
            if f == 'acoustic':
                x_.append(zscore(envs[k]))
            else:
                x_.append(wordlevels[k][:, wl_feats_id[f]])
        X.append(np.vstack(x_).T)
    if normalise is not None:
        # normalise feature to unit variance (so we do not remove the mean here...)
        var = np.var(np.vstack(X), 0)
        for x in X:
            if normalise=='story':
                x /= np.sqrt(np.var(x, 0)) # per story
            elif normalise=='all':
                x /= np.sqrt(var) # across all stories
    if verbose: print(f"Done. X shape: {X[0].shape}")
    return X

def normalize_complex(z):
    return z/np.abs(z)

def check_complex(amp, phase):
    if any(np.iscomplex(amp)):                                                                                                                              
        amp = abs(amp)                                                                                                                                      
    if any(np.iscomplex(phase)):                                                                                                                            
        phase = np.angle(phase)  
    return amp, phase
