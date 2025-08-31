"""
Set of helper functions for the CNSP25 tutorial.

Incldudes functions for:
- Lowpass/highpass/... filtering
- Convolving with a kernel (taking care of edge effects)
- SVD-based TRF estimation
- Conjugate gradient solver for ridge regression
"""
import numpy as np
from scipy.signal import convolve, butter, filtfilt
from scipy.linalg import svd
from typing import Tuple
from numpy.typing import NDArray
from scipy.sparse.linalg import LinearOperator, cg
from scipy.fft import rfft, rfftfreq
from typing import Union, List
from warnings import warn
from functools import reduce
from tqdm.auto import tqdm

def lowpass_filter(data, cutoff, fs, order=4):
    # Apply a low-pass Butterworth filter
    b, a = butter(order, cutoff / (fs / 2), btype='low')
    filtered_data = filtfilt(b, a, data)
    return filtered_data

def highpass_filter(data, cutoff, fs, order=4):
    # Apply a high-pass Butterworth filter
    b, a = butter(order, cutoff / (fs / 2), btype='high')
    filtered_data = filtfilt(b, a, data)
    return filtered_data

def convolve_with_kernel(data, kernel, tmin, tmax, fs) -> NDArray:
    # Convolve data with the given kernel
    pre = int(np.abs(tmin) * fs)
    post = int(np.abs(tmax) * fs)
    if pre < post:
        kernel = np.pad(kernel, (post-pre, 0), 'constant')
    else:
        kernel = np.pad(kernel, (0, pre - post), 'constant')
    return convolve(data, kernel, mode='same')

def lag_span(tmin, tmax, srate=125):
    """Create an array of lags spanning the time window [tmin, tmax].

    Parameters
    ----------
    tmin : float
        In seconds
    tmax : float
    srate : float
        Sampling rate

    Returns
    -------
    lags : 1d array
        Array of lags in _samples_

    """
    sample_min, sample_max = int(np.ceil(tmin * srate)), int(np.ceil(tmax * srate))
    return np.arange(sample_min, sample_max)

def lag_matrix(x, lags=(0,1), mode='full', fill_value=0., **kwargs):
    """Helper function to create a Toeplitz matrix of lagged time series.

    The lag can be arbitrarily spaced. Check other functions to create series of lags
    whether they are contiguous or sparsely spanning a time window :func:`lag_span` and
    :func:`lag_sparse`.

    Parameters
    ----------
    x : ndarray (nsamples x nfeats)
        Multivariate data
    lags : list
        Shift in _samples_ to be applied to data. Negative shifts are lagged in the past,
        positive shits in the future, and a shift of 0 represents the data array as it is
        in the input `data`.
    fill_value : float
        What value to use to fill entries which are not defined (Default: NaN).
    mode : str
       'valid' or 'full' (default: 'valid').
       'valid' returns only the part of the lagged matrix that is valid (i.e. no NaN values).
       'full' returns the full lagged matrix, including missing values, which are filled with `fill_value`.
    **kwargs : keyword arguments
        Additional arguments to be passed to the function for backward compatibility.
        For example, `filling` and `drop_missing` are deprecated and will be removed in future versions.

    Returns
    -------
    lagged : ndarray (nsamples_new x nfeats*len(lag_samples))
        Matrix of lagged time series.

    Raises
    ------
    ValueError
        If ``mode`` is not 'valid' or 'full'.

    Example
    -------
    >>> data = np.asarray([[1,2,3,4,5,6],[7,8,9,10,11,12]]).T
    >>> out = lag_matrix(data, (-1, 0, 2), mode='full')
    >>> out # doctest: +NORMALIZE_WHITESPACE
    array([[ 2,  1,  0,  8,  7,  0],
           [ 3,  2,  0,  9,  8,  0],
           [ 4,  3,  1, 10,  9,  7],
           [ 5,  4,  2, 11, 10,  8],
           [ 6,  5,  3, 12, 11,  9],
           [ 0,  6,  4,  0, 12, 10]])
    """
    if 'filling' in kwargs:
        fill_value = kwargs['filling']
    if 'drop_missing' in kwargs:
        if kwargs['drop_missing']:
            mode = 'valid'
        else:
            mode = 'full'
    if 'lag_samples' in kwargs:
        lags = kwargs['lag_samples']
            
    x = np.atleast_2d(np.asarray(x))
    if x.shape[0] == 1:
        x = x.T
    if x.shape[1] > 1:
        return np.concatenate([lag_matrix(x[:, i], lags, mode=mode, fill_value=fill_value) for i in range(x.shape[1])], axis=1)
    x = x.squeeze()
    n = len(x)
    lags = -np.asarray(lags) #TODO: minus signs to match expected behavior from TRFEstimator: maybe change in future
    min_lag, max_lag = lags.min(), lags.max()

    # Always compute full matrix first
    X_full = np.full((n, len(lags)), fill_value, dtype=x.dtype)

    for i, lag in enumerate(lags):
        if lag < 0:
            X_full[-lag:, i] = x[:n + lag]
        else:
            X_full[:n - lag, i] = x[lag:]

    if mode == 'full':
        return X_full
    elif mode == 'valid':
        start = max(0, -min_lag)
        end = n - max_lag
        return X_full[start:end]
    else:
        raise ValueError("mode must be 'valid' or 'full'")

def conjugate_gradient_solver(A, b, ridge=0.0):
    # Solve for x in the ridge regression problem using a conjugate gradient solver
    n_features = A.shape[1]
    def matvec(x):
        return A.T @ (A @ x) + ridge * x
    A_linop = LinearOperator((n_features, n_features), matvec=matvec)
    x, info = cg(A_linop, A.T @ b)
    if info != 0:
        print("Warning: Conjugate gradient solver did not converge.")
    return x

def svd_solver(X: NDArray, Y: NDArray, alpha: float=0.0, var_explained=None) -> NDArray:
    """SVD-based solver for ridge regression.

    This function is here for educational purposes, prefer using `_svd_regress` instead.

    Parameters
    ----------
    X : ndarray (nsamples, nfeats)
        Design matrix
    Y : ndarray (nsamples, nchans)
        Response matrix
    alpha : float
        Regularization parameter

    Returns
    -------
    betas : ndarray (nfeats, nchans)
        Coefficients

    """
    # Compute SVD of X, this is the most expensive step
    # U, s, Vt = svd(X, full_matrices=False)
    # S_inv = np.diag(s / (s**2 + alpha))
    # betas = Vt.T @ S_inv @ U.T @ Y

    # Instead, one can compute the SVD of X.T @ X which is smaller if nsamples > nfeats
    # but this is less stable numerically
    XtX = X.T @ X
    U, s, Vt = svd(XtX, full_matrices=False)
    # Regularization in the SVD space, removing components according to var_explained if given
    if var_explained is not None:
        s = s[s.cumsum() / s.sum() < var_explained]
        U = U[:, :len(s)]
        Vt = Vt[:len(s), :]
        S_inv = np.diag(1/s)
    # Regularization in the SVD space but keeping all components and shrinking them
    # according to ridge parameter alpha
    else:
        S_inv = np.diag(1 / (s + alpha))
    betas = U @ S_inv @ U.T @ X.T @ Y
    return betas

def _svd_regress(x: Union[np.ndarray, List[np.ndarray]],
                 y: Union[np.ndarray, List[np.ndarray]],
                 alpha: Union[float, np.ndarray],
                 verbose: bool = False) -> np.ndarray:
    """Linear regression using svd.

    Parameters
    ----------
    x : ndarray (nsamples, nfeats) or list of such
        If a list of such is given (with possibly different nsamples), covariance matrices
        will be computed by accumulating them for each trials. The number of samples must then be the same
        in both x and y per each trial.
    y : ndarray (nsamples, nchans) or list of such
        If a list of such arrays is given, each element of the
        list is treated as an individual subject, the resulting `betas` coefficients
        are thus computed on the averaged covariance matrices.
    alpha : float or array-like
        If array, will compute betas for every regularisation parameters at once

    Returns
    -------
    betas : ndarray (nfeats, nchans, len(alpha))
        Coefficients

    Raises
    ------
    ValueError
        If alpha < 0 (coefficient of L2 - regularization)
    AssertionError
        If trial length for each x and y differ.

    Notes
    -----
    A warning is shown in the case where nfeats > nsamples, if so the user
    should rather use partial regression.
    """
    # cast alpha in ndarray
    if np.isscalar(alpha):
        alpha = np.asarray([alpha], dtype=float)
    else:
        alpha = np.asarray(alpha)

    if not isinstance(x, list) and np.ndim(x) == 2:
        if x.shape[0] < x.shape[1]:
            warn("Less samples than features! The linear problem is not stable in that form. Consider using partial regression instead.")

    try:
        assert np.all(alpha >= 0), "Alpha must be positive"
    except AssertionError:
        raise ValueError

    if (len(x) == len(y)) and np.ndim(x[0])==2: # will accumulate covariances
        assert all([xtr.shape[0] == ytr.shape[0] for xtr, ytr in zip(x, y)]), "Inconsistent trial lengths!"
        XtX = reduce(lambda x, y: x + y, [xx.T @ xx for xx in x])
        [U, s, V] = np.linalg.svd(XtX, full_matrices=False) # here V = U.T
        XtY = np.zeros((XtX.shape[0], y[0].shape[1]), dtype=y[0].dtype)
        count = 1
        if verbose:
            pbar = tqdm(total=len(x), leave=False, desc='Covariance accumulation')
        for X, Y in zip(x, y):
            if verbose:
                print("Accumulating segment %d/%d", count, len(x))
                pbar.update()
            XtY += X.T @ Y
            count += 1
        if verbose: pbar.close()
        
        #betas = U @ np.diag(1/(s + alpha)) @ U.T @ XtY
        eigenvals_scaled = np.zeros((*V.shape, np.size(alpha)))
        eigenvals_scaled[range(len(V)), range(len(V)), :] = 1 / \
            (np.repeat(s[:, None], np.size(alpha), axis=1) + np.repeat(alpha[:, None].T, len(s), axis=0))
        Vsreg = np.dot(V.T, eigenvals_scaled) # np.diag(1/(s + alpha))
        betas = np.einsum('...jk, jl -> ...lk', Vsreg, U.T @ XtY) #Vsreg @ Ut
    else:
        [U, s, V] = np.linalg.svd(x, full_matrices=False)
        if np.ndim(y) == 3:
            Uty = np.zeros((U.shape[1], y.shape[2]))
            for Y in y:
                Uty += U.T @ Y
            Uty /= len(y)
        else:
            Uty = U.T @ y

        # Broadcast all alphas (regularization param) in a 3D matrix,
        # each slice being a diagonal matrix of s/(s**2+lambda)
        eigenvals_scaled = np.zeros((*V.shape, np.size(alpha)))
        eigenvals_scaled[range(len(V)), range(len(V)), :] = np.repeat(s[:, None], np.size(alpha), axis=1) / \
            (np.repeat(s[:, None]**2, np.size(alpha), axis=1) + np.repeat(alpha[:, None].T, len(s), axis=0))
        # A dot product instead of matmul allows to repeat multiplication alike across third dimension (alphas)
        Vsreg = np.dot(V.T, eigenvals_scaled) # np.diag(s/(s**2 + alpha))
        # Using einsum to control which access get multiplied, again leaving alpha's dimension "untouched"
        betas = np.einsum('...jk, jl -> ...lk', Vsreg, Uty) #Vsreg @ Uty
    
    return betas