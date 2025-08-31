import numpy as np
from scipy.fft import rfft, rfftfreq
import matplotlib.pyplot as plt

def plot_fft(signal, fs, ax=None, scale='amplitude'):
    """
    Plot the FFT of a signal.

    scale: 'amplitude', 'psd' (power spectral density, i.e. amp**2) or 'dB' (decibels, i.e. 20*log10(amplitude))
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    N = len(signal)
    yf = rfft(signal)
    xf = rfftfreq(N, 1 / fs)
    amp = np.abs(yf)
    if scale in ['dB', 'db']:
        amp = 20 * np.log10(amp)
        amp = amp - 20 * np.log10(N/2)  # Normalization 
        # DC component should not be doubled:
        amp[0] = 20 * np.log10(np.abs(yf[0])) - 20 * np.log10(N)
        ylabel = 'Power (dB)'
    elif scale in ['psd', 'power']:
        amp[0] /= 2
        amp = (2.0/N) * amp**2 / fs # with normalization
        ylabel = 'Power/Frequency (V**2/Hz)'
    else:
        amp = (2.0/N) * amp 
        ylabel = 'Amplitude'
        amp[0] /= 2  # DC component should not be doubled
    ax.plot(xf, amp)
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_title('FFT of the signal')
    ax.grid()
    return ax.figure