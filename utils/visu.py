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

def plot_phase_amp_distribution(bins, pmf, ax=None):
    if ax is None:
        f, ax = plt.subplots(1, 1)
    b = ax.bar(bins, pmf, alpha=0.8, width=bins[1]-bins[0])
    ax.margins(x=0)
    ax.set_xticks(np.linspace(0, 2*np.pi, 5))
    ax.set_xticklabels(['0', r'$\frac{\pi}{2}$', r'$\pi$', r'$3\frac{\pi}{2}$', r'$2\pi$'])
    ax.grid(axis='x', which='both')
    return b

def plot_sig(t, signal_fast, env_fast, ax=None):
    if ax is None:
        f, ax = plt.subplots(1, 1)
    l = ax.plot(t, np.asarray([signal_fast, env_fast]).T)
    ax.margins(x=0)
    ax.set_xlim([0, 1.5])
    return l

    
def multiple_formatter(denominator=2, number=np.pi, latex='\\pi'):
    def gcd(a, b):
        while b:
            a, b = b, a%b
        return a
    def _multiple_formatter(x, pos):
        den = denominator
        num = int(np.rint(den*x/number))
        com = gcd(num,den)
        (num,den) = (int(num/com),int(den/com))
        if den==1:
            if num==0:
                return r'$0$'
            if num==1:
                return r'$%s$'%latex
            elif num==-1:
                return r'$-%s$'%latex
            else:
                return r'$%s%s$'%(num,latex)
        else:
            if num==1:
                return r'$\frac{%s}{%s}$'%(latex,den)
            elif num==-1:
                return r'$\frac{-%s}{%s}$'%(latex,den)
            else:
                return r'$\frac{%s%s}{%s}$'%(num,latex,den)
    return _multiple_formatter