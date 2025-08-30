import numpy as np
from scipy.signal.windows import hann, gaussian
from scipy.signal import lfilter, butter, filtfilt

def simulate_background_eeg(alpha:float=0.7, duration: float=30., fs: int=100) -> np.ndarray:
    '''
    AR coefficient to start approximating 1/f over a range: alpha ~ 0.7 - 1.5
    '''
    n_samples = int(duration * fs)
    background_eeg = np.zeros(n_samples)
    # Generating AR(1) process
    white_noise = np.random.normal(size=n_samples)
    background_eeg = np.zeros(n_samples)
    for t in range(1, n_samples):
        background_eeg[t] = alpha * background_eeg[t-1] + white_noise[t]

    return background_eeg

def simulate_band_power(sampling_rate: int, duration: float, f_center: float, bandwidth: float) -> np.ndarray:
    N = int(sampling_rate * duration)
    # Generate white noise
    white_noise = np.random.normal(0, 1, N)
    frequencies = np.fft.fftfreq(N, d=1/sampling_rate)
    noise_spectrum = np.fft.fft(white_noise)
    # Gaussian filter in frequency domain
    gaussian_filter = np.exp(-0.5 * ((frequencies - f_center) / bandwidth) ** 2)
    filtered_spectrum = noise_spectrum * gaussian_filter
    random_signal = np.fft.ifft(filtered_spectrum).real
    return random_signal

def simulate_spike_events(n_samples: int, event_interval: int, spike_value: float = 1.0) -> np.ndarray:
    events = np.zeros(n_samples)
    events[::event_interval] = spike_value
    return events

def poisson_onsets_fixed_N(N, dur=1.0, seed=None):
    """Generate Poisson onsets with a fixed number of events.
    Instead of generating a Poisson process with a fixed rate, we generate a fixed number of events.

    Uniformly distribute them over [0,dur], which matches the order statistics of the Poisson process.

    .. seealso::
        :func:`poisson_onsets`, :func:`poisson_onsets_fixed_N`
    """
    rng = np.random.default_rng(seed)
    return np.sort(rng.uniform(0, dur, size=N))

def simulate_smooth_signal(n_samples: int, sample_rate: int, low_pass_freq: float = 12.0) -> np.ndarray:
    # Generate white noise
    noise = np.random.randn(n_samples)
    # Design a low-pass Butterworth filter
    b, a = butter(4, low_pass_freq / (sample_rate / 2), btype='low')
    smooth_signal = filtfilt(b, a, noise)
    return smooth_signal

def create_kernel(loc_peak: float = 0.1, spread: float = 0.05,
                  tmin: float = -0.1, tmax: float = 0.6,
                  sample_rate: int =100, bipolar: bool = False, normalise: bool=True) -> np.ndarray:
    t_kernel = np.arange(tmin, tmax, 1/sample_rate)
    kernel = np.exp(-0.5 * ((t_kernel - loc_peak) / spread) ** 2)
    
    # Optional: create a bipolar kernel by differentiating the single peak kernel
    if bipolar:
        kernel = np.gradient(kernel)
    
    if normalise:
        kernel /= np.sum(np.abs(kernel)**2)**0.5  # L2 norm
    return t_kernel, kernel

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from scipy.fft import rfft, rfftfreq

    fig, axs = plt.subplots(4, 2, figsize=(14, 8))

    # Example usage for background EEG:
    duration = 20  # Example number of samples
    sample_rate = 100  # 1000 Hz sample rate
    n_samples = int(duration * sample_rate)
    background_eeg = simulate_background_eeg(duration=duration, fs=sample_rate)
    axs[0, 0].plot(background_eeg)
    axs[0, 0].set_title("Simulated Background EEG")

    # Magnitude spectrum of background EEG
    spectrum_bg_eeg = np.abs(rfft(background_eeg))
    freqs_bg_eeg = rfftfreq(n_samples, d=1./sample_rate)
    axs[0, 1].plot(freqs_bg_eeg, spectrum_bg_eeg)
    axs[0, 1].set_title("Spectrum of Background EEG")

    # Example usage for event signals:
    event_interval = 100  # Event every 100 samples
    spike_events = simulate_spike_events(n_samples, event_interval)
    smooth_signal = simulate_smooth_signal(n_samples, sample_rate)
    kernel = create_kernel()
    convolved_events = np.convolve(spike_events, kernel, mode='same')
    axs[1, 0].plot(spike_events, label='Spike Events')
    axs[1, 0].plot(convolved_events, label='Convolved Events')
    axs[1, 0].set_title("Simulated Event Signals")
    axs[1, 0].legend()

    # Magnitude spectrum of convolved events
    spectrum_conv_events = np.abs(rfft(convolved_events))
    freqs_conv_events = rfftfreq(n_samples, d=1./sample_rate)
    axs[1, 1].plot(freqs_conv_events, spectrum_conv_events)
    axs[1, 1].set_title("Spectrum of Convolved Events")
    axs[2, 0].plot(smooth_signal)
    axs[2, 0].set_title("Simulated Smooth Signal")

    # Magnitude spectrum of smooth signal
    spectrum_smooth_signal = np.abs(rfft(smooth_signal))
    freqs_smooth_signal = rfftfreq(n_samples, d=1./sample_rate)
    axs[2, 1].plot(freqs_smooth_signal, spectrum_smooth_signal)
    axs[2, 1].set_title("Spectrum of Smooth Signal")
    axs[3, 0].plot(kernel)
    axs[3, 0].set_title("Kernel")

    # Magnitude spectrum of kernel
    spectrum_kernel = np.abs(rfft(kernel))
    freqs_kernel = rfftfreq(len(kernel), d=1./sample_rate)
    axs[3, 1].plot(freqs_kernel, spectrum_kernel)
    axs[3, 1].set_title("Spectrum of Kernel")

    plt.tight_layout()
    plt.show()
