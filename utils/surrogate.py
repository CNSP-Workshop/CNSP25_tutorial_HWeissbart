import numpy as np
from scipy.signal.windows import hann, gaussian
from scipy.signal import lfilter, butter, filtfilt

def simulate_background_eeg(alpha:float=0.9, duration: float=30., fs: int=100) -> np.ndarray:
    '''
    AR coefficient to start approximating 1/f over a range: alpha ~ 0.7 - 1.
    '''
    n_samples = int(duration * fs)
    background_eeg = np.zeros(n_samples)
    # Generating AR(1) process
    white_noise = np.random.normal(size=n_samples)
    background_eeg = np.zeros(n_samples)
    for t in range(1, n_samples):
        background_eeg[t] = alpha * background_eeg[t-1] + white_noise[t]

    return background_eeg

def simulate_band_power(sampling_rate: int=100, duration: float=30., f_center: float=10., bandwidth: float=2.) -> np.ndarray:
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

def simulate_random_events(duration=30, fs=100, rate=1, seed=None) -> np.ndarray:
    """Simulate spike events as a binary array with spikes occurring at a specified rate.
    
    Rate is in Hz, duration in seconds, fs is the sampling frequency in Hz.
    Rate is an average rate, the actual number of spikes will vary due to the Poisson process.
    """
    rng = np.random.default_rng(seed)
    n_samples = int(duration * fs)
    expected_n_spikes = int(rate * duration)
    spike_times = np.sort(rng.uniform(0, duration, size=expected_n_spikes))
    spike_indices = (spike_times * fs).astype(int)
    spike_indices = spike_indices[spike_indices < n_samples]  # Ensure indices are within bounds
    spikes = np.zeros(n_samples)
    spikes[spike_indices] = 1
    return spikes

def simulate_regular_events(duration=30, fs=100, interval=1, jitter=0.05, seed=None) -> np.ndarray:
    """Simulate spike events occurring at regular intervals with optional jitter.
    
    Interval is in seconds, duration in seconds, fs is the sampling frequency in Hz.
    Jitter is the maximum deviation from the regular interval in seconds.
    """
    rng = np.random.default_rng(seed)
    n_samples = int(duration * fs)
    n_intervals = int(duration / interval)
    spike_times = np.arange(0, n_intervals * interval, interval)
    if jitter > 0:
        jitter_values = rng.uniform(-jitter, jitter, size=spike_times.shape)
        spike_times += jitter_values
    spike_times = spike_times[(spike_times >= 0) & (spike_times < duration)]
    spike_indices = (spike_times * fs).astype(int)
    spikes = np.zeros(n_samples)
    spikes[spike_indices] = 1
    return spikes


def simulate_smooth_signal(n_samples: int, sample_rate: int, low_pass_freq: float = 12.0) -> np.ndarray:
    # Generate white noise
    noise = np.random.randn(n_samples)
    # Design a low-pass Butterworth filter
    b, a = butter(4, low_pass_freq / (sample_rate / 2), btype='low')
    smooth_signal = filtfilt(b, a, noise)
    return smooth_signal

def create_kernel(loc: float = 0.1, spread: float = 0.05,
                  tmin: float = -0.2, tmax: float = 0.6,
                  sample_rate: int =100, bipolar: bool = False, normalise: bool=True) -> np.ndarray:
    t_kernel = np.arange(tmin, tmax, 1/sample_rate)
    kernel = np.exp(-0.5 * ((t_kernel - loc) / spread) ** 2)
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
    spike_events = simulate_regular_events(n_samples, event_interval)
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
