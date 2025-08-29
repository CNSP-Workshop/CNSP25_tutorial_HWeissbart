import numpy as np
from scipy.signal import lfilter, hann
from scipy.signal import butter, filtfilt, gaussian

def simulate_background_eeg(n_samples: int, sample_rate: int, ar_order: int = 2, alpha_freq: tuple = (8, 12)) -> np.ndarray:
    # Simulate 1/f noise using AR model
    ar_params = np.random.rand(ar_order) - 0.5  # Random AR coefficients
    noise = np.random.randn(n_samples)
    background_eeg = lfilter([1], np.r_[1, -ar_params], noise)
    
    # Add Gaussian noise
    gaussian_noise = np.random.randn(n_samples)
    background_eeg += gaussian_noise
    
    # Add alpha component (8-12Hz)
    t = np.arange(n_samples) / sample_rate
    alpha_wave = np.sin(2 * np.pi * np.random.uniform(*alpha_freq) * t) * np.random.rand(n_samples)
    background_eeg += alpha_wave
    
    return background_eeg

# Example usage:
n_samples = 1000  # Example number of samples
sample_rate = 1000  # 1000 Hz sample rate
background_eeg = simulate_background_eeg(n_samples, sample_rate)

def simulate_spike_events(n_samples: int, event_interval: int, spike_value: float = 1.0) -> np.ndarray:
    events = np.zeros(n_samples)
    events[::event_interval] = spike_value
    return events

def simulate_smooth_signal(n_samples: int, sample_rate: int, low_pass_freq: float = 12.0) -> np.ndarray:
    # Generate white noise
    noise = np.random.randn(n_samples)

    # Design a low-pass Butterworth filter
    b, a = butter(4, low_pass_freq / (sample_rate / 2), btype='low')
    smooth_signal = filtfilt(b, a, noise)
    
    return smooth_signal

def create_kernel(sample_rate: int, peak_time: int = 100, bipolar: bool = False) -> np.ndarray:
    kernel_length = int(0.8 * sample_rate)  # 800 ms at the given sample rate
    kernel = hann(kernel_length)
    
    # Optional: create a bipolar kernel by differentiating the single peak kernel
    if bipolar:
        kernel = np.gradient(kernel)
    
    # Align peak
    kernel = np.roll(kernel, peak_time - kernel_length // 2)
    
    return kernel

# Example usage for event signals:
event_interval = 100  # Event every 100 samples
spike_events = simulate_spike_events(n_samples, event_interval)

smooth_signal = simulate_smooth_signal(n_samples, sample_rate)

kernel = create_kernel(sample_rate)