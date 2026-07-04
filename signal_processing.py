"""
Shared Signal Processing Module for the P300 BCI Speller.

This module is the "brain" of the data processing pipeline. It defines how we 
clean the raw electrical signals from the brain (EEG) and how we cut them 
into small pieces (epochs) that the AI can understand.

Fundamental Concepts:
1. FS (Sampling Rate): How many snapshots of brain activity we take per second.
2. Filtering: Removing noise like electricity from the walls (50Hz) or muscle movement.
3. Epoching: Cutting the continuous EEG stream into windows locked to a "flash" event.
4. Baseline Correction: Centering the signal so it starts at zero, removing DC offset.
"""

import os
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

# ============================================================================
# Paths & Environment
# ============================================================================
# Automatically find where this file is located so we can find the data folders.
_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_DATA_DIR = os.path.join(_BACKEND_DIR, "training_data")

# ============================================================================
# Hardware Constants — g.tec Unicorn Hybrid Black
# ============================================================================
FS = 250                            # The Unicorn headset sends 250 samples every second.
EPOCH_LEN = 0.8                     # We look at 800ms of data after every flash to find the P300.
                                    # 800ms is enough to see the "Peak" which usually happens at 300-500ms.
SAMPLES_PER_EPOCH = int(FS * EPOCH_LEN)  # 0.8 seconds * 250 samples/sec = 200 samples.
BASELINE_SAMPLES = int(FS * 0.1)         # We use 100ms of data BEFORE the flash to "zero" the signal.

# Artifact rejection threshold (peak-to-peak µV per channel)
# If a signal jumps more than 150 microvolts (like a blink or a cough), we throw it away.
ARTIFACT_THRESHOLD_UV = 100.0

# Epoch start offset (seconds after flash onset).
# We set this to 0.06 to start extraction at the 60ms mark.
# This ensures our window covers [60ms, 860ms] post-flash.
EPOCH_START_OFFSET_S = 0.06 

# Unicorn Hybrid Black electrode montage (8 channels)
# These are the standard names for where the sensors sit on the head.
UNICORN_CHANNELS = ['Fz', 'C3', 'Cz', 'C4', 'Pz', 'PO7', 'Oz', 'PO8']
PZ_INDEX = 4  # Pz is the most important channel for P300; it's right on top of the head.

# Active channel subset for classification.
# We use all 8 channels to give the AI as much information as possible.
ACTIVE_CHANNEL_INDICES = [0, 1, 2, 3, 4, 5, 6, 7] 

# The 6x6 matrix of characters shown on the screen.
MATRIX_CHARS = [
    'A', 'B', 'C', 'D', 'E', 'F',
    'G', 'H', 'I', 'J', 'K', 'L',
    'M', 'N', 'O', 'P', 'Q', 'R',
    'S', 'T', 'U', 'V', 'W', 'X',
    'Y', 'Z', '1', '2', '3', '4',
    '5', '6', '7', '8', '9', '_'
]

# ============================================================================
# Pre-computed Filter Coefficients
# ============================================================================
# We compute these once when the program starts so the math is faster later.
_nyq = 0.5 * FS  # Nyquist frequency (half the sampling rate)

# Bandpass Filter: We only care about brain waves between 0.5 Hz and 30 Hz.
# This removes very slow drifts and very fast "fuzz" (noise).
_bp_low = 0.5 / _nyq
_bp_high = 30.0 / _nyq
BP_B, BP_A = butter(4, [_bp_low, _bp_high], btype='band')

# Notch Filter: Electricity in the walls vibrates at 50 Hz (in Europe/Egypt).
# This acts like a "surgical strike" to remove exactly 50 Hz noise.
_notch_freq = 50.0 / _nyq
NOTCH_B, NOTCH_A = iirnotch(_notch_freq, 30.0)


def apply_preprocessing(data):
    """
    Apply the filters to a chunk of EEG data.
    
    Parameters:
        data: A numpy array of raw brain waves (samples x channels).
    Returns:
        The same data but cleaned and smoothed.
    """
    # filtfilt is "Zero-Phase" filtering — it processes the data forward and backward
    # so that the timing of the brain waves doesn't get shifted by the filter math.
    filtered = filtfilt(BP_B, BP_A, data, axis=0)
    filtered = filtfilt(NOTCH_B, NOTCH_A, filtered, axis=0)
    return filtered


def reject_artifacts(epoch, threshold_uv=ARTIFACT_THRESHOLD_UV):
    """
    Check if the user blinked or moved too much during a flash.
    
    Parameters:
        epoch: A single 800ms window of brain data.
    Returns:
        True if the data is "clean" (keep it).
        False if it's "noisy" (throw it away).
    """
    # Calculate the Peak-to-Peak (max value minus min value).
    # If the signal jumps more than our threshold, it's probably noise, not brain activity.
    if epoch.ndim == 2:
        if epoch.shape[0] < epoch.shape[1]:
            ptp = np.ptp(epoch, axis=1)
        else:
            ptp = np.ptp(epoch, axis=0)
    else:
        ptp = np.array([np.ptp(epoch)])
    
    return np.all(ptp < threshold_uv)


def extract_epoch(data_arr, time_arr, flash_time, apply_baseline=True):
    """
    Cut a specific 800ms slice of data out of the continuous stream.
    
    Parameters:
        data_arr: The continuous stream of cleaned EEG.
        time_arr: The timestamps for every sample in that stream.
        flash_time: Exactly when the light flashed on the screen.
    Returns:
        A (channels x samples) window of data, or None if the data hasn't arrived yet.
    """
    # 1. Find the "moment of the flash" in our data array.
    adjusted_time = flash_time + EPOCH_START_OFFSET_S
    idx = np.searchsorted(time_arr, adjusted_time)
    
    # 2. Check if we have enough data yet. 
    # We need 100ms before (for baseline) and 800ms after (for the P300).
    if idx - BASELINE_SAMPLES < 0 or idx + SAMPLES_PER_EPOCH >= len(data_arr):
        return None  # Data hasn't arrived in the buffer yet!
    
    # 3. Grab the baseline (the silence before the flash) and the epoch (the response).
    baseline = data_arr[idx - BASELINE_SAMPLES : idx]
    epoch = data_arr[idx : idx + SAMPLES_PER_EPOCH]
    
    # 4. Baseline Correction: Subtract the average of the "silence" from the response.
    # This ensures that every flash starts at "0 microvolts" regardless of slow drifts.
    if apply_baseline:
        baseline_mean = np.mean(baseline, axis=0)
        epoch = epoch - baseline_mean
    
    # Return it in the shape (channels, samples) which the AI models expect.
    return epoch.T
