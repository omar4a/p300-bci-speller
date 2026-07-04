import time
import math
import threading
import collections
import tkinter as tk
from tkinter import ttk

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch
from sklearn.cross_decomposition import CCA
from pylsl import StreamInlet, resolve_byprop, StreamInfo

# Configuration
TARGETS = [10.0, 12.0, 15.0]
NUM_HARMONICS = 3
SFREQ = 250.0
WINDOW_SECONDS = 1.0
CHANNEL_COLS = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]
ROI_CHANNELS = ["PO7", "Oz", "PO8"]


class SSVEPClassifier:
    def __init__(self, sfreq=SFREQ, targets=TARGETS, num_harmonics=NUM_HARMONICS):
        self.sfreq = sfreq
        self.targets = targets
        self.num_harmonics = num_harmonics
        self.nyq = sfreq / 2.0
        
        # 60 Hz Notch filter to remove power line noise
        self.b_notch, self.a_notch = iirnotch(60.0, 30.0, sfreq)
        
        # FBCCA Subbands (Lower cutoffs covering harmonics)
        # 3 Filter Banks: 8-88 Hz, 16-88 Hz, 24-88 Hz
        self.fbcca_subbands = []
        low_cuts = [8.0, 16.0, 24.0]
        for low in low_cuts:
            b, a = butter(4, [low / self.nyq, 88.0 / self.nyq], btype='bandpass')
            self.fbcca_subbands.append((b, a))
            
        # FBCCA subband weightings matching typical alpha formulations 
        # w(n) = n^(-1.25) + 0.25
        self.fbcca_weights = [(n**(-1.25) + 0.25) for n in range(1, len(low_cuts) + 1)]
        
        # Base filter for standard CCA mode equivalent to the broadest FBCCA band
        self.b_base, self.a_base = butter(4, [8.0 / self.nyq, 88.0 / self.nyq], btype='bandpass')

    def generate_references(self, num_samples):
        """Construct synthetic reference Sine/Cosine templates for each target and harmonic."""
        t = np.arange(num_samples) / self.sfreq
        Y_ref = {}
        for freq in self.targets:
            y = []
            for h in range(1, self.num_harmonics + 1):
                y.append(np.sin(2 * np.pi * h * freq * t))
                y.append(np.cos(2 * np.pi * h * freq * t))
            Y_ref[freq] = np.array(y).T  # Transpose to shape (samples, features)
        return Y_ref

    def apply_cca(self, X_filt, Y_ref):
        """Compute native Canonical Correlation against all available target frequencies."""
        max_corrs = []
        for freq in self.targets:
            cca = CCA(n_components=1)
            try:
                # X_filt is (channels, samples). sklearn expects (samples, features)
                X_c, Y_c = cca.fit_transform(X_filt.T, Y_ref[freq])
                # Extract strict Pearson correlation between canonical pairs
                corr = np.corrcoef(X_c[:, 0], Y_c[:, 0])[0, 1]
                max_corrs.append(abs(corr))
            except Exception:
                # If variance reaches 0 (sensors drop), strictly fallback to 0 correlation
                max_corrs.append(0.0)
        return np.array(max_corrs)

    def classify_cca(self, X):
        """
        Execute standard Canonical Correlation Analysis classification.
        Parameters: X matrix (channels, samples)
        Returns: Top frequency prediction, Full correlation vector
        """
        # Temporal filtering
        X_n = filtfilt(self.b_notch, self.a_notch, X, axis=1)
        X_filt = filtfilt(self.b_base, self.a_base, X_n, axis=1)
        
        Y_ref = self.generate_references(X.shape[1])
        corrs = self.apply_cca(X_filt, Y_ref)
        
        predicted_freq = self.targets[np.argmax(corrs)]
        return predicted_freq, corrs

    def classify_fbcca(self, X):
        """
        Execute mathematically robust Filter Bank CCA classification. 
        Calculates weighted sum of squared canonical correlations across filterbanks.
        Parameters: X matrix (channels, samples)
        Returns: Top frequency prediction, weighted correlation sum vector
        """
        X_n = filtfilt(self.b_notch, self.a_notch, X, axis=1)
        Y_ref = self.generate_references(X.shape[1])
        
        rho_squared_sum = np.zeros(len(self.targets))
        
        for i, (b, a) in enumerate(self.fbcca_subbands):
            X_filt = filtfilt(b, a, X_n, axis=1)
            corrs = self.apply_cca(X_filt, Y_ref)
            rho_squared_sum += self.fbcca_weights[i] * (corrs ** 2)
            
        predicted_freq = self.targets[np.argmax(rho_squared_sum)]
        return predicted_freq, rho_squared_sum


class LSLStreamer(threading.Thread):
    def __init__(self, callback):
        super().__init__(daemon=True)
        self.callback = callback
        self.classifier = SSVEPClassifier()
        self.algorithm = "CCA" 
        
        self.stop_event = threading.Event()
        self.history = collections.deque(maxlen=4)
        self.current_display_pred = "--"
        self.buffer = collections.deque(maxlen=int(SFREQ * WINDOW_SECONDS))
        self.roi_indices = []

    def set_algorithm(self, algo_name):
        self.algorithm = algo_name
        self.history.clear() # Purge the historical cache to prevent cross-contamination voting

    def stop(self):
        self.stop_event.set()

    def run(self):
        self.callback("status", "Searching for Unicorn LSL...")
        streams = resolve_byprop("name", "UnicornRecorderDataLSLStream", timeout=5.0)
        
        if not streams:
             # Fallback general pull if explicit Unicorn title not found
             streams = resolve_byprop("type", "EEG", timeout=5.0)
            
        if not streams:
            self.callback("status", "LSL Stream not found.")
            return
            
        inlet = StreamInlet(streams[0])
        info = inlet.info()
        self.callback("status", f"Connected: {info.name()}")
        
        channels = info.desc().child("channels")
        if not channels.empty():
             ch = channels.child("channel")
             labels = []
             while not ch.empty():
                 labels.append(ch.child_value("label").upper())
                 ch = ch.next_sibling("channel")
                 
             for roi in ROI_CHANNELS:
                 if roi.upper() in labels:
                     self.roi_indices.append(labels.index(roi.upper()))
        
        if len(self.roi_indices) < len(ROI_CHANNELS):
            # Assume strict Unicorn default ordering if LSL meta headers were dropped
            self.roi_indices = [CHANNEL_COLS.index(c) for c in ROI_CHANNELS]
            
        while not self.stop_event.is_set():
            # Minimal latency ping constraint (~128ms) tracking chunks
            samples, _ = inlet.pull_chunk(timeout=0.2, max_samples=32)
            for sample in samples:
                roi_data = [sample[idx] for idx in self.roi_indices]
                self.buffer.append(roi_data)
                
                if len(self.buffer) == self.buffer.maxlen:
                    X = np.array(self.buffer).T # Cast to (Channels, Time)
                    self.buffer.clear() # Reset array logic to enforce distinct 1 sec epoch bounds
                    
                    if self.algorithm == "CCA":
                        pred, _ = self.classifier.classify_cca(X)
                    else:
                        pred, _ = self.classifier.classify_fbcca(X)
                        
                    self.history.append(pred)
                    
                    # Implement strict 4-second rolling majority voter
                    if len(self.history) == self.history.maxlen:
                        counter = collections.Counter(self.history)
                        most_common_pred, count = counter.most_common(1)[0]
                        # Only lock in a new class if we have a strict majority (e.g., 3 or 4 votes out of 4)
                        if count > 2:
                            self.current_display_pred = most_common_pred
                            
                    if self.current_display_pred != "--":
                        self.callback("prediction", self.current_display_pred)


class SSVEPGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Real-Time SSVEP Algorithm Testbench")
        self.root.geometry("600x450")
        self.root.configure(bg="#2E2E38")
        
        tk.Label(
            root, text="Rolling 3-Second SSVEP Decoding", 
            font=("Segoe UI", 18, "bold"), bg="#2E2E38", fg="#F2F2F2"
        ).pack(pady=15)
        
        self.status_var = tk.StringVar(value="Status: Disconnected")
        self.status_label = tk.Label(
            root, textvariable=self.status_var, 
            font=("Segoe UI", 11, "italic"), bg="#2E2E38", fg="#8A8A96"
        )
        self.status_label.pack(pady=5)
        
        self.display_frame = tk.Frame(root, bg="#111116", bd=0, highlightthickness=2, highlightbackground="#00B8A9", highlightcolor="#00B8A9")
        self.display_frame.pack(pady=20, expand=True, fill="both", padx=60)
        
        self.pred_var = tk.StringVar(value="-- Hz")
        self.pred_label = tk.Label(
            self.display_frame, textvariable=self.pred_var, 
            font=("Segoe UI", 85, "bold"), bg="#111116", fg="#00B8A9"
        )
        self.pred_label.pack(expand=True, fill="both")
        
        self.controls_frame = tk.Frame(root, bg="#2E2E38")
        self.controls_frame.pack(pady=20)
        
        self.algo_var = tk.StringVar(value="CCA")
        
        tk.Label(self.controls_frame, text="Algorithm:", font=("Segoe UI", 14), bg="#2E2E38", fg="#D0D0D9").pack(side=tk.LEFT, padx=15)
        
        self.btn_cca = tk.Radiobutton(
            self.controls_frame, text="CCA", variable=self.algo_var, value="CCA", command=self.on_algo_switch,
            font=("Segoe UI", 13, "bold"), bg="#2E2E38", fg="#F2F2F2", selectcolor="#494959", activebackground="#2E2E38"
        )
        self.btn_cca.pack(side=tk.LEFT, padx=10)
        
        self.btn_fbcca = tk.Radiobutton(
            self.controls_frame, text="FBCCA", variable=self.algo_var, value="FBCCA", command=self.on_algo_switch,
            font=("Segoe UI", 13, "bold"), bg="#2E2E38", fg="#F2F2F2", selectcolor="#494959", activebackground="#2E2E38"
        )
        self.btn_fbcca.pack(side=tk.LEFT, padx=10)
        
        self.streamer = LSLStreamer(callback=self.handle_stream_event)
        self.streamer.start()

    def on_algo_switch(self):
        new_algo = self.algo_var.get()
        self.streamer.set_algorithm(new_algo)

    def handle_stream_event(self, event_type, value):
        if event_type == "status":
            self.root.after(0, lambda: self.status_var.set(f"Status: {value}"))
        elif event_type == "prediction":
            self.root.after(0, lambda: self._update_gui_prediction(value))
            
    def _update_gui_prediction(self, freq):
        self.pred_var.set(f"{freq:g} Hz")

def main():
    root = tk.Tk()
    app = SSVEPGUI(root)
    root.protocol("WM_DELETE_WINDOW", lambda: on_close(root, app.streamer))
    root.mainloop()

def on_close(root, streamer):
    streamer.stop()
    root.destroy()

if __name__ == "__main__":
    main()
