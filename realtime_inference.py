"""
Real-Time P300 Inference Engine.

This is the "decoder." It takes the models we trained and uses them to 
guess which letter you are looking at in real-time.

How it works:
1. Training: It loads your saved data and trains two AI models (LDA and MDM).
2. Listening: It buffers EEG data and flash markers from LSL.
3. Bayesian Accumulation: This is the secret sauce. For every flash, it calculates
   the probability that it was a "target." It keeps adding these probabilities 
   up until one letter stands out with > 95% confidence.
4. Output: Once it's sure, it sends the letter back to the UI.
"""

import os
import sys

# --- PSYCHOPY RUNNER ENVIRONMENT FIX ---
for version in ["310", "311", "312", "313", "314"]:
    _user_site = os.path.expanduser(fr"~\AppData\Roaming\Python\Python{version}\site-packages")
    if os.path.isdir(_user_site) and _user_site not in sys.path:
        sys.path.append(_user_site) # Append instead of insert(0) to avoid version clashes
sys.path.append(os.getcwd())

# --- PERSISTENT LOGGING ---
log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend_debug_log.txt")
debug_log = open(log_path, "w", buffering=1) # Line-buffered
sys.stdout = debug_log
sys.stderr = debug_log
# --------------------------

import numpy as np
import pylsl
print(f"\n--- BACKEND DEBUG LOG STARTED: {pylsl.local_clock()} ---")
import asyncio
from collections import deque
import pickle
import math
import scipy.special
import asrpy
import mne

from sklearn.pipeline import make_pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from pyriemann.spatialfilters import Xdawn
from pyriemann.estimation import XdawnCovariances
from pyriemann.classification import MDM
from mne.decoding import Vectorizer

# [NEW] Project 2 Paths: Inject Zeina_Branch and SSVEP Protocol
_backend_dir = os.path.dirname(os.path.abspath(__file__))
_root_dir = os.path.dirname(os.path.dirname(_backend_dir))

zeina_path = os.path.join(_root_dir, "Zeina_Branch")
if zeina_path not in sys.path:
    sys.path.append(zeina_path)

ssvep_path = os.path.join(_root_dir, "SSVEP Protocol")
if ssvep_path not in sys.path:
    sys.path.append(ssvep_path)

import _client
from speller import API
from ssvep_realtime import SSVEPClassifier

from signal_processing import (
    FS, EPOCH_LEN, SAMPLES_PER_EPOCH, BASELINE_SAMPLES,
    MATRIX_CHARS, ARTIFACT_THRESHOLD_UV, ACTIVE_CHANNEL_INDICES,
    TRAINING_DATA_DIR,
    apply_preprocessing, reject_artifacts, extract_epoch
)

# 120 seconds of EEG data is kept in memory. Old data is automatically deleted.
MAX_BUFFER_SAMPLES = FS * 120


class RealTimeInference:
    def __init__(self):
        # Ring buffers: Deque is like a pipe; when you push data in one end, 
        # old data falls out the other end once it's full.
        self.eeg_data = deque(maxlen=MAX_BUFFER_SAMPLES)
        self.eeg_times = deque(maxlen=MAX_BUFFER_SAMPLES)
        
        self.current_trial_flashes = []
        self.is_running = True
        
        # Synthetic clock tracking
        self.global_sample_count = 0
        self.global_t_anchor = None
        
        # Incremental decoding state: remembers how many flashes we've already
        # processed so we don't repeat work.
        self._processed_flash_idx = 0
        # Count of flashes that actually contributed to the accumulator
        # (extracted an epoch AND passed artifact rejection). Used as the
        # confidence denominator so artifacts don't suppress dynamic stop.
        self._n_decoder_updates = 0

        # This dictionary stores our "confidence" for every letter (A-Z, 1-9).
        # We start with every letter having an equal score of 0.
        self._accumulated_scores = {c: 0.0 for c in MATRIX_CHARS}
        
        # Recording arrays for post-hoc diagnosis
        self.recorded_eeg_data = []
        self.recorded_eeg_times = []
        self.recorded_flashes = []
        self.recorded_predictions = []
        
        self.model_lda = None
        self.kde_target = None
        self.kde_nontarget = None
        self.asr_filter = None
        
        # [NEW] Project 2: BCI States & Buffers
        self.bci_mode = "P300" # Toggles between P300 and SSVEP
        self.ssvep_targets = [8.0, 10.0, 12.0, 15.0, 17.0] # 5 optimized options. Unused frequencies act as 'Otherwise' noise catchers.
        self.ssvep_classifier = SSVEPClassifier(targets=self.ssvep_targets)
        self.ssvep_buffer = deque(maxlen=int(FS * 1.0)) # 1.0-second discrete window
        self.ssvep_window_size = 6
        self.ssvep_history = deque(maxlen=self.ssvep_window_size) # Dynamic voting window
        self.roi_indices = [5, 6, 7] # PO7, Oz, PO8 indices for SSVEP
        
        self.context_word = ""
        self.sentence_history = ""
        self.current_word = ""
        self.llm_threshold = 0.9
        
        # LSL Outlet: How we talk back to the PsychoPy UI.
        # We don't create this until the model is trained.
        self.decoded_outlet = None
        
        # Log file for debugging
        self.log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "spelled_text.txt")
        
        # [NEW] Project 2: LLM API Integration
        self.speller_api = API()
        
    def load_and_train_model(self):
        """
        Loads the training data and prepares the AI models.
        """
        print("Loading training data to train the live classifier...")
        x_path = os.path.join(TRAINING_DATA_DIR, "X_train.npy")
        y_path = os.path.join(TRAINING_DATA_DIR, "y_train.npy")
        
        if not os.path.exists(x_path):
            raise FileNotFoundError("X_train.npy not found. You must train first!")
            
        X = np.load(x_path) # Brain waves
        y = np.load(y_path) # 1 if Target, 0 if Not
        
        # Filter to use only our 8 EEG channels
        X = X[:, ACTIVE_CHANNEL_INDICES, :]
        
        # xDAWN + LDA Pipeline
        # xDAWN is a "spatial filter" that ignores noise and focuses on the P300.
        # LDA is a classic classifier that draws a line between Target and Non-Target.
        print("Training xDAWN + LDA Pipeline...")
        self.model_lda = make_pipeline(
            Xdawn(nfilter=4),
            Vectorizer(),
            LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto')
        )
        self.model_lda.fit(X, y)
        
        # Load KDE Distributions
        print("Loading Bayesian KDE Distributions...")
        kde_path = os.path.join(TRAINING_DATA_DIR, "kde_distributions.pkl")
        with open(kde_path, 'rb') as f:
            kdes = pickle.load(f)
            self.kde_target = kdes['target']
            self.kde_nontarget = kdes['nontarget']
            
        # [TASK 2] Load Pre-fitted ASR State
        print("Loading Pre-fitted ASR State...")
        asr_path = os.path.join(TRAINING_DATA_DIR, "asr_state.pkl")
        if os.path.exists(asr_path):
            with open(asr_path, 'rb') as f:
                self.asr_filter = pickle.load(f)
        else:
            print("WARNING: asr_state.pkl not found. ASR filtering will be unavailable.")

        print("Model, KDEs, and ASR State Loaded Successfully.")
        
        # Now publish the "Ready" signal for the UI
        info_dec = pylsl.StreamInfo('Speller_Decoded', 'Markers', 1, 0, 'string', 'bci_decoder_123')
        self.decoded_outlet = pylsl.StreamOutlet(info_dec)


    async def lsl_worker(self, inlet):
        """Continuously pulls EEG data into our 120-second ring buffer."""
        print("Synchronizing LSL clocks... (3 seconds)")
        await asyncio.sleep(3.0)
        print("[READY] Clock sync complete. Pulling data.")
        
        while self.is_running:
            chunk, timestamps = inlet.pull_chunk()
            if timestamps:
                timestamps = np.array(timestamps)
                n_samples = len(timestamps)
                
                # 1. Calculate ideal time offsets from the start of the session
                chunk_ideal_offsets = (self.global_sample_count + np.arange(n_samples)) / FS
                
                # 2. Find the anchor (minimum delay) for this chunk
                chunk_anchor = np.min(timestamps - chunk_ideal_offsets)
                
                # 3. Update the global true hardware clock anchor
                if self.global_t_anchor is None:
                    self.global_t_anchor = chunk_anchor
                else:
                    self.global_t_anchor = min(self.global_t_anchor, chunk_anchor)
                    
                # 4. Generate perfectly uniform timestamps using the true anchor
                synthetic_timestamps = self.global_t_anchor + chunk_ideal_offsets
                
                self.global_sample_count += n_samples
                self.eeg_data.extend(chunk)
                self.eeg_times.extend(synthetic_timestamps)
                self.recorded_eeg_data.extend(chunk)
                self.recorded_eeg_times.extend(synthetic_timestamps)
                
                # [NEW] Real-time SSVEP Processing
                if self.bci_mode == "SSVEP":
                    for sample in chunk:
                        roi_data = [sample[idx] for idx in self.roi_indices]
                        self.ssvep_buffer.append(roi_data)
                        
                        if len(self.ssvep_buffer) == self.ssvep_buffer.maxlen:
                            X = np.array(self.ssvep_buffer).T # (Channels, Samples)
                            self.ssvep_buffer.clear()
                            
                            # EXACT MATCH to ssvep_realtime.py: Standard CCA, no threshold
                            pred, _ = self.ssvep_classifier.classify_cca(X)
                            self.ssvep_history.append(pred)
                            
                            # Majority voter
                            if len(self.ssvep_history) == self.ssvep_history.maxlen:
                                from collections import Counter
                                counts = Counter(self.ssvep_history)
                                most_common, count = counts.most_common(1)[0]
                                
                                # Require 4 out of 6 majority
                                required_votes = 4
                                if count >= required_votes:
                                    print(f"*** SSVEP DETECTED: {most_common} Hz ***")
                                    self.decoded_outlet.push_sample([f"SSVEP_DECODED_{most_common}"], pylsl.local_clock())
                                    self.ssvep_history.clear()
                                    self.bci_mode = "P300" # Revert after selection
            await asyncio.sleep(0.01)

    async def marker_worker(self, marker_inlet):
        """
        Listens to the UI and decides when to try and decode a letter.
        """
        while self.is_running:
            marker, timestamp = marker_inlet.pull_sample(timeout=0.0)
            if timestamp is not None and marker is not None:
                m_str = marker[0]
                
                if m_str == "SESSION_START":
                    self._reset_trial_state()
                    print("New letter trial starting...")
                
                elif m_str.startswith("FLASH_GROUP_"):
                    # The UI tells us which letters just flashed
                    group = list(m_str.replace("FLASH_GROUP_", ""))
                    self.current_trial_flashes.append((timestamp, group))
                    self.recorded_flashes.append({"time": timestamp, "group": group})
                
                # [NEW] Project 2 Markers
                elif m_str.startswith("SET_THRESHOLD:"):
                    self.llm_threshold = float(m_str.replace("SET_THRESHOLD:", ""))
                    print(f"LLM Trigger Threshold Updated: {self.llm_threshold}")

                elif m_str.startswith("SET_CONTEXT:"):
                    self.context_word = m_str.replace("SET_CONTEXT:", "").strip()
                    # The UI sends this after the user finishes spelling the context word.
                    # We must completely wipe the sentence buffers so the context word
                    # isn't accidentally included in the main sentence to the LLM.
                    self.current_word = ""
                    self.sentence_history = ""
                    print(f"Global Context Updated: {self.context_word} (Speller Reset)")
                
                elif m_str.startswith("SSVEP_START"):
                    print(f"Backend switching to SSVEP mode... ({m_str})")
                    self.bci_mode = "SSVEP"
                    self.ssvep_window_size = 6
                    self.ssvep_buffer.clear()
                    self.ssvep_history = deque(maxlen=self.ssvep_window_size)
                
                elif m_str == "SSVEP_TIMEOUT":
                    print("Backend reverting to P300 mode (SSVEP Timeout)...")
                    # DO NOT clear self.current_word here. The user didn't pick anything,
                    # so the spelled letters so far stay in the prefix buffer.
                    self.bci_mode = "P300"
                    
                elif m_str.startswith("WORD_SELECTED:"):
                    selected_word = m_str.replace("WORD_SELECTED:", "").strip()
                    print(f"Backend reverting to P300 mode (Selected: {selected_word})")
                    # Update sentence history for future LLM predictions
                    self.sentence_history += (" " if self.sentence_history else "") + selected_word
                    # RESET the prefix buffer because a word was completed
                    self.current_word = ""
                    self.bci_mode = "P300"
                    
                elif m_str == "RESPONSE_ACK":
                    print("Backend reverting to P300 mode (Response Acknowledged)")
                    self.sentence_history = ""
                    self.current_word = ""
                    self.bci_mode = "P300"
                
                elif m_str == "EVALUATE":
                    # The UI wants to know if we've found the letter yet.
                    # Guard: Don't perform inference until model is trained
                    if self.model_lda is not None:
                        predicted_char, hit_threshold = self.decode_trial(check_threshold=True)
                        if hit_threshold and predicted_char:
                            print(f"*** DYNAMIC STOP *** FOUND: {predicted_char}")
                            self.decoded_outlet.push_sample([f"DECODED_{predicted_char}"], pylsl.local_clock())
                            self.recorded_predictions.append({"time": pylsl.local_clock(), "predicted": predicted_char, "type": "dynamic_stop"})
                            
                            # [NEW] Project 2: Update current word and call LLM
                            self._handle_character_decoded(predicted_char)
                            self._reset_trial_state()
                
                elif m_str == "TRIAL_END":
                    # Flashing is over! We MUST pick a letter now.
                    # But we must wait for the last flashes to actually arrive in the EEG buffer!
                    # (With 220ms offset + 800ms epoch, the last flash needs 1.02s of data).
                    print("TRIAL_END received. Waiting for last brainwaves to arrive...")
                    
                    predicted_char = None
                    # Wait up to 3 seconds for the EEG buffer to catch up
                    for _ in range(300): 
                        # Guard: Don't decode until model is trained
                        if self.model_lda is None:
                            await asyncio.sleep(0.01)
                            continue

                        predicted_char, _ = self.decode_trial(check_threshold=False)
                        # Check if we've processed all flashes in this trial
                        if self._processed_flash_idx >= len(self.current_trial_flashes):
                            break
                        await asyncio.sleep(0.01)
                    
                    if predicted_char:
                        print(f"---------> FINAL GUESS: {predicted_char} <---------")
                        self.decoded_outlet.push_sample([f"DECODED_{predicted_char}"], pylsl.local_clock())
                        self.recorded_predictions.append({"time": pylsl.local_clock(), "predicted": predicted_char, "type": "fallback"})
                        
                        # [NEW] Project 2: Update current word and call LLM
                        self._handle_character_decoded(predicted_char)
                    else:
                        print("ERROR: Failed to decode letter after 3s timeout.")
                        
                    self._reset_trial_state()
                    
                elif m_str == "SESSION_END":
                    print("\nSaving inference recording for post-hoc diagnosis...")
                    base_dir = os.path.dirname(os.path.abspath(__file__))
                    rec_dir = os.path.join(base_dir, "inference_recording")
                    os.makedirs(rec_dir, exist_ok=True)
                    np.save(os.path.join(rec_dir, "eeg_continuous.npy"), np.array(self.recorded_eeg_data))
                    np.save(os.path.join(rec_dir, "eeg_timestamps.npy"), np.array(self.recorded_eeg_times))
                    np.save(os.path.join(rec_dir, "flash_events.npy"), np.array(self.recorded_flashes, dtype=object))
                    np.save(os.path.join(rec_dir, "predictions.npy"), np.array(self.recorded_predictions, dtype=object))
                    print("Inference recording saved to 'inference_recording/' directory.\n")
                    self._reset_trial_state()
                    
            await asyncio.sleep(0.01)

    def _handle_character_decoded(self, char):
        """Internal logic to manage words/sentences and trigger LLM."""
        if char == "8": # Backspace moved to 8
            if self.current_word:
                self.current_word = self.current_word[:-1]
            elif self.sentence_history:
                parts = self.sentence_history.strip().split()
                if parts:
                    self.current_word = parts[-1]
                    self.sentence_history = " ".join(parts[:-1])
            print(f"  [BACKSPACE] Word: '{self.current_word}' | History: '{self.sentence_history}'")
        elif char == "9": # Submit to AI
            self.sentence_history += (" " if self.sentence_history else "") + self.current_word
            self.current_word = ""
            print(f"  [SUBMIT] Sending to LLM: '{self.sentence_history}'")
            asyncio.create_task(self.generate_response(self.sentence_history))
        elif char == "_": # Space / Add word directly
            self.sentence_history += (" " if self.sentence_history else "") + self.current_word
            self.current_word = ""
            print(f"  [SPACE] Added word to sentence. Sentence now: '{self.sentence_history}'")
        else:
            self.current_word += char
            # Trigger LLM async prediction for the current word fragment
            asyncio.create_task(self.predict_words(self.current_word))

    def decode_trial(self, check_threshold=False):
        """
        The Brain of the Brain-Computer Interface.
        Uses Bayesian math to combine all flashes into a single guess.
        """
        if not self.current_trial_flashes: return None, False
            
        # 1. Grab only the NEW flashes since the last time we checked.
        new_flashes = self.current_trial_flashes[self._processed_flash_idx:]
        if not new_flashes: return self._evaluate_accumulated(check_threshold)

        time_arr = np.array(self.eeg_times)
        data_arr = np.array(self.eeg_data)[:, :8]

        # Safety: deques are extended (data, then times) in lsl_worker with
        # no await between, so lengths should match. Clip just in case a
        # race or partial extension ever slips through.
        n_common = min(len(time_arr), len(data_arr))
        time_arr = time_arr[:n_common]
        data_arr = data_arr[:n_common]

        # 2. Filter the entire buffer to avoid filtfilt edge transients destroying the P300
        data_filtered = apply_preprocessing(data_arr)
        
        # [TASK 2] Fast Real-Time ASR (Transform Only)
        if self.asr_filter is None:
            raise RuntimeError("CRITICAL: ASR filter state not loaded. Run apply_offline_asr.py first!")

        # Transpose to [n_channels, n_samples] as expected by asrpy
        transposed_data = data_filtered.T
        
        # MNE constraint: asrpy.ASR.transform strictly requires an mne.Raw object
        info = mne.create_info(ch_names=[f'CH{i}' for i in range(8)], sfreq=FS, ch_types='eeg')
        raw = mne.io.RawArray(transposed_data, info, verbose=False)
        
        # Apply the pre-fitted filter (No fit() call here to avoid thread blocking)
        raw_clean = self.asr_filter.transform(raw)
        
        # Transpose back to [n_samples, n_channels]
        data_filtered = raw_clean.get_data().T

        X_test = []
        groups = []
        last_processed_idx = self._processed_flash_idx
        n_stale = 0  # flashes older than the buffer — permanently undecodable
        epoch_wall_spans = []  # for diagnostic on non-uniform LSL delivery

        # 3. For every flash, try to extract its 800ms brain wave
        for f_time, group in new_flashes:
            # Stale: flash timestamp has fallen off the back of the ring
            # buffer (can happen if decode_trial was blocked for > buffer
            # duration). Advance past it so we don't stall forever.
            if len(time_arr) and f_time < time_arr[0]:
                last_processed_idx += 1
                n_stale += 1
                continue

            epoch = extract_epoch(data_filtered, time_arr, f_time, apply_baseline=True)

            if epoch is None:
                # If data hasn't arrived in LSL yet, we stop and wait for it.
                break

            last_processed_idx += 1

            # Diagnostic: sample count-to-wall-time sanity. Unicorn Recorder's
            # LSL timestamps reflect delivery time (bursty), so the 200-sample
            # epoch window sometimes spans 0.5-1.1s instead of 0.8s, degrading
            # classifier input. See investigation in repo issues.
            idx = np.searchsorted(time_arr, f_time)
            if 0 <= idx < len(time_arr) - SAMPLES_PER_EPOCH:
                epoch_wall_spans.append(
                    float(time_arr[idx + SAMPLES_PER_EPOCH - 1] - time_arr[idx])
                )

            # Skip if user blinked
            if not reject_artifacts(epoch, ARTIFACT_THRESHOLD_UV):
                continue

            X_test.append(epoch[ACTIVE_CHANNEL_INDICES, :])
            groups.append(group)
            # Only count flashes that actually feed the accumulator so the
            # confidence denominator is correct.
            self._n_decoder_updates += 1

        if n_stale:
            print(f"[WARN] decode_trial dropped {n_stale} stale flash(es) "
                  f"older than buffer start.")
        if epoch_wall_spans:
            spans = np.asarray(epoch_wall_spans)
            bad = int((np.abs(spans - EPOCH_LEN) > 0.15).sum())
            if bad:
                print(f"[WARN] {bad}/{len(spans)} epochs span "
                      f"wall-time outside 0.65-0.95s "
                      f"(min={spans.min():.3f}s, max={spans.max():.3f}s). "
                      f"LSL EEG stream is bursty — classifier input is "
                      f"misaligned with training data.")
        
        self._processed_flash_idx = last_processed_idx
        if not X_test: return self._evaluate_accumulated(check_threshold)
        
        # 4. Extract continuous decision scores
        X_test = np.array(X_test)
        decision_scores = self.model_lda.decision_function(X_test)
        
        # 5. Bayesian Log-Probability Update using KDE
        for score, group in zip(decision_scores, groups):
            # Evaluate KDE and clip to prevent log(0) underflow
            lh_target = np.clip(self.kde_target.evaluate(score)[0], 1e-10, None)
            lh_nontarget = np.clip(self.kde_nontarget.evaluate(score)[0], 1e-10, None)
            
            log_lh_target = np.log(lh_target)
            log_lh_nontarget = np.log(lh_nontarget)
            
            for c in MATRIX_CHARS:
                if c in group:
                    self._accumulated_scores[c] += log_lh_target
                else:
                    self._accumulated_scores[c] += log_lh_nontarget
                    
        # 6. Normalize Log-Probabilities to prevent unbounded drift
        scores_arr = np.array(list(self._accumulated_scores.values()))
        logsum = scipy.special.logsumexp(scores_arr)
        for c in MATRIX_CHARS:
            self._accumulated_scores[c] -= logsum
        
        return self._evaluate_accumulated(check_threshold)

    def _evaluate_accumulated(self, check_threshold):
        """Check if any letter has a decisively high score."""
        if not self.current_trial_flashes:
            return None, False

        scores_arr = np.fromiter(self._accumulated_scores.values(), dtype=np.float64)

        # Guard: if no flash ever produced a valid epoch, the accumulator is
        # still at the uniform prior. Returning max(dict) here is a silent
        # bias toward MATRIX_CHARS[0] ('A'). Surface the failure instead so
        # the UI can prompt a repeat or the session can be flagged.
        if np.ptp(scores_arr) < 1e-9:
            return "!", False

        # Use successfully-updated-flash count as the denominator, not
        # _processed_flash_idx — the latter counts artifact-rejected flashes
        # that did NOT contribute to the accumulator, artificially suppressing
        # confidence in high-artifact sessions.
        flashes_processed = self._n_decoder_updates
        if flashes_processed == 0:
            return "!", False

        # Convert from normalized log-probabilities back to standard probabilities
        # Because we used logsumexp, these will sum to exactly 1.0
        probs = {c: math.exp(log_p) for c, log_p in self._accumulated_scores.items()}
        
        # Sort characters by their probability in descending order
        sorted_chars = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
        best_char = sorted_chars[0][0]
        p_top1 = sorted_chars[0][1]
        p_top2 = sorted_chars[1][1]

        # If the trial ended naturally (TRIAL_END), return the best guess
        if not check_threshold:
            return best_char, False

        # --- DYNAMIC STOPPING ---
        # Calculate the Uncertainty Ratio (UQ). 
        # A UQ of 98.0 means the top character is 98x more likely than the second.
        # This is an extremely conservative threshold preventing premature firing.
        uq_ratio = p_top1 / (p_top2 + 1e-12) # Add epsilon to prevent div by zero
        
        print(f"  [EVAL] {best_char}: P={p_top1:.3f} | UQ Ratio={uq_ratio:.1f}")

        if uq_ratio >= 98.0:
            return best_char, True
            
        return None, False

    # [NEW] Project 2: LLM Word Completion Hook
    async def predict_words(self, partial_word):
        """Async call to speller.API for word completion."""
        if not partial_word or partial_word == "_":
            return
            
        print(f"Fetching predictions for: '{partial_word}' (Context: {self.context_word})")
        
        try:
            import time
            start_t = time.time()
            
            # Offload blocking API call to a thread
            loop = asyncio.get_running_loop()
            results = await loop.run_in_executor(None, lambda: self.speller_api.predict_words(
                context=self.context_word,
                prefix=partial_word,
                sentence=self.sentence_history,
                return_probs=True
            ))
            
            end_t = time.time()
            print(f"  [SPELLER_LLM] Inference time: {(end_t - start_t)*1000:.1f}ms")
            
            # results is list[dict] with {"word": str, "prob": float}
            words = [r["word"] for r in results]
            probs = [r["prob"] for r in results]
            
            if len(words) < 4: 
                words += [""] * (4 - len(words))
                probs += [0.0] * (4 - len(probs))
            
            words = words[:4]
            probs = probs[:4]
            
            # Uncertainty Ratio calculation
            p_top1 = probs[0]
            p_top2 = probs[1]
            uq_ratio = p_top1 / (p_top2 + 1e-12)
            
            dynamic_ratio_threshold = self.llm_threshold / max(0.001, (1.0 - self.llm_threshold))
            
            print(f"  Predictions: {words} | Probabilities: {probs}")
            print(f"  Uncertainty Ratio (Top 1 / Top 2): {uq_ratio:.1f} (Required: {dynamic_ratio_threshold:.1f})")
            
            if uq_ratio >= (dynamic_ratio_threshold - 1e-4):
                print(f"  [LLM] High confidence (Ratio >= {dynamic_ratio_threshold:.1f}). Triggering SSVEP UI...")
                marker = f"SSVEP_PREDICTIONS:{','.join(words)}"
                self.decoded_outlet.push_sample([marker], pylsl.local_clock())
                
        except Exception as e:
            print(f"  [LLM ERROR] {e}")

    def _get_concentration_state(self):
        """Calculates a simple Beta/Alpha ratio from the last 10 seconds of EEG."""
        import scipy.signal
        if len(self.eeg_data) < FS * 10:
            return "Unknown"
            
        # Get last 10 seconds, average across all 8 channels
        recent_data = np.array(list(self.eeg_data)[-int(FS * 10):])[:, :8]
        
        # Calculate PSD using Welch
        freqs, psd = scipy.signal.welch(recent_data, fs=FS, axis=0, nperseg=int(FS*2))
        
        # Average PSD across all channels
        psd_mean = np.mean(psd, axis=1)
        
        alpha_power = np.sum(psd_mean[np.logical_and(freqs >= 8, freqs <= 13)])
        beta_power = np.sum(psd_mean[np.logical_and(freqs >= 13, freqs <= 30)])
        
        ratio = beta_power / (alpha_power + 1e-9)
        
        print(f"  [METRICS] Beta/Alpha Ratio: {ratio:.2f}")
        return "Focused" if ratio > 0.8 else "Fatigued"

    # [NEW] Project 2: LLM Sentence Response Hook
    async def generate_response(self, text):
        """Async call to speller.API for full sentence response."""
        if not text.strip():
            return
            
        concentration_state = self._get_concentration_state()
        print(f"Generating full response for: '{text}' (Context: {self.context_word}) | State: {concentration_state}")
        
        try:
            import time
            start_t = time.time()
            
            loop = asyncio.get_running_loop()
            content = await loop.run_in_executor(None, lambda: self.speller_api.respond(
                sentence=text,
                context=self.context_word,
                mental_state=concentration_state
            ))
            
            end_t = time.time()
            print(f"  [RESPONSE_LLM] Inference time: {(end_t - start_t)*1000:.1f}ms")
            
            if not content:
                content = "Error generating response."
            
            # Remove any colons or newlines to avoid LSL marker parsing issues
            content = content.replace(":", "-").replace("\n", " ")
            
            print(f"  [RESPONSE_LLM] Output: {content}")
            
            marker = f"LLM_RESPONSE:{content}"
            self.decoded_outlet.push_sample([marker], pylsl.local_clock())
                
        except Exception as e:
            print(f"  [RESPONSE_LLM ERROR] {e}")
            marker = f"LLM_RESPONSE:Error generating response."
            self.decoded_outlet.push_sample([marker], pylsl.local_clock())

    def _reset_trial_state(self):
        """Clear memory for a new letter."""
        self._processed_flash_idx = 0
        self._n_decoder_updates = 0
        self.current_trial_flashes = []
        # Initialize with uniform prior log-probabilities
        self._accumulated_scores = {c: 0.0 for c in MATRIX_CHARS}

    async def main_loop(self):
        print("Resolving LSL Streams...")
        loop = asyncio.get_running_loop()
        u_streams = await loop.run_in_executor(None, pylsl.resolve_byprop, 'name', 'UnicornRecorderLSLStream', 1, 10.0)
        m_streams = await loop.run_in_executor(None, pylsl.resolve_byprop, 'name', 'Speller_Markers', 1, 10.0)
        
        if not u_streams:
            print("\n" + "!" * 60)
            print("FATAL ERROR: Could not find UnicornRecorderLSLStream!")
            print("Did you forget to hit START in the Unicorn Recorder?")
            print("!" * 60 + "\n")
            return
            
        if not m_streams:
            print("\nFATAL ERROR: Could not find Speller_Markers stream!\n")
            return
        
        # Use stable processing flags
        p_flags = pylsl.proc_dejitter | pylsl.proc_threadsafe
        inlet = pylsl.StreamInlet(u_streams[0], max_buflen=360, processing_flags=p_flags)
        marker_inlet = pylsl.StreamInlet(m_streams[0], max_buflen=120, processing_flags=p_flags)

        # Start listening immediately in the background
        # This ensures we capture SESSION_START even if we are still training.
        lsl_task = asyncio.create_task(self.lsl_worker(inlet))
        marker_task = asyncio.create_task(self.marker_worker(marker_inlet))
        
        # Train the model in the background while the workers buffer data
        print("Training models in background...")
        await loop.run_in_executor(None, self.load_and_train_model)
        
        print("\n[READY] Engine active and models trained. Decoding dynamic markers...")
        await asyncio.gather(lsl_task, marker_task)

async def main():
    agent = RealTimeInference()
    # main_loop now handles model training internally
    await agent.main_loop()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"\nCRITICAL ENGINE FAILURE: {e}")
