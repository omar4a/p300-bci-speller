# Real-Time P300 BCI Speller 🧠⌨️

Real-time P300 BCI Speller using g.tec Unicorn Hybrid Black with PsychoPy visual stimulation, bandpass/notch filtering, and online single-trial classification.

## 🔬 Overview
A Brain-Computer Interface (BCI) speller leveraging the P300 Event-Related Potential (ERP). The system highlights rows and columns of a character matrix. When the user's target character flashes, a P300 wave is elicited. By classifying the EEG response in real-time, the system types the target character.

## 🏗️ Architecture
```mermaid
graph LR
    A[PsychoPy Stimulus] --> B[Unicorn EEG]
    B --> C[Preprocessing]
    C --> D[Feature Extraction]
    D --> E[SVM Classification]
    E --> F[Character Output]
```

## ⚙️ Signal Processing Pipeline
- **Filtering:** Bandpass (0.1-30Hz), Notch (50Hz)
- **Epoch Extraction & Baseline Correction**
- **Artifact Rejection:** Peak-to-peak thresholding

## 🛠️ Technologies
`Python` `PsychoPy` `SciPy` `NumPy` `BrainFlow` `g.tec Unicorn`
