# LSTM Audio to Text Conversion Using LSTM Layers

This project implements a Recurrent Neural Network (RNN) using Long Short-Term Memory (LSTM) layers for converting audio commands into text. The model is trained on a dataset of labeled audio commands and can perform inference to predict the spoken command from a given `.wav` file.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Folder Structure](#folder-structure)
3. [How the Code Works](#how-the-code-works)
   - [Preprocessing](#preprocessing)
   - [Model Architecture](#model-architecture)
   - [Training Modes](#training-modes)
   - [Inference](#inference)
   - [Profiling](#profiling)
4. [How to Run the Code](#how-to-run-the-code)
   - [Command-Line Arguments](#command-line-arguments)
5. [Dataset Details](#dataset-details)
6. [Why Two Training Modes?](#why-two-training-modes)
7. [Dependencies](#dependencies)

---

## Project Overview

This project uses LSTM layers to process audio commands and classify them into one of eight predefined categories: `down`, `go`, `left`, `no`, `right`, `stop`, `up`, and `yes`. The workflow includes:

1. Preprocessing audio files into MFCC (Mel-Frequency Cepstral Coefficients) features.
2. Training an LSTM-based model on the processed data.
3. Performing inference on new audio files to predict the spoken command.
4. Profiling the inference process for performance optimization.

---

## Folder Structure

The project folder is organized as follows:

```
LSTM_audio_to_text_SW/
├── lstm_wav2text_v3.py       # Main Python script for training and inference
├── mini_speech_commands/     # Dataset folder containing labeled audio files
│   ├── down/                 # Audio files for the "down" command
│   ├── go/                   # Audio files for the "go" command
│   ├── left/                 # Audio files for the "left" command
│   ├── no/                   # Audio files for the "no" command
│   ├── right/                # Audio files for the "right" command
│   ├── stop/                 # Audio files for the "stop" command
│   ├── up/                   # Audio files for the "up" command
│   ├── yes/                  # Audio files for the "yes" command
│   └── README.md             # Dataset description
```

### Dataset Usage in Code

The `mini_speech_commands` folder contains subfolders for each label (`down`, `go`, etc.), and each subfolder contains `.wav` files representing audio samples for that label. The script uses these files for training and inference:

- **Training**: The script loads audio files, preprocesses them into MFCC features, and trains the LSTM model.
- **Inference**: The script uses the trained model to predict the label of a given `.wav` file.

---

## How the Code Works

### Preprocessing

The `preprocess_audio` function converts audio files into MFCC features:

1. **Load Audio**: The `librosa` library loads the audio file at a sample rate of 16,000 Hz.
2. **Extract MFCC**: The MFCC features are extracted with 40 coefficients per frame.
3. **Padding**: The MFCC sequences are padded to a fixed length of 100 frames.

### Model Architecture

The model is built using TensorFlow's Keras API:

1. **Input Layer**: Accepts MFCC features with a shape of `(100, 40)`.
2. **Masking Layer**: Masks padded values to prevent them from affecting the model.
3. **LSTM Layers**: Three stacked LSTM layers with 1024, 2048, and 1024 units, respectively.
4. **Dense Layer**: Outputs probabilities for the 8 classes using a softmax activation.

### Training Modes

The script supports two training modes:

1. **Fast Training (`train_fast`)**:
   - Uses a limited number of files (default: 100) per label.
   - Trains for 2 epochs.
   - Suitable for quick experimentation.

2. **Full Training (`train_full`)**:
   - Uses 80% of the files per label for training.
   - Trains for 10 epochs.
   - Suitable for thorough training.

### Inference

The `run_inference` function performs the following steps:

1. Loads the trained model from `speech_model.keras`.
2. Preprocesses the input `.wav` file into MFCC features.
3. Pads the features and performs inference using the model.
4. Outputs the predicted label and inference time.

### Profiling

The `profile_inference` function uses TensorFlow's profiler to analyze the inference process. It generates logs that can be visualized using TensorBoard.

---

## How to Run the Code

### Command-Line Arguments

The script can be executed from the terminal with the following commands:

1. **Train Fast**:
   ```bash
   python lstm_wav2text_v3.py train_fast <num_per_label>
   ```
   - `<num_per_label>`: (Optional) Number of files per label to use for training. Default is 100.

2. **Train Full**:
   ```bash
   python lstm_wav2text_v3.py train_full
   ```
   - Trains the model using 80% of the dataset.

3. **Infer**:
   ```bash
   python lstm_wav2text_v3.py infer <path_to_wav>
   ```
   - `<path_to_wav>`: Path to the `.wav` file for inference.

4. **Profile**:
   ```bash
   python lstm_wav2text_v3.py prof
   ```
   - Profiles the inference process using TensorFlow's profiler.

---

## Dataset Details

The `mini_speech_commands` dataset contains audio files for eight labels:

- `down`, `go`, `left`, `no`, `right`, `stop`, `up`, `yes`

Each subfolder contains `.wav` files named numerically (e.g., `0.wav`, `1.wav`, etc.). These files are used for training and inference.

---

## Why Two Training Modes?

1. **Fast Training**:
   - Useful for quick testing and debugging.
   - Reduces training time by limiting the number of files and epochs.

2. **Full Training**:
   - Provides better model performance by using more data and training for more epochs.
   - Suitable for final model training.

---

## Dependencies

Install the required Python libraries using:

```bash
pip install tensorflow librosa numpy
```

---

## Notes

- Ensure the `mini_speech_commands` folder is in the same directory as the script.
- Use TensorBoard to visualize profiling logs:
  ```bash
  tensorboard --logdir=logs
  ```

