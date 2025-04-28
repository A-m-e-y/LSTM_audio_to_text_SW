import tensorflow as tf
tf.get_logger().setLevel('ERROR')
tf.config.run_functions_eagerly(True)

import librosa
import numpy as np
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
from glob import glob
from custom_lstm_cell import CustomLSTMCell
import cProfile
import pstats

# ========== Config ==========
SAMPLE_RATE = 16000
NUM_MFCC = 40
MAX_LEN = 100
NUM_CLASSES = 8

DATA_DIR = "mini_speech_commands"
LABELS = ['down', 'go', 'left', 'no', 'right', 'stop', 'up', 'yes']
LABEL2IDX = {label: i for i, label in enumerate(LABELS)}
IDX2LABEL = {i: label for label, i in LABEL2IDX.items()}

# ========== Preprocessing ==========
def preprocess_audio(path):
    y, sr = librosa.load(path, sr=SAMPLE_RATE)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=NUM_MFCC).T
    return mfcc

def load_data_from_disk(mode='fast', limit_per_label=100):
    mfccs = []
    labels = []

    for label in LABELS:
        dir_path = os.path.join(DATA_DIR, label)
        if not os.path.exists(dir_path):
            print(f"❌ Directory not found: {dir_path}")
            continue

        files = sorted(glob(os.path.join(dir_path, '*.wav')), key=lambda x: int(os.path.basename(x).split('.')[0]))

        if mode == 'fast':
            files_to_load = files[:limit_per_label]
        elif mode == 'full':
            files_to_load = files[:int(0.8 * len(files))]  # 80%
        else:
            raise ValueError("mode must be 'fast' or 'full'")

        for path in files_to_load:
            mfcc = preprocess_audio(path)
            mfccs.append(mfcc)
            labels.append(LABEL2IDX[label])

    mfccs = tf.keras.preprocessing.sequence.pad_sequences(
        mfccs, padding='post', maxlen=MAX_LEN, dtype='float32')
    labels = np.array(labels)

    return mfccs, labels

# ========== Model ==========
def build_model():
    inputs = tf.keras.Input(shape=(MAX_LEN, NUM_MFCC), name='mfcc_input')
    x = tf.keras.layers.Masking()(inputs)
    x = tf.keras.layers.LSTM(1024, return_sequences=True)(x)
    x = tf.keras.layers.LSTM(2048, return_sequences=True)(x)
    x = tf.keras.layers.LSTM(1024, return_sequences=False)(x)
    x = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)
    return tf.keras.Model(inputs, x)

# ========== Custom Model ==========
def build_custom_model():
    inputs = tf.keras.Input(shape=(MAX_LEN, NUM_MFCC), name='mfcc_input')
    x = tf.keras.layers.Masking()(inputs)
    cell_1 = CustomLSTMCell(1024)
    x = tf.keras.layers.RNN(cell_1, return_sequences=True)(x)
    cell_2 = CustomLSTMCell(2048)
    x = tf.keras.layers.RNN(cell_2, return_sequences=True)(x)
    cell_3 = CustomLSTMCell(1024)
    x = tf.keras.layers.RNN(cell_3, return_sequences=False)(x)
    x = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)
    return tf.keras.Model(inputs, x)

# ========== Training ==========
def train_fast(limit=100):
    print(f"⚡ Training FAST model using first {limit} files per label...")
    start_time = time.perf_counter()

    mfccs, labels = load_data_from_disk(mode='fast', limit_per_label=limit)
    # model = build_model()
    model = build_custom_model()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(mfccs, labels, batch_size=16, epochs=2)
    model.save('speech_model.keras')

    elapsed = time.perf_counter() - start_time
    print()
    print("##################################################################################")
    print(f"✅ Fast model saved.\n⏱️ Training time: {elapsed:.2f} seconds")
    print("##################################################################################")
    print()

def train_full():
    print(f"📚 Training FULL model using first 80% of files per label...")
    start_time = time.perf_counter()

    mfccs, labels = load_data_from_disk(mode='full')
    # model = build_model()
    model = build_custom_model()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(mfccs, labels, batch_size=16, epochs=10)
    model.save('speech_model.keras')

    elapsed = time.perf_counter() - start_time
    print()
    print("##################################################################################")
    print(f"✅ Full model saved.\n⏱️ Training time: {elapsed:.2f} seconds")
    print("##################################################################################")
    print()

# ========== Inference ==========
def run_inference(wav_path, batch=32):
    if not os.path.exists('speech_model.keras'):
        print("❌ Model not found. Please train the model first.")
        return

    model = tf.keras.models.load_model('speech_model.keras')
    mfcc = preprocess_audio(wav_path)
    mfcc = tf.keras.preprocessing.sequence.pad_sequences(
        [mfcc], maxlen=MAX_LEN, padding='post', dtype='float32')

    passes = np.repeat(mfcc, batch, axis=0)

    start_time = time.perf_counter()
    pred = model.predict(passes)
    elapsed = time.perf_counter() - start_time

    label_index = np.argmax(pred[0])
    print()
    print("##################################################################################")
    print(f"🎤 Prediction: {IDX2LABEL[label_index]}\n⏱️ Inference time: {elapsed:.4f} seconds")
    print("##################################################################################")
    print()

# ========== Profiling ==========
def profile_inference(wav_path, log_dir="logs"):
    import tensorflow as tf
    import numpy as np
    import os

    if not os.path.exists('speech_model.keras'):
        print("❌ Model not found. Train it first.")
        return

    if not os.path.exists(wav_path):
        print(f"❌ WAV file not found: {wav_path}")
        return

    print(f"🚀 Starting profiling...")

    # Prepare input
    mfcc = preprocess_audio(wav_path)
    mfcc = tf.keras.preprocessing.sequence.pad_sequences(
        [mfcc], maxlen=MAX_LEN, padding='post', dtype='float32')
    mfcc = tf.convert_to_tensor(mfcc)

    model = tf.keras.models.load_model('speech_model.keras')

    # Wrap the model call in a tf.function to enable tracing
    @tf.function
    def run_inference(x):
        return model(x, training=False)

    # Start profiling
    tf.profiler.experimental.start(log_dir)

    # Trigger a few warm-up calls
    for _ in range(3):
        _ = run_inference(mfcc)

    # Actual profiled run
    with tf.profiler.experimental.Trace("inference", step_num=0):
        _ = run_inference(mfcc)

    tf.profiler.experimental.stop()

    print(f"✅ Profiling complete.")
    print(f"👉 View it with: tensorboard --logdir={log_dir}")


# ========== CLI ==========
if __name__ == '__main__':
    
    # ➡️ Start profiling
    profiler = cProfile.Profile()
    profiler.enable()

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python lstm_speech_commands.py train_fast <num_per_label>")
        print("  python lstm_speech_commands.py train_full")
        print("  python lstm_speech_commands.py infer <path_to_wav>")
        sys.exit(1)

    mode = sys.argv[1]

    if mode == 'train_fast':
        if len(sys.argv) >= 3:
            try:
                n = int(sys.argv[2])
            except ValueError:
                print("❌ Please provide an integer for number of files per label.")
                sys.exit(1)
        else:
            n = 100
        train_fast(limit=n)

    elif mode == 'train_full':
        train_full()

    elif mode == 'infer':
        if len(sys.argv) < 4:
            print("❌ Please provide path to a WAV file.")
        else:
            run_inference(sys.argv[2], sys.argv[3])

    elif mode == 'prof':
        profile_inference("mini_speech_commands/yes/850.wav")

    else:
        print("❌ Unknown command. Use 'train_fast', 'train_full', or 'infer'")

        # ➡️ Stop profiling
    profiler.disable()
    profiler.dump_stats('profile_output.prof')  # Save profile output

    # ➡️ (Optional) Quick human-readable text output
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative').print_stats('custom_lstm_cell')  # Print stats for function named "sw_dot"

