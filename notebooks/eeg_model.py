#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Memory-Optimized EEG Real Data Training Script 
Train CNN model on real EEG data with different channel numbers including Alzheimer data
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mne
from scipy.signal import butter, filtfilt
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dropout, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping
import json
import gc  # Garbage collection

# Memory Optimization: Use mixed precision
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Configuration - Memory Optimized
DATA_DIR = 'models/data'
MODELS_DIR = 'models'
SFREQ = 256
WIN_SEC = 1.0  # Reduced from 2.0 to save 50% memory
STEP_SEC = 0.8  # Reduced overlap to save memory
CLASS_NAMES = ['Alzheimer', 'Epilepsy', 'SleepDisorders', 'Depression']  
CNN_EPOCHS = 40
BATCH_SIZE = 16
RANDOM_STATE = 42

# Ensure directories exist
os.makedirs(MODELS_DIR, exist_ok=True)

print(f"Using data from: {DATA_DIR}")
print(f"Saving models to: {MODELS_DIR}")
print(f"Class names: {CLASS_NAMES}")
print(f"Memory-optimized settings: WIN_SEC={WIN_SEC}, BATCH_SIZE={BATCH_SIZE}")

# Utility functions
def bandpass_filter(data, low=0.5, high=40.0, fs=SFREQ, order=4):
    """Apply bandpass filter to EEG data"""
    nyq = 0.5 * fs
    try:
        b, a = butter(order, [low/nyq, high/nyq], btype='band')
        return filtfilt(b, a, data, axis=0).astype(np.float32)
    except Exception as e:
        print(f"Filtering failed: {e}, returning original data")
        return data.astype(np.float32)

def normalize_zscore(windows):
    """Apply z-score normalization to windows - memory efficient"""
    mean = windows.mean(axis=(0, 1), keepdims=True)
    std = windows.std(axis=(0, 1), keepdims=True) + 1e-6
    normalized = ((windows - mean) / std).astype(np.float32)
    del mean, std
    return normalized

def standardize_channels(windows, target_channels=16):
    """Standardize number of channels by padding or selecting subset"""
    current_channels = windows.shape[2]

    if current_channels == target_channels:
        return windows
    elif current_channels < target_channels:
        # Pad with zeros
        print(f"    Padding from {current_channels} to {target_channels} channels")
        padding = np.zeros((windows.shape[0], windows.shape[1], 
                          target_channels - current_channels), dtype=np.float32)
        return np.concatenate([windows, padding], axis=2)
    else:
        # Select subset of channels (take first N channels)
        print(f"    Selecting first {target_channels} from {current_channels} channels")
        return windows[:, :, :target_channels]

def load_windows_mat_optimized(mat_path, target_channels=16, max_duration=300):
    """Load and process .mat file for Alzheimer EEG data"""
    try:
        print(f"Loading MAT file: {mat_path}")

        if not os.path.exists(mat_path):
            print(f"  Error: File not found: {mat_path}")
            return None

        # Load .mat file
        mat_data = loadmat(mat_path)
        print(f"  MAT file keys: {list(mat_data.keys())}")

        # Find the EEG data array (skip MATLAB metadata keys starting with '_')
        data_keys = [k for k in mat_data.keys() if not k.startswith('_')]

        if not data_keys:
            print(f"  Error: No data found in {mat_path}")
            return None

        # Use the first non-metadata key as EEG data
        data_key = data_keys[0]
        eeg_data = mat_data[data_key]
        print(f"  Using data key: {data_key}")
        print(f"  Original data shape: {eeg_data.shape}")

        # Handle different possible data arrangements
        # Common formats: (channels, samples), (samples, channels), (trials, channels, samples)
        if len(eeg_data.shape) == 3:
            # If 3D, likely (trials, channels, samples) or (channels, samples, trials)
            # Reshape to (samples, channels)
            if eeg_data.shape[0] < eeg_data.shape[2]:  # likely (trials, channels, samples)
                eeg_data = eeg_data.reshape(-1, eeg_data.shape[1]).T  # (samples, channels)
            else:  # likely (channels, samples, trials)
                eeg_data = eeg_data.reshape(eeg_data.shape[0], -1).T  # (samples, channels)
        elif len(eeg_data.shape) == 2:
            # If 2D, check if we need to transpose
            if eeg_data.shape[0] > eeg_data.shape[1]:
                # Already (samples, channels)
                pass
            else:
                # Transpose from (channels, samples) to (samples, channels)
                eeg_data = eeg_data.T
        else:
            print(f"  Error: Unexpected data shape: {eeg_data.shape}")
            return None

        eeg_data = eeg_data.astype(np.float32)
        print(f"  Reshaped data to: {eeg_data.shape} (samples, channels)")

        # Limit duration to save memory
        max_samples = int(max_duration * SFREQ)
        if eeg_data.shape[0] > max_samples:
            eeg_data = eeg_data[:max_samples]
            print(f"  Limited to {max_duration}s ({max_samples} samples)")

        # Apply bandpass filter
        eeg_data = bandpass_filter(eeg_data)
        print(f"  Applied bandpass filter (0.5-40 Hz)")

        # Create sliding windows
        win = int(WIN_SEC * SFREQ)
        step = int(STEP_SEC * SFREQ)

        # Handle too-short recordings
        if eeg_data.shape[0] < win:
            print(f"  Warning: Data too short ({eeg_data.shape[0]} samples), padding...")
            padding = np.zeros((win - eeg_data.shape[0], eeg_data.shape[1]), dtype=np.float32)
            eeg_data = np.vstack([eeg_data, padding])

        # Create windows efficiently
        windows = []
        for i in range(0, eeg_data.shape[0] - win + 1, step):
            windows.append(eeg_data[i:i+win])

            if len(windows) % 1000 == 0:
                print(f"    Created {len(windows)} windows...")

        if len(windows) == 0:
            print(f"  Error: No windows created from {mat_path}")
            return np.array([]).reshape(0, win, target_channels)

        windows = np.stack(windows, dtype=np.float32)
        print(f"  Created {len(windows)} windows of shape {windows[0].shape}")

        # Standardize number of channels
        windows = standardize_channels(windows, target_channels)
        print(f"  Standardized to shape: {windows[0].shape}")

        del eeg_data, mat_data
        gc.collect()

        return windows

    except Exception as e:
        print(f"Error loading {mat_path}: {e}")
        return None

def load_windows_edf_optimized(edf_path, picks='eeg', max_duration=300, target_channels=16):
    """Memory-optimized EDF loading with channel standardization"""
    try:
        print(f"Loading EDF: {edf_path}")

        if not os.path.exists(edf_path):
            print(f"  Error: File not found: {edf_path}")
            return None

        raw = mne.io.read_raw_edf(edf_path, preload=False, stim_channel=None, verbose=False)

        print(f"  Original sampling rate: {raw.info['sfreq']} Hz")
        print(f"  Number of channels: {len(raw.ch_names)}")

        # Limit duration to save memory
        duration = min(raw.times[-1], max_duration)
        raw.crop(tmax=duration)
        print(f"  Using duration: {duration:.1f} seconds")

        raw.load_data()

        # Resample if needed
        if raw.info['sfreq'] != SFREQ:
            raw.resample(SFREQ, verbose=False)
            print(f"  Resampled to {SFREQ} Hz")

        # Get EEG data: (channels, time) -> (time, channels)
        data = raw.get_data(picks=picks).T.astype(np.float32)
        print(f"  Data shape after transpose: {data.shape}")

        # Clear raw data from memory
        del raw
        gc.collect()

        # Apply bandpass filter
        data = bandpass_filter(data)
        print(f"  Applied bandpass filter (0.5-40 Hz)")

        # Create sliding windows
        win = int(WIN_SEC * SFREQ)
        step = int(STEP_SEC * SFREQ)

        # Handle too-short recordings
        if data.shape[0] < win:
            print(f"  Warning: Data too short ({data.shape[0]} samples), padding...")
            padding = np.zeros((win - data.shape[0], data.shape[1]), dtype=np.float32)
            data = np.vstack([data, padding])

        # Create windows efficiently
        windows = []
        for i in range(0, data.shape[0] - win + 1, step):
            windows.append(data[i:i+win])

            if len(windows) % 1000 == 0:
                print(f"    Created {len(windows)} windows...")

        if len(windows) == 0:
            print(f"  Error: No windows created from {edf_path}")
            return np.array([]).reshape(0, win, target_channels)

        windows = np.stack(windows, dtype=np.float32)
        print(f"  Created {len(windows)} windows of shape {windows[0].shape}")

        # Standardize number of channels
        windows = standardize_channels(windows, target_channels)
        print(f"  Standardized to shape: {windows[0].shape}")

        del data
        gc.collect()

        return windows

    except Exception as e:
        print(f"Error loading {edf_path}: {e}")
        return None

# Load data with memory management
print("\n=== LOADING DATA ===")

# Determine target channel count
TARGET_CHANNELS = 16  # Reasonable number that works for all datasets
print(f"Target channels for standardization: {TARGET_CHANNELS}")

eeg_datasets = []

# 1) Process Alzheimer MAT files (Class 0)
print("\n1. Loading Alzheimer MAT files...")
alzheimer_files = [
    ('models/data/AD4/1.mat', 0),
    ('models/data/AD4/2.mat', 0),
    ('models/data/AD4/3.mat', 0),
    ('models/data/AD4/4.mat', 0)
]

for mat_file, label in alzheimer_files:
    windows = load_windows_mat_optimized(mat_file, target_channels=TARGET_CHANNELS)
    if windows is not None and len(windows) > 0:
        windows = normalize_zscore(windows)
        labels = np.full(len(windows), label, dtype=int)
        eeg_datasets.append((windows, labels, f'Alzheimer_{os.path.basename(mat_file)}'))
        print(f"    Added {len(windows)} windows for {os.path.basename(mat_file)}")
        gc.collect()

# 2) Process Epilepsy files (Class 1)
print("\n2. Loading Epilepsy EDF files...")
epilepsy_files = [
    ('models/data/chb01_03.edf', 1),
    ('models/data/chb01_04.edf', 1)
]

for edf_file, label in epilepsy_files:
    windows = load_windows_edf_optimized(edf_file, target_channels=TARGET_CHANNELS)
    if windows is not None and len(windows) > 0:
        windows = normalize_zscore(windows)
        labels = np.full(len(windows), label, dtype=int)
        eeg_datasets.append((windows, labels, f'Epilepsy_{os.path.basename(edf_file)}'))
        print(f"    Added {len(windows)} windows for {os.path.basename(edf_file)}")
        gc.collect()

# 3) Process Sleep Disorders file (Class 2) - NO NORMAL CLASS
print("\n3. Loading Sleep Disorders EDF...")
sleep_windows = load_windows_edf_optimized('models/data/SC4001E0-PSG.edf', target_channels=TARGET_CHANNELS)
if sleep_windows is not None and len(sleep_windows) > 0:
    sleep_windows = normalize_zscore(sleep_windows)
    # All sleep data as sleep disorders (class 2) - NO normal class
    labels_sleep = np.full(len(sleep_windows), 2, dtype=int)
    eeg_datasets.append((sleep_windows, labels_sleep, 'Sleep_Disorders'))
    print(f"    Added {len(sleep_windows)} sleep disorder windows (no normal class)")
    del sleep_windows
    gc.collect()

# 4) Process Depression file (Class 3)
print("\n4. Loading Depression EDF...")
depression_windows = load_windows_edf_optimized('models/data/HS5TASK.edf', target_channels=TARGET_CHANNELS)
if depression_windows is not None and len(depression_windows) > 0:
    depression_windows = normalize_zscore(depression_windows)
    labels = np.full(len(depression_windows), 3, dtype=int)
    eeg_datasets.append((depression_windows, labels, 'Depression'))
    print(f"    Added {len(depression_windows)} depression windows")
    del depression_windows
    gc.collect()

# Combine datasets efficiently
print("\n=== COMBINING EEG DATA ===")
if len(eeg_datasets) == 0:
    print("  Error: No EEG data available for CNN training!")
    exit(1)

# Calculate total size
total_windows = sum(len(windows) for windows, labels, name in eeg_datasets)
print(f"  Total windows across all datasets: {total_windows}")

# Get dimensions from first dataset
sample_shape = eeg_datasets[0][0].shape[1:]
print(f"  Standardized window shape: {sample_shape}")

# Combine data efficiently
print("  Combining datasets...")
X_cnn = np.empty((total_windows, *sample_shape), dtype=np.float32)
y_cnn = np.empty(total_windows, dtype=int)

current_idx = 0
for windows, labels, name in eeg_datasets:
    end_idx = current_idx + len(windows)
    X_cnn[current_idx:end_idx] = windows
    y_cnn[current_idx:end_idx] = labels
    current_idx = end_idx
    print(f"    Added {name}: {len(windows)} windows")
    del windows, labels
    gc.collect()

print(f"  Final combined data: {X_cnn.shape}, labels: {y_cnn.shape}")
print(f"  Class distribution: {np.bincount(y_cnn)}")

# CRITICAL FIX: Convert numpy types to Python types for JSON serialization
unique_classes = [int(x) for x in sorted(set(y_cnn))]
print(f"  Classes present: {unique_classes}")

# Check if we have enough classes for training
if len(unique_classes) < 2:
    print(f"  Error: Need at least 2 classes for training, got {len(unique_classes)}")
    exit(1)

# Memory usage check
memory_usage_gb = X_cnn.nbytes / (1024**3)
print(f"  Memory usage: {memory_usage_gb:.2f} GB")

# Train/test split
print("\n=== SPLITTING DATA ===")
X_train, X_test, y_train, y_test = train_test_split(
    X_cnn, y_cnn, test_size=0.2, random_state=RANDOM_STATE, stratify=y_cnn
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Train class distribution: {np.bincount(y_train)}")
print(f"Test class distribution: {np.bincount(y_test)}")

# Clear full dataset to save memory
del X_cnn, y_cnn
gc.collect()

# Build CNN model
print("\n=== BUILDING ALZHEIMER-INCLUDED CNN MODEL ===")
input_shape = X_train.shape[1:]
print(f"Input shape: {input_shape}")

num_output_classes = len(unique_classes)
print(f"Number of output classes: {num_output_classes}")

model = Sequential([
    Conv1D(16, 5, activation='relu', input_shape=input_shape),
    Dropout(0.5),
    Conv1D(32, 3, activation='relu'),
    Dropout(0.5),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(num_output_classes, activation='softmax', dtype='float32')
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Memory-efficient training
print("\n=== TRAINING CNN ===")
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=CNN_EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop],
    verbose=1
)

# Plot training history
print("\n=== PLOTTING TRAINING HISTORY ===")
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('CNN Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('CNN Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(MODELS_DIR, 'training_history_alzheimer_fixed.png'))
plt.show()

# Evaluate CNN
print("\n=== EVALUATING CNN ===")
y_pred = model.predict(X_test, batch_size=BATCH_SIZE)
y_pred = np.argmax(y_pred, axis=1)

# Get class names for the classes we actually have
available_class_names = [CLASS_NAMES[i] for i in unique_classes]

print("CNN Classification Report:")
print(classification_report(y_test, y_pred, target_names=available_class_names))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', 
            xticklabels=available_class_names,
            yticklabels=available_class_names,
            cmap='Blues')
plt.title('Confusion Matrix - CNN')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig(os.path.join(MODELS_DIR, 'confusion_matrix_alzheimer_fixed.png'))
plt.show()

# Save model and configuration
print("\n=== SAVING MODEL ===")
model_path = os.path.join(MODELS_DIR, 'alzheimer_included_eeg_cnn_model.h5')
config_path = os.path.join(MODELS_DIR, 'alzheimer_included_eeg_config.json')

model.save(model_path)
print(f"Model saved to: {model_path}")

# FIXED: Save configuration with JSON-serializable types
config = {
    'class_names': CLASS_NAMES,
    'available_classes': unique_classes,  # Already converted to Python int
    'available_class_names': available_class_names,
    'num_timesteps': int(WIN_SEC * SFREQ),
    'num_channels': int(TARGET_CHANNELS),  # Ensure int
    'target_channels': int(TARGET_CHANNELS),  # Ensure int
    'sfreq': int(SFREQ),  # Ensure int
    'win_sec': float(WIN_SEC),  # Ensure float
    'step_sec': float(STEP_SEC),  # Ensure float
    'total_samples': int(len(X_train) + len(X_test)),
    'train_samples': int(len(X_train)),
    'test_samples': int(len(X_test)),
    'final_val_accuracy': float(history.history['val_accuracy'][-1]),
    'final_val_loss': float(history.history['val_loss'][-1]),
    'channel_standardized': True,
    'alzheimer_included': True,
    'memory_optimized': True,
    'batch_size': int(BATCH_SIZE),
    'epochs_trained': int(len(history.history['accuracy'])),
    'model_version': '2.0'
}

try:
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Configuration saved to: {config_path}")
except Exception as e:
    print(f"Configuration save error: {e}")

# FIXED: Save training results with JSON-serializable types
try:
    results = {
        'classification_report': classification_report(y_test, y_pred, 
                                                      target_names=available_class_names,
                                                      output_dict=True),
        'confusion_matrix': cm.tolist(),  # Convert numpy array to list
        'history': {
            'accuracy': [float(x) for x in history.history['accuracy']],
            'val_accuracy': [float(x) for x in history.history['val_accuracy']],
            'loss': [float(x) for x in history.history['loss']],
            'val_loss': [float(x) for x in history.history['val_loss']]
        },
        'memory_usage_gb': float(memory_usage_gb),
        'channel_info': {
            'target_channels': int(TARGET_CHANNELS),
            'original_alzheimer_channels': 'variable',
            'original_epilepsy_channels': 23,
            'original_sleep_channels': 7,
            'original_depression_channels': 22
        }
    }

    results_path = os.path.join(MODELS_DIR, 'alzheimer_included_training_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Training results saved to: {results_path}")
except Exception as e:
    print(f"Results save error: {e}")

print(f"\n=== TRAINING COMPLETE ===")
print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
print(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")
print(f"Classes trained: {available_class_names}")
print(f"Memory usage: {memory_usage_gb:.2f} GB")
print(f"Channels standardized to: {TARGET_CHANNELS}")
print(f"Alzheimer data included from 4 MAT files!")
print(f"Model ready for use in web application!")

# Final cleanup
del X_train, X_test, y_train, y_test
gc.collect()
print("Memory cleaned up successfully!")
