 
# 🌊 Signal Viewer - Multi-Domain Signal Analysis Platform

Welcome to **Signal Viewer**, a comprehensive platform for analyzing signals across medical, acoustic, and satellite domains. Whether you're a researcher, clinician, or engineer, this tool brings professional-grade signal processing to your fingertips.

---

## 🎯 What's This All About?

Signal Viewer is your one-stop shop for analyzing different types of signals:

- **Medical Signals**: ECG and EEG analysis with AI-powered disease detection
- **Sound Analysis**: Doppler shift effects and drone acoustic detection
- **Satellite Data**: InSAR displacement mapping from Sentinel-1 imagery

Think of it as a Swiss Army knife for signal processing - multiple tools, one unified interface.

---

## 🚀 Features That Matter

### 📊 **ECG Analysis**
- Upload and visualize electrocardiogram data (.dat/.hea files)
- Multiple visualization modes:
  - Default continuous-time viewer
  - XOR overlay (spot repeating patterns)
  - Polar graphs (see signals in a whole new way)
  - Recurrence plots (find hidden patterns)
- **AI Disease Detection**: Detects 6 cardiac abnormalities:
  - First-degree AV Block
  - Right/Left Bundle Branch Block
  - Sinus Bradycardia
  - Atrial Fibrillation
  - ST-segment Changes
- Real-time playback with speed control
- Pan, zoom, and explore your data interactively

### 🧠 **EEG Analysis**
- Support for EDF, CSV, and TXT formats
- Same powerful visualization modes as ECG
- **Dual AI Models**:
  - 1D Multi-Channel Model (time-series analysis)
  - 2D Recurrence Plot Model (spatial pattern detection)
- Detects brain conditions:
  - Normal brain activity
  - Epilepsy
  - Sleep disorders
  - Depression
- Animated playback for temporal analysis

### 🚁 **Drone Detection**
- Upload audio files (.wav, .mp3, .flac)
- YAMNet-based deep learning classification
- Sliding window analysis for temporal detection
- Real-time probability visualization
- Spectrogram and waveform analysis
- Detection timeline with customizable thresholds

### 🎵 **Doppler Shift Analysis**
- **Sound Generation**: Create realistic car pass-by simulations
  - Adjustable velocity, frequency, and duration
  - Authentic engine harmonics and environmental sounds
- **AI Prediction**: Upload sounds to predict:
  - Vehicle velocity
  - Doppler-shifted frequency
- Visual feedback with waveform and frequency plots

### 🛰️ **InSAR Viewer**
- Process Sentinel-1 NetCDF interferogram data
- Surface displacement mapping
- Statistical analysis with interactive heatmaps
- Histogram distribution visualization
- Perfect for monitoring ground movement and deformation

---

## 🛠️ Installation

### Prerequisites
You'll need Python 3.8 or later. That's pretty much it.

### Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/signal-viewer.git
cd signal-viewer
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up the models directory**
```
project_root/
├── models/
│   ├── ecg_model.hdf5
│   ├── EEG_Model.h5
│   ├── EEG_2D_Model.h5
│   ├── channel_standardized_eeg_config.json
│   ├── doppler_regressor.h5
│   ├── scaler (2).pkl
│   └── yamnet_drone_classifier.h5
├── uploads/
└── pages/
```

4. **Run the app**
```bash
python app.py
```

5. **Open your browser**
Navigate to `http://localhost:8050`

---

## 📦 Dependencies

Here's what makes it all work:

**Core Framework**
- Dash (interactive web apps)
- Plotly (beautiful visualizations)

**Signal Processing**
- NumPy, SciPy (the math wizards)
- librosa (audio analysis)
- wfdb, MNE (medical signal processing)
- xarray (satellite data handling)

**Machine Learning**
- TensorFlow/Keras (neural networks)
- TensorFlow Hub (pre-trained models)

**File Handling**
- pandas (data manipulation)
- soundfile (audio I/O)
- Pillow (image processing)

Full list in `requirements.txt`

---

## 🎮 How to Use

### ECG Analysis
1. Click **ECG** from the home screen
2. Upload both `.dat` and `.hea` files (they're a pair!)
3. Click **Process ECG**
4. View AI predictions and explore visualizations
5. Use playback controls to navigate through time
6. Try different visualization modes for unique insights

### EEG Analysis
1. Click **EEG** from home
2. Upload your EEG file (.edf, .csv, or .txt)
3. Select channels you want to analyze
4. **Process EEG** for filtering and visualization
5. Run **1D Detection** or **2D Detection** for AI analysis
6. Animate through your data with playback controls

### Drone Detection
1. Navigate to **Drone Detection**
2. Upload an audio file
3. Adjust window size, hop size, and threshold
4. Click **Analyze Audio**
5. Watch the detection timeline light up
6. Review spectrogram and detection statistics

### Doppler Analysis
1. Choose **Doppler** from home
2. **Generate Sound**:
   - Set velocity, frequency, and duration
   - Listen to the synthetic car pass-by
3. **Predict**:
   - Upload a sound file
   - Get AI predictions for velocity and frequency

### InSAR Viewer
1. Click **InSAR Viewer**
2. Upload a Sentinel-1 NetCDF file (.nc)
3. Click **Generate Displacement Map**
4. Explore surface deformation with statistics

---

## 🧪 Technical Details

### XOR Overlay Explained
The XOR mode is like a visual diff tool for signals. When chunks of your signal are identical, they cancel out (go to zero). When they differ, patterns emerge. It's perfect for spotting variations in what should be repetitive signals.

### Recurrence Plots
These reveal when your signal "remembers" its past. A dark spot at coordinates (i, j) means the signal at time i is similar to time j. Diagonal lines? That's a repeating pattern!

### Polar Visualization
Ever wondered what your signal looks like as a spiral? Polar graphs map amplitude to radius and time to angle. It's particularly useful for spotting cyclical patterns that are hard to see in standard time-series plots.

---

## ⚠️ Important Notes

- **Medical Disclaimer**: This tool is for research and educational purposes only. It's NOT a substitute for professional medical diagnosis. Always consult healthcare professionals.

- **Model Files**: The AI models are NOT included in the repository due to size. You'll need to train them or obtain them separately.

- **Performance**: Processing large files may take time. The app uses downsampling and chunking for responsiveness.

- **Browser Storage**: The app doesn't use localStorage due to platform limitations. All data is session-based.

---

## 🐛 Troubleshooting

**Models not loading?**
- Check that all model files are in the `models/` directory
- Verify file names match exactly (case-sensitive!)

**Upload fails?**
- Ensure file formats match requirements
- Check file size (very large files may timeout)

**Visualization glitches?**
- Try reducing the time window
- Select fewer channels for better performance

**Callbacks not working?**
- Check browser console for errors
- Ensure you're running the latest version

---

## 🤝 Contributing

Found a bug? Have an idea? Contributions are welcome!

1. Fork the repo
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 Project Structure

```
signal-viewer/
├── app.py                    # Main application entry point
├── pages/
│   ├── ecg_page.py          # ECG analysis interface
│   ├── eeg_page.py          # EEG analysis interface
│   ├── drone_page.py        # Drone detection interface
│   ├── doppler_page.py      # Doppler shift analysis
│   └── SAR_page.py          # InSAR displacement viewer
├── models/                   # AI model files (not included)
├── uploads/                  # Temporary file storage
└── requirements.txt          # Python dependencies
```

---

## 🎓 Learning Resources

New to signal processing? Check these out:
- [DSP Guide](https://www.dspguide.com/) - Digital Signal Processing fundamentals
- [PhysioNet](https://physionet.org/) - Medical signal databases
- [Librosa Documentation](https://librosa.org/) - Audio analysis tutorials

---

## 📜 License

This project is open source and available for educational and research purposes.

---

## 🙏 Acknowledgments

Built with:
- Dash by Plotly
- TensorFlow by Google
- YAMNet acoustic model
- WFDB-Python for ECG processing
- MNE-Python for EEG analysis
- The entire open-source community

---

## 📧 Contact

Questions? Issues? Feedback?
- Open an issue on GitHub
- Check the troubleshooting section
- Review the code documentation

---

**Happy Signal Processing! 📊🔊🧠**

*Remember: With great signal processing power comes great responsibility. Use wisely!*
