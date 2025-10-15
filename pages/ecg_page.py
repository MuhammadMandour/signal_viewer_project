from dash import html, dcc, Input, Output, State, callback
import os
import base64
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import wfdb
<<<<<<< HEAD
=======
from pyts.image import RecurrencePlot
>>>>>>> main
from PIL import Image
from scipy.signal import butter, filtfilt, resample
import io
import tensorflow as tf
from tensorflow import keras

# ----------------- Ensure uploads folder exists -----------------
os.makedirs("uploads", exist_ok=True)

# ----------------- Load Model -----------------
MODEL_PATH = os.path.join('models', 'ecg_model.hdf5')

# Global variable to store model
model = None

def load_classifier():
    """Load the trained model for 6 abnormality predictions"""
    global model
    try:
        if os.path.exists(MODEL_PATH):
            model = keras.models.load_model(MODEL_PATH, compile=False)
            print(f"Model loaded successfully from {MODEL_PATH}")
        else:
            print(f"Warning: Model file not found at {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")

# Load model on module import
load_classifier()

# ----------------- Layout -----------------
layout = html.Div(
    style={'backgroundColor': '#1a1a2e', 'minHeight': '100vh', 'fontFamily': 'Arial, sans-serif', 'padding': '30px'},
    children=[
        html.H1("ECG Signal Viewer", 
                style={'color':'#00d9ff','fontSize':'48px','marginBottom':'20px', 'textAlign': 'center'}),

        # Upload Section
        html.Div(
            style={'maxWidth': '1200px', 'margin': '0 auto', 'marginBottom': '30px'},
            children=[
                dcc.Upload(
                    id='upload-ecg',
                    children=html.Div(['Drag and Drop or ', html.A('Select .dat and .hea files')]),
                    style={
                        'width': '100%', 'height': '100px', 'lineHeight': '100px',
                        'borderWidth': '2px', 'borderStyle': 'dashed', 'borderRadius': '10px',
                        'textAlign': 'center', 'backgroundColor': '#16213e', 'color': '#00d9ff',
                        'cursor': 'pointer', 'border': '2px dashed #00d9ff'
                    },
                    multiple=True
                ),
                html.Div(id='file-list', style={'marginTop': '15px', 'fontSize':'16px','color':'#00d9ff', 'textAlign': 'center'}),
                
                html.Button("Process ECG", id='process-btn', n_clicks=0, style={
                    'marginTop':'20px', 'backgroundColor': '#00d9ff', 'color': '#1a1a2e',
                    'borderRadius': '8px', 'padding': '12px 40px', 'fontSize':'16px',
                    'border':'none', 'cursor':'pointer', 'fontWeight': 'bold',
                    'display': 'block', 'margin': '20px auto'
                }),
            ]
        ),

        # Processing Status
        dcc.Loading(
            id="loading-ecg",
            type="circle",
            color="#00d9ff",
            children=[
                html.Div(id='ecg-output', style={'marginTop': '20px','fontSize':'18px','color':'#00d9ff', 'textAlign': 'center'}),
            ]
        ),

        # Main Content Container
        html.Div(
            id='main-content',
            style={'display': 'none', 'maxWidth': '1800px', 'margin': '0 auto'},
            children=[
                # Health Prediction Section (Top)
                html.Div(
                    id='prediction-section',
                    style={'marginBottom': '30px'},
                    children=[
                        html.Div(
                            id='prediction-card',
                            style={
                                'backgroundColor': '#16213e',
                                'borderRadius': '15px',
                                'padding': '30px',
                                'maxWidth': '900px',
                                'margin': '0 auto',
                                'boxShadow': '0 4px 20px rgba(0, 217, 255, 0.3)',
                                'border': '2px solid #00d9ff'
                            },
                            children=[
                                html.H2("🏥 ECG Abnormality Analysis", style={'color': '#00d9ff', 'marginBottom': '25px', 'textAlign': 'center'}),
                                html.Div(id='prediction-result', style={'fontSize': '24px', 'fontWeight': 'bold', 'marginBottom': '20px', 'textAlign': 'center'}),
                                html.Div(id='prediction-probabilities', style={'marginTop': '20px'}),
                                html.Div(id='prediction-details', style={'fontSize': '16px', 'color': '#ccc', 'marginTop': '20px'})
                            ]
                        )
                    ]
                ),

                # Controls and Visualization Section
                html.Div(
                    style={'display': 'flex', 'gap': '20px', 'marginTop': '30px'},
                    children=[
                        # Left Panel - Controls
                        html.Div(
                            style={
                                'backgroundColor': '#16213e',
                                'borderRadius': '15px',
                                'padding': '25px',
                                'width': '350px',
                                'boxShadow': '0 4px 20px rgba(0, 217, 255, 0.2)',
                                'border': '1px solid #00d9ff',
                                'maxHeight': '800px',
                                'overflowY': 'auto'
                            },
                            children=[
                                html.H3("Viewer Controls", style={'color': '#00d9ff', 'marginBottom': '20px', 'fontSize': '24px'}),
                                
                                # Graph Mode Selection
                                html.Label("Viewer Type", style={'color': '#00d9ff', 'fontSize': '14px', 'marginBottom': '8px', 'display': 'block'}),
                                dcc.Dropdown(
                                    id='graph-mode',
                                    options=[
                                        {'label': '📊 Default Continuous Time', 'value': 'waveform'},
                                        {'label': '⚡ XOR Chunks', 'value': 'xor'},
                                        {'label': '🎯 Polar Graph', 'value': 'polar'},
                                        {'label': '🔄 Reoccurrence Graph', 'value': 'reoccurrence'}
                                    ],
                                    value='waveform',
                                    style={
                                        'marginBottom': '20px',
                                        'backgroundColor': '#0f1626',
                                        'color': '#1a1a2e',
                                    },
                                    className='custom-dropdown'
                                ),
                                
                                # Window Width Control
                                html.Label("Viewport/Chunk Width (seconds)", style={'color': '#00d9ff', 'fontSize': '14px', 'marginBottom': '8px', 'display': 'block'}),
                                dcc.Input(
                                    id='window-width',
                                    type='number',
                                    value=5,
                                    min=1,
                                    max=30,
                                    step=0.5,
                                    style={
                                        'width': '100%',
                                        'padding': '10px',
                                        'borderRadius': '8px',
                                        'border': '1px solid #00d9ff',
                                        'backgroundColor': '#0f1626',
                                        'color': '#00d9ff',
                                        'fontSize': '16px',
                                        'marginBottom': '20px'
                                    }
                                ),
                                
                                # Speed Control
                                html.Label("Playback Speed", style={'color': '#00d9ff', 'fontSize': '14px', 'marginBottom': '8px', 'display': 'block'}),
                                dcc.Slider(
                                    id='speed-control',
                                    min=0.5,
                                    max=5,
                                    step=0.5,
                                    value=1,
                                    marks={0.5: '0.5x', 1: '1x', 2: '2x', 3: '3x', 5: '5x'},
                                    tooltip={"placement": "bottom", "always_visible": False},
                                    className='custom-slider'
                                ),
                                html.Div(style={'marginBottom': '20px'}),
                                
                                # Zoom Control
                                html.Label("Amplitude Zoom", style={'color': '#00d9ff', 'fontSize': '14px', 'marginBottom': '8px', 'display': 'block'}),
                                dcc.Slider(
                                    id='zoom-control',
                                    min=0.5,
                                    max=3,
                                    step=0.25,
                                    value=1,
                                    marks={0.5: '0.5x', 1: '1x', 2: '2x', 3: '3x'},
                                    tooltip={"placement": "bottom", "always_visible": False},
                                    className='custom-slider'
                                ),
                                html.Div(style={'marginBottom': '20px'}),
                                
                                # Polar Mode (for polar graph)
                                html.Div(
                                    id='polar-mode-container',
                                    style={'display': 'none'},
                                    children=[
                                        html.Label("Polar Display Mode", style={'color': '#00d9ff', 'fontSize': '14px', 'marginBottom': '8px', 'display': 'block'}),
                                        dcc.RadioItems(
                                            id='polar-mode',
                                            options=[
                                                {'label': ' Latest Fixed Time', 'value': 'latest'},
                                                {'label': ' Cumulative', 'value': 'cumulative'}
                                            ],
                                            value='latest',
                                            style={'color': '#00d9ff', 'marginBottom': '20px'},
                                            labelStyle={'display': 'block', 'marginBottom': '8px'}
                                        ),
                                    ]
                                ),
                                
                                # Colormap Selection (for 2D representations)
                                html.Div(
                                    id='colormap-container',
                                    style={'display': 'none'},
                                    children=[
                                        html.Label("Colormap", style={'color': '#00d9ff', 'fontSize': '14px', 'marginBottom': '8px', 'display': 'block'}),
                                        dcc.Dropdown(
                                            id='colormap-selection',
                                            options=[
                                                {'label': 'Viridis', 'value': 'Viridis'},
                                                {'label': 'Plasma', 'value': 'Plasma'},
                                                {'label': 'Inferno', 'value': 'Inferno'},
                                                {'label': 'Hot', 'value': 'Hot'},
                                                {'label': 'Jet', 'value': 'Jet'},
                                                {'label': 'Rainbow', 'value': 'Rainbow'},
                                                {'label': 'Blues', 'value': 'Blues'},
                                                {'label': 'Reds', 'value': 'Reds'},
                                            ],
                                            value='Viridis',
                                            style={
                                                'marginBottom': '20px',
                                                'backgroundColor': '#0f1626',
                                                'color': '#1a1a2e',
                                            },
                                        ),
                                    ]
                                ),
                                
                                # Channel Selection
                                html.Label("Select Channels to Display", style={'color': '#00d9ff', 'fontSize': '14px', 'marginBottom': '12px', 'display': 'block'}),
                                dcc.Checklist(
                                    id='channel-selection',
                                    options=[],
                                    value=[],
                                    style={'color': '#00d9ff', 'marginBottom': '20px'},
                                    labelStyle={'display': 'block', 'marginBottom': '8px', 'cursor': 'pointer'}
                                ),
                                
                                # Reoccurrence Channel Selection (X and Y)
                                html.Div(
                                    id='reoccurrence-channel-container',
                                    style={'display': 'none'},
                                    children=[
                                        html.Label("X-Axis Channel", style={'color': '#00d9ff', 'fontSize': '14px', 'marginBottom': '8px', 'display': 'block'}),
                                        dcc.Dropdown(
                                            id='reoccurrence-x-channel',
                                            options=[],
                                            value=None,
                                            style={
                                                'marginBottom': '15px',
                                                'backgroundColor': '#0f1626',
                                                'color': '#1a1a2e',
                                            },
                                        ),
                                        html.Label("Y-Axis Channel", style={'color': '#00d9ff', 'fontSize': '14px', 'marginBottom': '8px', 'display': 'block'}),
                                        dcc.Dropdown(
                                            id='reoccurrence-y-channel',
                                            options=[],
                                            value=None,
                                            style={
                                                'marginBottom': '20px',
                                                'backgroundColor': '#0f1626',
                                                'color': '#1a1a2e',
                                            },
                                        ),
                                    ]
                                ),
                                
                                # Playback Controls
                                html.Div(
                                    style={'display': 'flex', 'gap': '10px', 'marginTop': '20px'},
                                    children=[
                                        html.Button(
                                            "▶ Play",
                                            id='stream-btn',
                                            n_clicks=0,
                                            style={
                                                'flex': '1',
                                                'backgroundColor': '#00d9ff',
                                                'color': '#1a1a2e',
                                                'borderRadius': '8px',
                                                'padding': '12px',
                                                'fontSize': '14px',
                                                'border': 'none',
                                                'cursor': 'pointer',
                                                'fontWeight': 'bold',
                                            }
                                        ),
                                        html.Button(
                                            "⏸ Pause",
                                            id='pause-btn',
                                            n_clicks=0,
                                            style={
                                                'flex': '1',
                                                'backgroundColor': '#ff006e',
                                                'color': '#fff',
                                                'borderRadius': '8px',
                                                'padding': '12px',
                                                'fontSize': '14px',
                                                'border': 'none',
                                                'cursor': 'pointer',
                                                'fontWeight': 'bold',
                                            }
                                        ),
                                    ]
                                ),
                                
                                # Pan Controls
                                html.Div(
                                    style={'display': 'flex', 'gap': '10px', 'marginTop': '15px'},
                                    children=[
                                        html.Button(
                                            "⏮ Reset",
                                            id='reset-btn',
                                            n_clicks=0,
                                            style={
                                                'flex': '1',
                                                'backgroundColor': '#8338ec',
                                                'color': '#fff',
                                                'borderRadius': '8px',
                                                'padding': '12px',
                                                'fontSize': '14px',
                                                'border': 'none',
                                                'cursor': 'pointer',
                                                'fontWeight': 'bold',
                                            }
                                        ),
                                        html.Button(
                                            "◀ Back",
                                            id='pan-back-btn',
                                            n_clicks=0,
                                            style={
                                                'flex': '1',
                                                'backgroundColor': '#2a9d8f',
                                                'color': '#fff',
                                                'borderRadius': '8px',
                                                'padding': '12px',
                                                'fontSize': '14px',
                                                'border': 'none',
                                                'cursor': 'pointer',
                                                'fontWeight': 'bold',
                                            }
                                        ),
                                        html.Button(
                                            "▶ Forward",
                                            id='pan-forward-btn',
                                            n_clicks=0,
                                            style={
                                                'flex': '1',
                                                'backgroundColor': '#2a9d8f',
                                                'color': '#fff',
                                                'borderRadius': '8px',
                                                'padding': '12px',
                                                'fontSize': '14px',
                                                'border': 'none',
                                                'cursor': 'pointer',
                                                'fontWeight': 'bold',
                                            }
                                        ),
                                    ]
                                ),
                            ]
                        ),
                        
                        # Right Panel - Visualization
                        html.Div(
                            style={'flex': '1'},
                            children=[
                                dcc.Graph(
                                    id='ecg-stream-graph',
                                    style={
                                        'height': '700px',
                                        'backgroundColor': '#16213e',
                                        'borderRadius': '15px',
                                        'boxShadow': '0 4px 20px rgba(0, 217, 255, 0.2)',
                                    },
                                    config={'displayModeBar': True, 'displaylogo': False}
                                )
                            ]
                        )
                    ]
                ),
                
                # Interval component for streaming
                dcc.Interval(
                    id='interval-component',
                    interval=100,
                    n_intervals=0,
                    disabled=True
                ),
                
                # Store components
                dcc.Store(id='ecg-data-store'),
                dcc.Store(id='stream-state', data={'streaming': False, 'position': 0, 'polar_data': [], 'xor_chunks': []}),
                dcc.Store(id='pan-clicks', data={'back': 0, 'forward': 0, 'reset': 0})
            ]
        ),

        html.Br(),
        html.Div(
            style={'textAlign': 'center', 'marginTop': '40px'},
            children=[
                dcc.Link(
                    html.Button("Back to Home", style={
                        'backgroundColor': '#00d9ff', 'color': '#1a1a2e', 'borderRadius': '8px',
                        'padding': '12px 40px', 'fontSize':'16px', 'border': 'none', 
                        'cursor': 'pointer', 'fontWeight': 'bold'
                    }), 
                    href='/'
                )
            ]
        )
    ]
)

# ----------------- ECG Functions -----------------
def bandpass(sig, fs, low=0.5, high=40, order=4):
    """Apply bandpass filter to ECG signal"""
    nyq = 0.5 * fs
    low_normalized = low / nyq
    high_normalized = high / nyq
    
    low_normalized = max(0.001, min(low_normalized, 0.999))
    high_normalized = max(low_normalized + 0.001, min(high_normalized, 0.999))
    
    b, a = butter(order, [low_normalized, high_normalized], btype='band')
    return filtfilt(b, a, sig)

def read_wfdb_files(dat_path, hea_path):
    """Read WFDB files with comprehensive error handling"""
    try:
        base_path = dat_path.replace('.dat', '')
        record = wfdb.rdrecord(base_path)
        print(f"Successfully read record: {record.record_name}")
        return record
    except Exception as e1:
        print(f"Method 1 failed: {e1}")
        try:
            with open(hea_path, 'r') as f:
                header_lines = f.readlines()
            
            first_line = header_lines[0].strip().split()
            record_name = first_line[0]
            n_signals = int(first_line[1])
            fs = float(first_line[2])
            
            signal_info = []
            sig_names = []
            for i in range(1, min(n_signals + 1, len(header_lines))):
                line = header_lines[i].strip()
                if line and not line.startswith('#'):
                    parts = line.split()
                    if len(parts) >= 2:
                        if len(parts) > 8:
                            sig_name = ' '.join(parts[8:])
                        elif len(parts) > 4:
                            sig_name = parts[-1]
                        else:
                            sig_name = f'Lead {i}'
                        
                        sig_names.append(sig_name)
                        signal_info.append({
                            'gain': float(parts[2]) if len(parts) > 2 and parts[2] not in ['0', ''] else 200.0,
                            'baseline': int(parts[3]) if len(parts) > 3 else 0,
                        })
            
            with open(dat_path, 'rb') as f:
                data = f.read()
            
            raw_data = np.frombuffer(data, dtype=np.int16)
            
            if n_signals > 1:
                if len(raw_data) % n_signals == 0:
                    raw_data = raw_data.reshape(-1, n_signals)
                else:
                    truncate_length = (len(raw_data) // n_signals) * n_signals
                    raw_data = raw_data[:truncate_length].reshape(-1, n_signals)
            else:
                raw_data = raw_data.reshape(-1, 1)
            
            signals = np.zeros_like(raw_data, dtype=np.float64)
            for i in range(min(n_signals, len(signal_info))):
                gain = signal_info[i]['gain']
                baseline = signal_info[i]['baseline']
                if gain != 0:
                    signals[:, i] = (raw_data[:, i] - baseline) / gain
                else:
                    signals[:, i] = raw_data[:, i] - baseline
            
            class Record:
                def __init__(self, signals, fs, sig_names):
                    self.p_signal = signals
                    self.fs = fs
                    self.sig_name = sig_names if sig_names else [f'Lead {i+1}' for i in range(signals.shape[1])]
                    self.n_sig = signals.shape[1]
            
            return Record(signals, fs, sig_names)
        except Exception as e2:
            raise Exception(f"All methods failed. {e1}, {e2}")

def preprocess_for_prediction(signals, fs, lead_names, target_fs=400, target_length=4096):
    """Preprocess ECG for 6-abnormality model prediction"""
    mapping = {
        'I': 'DI', 'II': 'DII', 'III': 'DIII',
        'AVR': 'AVR', 'AVL': 'AVL', 'AVF': 'AVF',
        'V1': 'V1', 'V2': 'V2', 'V3': 'V3',
        'V4': 'V4', 'V5': 'V5', 'V6': 'V6'
    }
    
    mapped_leads = [mapping.get(lead.upper(), None) for lead in lead_names]
    
    model_leads = ['DI', 'DII', 'DIII', 'AVR', 'AVL', 'AVF',
                   'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    indices = []
    for lead in model_leads:
        if lead in mapped_leads:
            indices.append(mapped_leads.index(lead))
        else:
            indices.append(-1)
    
    signals_selected = np.zeros((signals.shape[0], 12))
    for i, idx in enumerate(indices):
        if idx >= 0:
            signals_selected[:, i] = signals[:, idx]
    
    if fs != target_fs:
        num_samples = int(signals_selected.shape[0] * target_fs / fs)
        signals_selected = resample(signals_selected, num_samples, axis=0)
    
    curr_len = signals_selected.shape[0]
    if curr_len > target_length:
        signals_selected = signals_selected[:target_length, :]
    elif curr_len < target_length:
        padding = np.zeros((target_length - curr_len, 12))
        signals_selected = np.vstack([signals_selected, padding])
    
    signals_norm = (signals_selected - np.mean(signals_selected, axis=0)) / (np.std(signals_selected, axis=0) + 1e-8)
    
    input_tensor = np.expand_dims(signals_norm, axis=0)
    return input_tensor

def predict_abnormalities(signals, fs, lead_names):
    """Predict 6 ECG abnormalities"""
    global model
    
    if model is None:
        return None, "Model not loaded"
    
    try:
        input_tensor = preprocess_for_prediction(signals, fs, lead_names)
        predictions = model.predict(input_tensor, verbose=0)[0]
        
        abnormalities = ["1dAVb", "RBBB", "LBBB", "SB", "AF", "ST"]
        
        results = {}
        for ab, prob in zip(abnormalities, predictions):
            results[ab] = float(prob)
        
        return results, None
    except Exception as e:
        return None, f"Prediction error: {str(e)}"

def process_ecg_signals(record):
    """Process all ECG signals"""
    signals = record.p_signal
    fs = record.fs
    lead_names = record.sig_name
    
    processed_signals = []
    for i in range(signals.shape[1]):
        signal = signals[:, i]
        valid_indices = ~(np.isnan(signal) | np.isinf(signal))
        if not np.any(valid_indices):
            processed_signals.append(np.zeros_like(signal))
            continue
        if not np.all(valid_indices):
            median_val = np.median(signal[valid_indices])
            signal[~valid_indices] = median_val
        
        if np.std(signal) > 1e-6:
            try:
                signal_filtered = bandpass(signal, fs)
            except:
                signal_filtered = signal
        else:
            signal_filtered = signal
        
        processed_signals.append(signal_filtered)
    
    processed_signals = np.column_stack(processed_signals)
    
    target_fs = 250
    if fs != target_fs and fs > 0:
        try:
            n_samples = int(processed_signals.shape[0] * target_fs / fs)
            resampled_signals = []
            for i in range(processed_signals.shape[1]):
                resampled_sig = resample(processed_signals[:, i], n_samples)
                resampled_signals.append(resampled_sig)
            processed_signals = np.column_stack(resampled_signals)
            fs_final = target_fs
        except:
            fs_final = fs
    else:
        fs_final = fs
    
    return processed_signals, fs_final, lead_names
<<<<<<< HEAD
def apply_xor_to_chunks(signals, chunk_samples, current_position):
    """
    Apply XOR-like overlay to signal chunks where identical patterns cancel out.
    
    Logic:
    - Each chunk is normalized to [-1, 1]
    - Chunks accumulate: if waveforms align (same polarity) → they reinforce
    - If waveforms oppose (opposite polarity) → they cancel toward 0
    - Identical chunks → flat line, different chunks → visible patterns
    """
    total_chunks = signals.shape[0] // chunk_samples
    current_chunk_idx = min(current_position // chunk_samples, total_chunks - 1)
    
    if current_chunk_idx == 0:
        # First chunk - normalize it
        chunk_end = min(chunk_samples, signals.shape[0])
        chunk_data = signals[:chunk_end]
        
        if len(chunk_data) < chunk_samples:
            chunk_data = np.pad(chunk_data, (0, chunk_samples - len(chunk_data)), mode='constant')
        
        # Normalize to [-1, 1]
        if np.std(chunk_data) > 1e-6:
            chunk_norm = (chunk_data - np.mean(chunk_data)) / (np.std(chunk_data) * 3)
            chunk_norm = np.clip(chunk_norm, -1, 1)
        else:
            chunk_norm = chunk_data * 0
        
        return chunk_norm
    
    # Initialize with first chunk
    chunk_end = min(chunk_samples, signals.shape[0])
    first_chunk = signals[:chunk_end]
    
    if len(first_chunk) < chunk_samples:
        first_chunk = np.pad(first_chunk, (0, chunk_samples - len(first_chunk)), mode='constant')
    
    if np.std(first_chunk) > 1e-6:
        xor_accumulator = (first_chunk - np.mean(first_chunk)) / (np.std(first_chunk) * 3)
        xor_accumulator = np.clip(xor_accumulator, -1, 1)
    else:
        xor_accumulator = first_chunk * 0
    
    # XOR overlay: multiply accumulated result by each new chunk
    # Same polarity → positive product → they differ
    # Opposite polarity → negative product → they cancel
    # Limit to last 20 chunks for performance
    start_chunk = max(1, current_chunk_idx - 19)
    
    for chunk_i in range(start_chunk, current_chunk_idx + 1):
        chunk_start = chunk_i * chunk_samples
        chunk_end = min(chunk_start + chunk_samples, signals.shape[0])
        chunk_data = signals[chunk_start:chunk_end]
        
        if len(chunk_data) < chunk_samples:
            chunk_data = np.pad(chunk_data, (0, chunk_samples - len(chunk_data)), mode='constant')
        
        # Normalize current chunk
        if np.std(chunk_data) > 1e-6:
            chunk_norm = (chunk_data - np.mean(chunk_data)) / (np.std(chunk_data) * 3)
            chunk_norm = np.clip(chunk_norm, -1, 1)
        else:
            chunk_norm = chunk_data * 0
        
        # XOR-like operation: multiply and take absolute difference
        # Where chunks match → low values, where they differ → high values
        difference = np.abs(xor_accumulator - chunk_norm)
        
        # Update accumulator with the difference
        xor_accumulator = difference
    
    # Scale to [0, 1] for display
    if np.max(xor_accumulator) > 0:
        xor_accumulator = xor_accumulator / np.max(xor_accumulator)
    
    return xor_accumulator
=======

>>>>>>> main
# ----------------- Callbacks -----------------

@callback(
    Output('file-list', 'children'),
    Input('upload-ecg', 'contents'),
    State('upload-ecg', 'filename')
)
def show_file_list(contents, filenames):
    if not contents or not filenames:
        return "No files uploaded yet."
    if isinstance(filenames, str):
        filenames = [filenames]
    return "Uploaded files: " + ", ".join(filenames)

@callback(
    [Output('ecg-data-store', 'data'),
     Output('ecg-output', 'children'),
     Output('main-content', 'style'),
     Output('channel-selection', 'options'),
     Output('channel-selection', 'value'),
     Output('reoccurrence-x-channel', 'options'),
     Output('reoccurrence-y-channel', 'options'),
     Output('reoccurrence-x-channel', 'value'),
     Output('reoccurrence-y-channel', 'value'),
     Output('prediction-result', 'children'),
     Output('prediction-probabilities', 'children'),
     Output('prediction-details', 'children')],
    Input('process-btn', 'n_clicks'),
    [State('upload-ecg', 'contents'),
     State('upload-ecg', 'filename')]
)
def process_ecg(n_clicks, contents, filenames):
    empty_outputs = (None, "", {'display':'none'}, [], [], [], [], None, None, "", "", "")
    
    if n_clicks == 0 or not contents or not filenames:
        return empty_outputs
    
    if isinstance(contents, str):
        contents = [contents]
    if isinstance(filenames, str):
        filenames = [filenames]
    
    for filename in os.listdir("uploads"):
        try:
            os.unlink(os.path.join("uploads", filename))
        except:
            pass
    
    for content, filename in zip(contents, filenames):
        try:
            data = content.split(',')[1] if ',' in content else content
            with open(os.path.join("uploads", filename), "wb") as f:
                f.write(base64.b64decode(data))
        except Exception as e:
            return (None, f"Error saving {filename}: {e}", {'display':'none'}, [], [], [], [], None, None, "", "", "")
    
    saved_files = os.listdir("uploads")
    dat_files = [f for f in saved_files if f.endswith('.dat')]
    hea_files = [f for f in saved_files if f.endswith('.hea')]
    
    if not dat_files or not hea_files:
        return (None, "Missing .dat or .hea files", {'display':'none'}, [], [], [], [], None, None, "", "", "")
    
    dat_path = None
    hea_path = None
    for dat_file in dat_files:
        base_name = dat_file.replace('.dat', '')
        matching_hea = f"{base_name}.hea"
        if matching_hea in hea_files:
            dat_path = os.path.join("uploads", dat_file)
            hea_path = os.path.join("uploads", matching_hea)
            break
    
    if not dat_path:
        return (None, "Could not find matching .dat/.hea pair", {'display':'none'}, [], [], [], [], None, None, "", "", "")
    
    try:
        record = read_wfdb_files(dat_path, hea_path)
        signals, fs, lead_names = process_ecg_signals(record)
    except Exception as e:
        return (None, f"Error processing ECG: {e}", {'display':'none'}, [], [], [], [], None, None, "", "", "")
    
    ecg_data = {
        'signals': signals.tolist(),
        'fs': fs,
        'lead_names': lead_names,
    }
    
    # Create channel options for all channels
    channel_options = [{'label': f'Channel {i+1}: {name}', 'value': i} for i, name in enumerate(lead_names)]
    default_channels = list(range(min(3, len(lead_names))))
    
    # Reoccurrence channel options
    reoccurrence_options = [{'label': f'Ch{i+1}: {name}', 'value': i} for i, name in enumerate(lead_names)]
    default_x = 0 if len(lead_names) > 0 else None
    default_y = 1 if len(lead_names) > 1 else None
    
    # Make prediction
    results, error_msg = predict_abnormalities(record.p_signal, record.fs, record.sig_name)
    
    if error_msg or results is None:
<<<<<<< HEAD
        prediction_result = html.Div("⚠ Prediction Unavailable", style={'color': '#FFA500'})
        prediction_probs = html.Div("Model not available", style={'color': '#ccc', 'textAlign': 'center'})
        prediction_det = "Please ensure model file (ecg_model.hdf5) is in models/ directory"
=======
        prediction_result = html.Div("⚠️ Prediction Unavailable", style={'color': '#FFA500'})
        prediction_probs = html.Div("Model not available", style={'color': '#ccc', 'textAlign': 'center'})
        prediction_det = "Please ensure model file (model.hdf5) is in models/ directory"
>>>>>>> main
    else:
        max_abnormality = max(results, key=results.get)
        max_prob = results[max_abnormality]
        
        if max_prob > 0.5:
            prediction_result = html.Div(
<<<<<<< HEAD
                f"⚠ Detected: {max_abnormality}",
=======
                f"⚠️ Detected: {max_abnormality}",
>>>>>>> main
                style={'color': '#E74C3C'}
            )
        else:
            prediction_result = html.Div(
                "✅ No significant abnormalities detected",
                style={'color': '#2ECC71'}
            )
        
        abnormality_names = {
            "1dAVb": "First-degree AV Block",
            "RBBB": "Right Bundle Branch Block",
            "LBBB": "Left Bundle Branch Block",
            "SB": "Sinus Bradycardia",
            "AF": "Atrial Fibrillation",
            "ST": "ST-segment Changes"
        }
        
        prob_bars = []
        for ab in ["1dAVb", "RBBB", "LBBB", "SB", "AF", "ST"]:
            prob = results[ab]
            color = '#E74C3C' if prob > 0.5 else '#00d9ff'
            
            prob_bars.append(
                html.Div([
                    html.Div(
                        f"{abnormality_names[ab]} ({ab})",
                        style={'color': '#00d9ff', 'marginBottom': '5px', 'fontSize': '14px'}
                    ),
                    html.Div([
                        html.Div(
                            style={
                                'width': f'{prob*100}%',
                                'height': '25px',
                                'backgroundColor': color,
                                'borderRadius': '5px',
                                'transition': 'width 0.5s ease'
                            }
                        ),
                        html.Div(
                            f'{prob*100:.1f}%',
                            style={
                                'position': 'absolute',
                                'right': '10px',
                                'top': '50%',
                                'transform': 'translateY(-50%)',
                                'color': '#fff',
                                'fontSize': '12px',
                                'fontWeight': 'bold'
                            }
                        )
                    ], style={
                        'width': '100%',
                        'backgroundColor': '#0f1626',
                        'borderRadius': '5px',
                        'position': 'relative',
                        'height': '25px',
                        'marginBottom': '15px'
                    })
                ])
            )
        
        prediction_probs = html.Div(prob_bars, style={'padding': '20px'})
        
        high_risk = [ab for ab, prob in results.items() if prob > 0.5]
        if high_risk:
            prediction_det = html.Div([
<<<<<<< HEAD
                html.P("⚕ Clinical Recommendations:", style={'fontWeight': 'bold', 'color': '#00d9ff', 'marginBottom': '10px'}),
=======
                html.P("⚕️ Clinical Recommendations:", style={'fontWeight': 'bold', 'color': '#00d9ff', 'marginBottom': '10px'}),
>>>>>>> main
                html.Ul([
                    html.Li(f"High probability detected for: {', '.join(high_risk)}", style={'color': '#E74C3C'}),
                    html.Li("Consult a cardiologist for detailed evaluation"),
                    html.Li("Further diagnostic tests may be recommended"),
                    html.Li("This is a screening tool, not a definitive diagnosis")
                ], style={'textAlign': 'left', 'color': '#ccc'})
            ])
        else:
            prediction_det = html.Div([
                html.P("📋 Analysis Summary:", style={'fontWeight': 'bold', 'color': '#00d9ff', 'marginBottom': '10px'}),
                html.Ul([
                    html.Li("No abnormalities with probability > 50%"),
                    html.Li("ECG pattern appears normal"),
                    html.Li("Continue regular health check-ups"),
                    html.Li("Consult physician if symptoms present")
                ], style={'textAlign': 'left', 'color': '#ccc'})
            ])
    
    return (ecg_data, 
            "✅ ECG processed successfully! Configure controls and click 'Play'",
            {'display': 'block'},
            channel_options,
            default_channels,
            reoccurrence_options,
            reoccurrence_options,
            default_x,
            default_y,
            prediction_result,
            prediction_probs,
            prediction_det)

# Toggle visibility of mode-specific controls
@callback(
    [Output('polar-mode-container', 'style'),
     Output('colormap-container', 'style'),
     Output('reoccurrence-channel-container', 'style')],
    Input('graph-mode', 'value')
)
def toggle_mode_controls(graph_mode):
    polar_style = {'display': 'block'} if graph_mode == 'polar' else {'display': 'none'}
    colormap_style = {'display': 'block'} if graph_mode == 'reoccurrence' else {'display': 'none'}
    reoccurrence_style = {'display': 'block'} if graph_mode == 'reoccurrence' else {'display': 'none'}
    
    return polar_style, colormap_style, reoccurrence_style

@callback(
    [Output('stream-btn', 'children'),
     Output('interval-component', 'disabled'),
     Output('stream-state', 'data')],
    [Input('stream-btn', 'n_clicks'),
     Input('pause-btn', 'n_clicks'),
     Input('reset-btn', 'n_clicks')],
    State('stream-state', 'data')
)
def control_streaming(play_clicks, pause_clicks, reset_clicks, stream_state):
    from dash import callback_context
    
    if not callback_context.triggered:
        return "▶ Play", True, {'streaming': False, 'position': 0, 'polar_data': [], 'xor_chunks': []}
    
    button_id = callback_context.triggered[0]['prop_id'].split('.')[0]
    
    if stream_state is None:
        stream_state = {'streaming': False, 'position': 0, 'polar_data': [], 'xor_chunks': []}
    
    if button_id == 'stream-btn':
        return "⏸ Pause", False, {**stream_state, 'streaming': True}
    elif button_id == 'pause-btn':
        return "▶ Play", True, {**stream_state, 'streaming': False}
    elif button_id == 'reset-btn':
        return "▶ Play", True, {'streaming': False, 'position': 0, 'polar_data': [], 'xor_chunks': []}
    
    return "▶ Play", True, stream_state

@callback(
    Output('pan-clicks', 'data'),
    [Input('pan-back-btn', 'n_clicks'),
     Input('pan-forward-btn', 'n_clicks'),
     Input('reset-btn', 'n_clicks')],
    State('pan-clicks', 'data')
)
def update_pan_clicks(back, forward, reset, pan_data):
    if pan_data is None:
        pan_data = {'back': 0, 'forward': 0, 'reset': 0}
    
    from dash import callback_context
    if not callback_context.triggered:
        return pan_data
    
    button_id = callback_context.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == 'pan-back-btn':
        pan_data['back'] = back
    elif button_id == 'pan-forward-btn':
        pan_data['forward'] = forward
    elif button_id == 'reset-btn':
        pan_data['reset'] = reset
    
    return pan_data

@callback(
    [Output('ecg-stream-graph', 'figure'),
     Output('stream-state', 'data', allow_duplicate=True)],
    [Input('interval-component', 'n_intervals'),
     Input('graph-mode', 'value'),
     Input('window-width', 'value'),
     Input('channel-selection', 'value'),
     Input('speed-control', 'value'),
     Input('zoom-control', 'value'),
     Input('polar-mode', 'value'),
     Input('colormap-selection', 'value'),
     Input('reoccurrence-x-channel', 'value'),
     Input('reoccurrence-y-channel', 'value'),
     Input('pan-clicks', 'data')],
    [State('ecg-data-store', 'data'),
     State('stream-state', 'data')],
    prevent_initial_call=True
)
def update_stream(n_intervals, graph_mode, window_width, selected_channels, speed, zoom, 
                  polar_mode, colormap, reoccurrence_x, reoccurrence_y, pan_clicks,
                  ecg_data, stream_state):
    
    if ecg_data is None:
        return go.Figure(), stream_state
    
    signals = np.array(ecg_data['signals'])
    fs = ecg_data['fs']
    lead_names = ecg_data['lead_names']
    
    if stream_state is None:
        stream_state = {'streaming': True, 'position': 0, 'polar_data': [], 'xor_chunks': []}
    
    position = stream_state.get('position', 0)
    is_streaming = stream_state.get('streaming', False)
    
    # Handle panning
    from dash import callback_context
    if callback_context.triggered:
        trigger_id = callback_context.triggered[0]['prop_id'].split('.')[0]
        if trigger_id == 'pan-clicks' and pan_clicks:
<<<<<<< HEAD
            pan_amount = int(window_width * fs * 0.5)
=======
            pan_amount = int(window_width * fs * 0.5)  # Pan by half window
>>>>>>> main
            if pan_clicks.get('back', 0) > stream_state.get('last_back', 0):
                position = max(0, position - pan_amount)
                stream_state['last_back'] = pan_clicks['back']
            elif pan_clicks.get('forward', 0) > stream_state.get('last_forward', 0):
                position = min(signals.shape[0] - int(window_width * fs), position + pan_amount)
                stream_state['last_forward'] = pan_clicks['forward']
            elif pan_clicks.get('reset', 0) > stream_state.get('last_reset', 0):
                position = 0
                stream_state = {'streaming': False, 'position': 0, 'polar_data': [], 'xor_chunks': []}
                stream_state['last_reset'] = pan_clicks['reset']
    
    # Update position for streaming
    if is_streaming:
<<<<<<< HEAD
        samples_per_update = int(fs * 0.1 * speed)
=======
        samples_per_update = int(fs * 0.1 * speed)  # Speed multiplier
>>>>>>> main
        position += samples_per_update
        
        if position >= signals.shape[0]:
            position = 0
            stream_state['polar_data'] = []
            stream_state['xor_chunks'] = []
    
    window_samples = int(window_width * fs)
    start = position
    end = min(position + window_samples, signals.shape[0])
    
    if end >= signals.shape[0]:
        position = 0
        start = 0
        end = min(window_samples, signals.shape[0])
    
    time_window = np.arange(start, end) / fs
    
    # Generate appropriate visualization based on mode
    if graph_mode == 'waveform':
<<<<<<< HEAD
=======
        # Default continuous-time signal viewer
>>>>>>> main
        if not selected_channels:
            fig = go.Figure()
        else:
            n_channels = len(selected_channels)
            fig = make_subplots(
                rows=n_channels, cols=1,
                subplot_titles=[f'{lead_names[ch]}' for ch in selected_channels if ch < len(lead_names)],
                vertical_spacing=0.08,
                shared_xaxes=True
            )
            
            colors = ['#00d9ff', '#ff006e', '#8338ec', '#2a9d8f', '#e9c46a', '#f4a261']
            for idx, ch in enumerate(selected_channels):
                if ch < signals.shape[1]:
                    signal_window = signals[start:end, ch] * zoom
                    fig.add_trace(
                        go.Scatter(
                            x=time_window,
                            y=signal_window,
                            mode='lines',
                            line=dict(color=colors[idx % len(colors)], width=2),
                            showlegend=False,
                            name=lead_names[ch] if ch < len(lead_names) else f'Ch {ch+1}'
                        ),
                        row=idx+1, col=1
                    )
            
            fig.update_layout(
                title=f"ECG Waveform - Continuous Time Viewer (Speed: {speed}x, Zoom: {zoom}x)",
                plot_bgcolor='#0f1626',
                paper_bgcolor='#16213e',
                font=dict(color='#00d9ff'),
                height=max(500, n_channels * 180),
                margin=dict(l=60, r=40, t=80, b=60)
            )
            
            for i in range(n_channels):
                fig.update_xaxes(gridcolor='#2a2a4e', row=i+1, col=1)
                fig.update_yaxes(gridcolor='#2a2a4e', title_text="Amplitude (mV)", row=i+1, col=1)
            
            fig.update_xaxes(title_text="Time (seconds)", row=n_channels, col=1)
    
    elif graph_mode == 'xor':
<<<<<<< HEAD
        # True XOR chunks visualization - single combined graph like EEG
=======
        # XOR chunks visualization
>>>>>>> main
        if not selected_channels:
            fig = go.Figure()
        else:
            chunk_samples = int(window_width * fs)
            total_chunks = signals.shape[0] // chunk_samples
<<<<<<< HEAD
            current_chunk_idx = position // chunk_samples
            
            fig = go.Figure()
            
            colors = ['#00d9ff', '#ff006e', '#8338ec', '#2a9d8f', '#e9c46a', '#f4a261']
            time_chunk = np.arange(chunk_samples) / fs
            
            # Add zero reference line first (behind other traces)
            fig.add_trace(
                go.Scatter(
                    x=time_chunk,
                    y=np.zeros(chunk_samples),
                    mode='lines',
                    line=dict(color='#666666', width=1, dash='dot'),
                    showlegend=False,
                    name='Zero Reference'
                )
            )
            
            # Plot XOR result for each selected channel
            for idx, ch in enumerate(selected_channels):
                if ch < signals.shape[1]:
                    # Apply true XOR operation with current position
                    xor_result = apply_xor_to_chunks(signals[:, ch], chunk_samples, position)
                    
                    # Apply zoom
                    xor_result_scaled = xor_result * zoom
                    
                    lead_name = lead_names[ch] if ch < len(lead_names) else f'Ch {ch+1}'
                    
                    # Plot the XOR result with fill
                    fig.add_trace(
                        go.Scatter(
                            x=time_chunk,
                            y=xor_result_scaled,
                            mode='lines',
                            line=dict(color=colors[idx % len(colors)], width=2),
                            name=lead_name,
                            fill='tozeroy',
                            fillcolor=f'rgba({int(colors[idx % len(colors)][1:3], 16)}, {int(colors[idx % len(colors)][3:5], 16)}, {int(colors[idx % len(colors)][5:7], 16)}, 0.2)'
                        )
                    )
            
            chunks_processed = min(current_chunk_idx + 1, total_chunks)
            fig.update_layout(
                title=f"XOR Overlay Graph - Identical Chunks Cancel Out<br><sub>Chunks analyzed: {chunks_processed}/{total_chunks} | Chunk width: {window_width}s | Speed: {speed}x</sub>",
                xaxis_title="Time within chunk (seconds)",
                yaxis_title="XOR Amplitude (0 = cancelled)",
                plot_bgcolor='#0f1626',
                paper_bgcolor='#16213e',
                font=dict(color='#00d9ff'),
                height=700,
                margin=dict(l=60, r=40, t=100, b=60),
                showlegend=True,
                legend=dict(
                    bgcolor='#0f1626',
                    bordercolor='#00d9ff',
                    borderwidth=1,
                    x=1.02,
                    y=1
                ),
                xaxis=dict(gridcolor='#2a2a4e'),
                yaxis=dict(gridcolor='#2a2a4e', range=[-0.1, 1.1])
            )
    
    elif graph_mode == 'polar':
=======
            
            # Collect all chunks up to current position
            current_chunk_idx = position // chunk_samples
            
            n_channels = len(selected_channels)
            fig = make_subplots(
                rows=n_channels, cols=1,
                subplot_titles=[f'{lead_names[ch]} - XOR Overlay' for ch in selected_channels if ch < len(lead_names)],
                vertical_spacing=0.08,
                shared_xaxes=True
            )
            
            colors = ['#00d9ff', '#ff006e', '#8338ec', '#2a9d8f', '#e9c46a', '#f4a261']
            
            for idx, ch in enumerate(selected_channels):
                if ch < signals.shape[1]:
                    # Create XOR effect by overlaying chunks
                    time_chunk = np.arange(chunk_samples) / fs
                    
                    # Start with zeros
                    xor_result = np.zeros(chunk_samples)
                    
                    # XOR logic: if chunks overlap (similar values), they cancel out
                    for chunk_i in range(min(current_chunk_idx + 1, total_chunks)):
                        chunk_start = chunk_i * chunk_samples
                        chunk_end = min(chunk_start + chunk_samples, signals.shape[0])
                        chunk_data = signals[chunk_start:chunk_end, ch]
                        
                        if len(chunk_data) == chunk_samples:
                            # Normalize chunk
                            chunk_norm = (chunk_data - np.mean(chunk_data)) / (np.std(chunk_data) + 1e-8)
                            
                            # XOR operation: where signals are similar, they cancel
                            xor_result = xor_result + chunk_norm - 2 * xor_result * (chunk_norm > 0)
                    
                    xor_result *= zoom
                    
                    fig.add_trace(
                        go.Scatter(
                            x=time_chunk[:len(xor_result)],
                            y=xor_result,
                            mode='lines',
                            line=dict(color=colors[idx % len(colors)], width=2),
                            fill='tonexty' if idx == 0 else None,
                            showlegend=False
                        ),
                        row=idx+1, col=1
                    )
            
            fig.update_layout(
                title=f"XOR Chunks Viewer - Chunk {current_chunk_idx + 1}/{total_chunks} (Width: {window_width}s)",
                plot_bgcolor='#0f1626',
                paper_bgcolor='#16213e',
                font=dict(color='#00d9ff'),
                height=max(500, n_channels * 180),
                margin=dict(l=60, r=40, t=80, b=60)
            )
            
            for i in range(n_channels):
                fig.update_xaxes(gridcolor='#2a2a4e', row=i+1, col=1)
                fig.update_yaxes(gridcolor='#2a2a4e', title_text="XOR Amplitude", row=i+1, col=1)
            
            fig.update_xaxes(title_text="Time (seconds)", row=n_channels, col=1)
    
    elif graph_mode == 'polar':
        # Polar graph representation
>>>>>>> main
        if not selected_channels:
            fig = go.Figure()
        else:
            fig = go.Figure()
            
            colors = ['#00d9ff', '#ff006e', '#8338ec', '#2a9d8f', '#e9c46a', '#f4a261']
            
            for idx, ch in enumerate(selected_channels):
                if ch < signals.shape[1]:
                    if polar_mode == 'cumulative':
<<<<<<< HEAD
                        signal_data = signals[:end, ch] * zoom
                        theta = np.linspace(0, 360 * (end / window_samples), len(signal_data))
                    else:
                        signal_data = signals[start:end, ch] * zoom
                        theta = np.linspace(0, 360, len(signal_data))
                    
=======
                        # Cumulative: show all data up to current position
                        signal_data = signals[:end, ch] * zoom
                        theta = np.linspace(0, 360 * (end / window_samples), len(signal_data))
                    else:
                        # Latest fixed time: show only current window
                        signal_data = signals[start:end, ch] * zoom
                        theta = np.linspace(0, 360, len(signal_data))
                    
                    # Ensure positive radius
>>>>>>> main
                    r_min = np.min(signal_data)
                    r_offset = abs(r_min) + 0.5 if r_min < 0 else 0
                    r_values = signal_data + r_offset
                    
                    fig.add_trace(go.Scatterpolar(
                        r=r_values,
                        theta=theta,
                        mode='lines',
                        name=lead_names[ch] if ch < len(lead_names) else f'Ch {ch+1}',
                        line=dict(color=colors[idx % len(colors)], width=2)
                    ))
            
            mode_text = "Cumulative" if polar_mode == 'cumulative' else "Latest Window"
            fig.update_layout(
                title=f"Polar Graph - {mode_text} (Zoom: {zoom}x)",
                polar=dict(
                    bgcolor='#0f1626',
                    radialaxis=dict(
                        visible=True,
                        gridcolor='#2a2a4e',
                        color='#00d9ff'
                    ),
                    angularaxis=dict(
                        visible=True,
                        gridcolor='#2a2a4e',
                        color='#00d9ff',
                        rotation=90,
                        direction='clockwise'
                    )
                ),
                paper_bgcolor='#16213e',
                font=dict(color='#00d9ff'),
                height=700,
                showlegend=True,
                legend=dict(
                    bgcolor='#0f1626',
                    bordercolor='#00d9ff',
                    borderwidth=1
                )
            )
    
    elif graph_mode == 'reoccurrence':
<<<<<<< HEAD
=======
        # Reoccurrence scatter plot (cumulative)
>>>>>>> main
        if reoccurrence_x is None or reoccurrence_y is None:
            fig = go.Figure()
            fig.add_annotation(
                text="Please select X and Y channels",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=20, color='#00d9ff')
            )
        elif reoccurrence_x >= signals.shape[1] or reoccurrence_y >= signals.shape[1]:
            fig = go.Figure()
        else:
<<<<<<< HEAD
            x_data = signals[:end, reoccurrence_x] * zoom
            y_data = signals[:end, reoccurrence_y] * zoom
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=x_data,
                y=y_data,
                mode='markers',
                marker=dict(
                    size=2,
                    color=np.arange(len(x_data)),
                    colorscale=colormap,
                    showscale=True,
                    colorbar=dict(
                        title=dict(
                            text="Time",
                            side="right"
                        ),
                        tickfont=dict(color='#00d9ff')
                    ),
                    opacity=0.6
                ),
                showlegend=False
=======
            # Cumulative scatter plot
            x_data = signals[:end, reoccurrence_x] * zoom
            y_data = signals[:end, reoccurrence_y] * zoom
            
            # Create 2D histogram for density representation
            fig = go.Figure()
            
            fig.add_trace(go.Histogram2d(
                x=x_data,
                y=y_data,
                colorscale=colormap,
                showscale=True,
                colorbar=dict(
                    title="Density",
                    titleside="right",
                    tickfont=dict(color='#00d9ff'),
                    titlefont=dict(color='#00d9ff')
                ),
                nbinsx=50,
                nbinsy=50
>>>>>>> main
            ))
            
            x_name = lead_names[reoccurrence_x] if reoccurrence_x < len(lead_names) else f'Ch {reoccurrence_x+1}'
            y_name = lead_names[reoccurrence_y] if reoccurrence_y < len(lead_names) else f'Ch {reoccurrence_y+1}'
            
            fig.update_layout(
<<<<<<< HEAD
                title=f"Reoccurrence Scatter Plot - {x_name} vs {y_name} (Cumulative)",
=======
                title=f"Reoccurrence Graph - {x_name} vs {y_name} (Cumulative)",
>>>>>>> main
                plot_bgcolor='#0f1626',
                paper_bgcolor='#16213e',
                font=dict(color='#00d9ff'),
                height=700,
                xaxis=dict(
                    title=f"{x_name} Amplitude",
                    gridcolor='#2a2a4e',
                    color='#00d9ff'
                ),
                yaxis=dict(
                    title=f"{y_name} Amplitude",
                    gridcolor='#2a2a4e',
                    color='#00d9ff'
                ),
                margin=dict(l=80, r=80, t=80, b=80)
            )
    
    else:
        fig = go.Figure()
    
<<<<<<< HEAD
=======
    # Update stream state
>>>>>>> main
    stream_state['position'] = position
    
    return fig, stream_state