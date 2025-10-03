from dash import html, dcc, Input, Output, State, callback
import os
import base64
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import wfdb
from pyts.image import RecurrencePlot
from PIL import Image
from scipy.signal import butter, filtfilt, resample
import io
import tensorflow as tf
from tensorflow import keras

# ----------------- Ensure uploads folder exists -----------------
os.makedirs("uploads", exist_ok=True)

# ----------------- Load Model -----------------
MODEL_PATH = os.path.join('models', 'model.hdf5')

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
            style={'display': 'none', 'maxWidth': '1600px', 'margin': '0 auto'},
            children=[
                # Health Prediction Section (Top) - Modified for 6 abnormalities
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
                                'width': '300px',
                                'boxShadow': '0 4px 20px rgba(0, 217, 255, 0.2)',
                                'border': '1px solid #00d9ff'
                            },
                            children=[
                                html.H3("ECG Controls", style={'color': '#00d9ff', 'marginBottom': '20px', 'fontSize': '24px'}),
                                
                                # Window Width Control
                                html.Label("Window width (seconds)", style={'color': '#00d9ff', 'fontSize': '14px', 'marginBottom': '8px', 'display': 'block'}),
                                dcc.Input(
                                    id='window-width',
                                    type='number',
                                    value=5,
                                    min=1,
                                    max=30,
                                    step=1,
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
                                
                                # Graph Mode Selection
                                html.Label("Graph Mode", style={'color': '#00d9ff', 'fontSize': '14px', 'marginBottom': '8px', 'display': 'block'}),
                                dcc.Dropdown(
                                    id='graph-mode',
                                    options=[
                                        {'label': 'ECG Waveform', 'value': 'waveform'},
                                        {'label': 'Polar Graph', 'value': 'polar'},
                                        {'label': 'Recurrence Plot', 'value': 'recurrence'}
                                    ],
                                    value='waveform',
                                    style={
                                        'marginBottom': '20px',
                                        'backgroundColor': '#0f1626',
                                        'color': '#1a1a2e',
                                    },
                                    className='custom-dropdown'
                                ),
                                
                                # Channel Selection
                                html.Label("Select Channels", style={'color': '#00d9ff', 'fontSize': '14px', 'marginBottom': '12px', 'display': 'block'}),
                                dcc.Checklist(
                                    id='channel-selection',
                                    options=[],
                                    value=[],
                                    style={'color': '#00d9ff', 'marginBottom': '20px'},
                                    labelStyle={'display': 'block', 'marginBottom': '8px', 'cursor': 'pointer'}
                                ),
                                
                                # Start/Stop Streaming Button
                                html.Button(
                                    "Start Streaming",
                                    id='stream-btn',
                                    n_clicks=0,
                                    style={
                                        'width': '100%',
                                        'backgroundColor': '#00d9ff',
                                        'color': '#1a1a2e',
                                        'borderRadius': '8px',
                                        'padding': '15px',
                                        'fontSize': '16px',
                                        'border': 'none',
                                        'cursor': 'pointer',
                                        'fontWeight': 'bold',
                                        'marginTop': '10px'
                                    }
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
                                        'height': '600px',
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
                dcc.Store(id='stream-state', data={'streaming': False, 'position': 0})
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
    # Map original lead names to model expected names
    mapping = {
        'I': 'DI', 'II': 'DII', 'III': 'DIII',
        'AVR': 'AVR', 'AVL': 'AVL', 'AVF': 'AVF',
        'V1': 'V1', 'V2': 'V2', 'V3': 'V3',
        'V4': 'V4', 'V5': 'V5', 'V6': 'V6'
    }
    
    mapped_leads = [mapping.get(lead.upper(), None) for lead in lead_names]
    
    # Model lead order
    model_leads = ['DI', 'DII', 'DIII', 'AVR', 'AVL', 'AVF',
                   'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    # Find indices to reorder leads
    indices = []
    for lead in model_leads:
        if lead in mapped_leads:
            indices.append(mapped_leads.index(lead))
        else:
            # If lead missing, use zeros
            indices.append(-1)
    
    # Reorder or fill missing leads
    signals_selected = np.zeros((signals.shape[0], 12))
    for i, idx in enumerate(indices):
        if idx >= 0:
            signals_selected[:, i] = signals[:, idx]
    
    # Resample to target frequency
    if fs != target_fs:
        num_samples = int(signals_selected.shape[0] * target_fs / fs)
        signals_selected = resample(signals_selected, num_samples, axis=0)
    
    # Crop or pad to target length
    curr_len = signals_selected.shape[0]
    if curr_len > target_length:
        signals_selected = signals_selected[:target_length, :]
    elif curr_len < target_length:
        padding = np.zeros((target_length - curr_len, 12))
        signals_selected = np.vstack([signals_selected, padding])
    
    # Normalize
    signals_norm = (signals_selected - np.mean(signals_selected, axis=0)) / (np.std(signals_selected, axis=0) + 1e-8)
    
    # Add batch dimension
    input_tensor = np.expand_dims(signals_norm, axis=0)
    return input_tensor

def predict_abnormalities(signals, fs, lead_names):
    """Predict 6 ECG abnormalities"""
    global model
    
    if model is None:
        return None, "Model not loaded"
    
    try:
        # Preprocess for model
        input_tensor = preprocess_for_prediction(signals, fs, lead_names)
        
        # Predict
        predictions = model.predict(input_tensor, verbose=0)[0]
        
        # Abnormality names
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

def generate_rp_data(signal, max_samples=400):
    """Generate recurrence plot data"""
    try:
        if len(signal) > max_samples:
            start_idx = len(signal) // 4
            signal_subset = signal[start_idx:start_idx + max_samples]
        else:
            signal_subset = signal
        
        signal_mean = np.mean(signal_subset)
        signal_std = np.std(signal_subset)
        
        if signal_std > 1e-8:
            signal_normalized = (signal_subset - signal_mean) / signal_std
        else:
            signal_normalized = signal_subset - signal_mean
        
        signal_2d = signal_normalized.reshape(1, -1)
        rp = RecurrencePlot(dimension=2, time_delay=2, percentage=15)
        X_rp = rp.fit_transform(signal_2d)[0]
        
        return X_rp
    except Exception as e:
        print(f"Error generating RP: {e}")
        return np.zeros((100, 100))

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
     Output('prediction-result', 'children'),
     Output('prediction-probabilities', 'children'),
     Output('prediction-details', 'children')],
    Input('process-btn', 'n_clicks'),
    [State('upload-ecg', 'contents'),
     State('upload-ecg', 'filename')]
)
def process_ecg(n_clicks, contents, filenames):
    empty_outputs = (None, "", {'display':'none'}, [], [], "", "", "")
    
    if n_clicks == 0 or not contents or not filenames:
        return empty_outputs
    
    if isinstance(contents, str):
        contents = [contents]
    if isinstance(filenames, str):
        filenames = [filenames]
    
    # Clear uploads directory
    for filename in os.listdir("uploads"):
        try:
            os.unlink(os.path.join("uploads", filename))
        except:
            pass
    
    # Save files
    for content, filename in zip(contents, filenames):
        try:
            data = content.split(',')[1] if ',' in content else content
            with open(os.path.join("uploads", filename), "wb") as f:
                f.write(base64.b64decode(data))
        except Exception as e:
            return (None, f"Error saving {filename}: {e}", {'display':'none'}, [], [], "", "", "")
    
    # Find files
    saved_files = os.listdir("uploads")
    dat_files = [f for f in saved_files if f.endswith('.dat')]
    hea_files = [f for f in saved_files if f.endswith('.hea')]
    
    if not dat_files or not hea_files:
        return (None, "Missing .dat or .hea files", {'display':'none'}, [], [], "", "", "")
    
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
        return (None, "Could not find matching .dat/.hea pair", {'display':'none'}, [], [], "", "", "")
    
    try:
        record = read_wfdb_files(dat_path, hea_path)
        signals, fs, lead_names = process_ecg_signals(record)
    except Exception as e:
        return (None, f"Error processing ECG: {e}", {'display':'none'}, [], [], "", "", "")
    
    # Store data
    ecg_data = {
        'signals': signals.tolist(),
        'fs': fs,
        'lead_names': lead_names,
    }
    
    # Create channel options
    display_leads = lead_names[:3] if len(lead_names) >= 3 else lead_names
    channel_options = [{'label': f'ECG Lead {name}', 'value': i} for i, name in enumerate(display_leads)]
    default_channels = [0] if len(display_leads) > 0 else []
    
    # Make prediction for all 6 abnormalities
    results, error_msg = predict_abnormalities(record.p_signal, record.fs, record.sig_name)
    
    if error_msg or results is None:
        prediction_result = html.Div("⚠️ Prediction Unavailable", style={'color': '#FFA500'})
        prediction_probs = html.Div("Model not available", style={'color': '#ccc', 'textAlign': 'center'})
        prediction_det = "Please ensure model file (model.hdf5) is in models/ directory"
    else:
        # Find highest probability abnormality
        max_abnormality = max(results, key=results.get)
        max_prob = results[max_abnormality]
        
        # Status message
        if max_prob > 0.5:
            prediction_result = html.Div(
                f"⚠️ Detected: {max_abnormality}",
                style={'color': '#E74C3C'}
            )
        else:
            prediction_result = html.Div(
                "✅ No significant abnormalities detected",
                style={'color': '#2ECC71'}
            )
        
        # Create probability bars
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
        
        # Details
        high_risk = [ab for ab, prob in results.items() if prob > 0.5]
        if high_risk:
            prediction_det = html.Div([
                html.P("⚕️ Clinical Recommendations:", style={'fontWeight': 'bold', 'color': '#00d9ff', 'marginBottom': '10px'}),
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
            "✅ ECG processed successfully! Configure controls and click 'Start Streaming'",
            {'display': 'block'},
            channel_options,
            default_channels,
            prediction_result,
            prediction_probs,
            prediction_det)

@callback(
    [Output('stream-btn', 'children'),
     Output('interval-component', 'disabled'),
     Output('stream-state', 'data')],
    Input('stream-btn', 'n_clicks'),
    State('stream-state', 'data')
)
def toggle_streaming(n_clicks, stream_state):
    if n_clicks == 0:
        return "Start Streaming", True, {'streaming': False, 'position': 0}
    
    if stream_state is None:
        stream_state = {'streaming': False, 'position': 0}
    
    is_streaming = stream_state.get('streaming', False)
    
    if is_streaming:
        return "Start Streaming", True, {'streaming': False, 'position': 0}
    else:
        return "Stop Streaming", False, {'streaming': True, 'position': 0}

@callback(
    [Output('ecg-stream-graph', 'figure'),
     Output('stream-state', 'data', allow_duplicate=True)],
    [Input('interval-component', 'n_intervals'),
     Input('graph-mode', 'value'),
     Input('window-width', 'value'),
     Input('channel-selection', 'value')],
    [State('ecg-data-store', 'data'),
     State('stream-state', 'data')],
    prevent_initial_call=True
)
def update_stream(n_intervals, graph_mode, window_width, selected_channels, ecg_data, stream_state):
    if ecg_data is None or not selected_channels:
        return go.Figure(), stream_state
    
    signals = np.array(ecg_data['signals'])
    fs = ecg_data['fs']
    lead_names = ecg_data['lead_names']
    
    # Limit to 3 displayed leads
    display_lead_names = lead_names[:3]
    
    # Update streaming position
    if stream_state is None:
        stream_state = {'streaming': True, 'position': 0}
    
    position = stream_state.get('position', 0)
    samples_per_update = int(fs * 0.1)  # 100ms of data
    window_samples = int(window_width * fs)
    
    # Calculate window
    start = position
    end = min(position + window_samples, signals.shape[0])
    
    # Loop if reached end
    if end >= signals.shape[0]:
        position = 0
        start = 0
        end = min(window_samples, signals.shape[0])
    
    # Advance position
    new_position = position + samples_per_update
    if new_position >= signals.shape[0]:
        new_position = 0
    
    time_window = np.arange(start, end) / fs
    
    if graph_mode == 'waveform':
        # Create waveform plot
        n_channels = len(selected_channels)
        fig = make_subplots(
            rows=n_channels, cols=1,
            subplot_titles=[f'{display_lead_names[ch]}' for ch in selected_channels if ch < len(display_lead_names)],
            vertical_spacing=0.1,
            shared_xaxes=True
        )
        
        colors = ['#00d9ff', '#ff006e', '#8338ec']
        for idx, ch in enumerate(selected_channels):
            if ch < signals.shape[1]:
                signal_window = signals[start:end, ch]
                fig.add_trace(
                    go.Scatter(
                        x=time_window,
                        y=signal_window,
                        mode='lines',
                        line=dict(color=colors[idx % 3], width=2),
                        showlegend=False
                    ),
                    row=idx+1, col=1
                )
        
        fig.update_layout(
            title="ECG Waveform (Streaming)",
            plot_bgcolor='#0f1626',
            paper_bgcolor='#16213e',
            font=dict(color='#00d9ff'),
            height=600,
            margin=dict(l=60, r=40, t=80, b=60)
        )
        
        for i in range(n_channels):
            fig.update_xaxes(gridcolor='#2a2a4e', row=i+1, col=1)
            fig.update_yaxes(gridcolor='#2a2a4e', title_text="mV", row=i+1, col=1)
        
        fig.update_xaxes(title_text="Time (seconds)", row=n_channels, col=1)
    
    elif graph_mode == 'polar':
        # Create polar plot for selected channels
        fig = go.Figure()
        
        colors = ['#00d9ff', '#ff006e', '#8338ec']
        for idx, ch in enumerate(selected_channels):
            if ch < signals.shape[1]:
                signal_window = signals[start:end, ch]
                theta = np.linspace(0, 360, len(signal_window))
                
                # Offset to ensure positive radius
                r_min = np.min(signal_window)
                r_offset = abs(r_min) + 0.1 if r_min < 0 else 0
                r_values = signal_window + r_offset
                
                fig.add_trace(go.Scatterpolar(
                    r=r_values,
                    theta=theta,
                    mode='lines',
                    name=display_lead_names[ch] if ch < len(display_lead_names) else f'Lead {ch+1}',
                    line=dict(color=colors[idx % 3], width=2)
                ))
        
        fig.update_layout(
            title="ECG Polar Representation (Streaming)",
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
                    color='#00d9ff'
                )
            ),
            paper_bgcolor='#16213e',
            font=dict(color='#00d9ff'),
            height=600,
            showlegend=True,
            legend=dict(
                bgcolor='#0f1626',
                bordercolor='#00d9ff',
                borderwidth=1
            )
        )
    
    elif graph_mode == 'recurrence':
        # Create recurrence plot for selected channels
        n_channels = len(selected_channels)
        
        if n_channels == 1:
            fig = go.Figure()
            ch = selected_channels[0]
            if ch < signals.shape[1]:
                signal_window = signals[start:end, ch]
                rp_data = generate_rp_data(signal_window)
                
                fig.add_trace(go.Heatmap(
                    z=rp_data,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(
                        title="Recurrence",
                        titleside="right",
                        tickfont=dict(color='#00d9ff'),
                        titlefont=dict(color='#00d9ff')
                    )
                ))
        else:
            # Multiple recurrence plots in grid
            cols = min(2, n_channels)
            rows = (n_channels + cols - 1) // cols
            
            fig = make_subplots(
                rows=rows, cols=cols,
                subplot_titles=[display_lead_names[ch] if ch < len(display_lead_names) else f'Lead {ch+1}' 
                               for ch in selected_channels],
                vertical_spacing=0.1,
                horizontal_spacing=0.1
            )
            
            for idx, ch in enumerate(selected_channels):
                row = idx // cols + 1
                col = idx % cols + 1
                
                if ch < signals.shape[1]:
                    signal_window = signals[start:end, ch]
                    rp_data = generate_rp_data(signal_window)
                    
                    fig.add_trace(
                        go.Heatmap(
                            z=rp_data,
                            colorscale='Viridis',
                            showscale=False
                        ),
                        row=row, col=col
                    )
        
        fig.update_layout(
            title="Recurrence Plot (Streaming)",
            plot_bgcolor='#0f1626',
            paper_bgcolor='#16213e',
            font=dict(color='#00d9ff'),
            height=600,
            margin=dict(l=40, r=40, t=80, b=40)
        )
        
        # Remove axis labels for recurrence plots
        fig.update_xaxes(showticklabels=False, showgrid=False)
        fig.update_yaxes(showticklabels=False, showgrid=False)
    
    else:
        fig = go.Figure()
    
    return fig, {'streaming': True, 'position': new_position}
