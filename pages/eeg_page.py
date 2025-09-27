from dash import html, dcc, Input, Output, State
import os
import base64
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import mne
from PIL import Image
import tensorflow as tf
from scipy.signal import butter, filtfilt, welch
import json
import pandas as pd

# ----------------- Constants & Configuration -----------------
os.makedirs("uploads", exist_ok=True)

# EEG Model Configuration
EEG_CONFIG_PATH = 'models/channel_standardized_eeg_config.json'
EEG_MODEL_PATH = 'models/EEG_Model.h5'

# Load EEG model configuration
try:
    with open(EEG_CONFIG_PATH, 'r') as f:
        EEG_CONFIG = json.load(f)
except FileNotFoundError:
    EEG_CONFIG = {
        'class_names': ['Normal', 'Epilepsy', 'SleepDisorders', 'Depression'],
        'num_timesteps': 256,
        'num_channels': 16,
        'sfreq': 256,
        'win_sec': 1.0
    }

# Disease information for educational purposes
DISEASE_INFO = {
    'Normal': {
        'description': 'Healthy brain activity with normal neural patterns.',
        'color': '#2E8B57'
    },
    'Epilepsy': {
        'description': 'Neurological disorder characterized by recurrent seizures.',
        'color': '#DC143C'
    },
    'SleepDisorders': {
        'description': 'Disrupted sleep patterns affecting brain wave activity.',
        'color': '#4169E1'
    },
    'Depression': {
        'description': 'Mental health condition affecting mood and cognitive function.',
        'color': '#8B4513'
    }
}

# ----------------- Layout -----------------
layout = html.Div(
    style={
        'backgroundColor': '#F0F8FF', 
        'fontFamily': 'Arial, sans-serif', 
        'minHeight': '100vh',
        'padding': '20px'
    },
    children=[
        # Header Section
        html.Div(
            style={'textAlign': 'center', 'marginBottom': '40px'},
            children=[
                html.H1(
                    "EEG Signal Analysis & Disease Detection",
                    style={
                        'color': '#1E3A8A',
                        'fontSize': '48px',
                        'fontWeight': 'bold',
                        'marginBottom': '20px',
                        'textShadow': '2px 2px 4px rgba(0,0,0,0.1)'
                    }
                ),
                html.P(
                    "Advanced EEG analysis with multiple visualization modes and AI-powered disease classification",
                    style={
                        'fontSize': '20px',
                        'color': '#374151',
                        'fontStyle': 'italic'
                    }
                )
            ]
        ),

        # Upload Section
        html.Div(
            style={
                'backgroundColor': '#FFFFFF',
                'borderRadius': '15px',
                'padding': '30px',
                'boxShadow': '0 4px 15px rgba(0,0,0,0.1)',
                'marginBottom': '30px'
            },
            children=[
                html.H2("Upload EEG Data", style={'color': '#1E3A8A', 'marginBottom': '20px'}),
                
                dcc.Upload(
                    id='upload-eeg',
                    children=html.Div([
                        html.Br(),
                        'Drag and Drop or Click to Select EEG Files',
                        html.Br(),
                        html.Small('Supported: .edf, .csv, .txt', style={'color': '#6B7280'})
                    ]),
                    style={
                        'width': '100%',
                        'height': '120px',
                        'lineHeight': '40px',
                        'borderWidth': '3px',
                        'borderStyle': 'dashed',
                        'borderColor': '#3B82F6',
                        'borderRadius': '15px',
                        'textAlign': 'center',
                        'backgroundColor': '#F0F9FF',
                        'cursor': 'pointer',
                        'transition': 'all 0.3s ease'
                    },
                    multiple=False
                ),
                
                html.Div(id='eeg-file-status', style={
                    'marginTop': '15px', 
                    'fontSize': '16px', 
                    'color': '#374151',
                    'textAlign': 'center'
                }, children="No file uploaded yet")
            ]
        ),

        # Control Panel
        html.Div(
            style={
                'backgroundColor': '#FFFFFF',
                'borderRadius': '15px',
                'padding': '25px',
                'boxShadow': '0 4px 15px rgba(0,0,0,0.1)',
                'marginBottom': '30px'
            },
            children=[
                html.H3("Analysis Controls", style={'color': '#1E3A8A', 'marginBottom': '20px'}),
                
                html.Div(
                    style={'display': 'flex', 'justifyContent': 'space-around', 'flexWrap': 'wrap', 'gap': '15px'},
                    children=[
                        html.Button(
                            "Process EEG",
                            id='process-eeg-btn',
                            n_clicks=0,
                            style={
                                'backgroundColor': '#10B981',
                                'color': 'white',
                                'border': 'none',
                                'borderRadius': '10px',
                                'padding': '12px 25px',
                                'fontSize': '16px',
                                'fontWeight': 'bold',
                                'cursor': 'pointer',
                                'transition': 'all 0.3s ease',
                                'boxShadow': '0 2px 8px rgba(16,185,129,0.3)'
                            }
                        ),
                        
                        html.Button(
                            "Detect Disease",
                            id='detect-disease-btn',
                            n_clicks=0,
                            style={
                                'backgroundColor': '#F59E0B',
                                'color': 'white',
                                'border': 'none',
                                'borderRadius': '10px',
                                'padding': '12px 25px',
                                'fontSize': '16px',
                                'fontWeight': 'bold',
                                'cursor': 'pointer',
                                'transition': 'all 0.3s ease',
                                'boxShadow': '0 2px 8px rgba(245,158,11,0.3)'
                            }
                        )
                    ]
                )
            ]
        ),

        # Loading Component
        dcc.Loading(
            id="loading-eeg-analysis",
            type="cube",
            color="#3B82F6",
            children=[
                # Results Section
                html.Div(id='eeg-results-container'),
                
                # Visualization Tabs
                html.Div(id='eeg-visualizations')
            ]
        ),

        # Hidden store for uploaded data
        dcc.Store(id='uploaded-eeg-data'),

        # Navigation
        html.Div(
            style={'textAlign': 'center', 'marginTop': '40px'},
            children=[
                dcc.Link(
                    html.Button(
                        "Back to Home",
                        style={
                            'backgroundColor': '#6366F1',
                            'color': 'white',
                            'border': 'none',
                            'borderRadius': '10px',
                            'padding': '15px 30px',
                            'fontSize': '18px',
                            'fontWeight': 'bold',
                            'cursor': 'pointer',
                            'boxShadow': '0 2px 8px rgba(99,102,241,0.3)'
                        }
                    ), 
                    href='/'
                )
            ]
        )
    ]
)

# ----------------- EEG Processing Functions -----------------
def bandpass_filter_eeg(data, fs, low=0.5, high=40.0, order=4):
    """Apply bandpass filter to EEG data"""
    nyq = 0.5 * fs
    try:
        b, a = butter(order, [low/nyq, high/nyq], btype='band')
        return filtfilt(b, a, data, axis=0)
    except Exception as e:
        print(f"Filtering error: {e}")
        return data

def load_eeg_data(file_path):
    """Load EEG data from various formats"""
    try:
        if file_path.endswith('.edf'):
            raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
            data = raw.get_data().T  # Transpose to (time, channels)
            fs = raw.info['sfreq']
            ch_names = raw.ch_names
            return data, fs, ch_names
        
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
            data = df.values
            fs = 256  # Default sampling rate
            ch_names = [f'Ch{i+1}' for i in range(data.shape[1])]
            return data, fs, ch_names
        
        else:
            raise ValueError("Unsupported file format")
    
    except Exception as e:
        raise Exception(f"Error loading EEG data: {str(e)}")

def create_eeg_windows(data, fs, win_sec=1.0, step_sec=0.8):
    """Create sliding windows from EEG data"""
    win_samples = int(win_sec * fs)
    step_samples = int(step_sec * fs)
    
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)
    
    windows = []
    for i in range(0, data.shape[0] - win_samples + 1, step_samples):
        window = data[i:i + win_samples]
        windows.append(window)
    
    return np.array(windows)

def standardize_channels(data, target_channels=16):
    """Standardize number of channels"""
    current_channels = data.shape[-1] if len(data.shape) > 1 else 1
    
    if current_channels == target_channels:
        return data
    elif current_channels < target_channels:
        # Pad with zeros
        if len(data.shape) == 3:  # (windows, time, channels)
            padding = np.zeros((data.shape[0], data.shape[1], target_channels - current_channels))
            return np.concatenate([data, padding], axis=2)
        else:  # (time, channels)
            padding = np.zeros((data.shape[0], target_channels - current_channels))
            return np.concatenate([data, padding], axis=1)
    else:
        # Select first target_channels
        if len(data.shape) == 3:
            return data[:, :, :target_channels]
        else:
            return data[:, :target_channels]

def generate_eeg_features(data, fs):
    """Generate frequency domain features"""
    freqs, psd = welch(data, fs, axis=0)
    
    # Define frequency bands
    bands = {
        'Delta (0.5-4 Hz)': (0.5, 4),
        'Theta (4-8 Hz)': (4, 8),
        'Alpha (8-13 Hz)': (8, 13),
        'Beta (13-30 Hz)': (13, 30),
        'Gamma (30-40 Hz)': (30, 40)
    }
    
    band_powers = {}
    for band_name, (low, high) in bands.items():
        mask = (freqs >= low) & (freqs <= high)
        band_power = np.mean(psd[mask], axis=0)
        band_powers[band_name] = band_power
    
    return freqs, psd, band_powers

def create_eeg_visualizations(data, fs, ch_names):
    """Create comprehensive EEG visualizations"""
    
    # Time domain plot
    time_axis = np.arange(len(data)) / fs
    
    # Select first few channels for visualization
    n_channels = min(8, len(ch_names))
    
    fig_time = make_subplots(
        rows=n_channels, cols=1,
        subplot_titles=[f"Channel {ch_names[i]}" for i in range(n_channels)],
        shared_xaxes=True,
        vertical_spacing=0.02
    )
    
    for i in range(n_channels):
        fig_time.add_trace(
            go.Scatter(
                x=time_axis,
                y=data[:, i],
                name=f'Ch {i+1}',
                line=dict(width=1)
            ),
            row=i+1, col=1
        )
    
    fig_time.update_layout(
        title="EEG Time Domain Analysis",
        xaxis_title="Time (seconds)",
        height=600,
        showlegend=False
    )
    
    # Frequency domain analysis
    freqs, psd, band_powers = generate_eeg_features(data, fs)
    
    fig_freq = go.Figure()
    for i in range(min(4, data.shape[1])):
        fig_freq.add_trace(
            go.Scatter(
                x=freqs,
                y=10 * np.log10(psd[:, i]),
                name=f'Channel {i+1}',
                line=dict(width=2)
            )
        )
    
    fig_freq.update_layout(
        title="Power Spectral Density",
        xaxis_title="Frequency (Hz)",
        yaxis_title="Power (dB)",
        height=400
    )
    
    # Band power visualization
    bands = list(band_powers.keys())
    powers = [np.mean(band_powers[band]) for band in bands]
    
    fig_bands = go.Figure(data=[
        go.Bar(
            x=bands,
            y=powers,
            marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
        )
    ])
    
    fig_bands.update_layout(
        title="EEG Frequency Bands Power",
        xaxis_title="Frequency Bands",
        yaxis_title="Average Power",
        height=400
    )
    
    # Polar plot (circular representation)
    if data.shape[1] >= 1:
        sample_channel = data[:min(1000, len(data)), 0]  # First 1000 samples
        theta = np.linspace(0, 2*np.pi, len(sample_channel))
        
        fig_polar = go.Figure()
        fig_polar.add_trace(
            go.Scatterpolar(
                r=sample_channel,
                theta=theta * 180/np.pi,
                mode='lines',
                name='EEG Polar',
                line=dict(width=2, color='#3B82F6')
            )
        )
        
        fig_polar.update_layout(
            title="EEG Polar Visualization",
            height=500
        )
    else:
        fig_polar = go.Figure()
    
    return html.Div([
        html.Div(
            style={
                'backgroundColor': '#FFFFFF',
                'borderRadius': '15px',
                'padding': '20px',
                'boxShadow': '0 4px 15px rgba(0,0,0,0.1)',
                'marginBottom': '20px'
            },
            children=[
                dcc.Graph(figure=fig_time),
                html.Hr(),
                html.Div([
                    html.Div([dcc.Graph(figure=fig_freq)], style={'width': '50%', 'display': 'inline-block'}),
                    html.Div([dcc.Graph(figure=fig_bands)], style={'width': '50%', 'display': 'inline-block'})
                ]),
                html.Hr(),
                dcc.Graph(figure=fig_polar)
            ]
        )
    ])

def perform_disease_detection(data, fs):
    """Perform AI-based disease detection"""
    try:
        # Load the trained model
        model = tf.keras.models.load_model(EEG_MODEL_PATH)
        
        # Preprocess data for model
        windows = create_eeg_windows(data, fs, EEG_CONFIG['win_sec'])
        if len(windows) == 0:
            raise ValueError("Not enough data to create windows")
        
        # Standardize channels
        windows = standardize_channels(windows, EEG_CONFIG['num_channels'])
        
        # Normalize
        mean = windows.mean(axis=(0, 1), keepdims=True)
        std = windows.std(axis=(0, 1), keepdims=True) + 1e-6
        windows = (windows - mean) / std
        
        # Predict
        predictions = model.predict(windows.astype(np.float32))
        
        # Average predictions across windows
        avg_prediction = np.mean(predictions, axis=0)
        predicted_class = np.argmax(avg_prediction)
        confidence = avg_prediction[predicted_class]
        
        class_name = EEG_CONFIG['class_names'][predicted_class]
        disease_info = DISEASE_INFO[class_name]
        
        # Create results visualization
        fig_prediction = go.Figure(data=[
            go.Bar(
                x=EEG_CONFIG['class_names'],
                y=avg_prediction * 100,
                marker_color=[disease_info['color'] if i == predicted_class else '#E5E7EB' 
                             for i in range(len(EEG_CONFIG['class_names']))],
                text=[f"{p*100:.1f}%" for p in avg_prediction],
                textposition='auto',
            )
        ])
        
        fig_prediction.update_layout(
            title="Disease Detection Results",
            xaxis_title="Conditions",
            yaxis_title="Confidence (%)",
            height=400
        )
        
        return html.Div(
            style={
                'backgroundColor': '#FFFFFF',
                'borderRadius': '15px',
                'padding': '25px',
                'boxShadow': '0 4px 15px rgba(0,0,0,0.1)',
                'marginBottom': '20px'
            },
            children=[
                html.H3("AI Disease Detection Results", style={'color': '#1E3A8A'}),
                
                # Main result card
                html.Div(
                    style={
                        'backgroundColor': disease_info['color'],
                        'color': 'white',
                        'borderRadius': '10px',
                        'padding': '20px',
                        'marginBottom': '20px',
                        'textAlign': 'center'
                    },
                    children=[
                        html.H2(f"{class_name}"),
                        html.H3(f"Confidence: {confidence*100:.1f}%"),
                        html.P(disease_info['description'])
                    ]
                ),
                
                # Detailed breakdown
                dcc.Graph(figure=fig_prediction),
                
                # Analysis info
                html.Div([
                    html.P(f"Analyzed {len(windows)} time windows"),
                    html.P(f"Model Version: {EEG_CONFIG.get('model_version', '1.0')}"),
                    html.P("This is for educational purposes only. Consult healthcare professionals for medical diagnosis.")
                ], style={'fontSize': '14px', 'color': '#6B7280'})
            ]
        )
        
    except Exception as e:
        return html.Div(
            style={
                'backgroundColor': '#FEF2F2',
                'border': '1px solid #FECACA',
                'borderRadius': '10px',
                'padding': '20px',
                'color': '#B91C1C'
            },
            children=[
                html.H4("Disease Detection Error"),
                html.P(f"Error: {str(e)}"),
                html.P("Please ensure your EEG data is in the correct format.")
            ]
        )

# ----------------- Callback Registration Function -----------------
def register_callbacks(app):
    """Register callbacks with the app instance"""
    
    @app.callback(
        Output('eeg-file-status', 'children'),
        Output('uploaded-eeg-data', 'data'),
        Input('upload-eeg', 'contents'),
        State('upload-eeg', 'filename')
    )
    def show_eeg_file_status(contents, filename):
        if not contents:
            return "No file uploaded yet", None
        
        return html.Div([
            f"File uploaded: {filename}",
            html.Br(),
            html.Small(f"Ready for processing", style={'color': '#6B7280'})
        ]), {'contents': contents, 'filename': filename}
    
    @app.callback(
        Output('eeg-results-container', 'children'),
        Output('eeg-visualizations', 'children'),
        Input('process-eeg-btn', 'n_clicks'),
        Input('detect-disease-btn', 'n_clicks'),
        State('uploaded-eeg-data', 'data')
    )
    def process_eeg_analysis(process_clicks, detect_clicks, uploaded_data):
        if (process_clicks == 0 and detect_clicks == 0) or not uploaded_data:
            return "", ""
        
        contents = uploaded_data['contents']
        filename = uploaded_data['filename']
        
        try:
            # Save uploaded file
            data = contents.split(',')[1]
            file_path = f"uploads/{filename}"
            with open(file_path, "wb") as f:
                f.write(base64.b64decode(data))
            
            # Load EEG data
            eeg_data, fs, ch_names = load_eeg_data(file_path)
            
            # Apply bandpass filter
            eeg_filtered = bandpass_filter_eeg(eeg_data, fs)
            
            # Generate basic visualizations
            visualizations = create_eeg_visualizations(eeg_filtered, fs, ch_names)
            
            results_content = html.Div([
                html.Div(
                    style={
                        'backgroundColor': '#FFFFFF',
                        'borderRadius': '15px',
                        'padding': '25px',
                        'boxShadow': '0 4px 15px rgba(0,0,0,0.1)',
                        'marginBottom': '20px'
                    },
                    children=[
                        html.H3("EEG Analysis Results", style={'color': '#1E3A8A'}),
                        html.P(f"Successfully processed {len(ch_names)} channels"),
                        html.P(f"Sampling Rate: {fs} Hz"),
                        html.P(f"Duration: {len(eeg_data)/fs:.1f} seconds"),
                        html.P(f"Data Points: {len(eeg_data):,}")
                    ]
                )
            ])
            
            # Disease detection if requested
            if detect_clicks > 0:
                disease_results = perform_disease_detection(eeg_filtered, fs)
                results_content.children.append(disease_results)
            
            return results_content, visualizations
            
        except Exception as e:
            error_msg = html.Div([
                html.Div(
                    style={
                        'backgroundColor': '#FEF2F2',
                        'border': '1px solid #FECACA',
                        'borderRadius': '10px',
                        'padding': '20px',
                        'color': '#B91C1C'
                    },
                    children=[
                        html.H4("Analysis Error"),
                        html.P(f"Error: {str(e)}"),
                        html.P("Please check your file format and try again.")
                    ]
                )
            ])
            return error_msg, ""