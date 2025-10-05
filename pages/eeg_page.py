# pages/eeg_page.py - OPTIMIZED VERSION WITH FIXED PERFORMANCE

from dash import html, dcc, Input, Output, State, callback_context, MATCH, ALL, no_update
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

# Disease information
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

        # Visualization Mode Selection
        html.Div(
            style={
                'backgroundColor': '#FFFFFF',
                'borderRadius': '15px',
                'padding': '25px',
                'boxShadow': '0 4px 15px rgba(0,0,0,0.1)',
                'marginBottom': '30px'
            },
            children=[
                html.H3("Visualization Mode", style={'color': '#1E3A8A', 'marginBottom': '20px'}),
                
                dcc.RadioItems(
                    id='viz-mode',
                    options=[
                        {'label': ' Default Continuous-Time Viewer', 'value': 'default'},
                        {'label': ' XOR Graph', 'value': 'xor'},
                        {'label': ' Polar Graph', 'value': 'polar'},
                        {'label': ' Recurrence Graph', 'value': 'recurrence'}
                    ],
                    value='default',
                    inline=True,
                    style={'fontSize': '16px'},
                    labelStyle={'marginRight': '20px', 'cursor': 'pointer'}
                )
            ]
        ),

        # Interactive Controls
        html.Div(
            style={
                'backgroundColor': '#FFFFFF',
                'borderRadius': '15px',
                'padding': '25px',
                'boxShadow': '0 4px 15px rgba(0,0,0,0.1)',
                'marginBottom': '30px'
            },
            children=[
                html.H3("Interactive Controls", style={'color': '#1E3A8A', 'marginBottom': '20px'}),
                
                html.Div(
                    style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr 1fr', 'gap': '20px', 'marginBottom': '20px'},
                    children=[
                        html.Div([
                            html.Label("Channel Selection:", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                            dcc.Dropdown(
                                id='channel-select',
                                multi=True,
                                placeholder="Select channels...",
                                style={'width': '100%'}
                            )
                        ]),
                        html.Div([
                            html.Label("Time Window (seconds):", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                            dcc.Input(
                                id='time-window',
                                type='number',
                                value=10,
                                min=1,
                                max=60,
                                step=1,
                                style={
                                    'width': '100%',
                                    'padding': '8px',
                                    'borderRadius': '5px',
                                    'border': '1px solid #D1D5DB'
                                }
                            )
                        ]),
                        html.Div([
                            html.Label("Chunk Size (for XOR):", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                            dcc.Input(
                                id='chunk-size',
                                type='number',
                                value=5,
                                min=1,
                                max=20,
                                step=0.5,
                                style={
                                    'width': '100%',
                                    'padding': '8px',
                                    'borderRadius': '5px',
                                    'border': '1px solid #D1D5DB'
                                }
                            )
                        ])
                    ]
                ),
                
                # Playback Controls (for Default Viewer)
                html.Div(
                    id='playback-controls-container',
                    style={'marginTop': '20px', 'padding': '15px', 'backgroundColor': '#F3F4F6', 'borderRadius': '10px'},
                    children=[
                        html.H4("Playback Controls (Default Viewer)", style={'marginBottom': '15px'}),
                        html.Div(
                            style={'display': 'flex', 'alignItems': 'center', 'gap': '20px', 'flexWrap': 'wrap'},
                            children=[
                                html.Button(
                                    "Play",
                                    id='play-pause-btn',
                                    n_clicks=0,
                                    style={
                                        'backgroundColor': '#10B981',
                                        'color': 'white',
                                        'border': 'none',
                                        'borderRadius': '8px',
                                        'padding': '10px 20px',
                                        'cursor': 'pointer',
                                        'fontWeight': 'bold'
                                    }
                                ),
                                html.Div([
                                    html.Label("Speed: ", style={'marginRight': '10px', 'fontWeight': 'bold'}),
                                    dcc.Slider(
                                        id='playback-speed',
                                        min=0.5,
                                        max=5,
                                        step=0.5,
                                        value=1,
                                        marks={i: f'{i}x' for i in [0.5, 1, 2, 3, 4, 5]},
                                        tooltip={"placement": "bottom", "always_visible": True}
                                    )
                                ], style={'flex': '1', 'minWidth': '300px'}),
                                html.Div([
                                    html.Button("Pan Left", id='pan-left-btn', n_clicks=0, 
                                               style={'padding': '8px 15px', 'margin': '0 5px', 'cursor': 'pointer'}),
                                    html.Button("Pan Right", id='pan-right-btn', n_clicks=0,
                                               style={'padding': '8px 15px', 'margin': '0 5px', 'cursor': 'pointer'}),
                                    html.Button("Zoom In", id='zoom-in-btn', n_clicks=0,
                                               style={'padding': '8px 15px', 'margin': '0 5px', 'cursor': 'pointer'}),
                                    html.Button("Zoom Out", id='zoom-out-btn', n_clicks=0,
                                               style={'padding': '8px 15px', 'margin': '0 5px', 'cursor': 'pointer'})
                                ])
                            ]
                        )
                    ]
                ),
                
                # Polar Mode Selection
                html.Div(
                    id='polar-mode-container',
                    style={'marginTop': '20px', 'padding': '15px', 'backgroundColor': '#F3F4F6', 'borderRadius': '10px'},
                    children=[
                        html.H4("Polar Graph Mode", style={'marginBottom': '10px'}),
                        dcc.RadioItems(
                            id='polar-mode',
                            options=[
                                {'label': ' Rolling Window (Latest Fixed Time)', 'value': 'rolling'},
                                {'label': ' Cumulative (Full History)', 'value': 'cumulative'}
                            ],
                            value='rolling',
                            inline=True,
                            style={'fontSize': '14px'}
                        )
                    ]
                ),
                
                # Color map selection
                html.Div(
                    style={'marginTop': '20px'},
                    children=[
                        html.Label("Color Map (for 2D plots):", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                        dcc.Dropdown(
                            id='colormap-select',
                            options=[
                                {'label': 'Viridis', 'value': 'Viridis'},
                                {'label': 'Jet', 'value': 'Jet'},
                                {'label': 'Hot', 'value': 'Hot'},
                                {'label': 'Cool', 'value': 'Cool'},
                                {'label': 'Rainbow', 'value': 'Rainbow'}
                            ],
                            value='Viridis',
                            style={'width': '200px'}
                        )
                    ]
                )
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
                            "Detect Disease (1D Model)",
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
                        ),
                        
                        # Placeholder for 2D model
                        html.Button(
                            "Detect from 2D Model (Coming Soon)",
                            id='detect-2d-btn',
                            n_clicks=0,
                            disabled=True,
                            style={
                                'backgroundColor': '#9CA3AF',
                                'color': 'white',
                                'border': 'none',
                                'borderRadius': '10px',
                                'padding': '12px 25px',
                                'fontSize': '16px',
                                'fontWeight': 'bold',
                                'cursor': 'not-allowed',
                                'boxShadow': '0 2px 8px rgba(156,163,175,0.3)'
                            }
                        )
                    ]
                )
            ]
        ),

        # Animation interval component - REDUCED FREQUENCY
        dcc.Interval(
            id='animation-interval',
            interval=200,  # Increased from 100ms to 200ms
            disabled=True,
            n_intervals=0
        ),

        # Loading Component
        dcc.Loading(
            id="loading-eeg-analysis",
            type="cube",
            color="#3B82F6",
            children=[
                html.Div(id='eeg-results-container'),
                
                # Persistent visualization container with graph component
                html.Div(
                    id='eeg-visualizations',
                    style={'marginTop': '20px'},
                    children=[
                        html.Div(
                            id='viz-container',
                            style={
                                'backgroundColor': '#FFFFFF',
                                'borderRadius': '15px',
                                'padding': '20px',
                                'boxShadow': '0 4px 15px rgba(0,0,0,0.1)',
                                'marginBottom': '20px',
                                'display': 'none'  # Hidden until data is processed
                            },
                            children=[
                                html.H3(id='viz-title', style={'color': '#1E3A8A', 'marginBottom': '20px'}),
                                dcc.Graph(id='eeg-graph', config={'displayModeBar': False})
                            ]
                        )
                    ]
                )
            ]
        ),

        # Hidden stores
        dcc.Store(id='uploaded-eeg-data'),
        dcc.Store(id='processed-eeg-data'),  # Stores metadata only
        dcc.Store(id='playback-state', data={'playing': False, 'position': 0, 'time_window': 10}),

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
            data = raw.get_data().T
            fs = raw.info['sfreq']
            ch_names = raw.ch_names
            return data, fs, ch_names
        
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
            data = df.values
            fs = 256
            ch_names = [f'Ch{i+1}' for i in range(data.shape[1])]
            return data, fs, ch_names
        
        else:
            raise ValueError("Unsupported file format")
    
    except Exception as e:
        raise Exception(f"Error loading EEG data: {str(e)}")

def create_default_viewer(data, fs, ch_names, selected_channels, time_window, start_pos=0):
    """Create default continuous-time viewer"""
    start_sample = int(start_pos * fs)
    n_samples = int(time_window * fs)
    end_sample = min(start_sample + n_samples, len(data))
    
    data_window = data[start_sample:end_sample]
    time_axis = np.arange(len(data_window)) / fs + start_pos
    
    channels = selected_channels if selected_channels else list(range(min(8, len(ch_names))))
    n_channels = len(channels)
    
    fig = make_subplots(
        rows=n_channels, cols=1,
        subplot_titles=[f"Channel {ch_names[i]}" for i in channels],
        shared_xaxes=True,
        vertical_spacing=0.02
    )
    
    for idx, ch_idx in enumerate(channels):
        fig.add_trace(
            go.Scatter(
                x=time_axis,
                y=data_window[:, ch_idx],
                name=f'Ch {ch_idx+1}',
                line=dict(width=1, color=f'hsl({ch_idx*40}, 70%, 50%)')
            ),
            row=idx+1, col=1
        )
    
    fig.update_layout(
        title=f"Default Continuous-Time EEG Viewer (Position: {start_pos:.1f}s)",
        xaxis_title="Time (seconds)",
        height=150*n_channels,
        showlegend=False
    )
    
    return fig

def create_xor_graph(data, fs, ch_names, selected_channels, chunk_size):
    """Create XOR graph"""
    channels = selected_channels if selected_channels else [0]
    chunk_samples = int(chunk_size * fs)
    
    fig = make_subplots(
        rows=len(channels), cols=1,
        subplot_titles=[f"Channel {ch_names[ch_idx]} - XOR Differences" for ch_idx in channels],
        shared_xaxes=True
    )
    
    for row_idx, ch_idx in enumerate(channels, start=1):
        channel_data = data[:, ch_idx]
        n_chunks = len(channel_data) // chunk_samples
        
        time_axis = np.linspace(0, chunk_size, chunk_samples)
        
        if n_chunks > 0:
            reference_chunk = channel_data[:chunk_samples]
            
            for i in range(min(n_chunks, 20)):
                chunk = channel_data[i*chunk_samples:(i+1)*chunk_samples]
                if len(chunk) == chunk_samples:
                    xor_result = chunk - reference_chunk
                    
                    if np.abs(xor_result).max() > 0.1 * np.abs(chunk).max():
                        fig.add_trace(
                            go.Scatter(
                                x=time_axis,
                                y=xor_result,
                                mode='lines',
                                name=f'Chunk {i+1}',
                                line=dict(width=1),
                                showlegend=(row_idx == 1 and i < 5)
                            ),
                            row=row_idx, col=1
                        )
                    
                    reference_chunk = chunk
    
    fig.update_layout(
        title=f"XOR Graph - Differences Between Chunks (Chunk Size: {chunk_size}s)",
        xaxis_title="Time within chunk (seconds)",
        yaxis_title="XOR Difference",
        height=400 * len(channels),
        template='plotly_white'
    )
    
    return fig

def create_polar_graph(data, fs, ch_names, selected_channels, time_window, mode='rolling'):
    """Create polar graph"""
    channels = selected_channels if selected_channels else [0]
    
    fig = go.Figure()
    
    for ch_idx in channels:
        if mode == 'rolling':
            n_samples = int(time_window * fs)
            channel_data = data[-n_samples:, ch_idx]
            theta = np.linspace(0, 360, len(channel_data))
        else:
            channel_data = data[:, ch_idx]
            theta = np.linspace(0, 360 * (len(channel_data) / (time_window * fs)), len(channel_data))
        
        fig.add_trace(go.Scatterpolar(
            r=np.abs(channel_data),
            theta=theta,
            mode='lines',
            name=f'Channel {ch_names[ch_idx]}',
            line=dict(width=1.5)
        ))
    
    mode_text = "Rolling Window" if mode == 'rolling' else "Cumulative"
    fig.update_layout(
        title=f"Polar Graph EEG Visualization ({mode_text} Mode)",
        height=600,
        polar=dict(radialaxis=dict(showticklabels=True, ticks='outside'))
    )
    
    return fig

def create_recurrence_graph(data, fs, ch_names, selected_channels, colormap):
    """Create recurrence graph"""
    if not selected_channels or len(selected_channels) < 2:
        ch_x, ch_y = 0, 1
    else:
        ch_x, ch_y = selected_channels[0], selected_channels[1]
    
    max_samples = min(5000, len(data))
    data_x = data[:max_samples, ch_x]
    data_y = data[:max_samples, ch_y]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=data_x,
        y=data_y,
        mode='markers',
        marker=dict(
            size=2,
            color=np.arange(len(data_x)),
            colorscale=colormap,
            showscale=True,
            colorbar=dict(title="Time")
        ),
        name='Recurrence'
    ))
    
    fig.update_layout(
        title=f"Recurrence Graph: {ch_names[ch_x]} vs {ch_names[ch_y]}",
        xaxis_title=f"Channel {ch_names[ch_x]} Amplitude",
        yaxis_title=f"Channel {ch_names[ch_y]} Amplitude",
        height=600,
        template='plotly_white'
    )
    
    return fig

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
        if len(data.shape) == 3:
            padding = np.zeros((data.shape[0], data.shape[1], target_channels - current_channels))
            return np.concatenate([data, padding], axis=2)
        else:
            padding = np.zeros((data.shape[0], target_channels - current_channels))
            return np.concatenate([data, padding], axis=1)
    else:
        if len(data.shape) == 3:
            return data[:, :, :target_channels]
        else:
            return data[:, :target_channels]

def perform_disease_detection(processed_path, fs):
    """Perform AI-based disease detection - loads from disk only once"""
    try:
        # Load processed data from disk
        eeg_filtered = np.load(processed_path)
        
        model = tf.keras.models.load_model(EEG_MODEL_PATH)
        
        windows = create_eeg_windows(eeg_filtered, fs, EEG_CONFIG['win_sec'])
        if len(windows) == 0:
            raise ValueError("Not enough data to create windows")
        
        windows = standardize_channels(windows, EEG_CONFIG['num_channels'])
        
        mean = windows.mean(axis=(0, 1), keepdims=True)
        std = windows.std(axis=(0, 1), keepdims=True) + 1e-6
        windows = (windows - mean) / std
        
        predictions = model.predict(windows.astype(np.float32), verbose=0)
        
        avg_prediction = np.mean(predictions, axis=0)
        predicted_class = np.argmax(avg_prediction)
        confidence = avg_prediction[predicted_class]
        
        class_name = EEG_CONFIG['class_names'][predicted_class]
        disease_info = DISEASE_INFO[class_name]
        
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
            title="Disease Detection Results (1D Multi-Channel Model)",
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
                html.H3("AI Disease Detection Results (1D Model)", style={'color': '#1E3A8A'}),
                
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
                
                dcc.Graph(figure=fig_prediction),
                
                html.Div([
                    html.P(f"Analyzed {len(windows)} time windows"),
                    html.P(f"Model Type: 1D Multi-Channel Time-Series CNN"),
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
    print("\n" + "="*60)
    print("REGISTERING EEG PAGE CALLBACKS")
    print("="*60 + "\n")
    
    # Callback 1: File upload - only handles upload
    @app.callback(
        Output('eeg-file-status', 'children'),
        Output('uploaded-eeg-data', 'data'),
        Output('channel-select', 'options'),
        Input('upload-eeg', 'contents'),
        State('upload-eeg', 'filename')
    )
    def handle_file_upload(contents, filename):
        print("\n=== FILE UPLOAD CALLBACK TRIGGERED ===")
        
        if not contents:
            print("No file uploaded yet")
            return "No file uploaded yet", None, []
        
        try:
            print(f"UPLOADING FILE: {filename}")
            data = contents.split(',')[1]
            file_path = f"uploads/{filename}"
            with open(file_path, "wb") as f:
                f.write(base64.b64decode(data))
            print(f"FILE SAVED TO: {file_path}")
            
            print("READING CHANNEL NAMES...")
            _, _, ch_names = load_eeg_data(file_path)
            print(f"FOUND {len(ch_names)} CHANNELS")
            
            options = [{'label': f'Channel {i+1}: {ch_names[i]}', 'value': i} 
                      for i in range(len(ch_names))]
            
            uploaded_data = {'filename': filename}
            print(f"FILE UPLOAD COMPLETE - Data stored: {uploaded_data}")
            
            return html.Div([
                f"File uploaded: {filename}",
                html.Br(),
                html.Small(f"Ready for processing ({len(ch_names)} channels)", style={'color': '#6B7280'})
            ]), uploaded_data, options
            
        except Exception as e:
            print(f"FILE UPLOAD ERROR: {e}")
            import traceback
            print(traceback.format_exc())
            return html.Div([
                f"File uploaded: {filename}",
                html.Br(),
                html.Small(f"Error: {str(e)}", style={'color': '#DC2626'})
            ]), {'filename': filename}, []
    
    # Callback 2: Process EEG - only runs when process button clicked
    @app.callback(
        Output('processed-eeg-data', 'data'),
        Output('eeg-results-container', 'children'),
        Output('eeg-graph', 'figure'),
        Output('viz-title', 'children'),
        Output('viz-container', 'style'),
        Input('process-eeg-btn', 'n_clicks'),
        State('uploaded-eeg-data', 'data'),
        State('channel-select', 'value'),
        State('time-window', 'value'),
        State('viz-mode', 'value'),
        prevent_initial_call=True
    )
    def process_eeg_file(n_clicks, uploaded_data, selected_channels, time_window, viz_mode):
        print("\n=== PROCESS EEG BUTTON CLICKED ===")
        
        if not uploaded_data or n_clicks == 0:
            print("ERROR: No uploaded data or n_clicks=0")
            return None, "", {}, "", {'display': 'none'}
        
        try:
            filename = uploaded_data['filename']
            file_path = f"uploads/{filename}"
            print(f"PROCESSING: {filename}")
            
            # Load and process data ONCE
            print("LOADING EEG DATA...")
            eeg_data, fs, ch_names = load_eeg_data(file_path)
            print(f"LOADED: {len(eeg_data)} samples, {len(ch_names)} channels, {fs} Hz")
            
            print("APPLYING BANDPASS FILTER...")
            eeg_filtered = bandpass_filter_eeg(eeg_data, fs)
            print("FILTER APPLIED")
            
            # Save processed data to disk
            processed_path = f"uploads/processed_{filename}.npy"
            print(f"SAVING TO: {processed_path}")
            np.save(processed_path, eeg_filtered)
            print("SAVED SUCCESSFULLY")
            
            # Store only metadata, not the actual data
            metadata = {
                'filename': filename,
                'processed_path': processed_path,
                'fs': float(fs),
                'ch_names': ch_names,
                'n_samples': int(len(eeg_filtered)),
                'duration': float(len(eeg_filtered) / fs)
            }
            print(f"METADATA: Duration={metadata['duration']:.2f}s")
            
            results = html.Div(
                style={
                    'backgroundColor': '#FFFFFF',
                    'borderRadius': '15px',
                    'padding': '25px',
                    'boxShadow': '0 4px 15px rgba(0,0,0,0.1)',
                    'marginBottom': '20px'
                },
                children=[
                    html.H3("EEG Processing Complete", style={'color': '#1E3A8A'}),
                    html.P(f"Successfully processed {len(ch_names)} channels"),
                    html.P(f"Sampling Rate: {fs} Hz"),
                    html.P(f"Duration: {len(eeg_data)/fs:.1f} seconds"),
                    html.P(f"Data Points: {len(eeg_data):,}"),
                    html.P(f"Bandpass Filter Applied: 0.5-40 Hz")
                ]
            )
            
            # Create initial visualization
            time_window = time_window or 10
            print(f"CREATING INITIAL VISUALIZATION: {viz_mode}, window={time_window}s")
            
            if viz_mode == 'default':
                fig = create_default_viewer(eeg_filtered, fs, ch_names, selected_channels, time_window, 0)
            elif viz_mode == 'xor':
                fig = create_xor_graph(eeg_filtered, fs, ch_names, selected_channels, 5)
            elif viz_mode == 'polar':
                fig = create_polar_graph(eeg_filtered, fs, ch_names, selected_channels, time_window, 'rolling')
            elif viz_mode == 'recurrence':
                fig = create_recurrence_graph(eeg_filtered, fs, ch_names, selected_channels, 'Viridis')
            
            print("INITIAL VISUALIZATION CREATED")
            
            viz_title = f"{viz_mode.upper()} Mode Visualization"
            viz_style = {
                'backgroundColor': '#FFFFFF',
                'borderRadius': '15px',
                'padding': '20px',
                'boxShadow': '0 4px 15px rgba(0,0,0,0.1)',
                'marginBottom': '20px',
                'display': 'block'
            }
            
            print("PROCESSING COMPLETE - RETURNING RESULTS")
            return metadata, results, fig, viz_title, viz_style
            
        except Exception as e:
            import traceback
            print(f"PROCESSING ERROR: {e}")
            print(traceback.format_exc())
            error_msg = html.Div(
                style={
                    'backgroundColor': '#FEF2F2',
                    'border': '1px solid #FECACA',
                    'borderRadius': '10px',
                    'padding': '20px',
                    'color': '#B91C1C'
                },
                children=[
                    html.H4("Processing Error"),
                    html.P(f"Error: {str(e)}"),
                    html.Pre(traceback.format_exc(), 
                            style={'fontSize': '10px', 'overflow': 'auto', 'maxHeight': '200px'})
                ]
            )
            return None, error_msg, {}, "", {'display': 'none'}
    
    # Callback 3: Disease Detection - separate from visualization
    @app.callback(
        Output('eeg-results-container', 'children', allow_duplicate=True),
        Input('detect-disease-btn', 'n_clicks'),
        State('processed-eeg-data', 'data'),
        State('eeg-results-container', 'children'),
        prevent_initial_call=True
    )
    def detect_disease(n_clicks, metadata, current_results):
        if not metadata or n_clicks == 0:
            return current_results
        
        try:
            disease_results = perform_disease_detection(metadata['processed_path'], metadata['fs'])
            
            # Append disease results to current results
            if current_results:
                if isinstance(current_results, list):
                    return current_results + [disease_results]
                else:
                    return [current_results, disease_results]
            else:
                return disease_results
                
        except Exception as e:
            error_msg = html.Div(
                style={
                    'backgroundColor': '#FEF2F2',
                    'border': '1px solid #FECACA',
                    'borderRadius': '10px',
                    'padding': '20px',
                    'color': '#B91C1C',
                    'marginTop': '20px'
                },
                children=[
                    html.H4("Detection Error"),
                    html.P(f"Error: {str(e)}")
                ]
            )
            if current_results:
                if isinstance(current_results, list):
                    return current_results + [error_msg]
                else:
                    return [current_results, error_msg]
            return error_msg
    
    # Callback 4: Play/Pause control - ONLY ENABLE IF DATA IS PROCESSED
    @app.callback(
        Output('animation-interval', 'disabled'),
        Output('play-pause-btn', 'children'),
        Output('play-pause-btn', 'style'),
        Output('playback-state', 'data', allow_duplicate=True),
        Input('play-pause-btn', 'n_clicks'),
        State('playback-state', 'data'),
        State('processed-eeg-data', 'data'),
        State('viz-mode', 'value'),
        prevent_initial_call=True
    )
    def toggle_playback(n_clicks, state, metadata, viz_mode):
        print("\n=== PLAY/PAUSE BUTTON CLICKED ===")
        print(f"Current state: {state}")
        print(f"Metadata available: {metadata is not None}")
        print(f"Viz mode: {viz_mode}")
        
        # Check if data is processed
        if not metadata:
            print("ERROR: Cannot play - no processed data!")
            return True, "Play (Process EEG First)", {
                'backgroundColor': '#9CA3AF',
                'color': 'white',
                'border': 'none',
                'borderRadius': '8px',
                'padding': '10px 20px',
                'cursor': 'not-allowed',
                'fontWeight': 'bold'
            }, state
        
        # Check if in default mode
        if viz_mode != 'default':
            print("ERROR: Play only works in Default mode")
            return True, "Play (Default Mode Only)", {
                'backgroundColor': '#9CA3AF',
                'color': 'white',
                'border': 'none',
                'borderRadius': '8px',
                'padding': '10px 20px',
                'cursor': 'not-allowed',
                'fontWeight': 'bold'
            }, state
        
        is_playing = state.get('playing', False)
        new_state = not is_playing
        
        print(f"Was playing: {is_playing}, Now playing: {new_state}")
        
        if new_state:
            state['playing'] = True
            print("STARTING PLAYBACK")
            return False, "Pause", {
                'backgroundColor': '#EF4444',
                'color': 'white',
                'border': 'none',
                'borderRadius': '8px',
                'padding': '10px 20px',
                'cursor': 'pointer',
                'fontWeight': 'bold'
            }, state
        else:
            state['playing'] = False
            print("STOPPING PLAYBACK")
            return True, "Play", {
                'backgroundColor': '#10B981',
                'color': 'white',
                'border': 'none',
                'borderRadius': '8px',
                'padding': '10px 20px',
                'cursor': 'pointer',
                'fontWeight': 'bold'
            }, state
    
    # Callback 5: Update visualization - optimized without page refresh
    @app.callback(
        Output('eeg-graph', 'figure', allow_duplicate=True),
        Output('viz-title', 'children', allow_duplicate=True),
        Output('playback-state', 'data', allow_duplicate=True),
        Input('viz-mode', 'value'),
        Input('animation-interval', 'n_intervals'),
        Input('pan-left-btn', 'n_clicks'),
        Input('pan-right-btn', 'n_clicks'),
        Input('zoom-in-btn', 'n_clicks'),
        Input('zoom-out-btn', 'n_clicks'),
        State('processed-eeg-data', 'data'),
        State('channel-select', 'value'),
        State('time-window', 'value'),
        State('chunk-size', 'value'),
        State('colormap-select', 'value'),
        State('polar-mode', 'value'),
        State('playback-speed', 'value'),
        State('playback-state', 'data'),
        prevent_initial_call=True
    )
    def update_visualization(viz_mode, n_intervals, pan_left, pan_right, zoom_in, zoom_out,
                           metadata, selected_channels, time_window, chunk_size, 
                           colormap, polar_mode, speed, playback_state):
        
        print("\n=== VISUALIZATION CALLBACK TRIGGERED ===")
        
        if not metadata:
            print("ERROR: No metadata available - skipping update")
            return no_update, no_update, playback_state
        
        ctx = callback_context
        if not ctx.triggered:
            print("ERROR: No trigger context")
            return no_update, no_update, playback_state
        
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        print(f"TRIGGER: {button_id}")
        
        try:
            # IMPORTANT: Only update for animation if in default mode and playing
            if button_id == 'animation-interval':
                if viz_mode != 'default':
                    print("SKIP: Animation only works in default mode")
                    return no_update, no_update, playback_state
                
                if not playback_state.get('playing', False):
                    print("SKIP: Not playing")
                    return no_update, no_update, playback_state
            
            # Load processed data from disk
            print(f"LOADING: {metadata['processed_path']}")
            eeg_filtered = np.load(metadata['processed_path'])
            fs = metadata['fs']
            ch_names = metadata['ch_names']
            
            # Update playback state
            position = playback_state.get('position', 0)
            time_window = time_window or playback_state.get('time_window', 10)
            playback_state['time_window'] = time_window
            
            # Handle different triggers
            if button_id == 'animation-interval' and playback_state.get('playing', False):
                position += (speed * 0.2)
                max_position = metadata['duration'] - time_window
                if position > max_position:
                    position = 0
                    print("LOOP: Resetting to start")
                playback_state['position'] = position
                print(f"POSITION: {position:.2f}s")
            
            elif button_id == 'pan-left-btn':
                position = max(0, position - 1)
                playback_state['position'] = position
                print(f"PAN LEFT: {position:.2f}s")
            
            elif button_id == 'pan-right-btn':
                max_position = metadata['duration'] - time_window
                position = min(max_position, position + 1)
                playback_state['position'] = position
                print(f"PAN RIGHT: {position:.2f}s")
            
            elif button_id == 'zoom-in-btn':
                time_window = max(1, time_window - 2)
                playback_state['time_window'] = time_window
                print(f"ZOOM IN: {time_window}s")
            
            elif button_id == 'zoom-out-btn':
                time_window = min(60, time_window + 2)
                playback_state['time_window'] = time_window
                print(f"ZOOM OUT: {time_window}s")
            
            # Create visualization figure only
            print(f"UPDATING FIGURE: {viz_mode}")
            if viz_mode == 'default':
                fig = create_default_viewer(eeg_filtered, fs, ch_names, selected_channels, 
                                          time_window, position)
            elif viz_mode == 'xor':
                fig = create_xor_graph(eeg_filtered, fs, ch_names, selected_channels, chunk_size or 5)
            elif viz_mode == 'polar':
                fig = create_polar_graph(eeg_filtered, fs, ch_names, selected_channels, 
                                       time_window, polar_mode)
            elif viz_mode == 'recurrence':
                fig = create_recurrence_graph(eeg_filtered, fs, ch_names, selected_channels, 
                                            colormap or 'Viridis')
            
            viz_title = f"{viz_mode.upper()} Mode Visualization"
            
            print("FIGURE UPDATED")
            return fig, viz_title, playback_state
            
        except Exception as e:
            import traceback
            print(f"VISUALIZATION ERROR: {e}")
            print(traceback.format_exc())
            return no_update, no_update, playback_state