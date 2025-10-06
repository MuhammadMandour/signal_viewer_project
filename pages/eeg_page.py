# pages/eeg_page.py - CORRECTED COMPLETE VERSION

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
from scipy.spatial.distance import pdist, squareform
import json
import pandas as pd
import io

# ----------------- Constants & Configuration -----------------
os.makedirs("uploads", exist_ok=True)

# EEG Model Configuration
EEG_CONFIG_PATH = 'models/channel_standardized_eeg_config.json'
EEG_MODEL_PATH = 'models/EEG_Model.h5'
EEG_2D_MODEL_PATH = 'models/EEG_2D_Model.h5'  # For 2D model

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
                        {'label': ' XOR Overlay Graph (Identical Chunks Cancel)', 'value': 'xor'},
                        {'label': ' Polar Graph (Magnitude vs Time)', 'value': 'polar'},
                        {'label': ' Reoccurrence Graph (Channel X vs Channel Y)', 'value': 'reoccurrence'},
                        {'label': ' Recurrence Plot (Time-based Similarity Matrix)', 'value': 'recurrence'}
                    ],
                    value='default',
                    inline=False,
                    style={'fontSize': '16px'},
                    labelStyle={'marginBottom': '10px', 'cursor': 'pointer', 'display': 'block'}
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
                
                # Playback Controls
                html.Div(
                    id='playback-controls-container',
                    style={'marginTop': '20px', 'padding': '15px', 'backgroundColor': '#F3F4F6', 'borderRadius': '10px'},
                    children=[
                        html.H4("Playback Controls (Works with All Visualization Modes)", style={'marginBottom': '15px'}),
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
                        
                        html.Button(
                            "Detect from 2D Recurrence Plot",
                            id='detect-2d-btn',
                            n_clicks=0,
                            style={
                                'backgroundColor': '#8B5CF6',
                                'color': 'white',
                                'border': 'none',
                                'borderRadius': '10px',
                                'padding': '12px 25px',
                                'fontSize': '16px',
                                'fontWeight': 'bold',
                                'cursor': 'pointer',
                                'boxShadow': '0 2px 8px rgba(139,92,246,0.3)'
                            }
                        )
                    ]
                )
            ]
        ),

        # Animation interval
        dcc.Interval(
            id='animation-interval',
            interval=100,  # Reduced for smoother animation
            disabled=True,
            n_intervals=0
        ),

        # Results container with loading only for initial processing
        dcc.Loading(
            id="loading-eeg-analysis",
            type="cube",
            color="#3B82F6",
            children=[
                html.Div(id='eeg-results-container')
            ]
        ),
        
        # Visualization container WITHOUT loading wrapper for smooth animation
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
                        'display': 'none'
                    },
                    children=[
                        html.H3(id='viz-title', style={'color': '#1E3A8A', 'marginBottom': '20px'}),
                        dcc.Graph(
                            id='eeg-graph', 
                            config={'displayModeBar': False},
                            style={'transition': 'none'}  # Disable transitions for smoother updates
                        )
                    ]
                )
            ]
        ),

        # Hidden stores
        dcc.Store(id='uploaded-eeg-data'),
        dcc.Store(id='processed-eeg-data'),
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

def create_xor_graph(data, fs, ch_names, selected_channels, chunk_size, n_chunks_to_show=None):
    """
    Create XOR overlay graph where identical chunks cancel out.
    Each chunk is plotted on top of previous ones with XOR logic:
    - If chunks have the same value at a time point → result is 0 (erased)
    - If chunks differ → result is visible
    n_chunks_to_show: if specified, only process up to this many chunks (for animation)
    """
    channels = selected_channels if selected_channels else [0]
    chunk_samples = int(chunk_size * fs)
    
    fig = make_subplots(
        rows=len(channels), cols=1,
        subplot_titles=[f"Channel {ch_names[ch_idx]} - XOR Overlay (Identical Chunks Erase)" for ch_idx in channels],
        shared_xaxes=True,
        vertical_spacing=0.08
    )
    
    time_axis = np.linspace(0, chunk_size, chunk_samples)
    
    for row_idx, ch_idx in enumerate(channels, start=1):
        channel_data = data[:, ch_idx]
        n_chunks = len(channel_data) // chunk_samples
        
        if n_chunks == 0:
            continue
        
        # Collect chunks
        chunks = []
        max_chunks = min(n_chunks, 50)  # Limit for performance
        if n_chunks_to_show is not None:
            max_chunks = min(max_chunks, n_chunks_to_show)
        
        for i in range(max_chunks):
            chunk = channel_data[i*chunk_samples:(i+1)*chunk_samples]
            
            if len(chunk) < chunk_samples:
                continue
            
            # Normalize each chunk to [0, 1] range for comparison
            chunk_norm = (chunk - chunk.min()) / (chunk.max() - chunk.min() + 1e-6)
            chunks.append(chunk_norm)
        
        if len(chunks) == 0:
            continue
        
        # XOR operation: Start with first chunk
        xor_result = chunks[0].copy()
        
        # Apply XOR with each subsequent chunk
        for chunk in chunks[1:]:
            # Binarize at threshold (0.5 for normalized data)
            bin_result = (xor_result > 0.5).astype(int)
            bin_chunk = (chunk > 0.5).astype(int)
            
            # XOR: same values → 0, different values → 1
            xor_binary = np.logical_xor(bin_result, bin_chunk).astype(float)
            
            # Apply back to amplitude (keep the amplitude where XOR is 1)
            xor_result = xor_binary * np.abs(chunk - xor_result)
        
        # Normalize result for visualization
        if xor_result.max() > 0:
            xor_result = xor_result / xor_result.max()
        
        # Plot the XOR result
        fig.add_trace(
            go.Scatter(
                x=time_axis,
                y=xor_result,
                mode='lines',
                fill='tozeroy',
                name=f'XOR Result Ch{ch_idx+1}',
                line=dict(width=2, color=f'hsl({ch_idx*40}, 70%, 50%)'),
                fillcolor=f'hsla({ch_idx*40}, 70%, 50%, 0.4)'
            ),
            row=row_idx, col=1
        )
        
        # Add reference line
        fig.add_hline(y=0, line_dash="dash", line_color="gray", 
                     annotation_text="Zero (Cancelled)", row=row_idx, col=1)
    
    fig.update_layout(
        title=f"XOR Overlay Graph - Identical Chunks Cancel Out (Chunk Size: {chunk_size}s, {max_chunks} chunks)",
        xaxis_title="Time within chunk (seconds)",
        yaxis_title="XOR Amplitude (0 = cancelled)",
        height=600 * len(channels),
        template='plotly_white'
    )
    
    return fig

def create_polar_graph(data, fs, ch_names, selected_channels, time_window, mode='rolling', animation_position=None):
    """
    Create polar graph where r = magnitude and theta = time
    animation_position: if specified, shows data up to this time point (for animation)
    """
    channels = selected_channels if selected_channels else [0]
    
    fig = go.Figure()
    
    for ch_idx in channels:
        if mode == 'rolling':
            # Latest fixed time window
            n_samples = int(time_window * fs)
            if animation_position is not None:
                # For animation: show window ending at animation_position
                end_sample = int(animation_position * fs)
                start_sample = max(0, end_sample - n_samples)
                channel_data = data[start_sample:end_sample, ch_idx]
            else:
                channel_data = data[-n_samples:, ch_idx]
            theta = np.linspace(0, 360, len(channel_data))
        else:
            # Cumulative - all data or up to animation position
            if animation_position is not None:
                end_sample = int(animation_position * fs)
                channel_data = data[:end_sample, ch_idx]
            else:
                channel_data = data[:, ch_idx]
            # Wrap around multiple times if data is longer than time window
            theta = np.linspace(0, 360 * (len(channel_data) / (time_window * fs)), len(channel_data))
        
        fig.add_trace(go.Scatterpolar(
            r=np.abs(channel_data),
            theta=theta,
            mode='lines',
            name=f'Channel {ch_names[ch_idx]}',
            line=dict(width=1.5)
        ))
    
    mode_text = "Rolling Window" if mode == 'rolling' else "Cumulative"
    pos_text = f" (Position: {animation_position:.1f}s)" if animation_position is not None else ""
    fig.update_layout(
        title=f"Polar Graph: r=Magnitude, θ=Time ({mode_text} Mode){pos_text}",
        height=700,
        polar=dict(
            radialaxis=dict(showticklabels=True, ticks='outside'),
            angularaxis=dict(showticklabels=True, ticks='outside')
        )
    )
    
    return fig

def create_reoccurrence_graph(data, fs, ch_names, selected_channels, colormap, animation_position=None):
    """
    Create reoccurrence graph: cumulative scatter plot of chX vs chY
    This shows the relationship between two channels over time
    animation_position: if specified, shows data up to this time point (for animation)
    """
    if not selected_channels or len(selected_channels) < 2:
        ch_x, ch_y = 0, min(1, data.shape[1]-1)
    else:
        ch_x, ch_y = selected_channels[0], selected_channels[1]
    
    # Use all available data (cumulative) or up to animation position
    max_samples = min(10000, len(data))  # Limit for performance
    
    if animation_position is not None:
        end_sample = min(int(animation_position * fs), max_samples)
    else:
        end_sample = max_samples
    
    # Ensure we have at least some data
    end_sample = max(10, end_sample)
    
    data_x = data[:end_sample, ch_x]
    data_y = data[:end_sample, ch_y]
    
    fig = go.Figure()
    
    # Cumulative scatter plot with time-based coloring
    fig.add_trace(go.Scatter(
        x=data_x,
        y=data_y,
        mode='markers',
        marker=dict(
            size=4,
            color=np.arange(len(data_x)),
            colorscale=colormap,
            showscale=True,
            colorbar=dict(
                title=dict(
                    text="Time Index",
                    side="right"
                ),
                thickness=15,
                len=0.7
            ),
            opacity=0.7
        ),
        name='Reoccurrence',
        hovertemplate='X: %{x:.3f}<br>Y: %{y:.3f}<br>Time: %{marker.color}<extra></extra>'
    ))
    
    pos_text = f" (Position: {animation_position:.1f}s)" if animation_position is not None else ""
    fig.update_layout(
        title=f"Reoccurrence Graph: {ch_names[ch_x]} vs {ch_names[ch_y]} (Cumulative){pos_text}",
        xaxis_title=f"Channel {ch_names[ch_x]} Amplitude (mV)",
        yaxis_title=f"Channel {ch_names[ch_y]} Amplitude (mV)",
        height=800,
        width=850,
        template='plotly_white',
        hovermode='closest'
    )
    
    # Add grid
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    return fig

def create_recurrence_plot(data, fs, ch_names, selected_channels, colormap, animation_position=None):
    """
    Create TRUE recurrence plot: time-based similarity matrix.
    Shows when signal patterns repeat over time.
    X-axis: time point i
    Y-axis: time point j
    Color: similarity between signal at time i and time j
    animation_position: if specified, shows matrix up to this time point (for animation)
    """
    ch_idx = selected_channels[0] if selected_channels else 0
    
    # Use limited samples for performance
    max_points = min(500, len(data))
    
    if animation_position is not None:
        end_point = min(int(animation_position * fs / (len(data) / max_points)), max_points)
        end_point = max(10, end_point)  # Show at least 10 points
    else:
        end_point = max_points
    
    signal = data[:end_point, ch_idx]
    
    # Normalize signal for better recurrence detection
    signal_normalized = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
    
    # Compute recurrence matrix (distance matrix)
    from scipy.spatial.distance import cdist
    
    # Reshape for distance computation
    signal_reshaped = signal_normalized.reshape(-1, 1)
    
    # Compute pairwise Euclidean distances
    distance_matrix = cdist(signal_reshaped, signal_reshaped, metric='euclidean')
    
    # Convert to similarity (inverse of distance)
    max_dist = distance_matrix.max()
    if max_dist > 0:
        similarity_matrix = 1 - (distance_matrix / max_dist)
    else:
        similarity_matrix = np.ones_like(distance_matrix)
    
    # Apply threshold for binary recurrence plot
    # Threshold: points are "recurrent" if similarity > 70%
    threshold = 0.7
    recurrence_binary = (similarity_matrix > threshold).astype(float)
    
    fig = go.Figure()
    
    # Create time axis labels
    time_points = np.arange(len(signal)) / fs
    
    # Plot as heatmap
    fig.add_trace(go.Heatmap(
        z=recurrence_binary,
        x=time_points,
        y=time_points,
        colorscale=[[0, 'white'], [1, 'black']],  # White = non-recurrent, Black = recurrent
        showscale=True,
        colorbar=dict(
            title=dict(
                text="Recurrence",
                side="right"
            ),
            thickness=15,
            len=0.7,
            tickvals=[0, 1],
            ticktext=['No', 'Yes']
        ),
        hovertemplate='Time i: %{x:.2f}s<br>Time j: %{y:.2f}s<br>Recurrent: %{z}<extra></extra>'
    ))
    
    pos_text = f" (Position: {animation_position:.1f}s)" if animation_position is not None else ""
    fig.update_layout(
        title=f"Recurrence Plot - Channel {ch_names[ch_idx]}{pos_text}<br><sub>Shows when signal patterns repeat (black = recurrent)</sub>",
        xaxis_title="Time Point i (seconds)",
        yaxis_title="Time Point j (seconds)",
        height=800,
        width=850,
        template='plotly_white'
    )
    
    # Make it square and add grid
    fig.update_xaxes(constrain='domain', showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(scaleanchor='x', scaleratio=1, showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    return fig

def generate_recurrence_plot_image(data, fs, ch_idx=0, size=(128, 128)):
    """
    Generate recurrence plot as image for 2D model input
    Returns: numpy array of shape (128, 128, 1) normalized to [0, 1]
    """
    max_points = min(size[0], len(data))
    signal = data[:max_points, ch_idx]
    
    from scipy.spatial.distance import cdist
    signal_reshaped = signal.reshape(-1, 1)
    distance_matrix = cdist(signal_reshaped, signal_reshaped, metric='euclidean')
    
    # Normalize
    max_dist = distance_matrix.max()
    if max_dist > 0:
        recurrence = 1 - (distance_matrix / max_dist)
    else:
        recurrence = np.ones_like(distance_matrix)
    
    # Resize if needed
    from scipy.ndimage import zoom
    if recurrence.shape != size:
        zoom_factors = (size[0] / recurrence.shape[0], size[1] / recurrence.shape[1])
        recurrence = zoom(recurrence, zoom_factors, order=1)
    
    # Add channel dimension
    recurrence = recurrence.reshape(size[0], size[1], 1)
    
    return recurrence.astype(np.float32)

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
    """Perform AI-based disease detection from 1D multi-channel model"""
    try:
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

def perform_2d_disease_detection(processed_path, fs):
    """Perform AI-based disease detection from 2D recurrence plot model"""
    try:
        eeg_filtered = np.load(processed_path)
        
        if not os.path.exists(EEG_2D_MODEL_PATH):
            return html.Div(
                style={
                    'backgroundColor': '#FEF9E7',
                    'border': '1px solid #F7DC6F',
                    'borderRadius': '10px',
                    'padding': '20px',
                    'color': '#7D6608'
                },
                children=[
                    html.H4("2D Model Not Found"),
                    html.P(f"The 2D model file was not found at: {EEG_2D_MODEL_PATH}"),
                    html.P("Please train the 2D model first using the training script."),
                    html.P("The model should be trained on recurrence plot images.")
                ]
            )
        
        model = tf.keras.models.load_model(EEG_2D_MODEL_PATH)
        
        # Generate recurrence plots for multiple channels
        recurrence_images = []
        num_channels_to_use = min(4, eeg_filtered.shape[1])
        
        for ch_idx in range(num_channels_to_use):
            rec_img = generate_recurrence_plot_image(eeg_filtered, fs, ch_idx, size=(128, 128))
            recurrence_images.append(rec_img)
        
        recurrence_batch = np.array(recurrence_images)
        
        predictions = model.predict(recurrence_batch, verbose=0)
        
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
            title="Disease Detection Results (2D Recurrence Plot Model)",
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
                html.H3("AI Disease Detection Results (2D Model)", style={'color': '#1E3A8A'}),
                
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
                    html.P(f"Analyzed recurrence plots from {num_channels_to_use} channels"),
                    html.P(f"Model Type: 2D CNN on Recurrence Plots"),
                    html.P("This is for educational purposes only. Consult healthcare professionals for medical diagnosis.")
                ], style={'fontSize': '14px', 'color': '#6B7280'})
            ]
        )
        
    except Exception as e:
        import traceback
        return html.Div(
            style={
                'backgroundColor': '#FEF2F2',
                'border': '1px solid #FECACA',
                'borderRadius': '10px',
                'padding': '20px',
                'color': '#B91C1C'
            },
            children=[
                html.H4("2D Detection Error"),
                html.P(f"Error: {str(e)}"),
                html.Pre(traceback.format_exc(), style={'fontSize': '10px', 'overflow': 'auto'})
            ]
        )

# ----------------- Callback Registration Function -----------------
def register_callbacks(app):
    """Register callbacks with the app instance"""
    print("\n" + "="*60)
    print("REGISTERING EEG PAGE CALLBACKS")
    print("="*60 + "\n")
    
    # Callback 1: File upload
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
            return "No file uploaded yet", None, []
        
        try:
            print(f"UPLOADING FILE: {filename}")
            data = contents.split(',')[1]
            file_path = f"uploads/{filename}"
            with open(file_path, "wb") as f:
                f.write(base64.b64decode(data))
            
            _, _, ch_names = load_eeg_data(file_path)
            options = [{'label': f'Channel {i+1}: {ch_names[i]}', 'value': i} 
                      for i in range(len(ch_names))]
            
            uploaded_data = {'filename': filename}
            
            return html.Div([
                f"File uploaded: {filename}",
                html.Br(),
                html.Small(f"Ready for processing ({len(ch_names)} channels)", style={'color': '#6B7280'})
            ]), uploaded_data, options
            
        except Exception as e:
            print(f"FILE UPLOAD ERROR: {e}")
            return html.Div([
                f"File uploaded: {filename}",
                html.Br(),
                html.Small(f"Error: {str(e)}", style={'color': '#DC2626'})
            ]), {'filename': filename}, []
    
    # Callback 2: Process EEG
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
            return None, "", {}, "", {'display': 'none'}
        
        try:
            filename = uploaded_data['filename']
            file_path = f"uploads/{filename}"
            
            eeg_data, fs, ch_names = load_eeg_data(file_path)
            eeg_filtered = bandpass_filter_eeg(eeg_data, fs)
            
            processed_path = f"uploads/processed_{filename}.npy"
            np.save(processed_path, eeg_filtered)
            
            metadata = {
                'filename': filename,
                'processed_path': processed_path,
                'fs': float(fs),
                'ch_names': ch_names,
                'n_samples': int(len(eeg_filtered)),
                'duration': float(len(eeg_filtered) / fs)
            }
            
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
            
            time_window = time_window or 10
            
            if viz_mode == 'default':
                fig = create_default_viewer(eeg_filtered, fs, ch_names, selected_channels, time_window, 0)
            elif viz_mode == 'xor':
                fig = create_xor_graph(eeg_filtered, fs, ch_names, selected_channels, 5)
            elif viz_mode == 'polar':
                fig = create_polar_graph(eeg_filtered, fs, ch_names, selected_channels, time_window, 'rolling')
            elif viz_mode == 'reoccurrence':
                fig = cr(eeg_filtered, fs, ch_names, selected_channels, 'Viridis')
            elif viz_mode == 'recurrence':
                fig = create_recurrence_plot(eeg_filtered, fs, ch_names, selected_channels, 'Viridis')
            
            viz_title = f"{viz_mode.upper()} Mode Visualization"
            viz_style = {
                'backgroundColor': '#FFFFFF',
                'borderRadius': '15px',
                'padding': '20px',
                'boxShadow': '0 4px 15px rgba(0,0,0,0.1)',
                'marginBottom': '20px',
                'display': 'block'
            }
            
            return metadata, results, fig, viz_title, viz_style
            
        except Exception as e:
            import traceback
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
    
    # Callback 3: 1D Disease Detection
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
                    html.H4("1D Detection Error"),
                    html.P(f"Error: {str(e)}")
                ]
            )
            if current_results:
                if isinstance(current_results, list):
                    return current_results + [error_msg]
                else:
                    return [current_results, error_msg]
            return error_msg
    
    # Callback 4: 2D Disease Detection
    @app.callback(
        Output('eeg-results-container', 'children', allow_duplicate=True),
        Input('detect-2d-btn', 'n_clicks'),
        State('processed-eeg-data', 'data'),
        State('eeg-results-container', 'children'),
        prevent_initial_call=True
    )
    def detect_disease_2d(n_clicks, metadata, current_results):
        if not metadata or n_clicks == 0:
            return current_results
        
        try:
            disease_results = perform_2d_disease_detection(metadata['processed_path'], metadata['fs'])
            
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
                    html.H4("2D Detection Error"),
                    html.P(f"Error: {str(e)}")
                ]
            )
            if current_results:
                if isinstance(current_results, list):
                    return current_results + [error_msg]
                else:
                    return [current_results, error_msg]
            return error_msg
    
    # Callback 5: Play/Pause control
    @app.callback(
        Output('animation-interval', 'disabled'),
        Output('play-pause-btn', 'children'),
        Output('play-pause-btn', 'style'),
        Output('playback-state', 'data', allow_duplicate=True),
        Input('play-pause-btn', 'n_clicks'),
        State('playback-state', 'data'),
        State('processed-eeg-data', 'data'),
        prevent_initial_call=True
    )
    def toggle_playback(n_clicks, state, metadata):
        if not metadata:
            return True, "Play (Process EEG First)", {
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
        
        if new_state:
            state['playing'] = True
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
            return True, "Play", {
                'backgroundColor': '#10B981',
                'color': 'white',
                'border': 'none',
                'borderRadius': '8px',
                'padding': '10px 20px',
                'cursor': 'pointer',
                'fontWeight': 'bold'
            }, state
    
    # Callback 6: Update visualization
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
        
        if not metadata:
            return no_update, no_update, playback_state
        
        ctx = callback_context
        if not ctx.triggered:
            return no_update, no_update, playback_state
        
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        try:
            # Handle animation updates
            if button_id == 'animation-interval':
                if not playback_state.get('playing', False):
                    return no_update, no_update, playback_state
            
            eeg_filtered = np.load(metadata['processed_path'])
            fs = metadata['fs']
            ch_names = metadata['ch_names']
            
            position = playback_state.get('position', 0)
            time_window = time_window or playback_state.get('time_window', 10)
            playback_state['time_window'] = time_window
            
            # Update position based on button clicks or animation
            if button_id == 'animation-interval' and playback_state.get('playing', False):
                position += (speed * 0.2)
                max_position = metadata['duration'] - time_window
                if position > max_position:
                    position = 0
                playback_state['position'] = position
            
            elif button_id == 'pan-left-btn':
                position = max(0, position - 1)
                playback_state['position'] = position
            
            elif button_id == 'pan-right-btn':
                max_position = metadata['duration'] - time_window
                position = min(max_position, position + 1)
                playback_state['position'] = position
            
            elif button_id == 'zoom-in-btn':
                time_window = max(1, time_window - 2)
                playback_state['time_window'] = time_window
            
            elif button_id == 'zoom-out-btn':
                time_window = min(60, time_window + 2)
                playback_state['time_window'] = time_window
            
            # Create visualizations based on mode
            # Pass animation position to all modes for consistent animation
            if viz_mode == 'default':
                fig = create_default_viewer(eeg_filtered, fs, ch_names, selected_channels, 
                                          time_window, position)
            
            elif viz_mode == 'xor':
                # For XOR, animate by showing progressive chunks
                chunk_samples = int((chunk_size or 5) * fs)
                total_chunks = len(eeg_filtered) // chunk_samples
                chunks_to_show = int((position / metadata['duration']) * total_chunks) + 1
                chunks_to_show = max(1, min(chunks_to_show, 50))
                fig = create_xor_graph(eeg_filtered, fs, ch_names, selected_channels, 
                                     chunk_size or 5, chunks_to_show)
            
            elif viz_mode == 'polar':
                fig = create_polar_graph(eeg_filtered, fs, ch_names, selected_channels, 
                                       time_window, polar_mode, position)
            
            elif viz_mode == 'reoccurrence':
                fig = create_reoccurrence_graph(eeg_filtered, fs, ch_names, selected_channels, 
                                              colormap or 'Viridis', position)
            
            elif viz_mode == 'recurrence':
                fig = create_recurrence_plot(eeg_filtered, fs, ch_names, selected_channels, 
                                            colormap or 'Viridis', position)
            
            viz_title = f"{viz_mode.upper()} Mode Visualization"
            
            return fig, viz_title, playback_state
            
        except Exception as e:
            import traceback
            print(f"VISUALIZATION ERROR: {e}")
            print(traceback.format_exc())
            return no_update, no_update, playback_state