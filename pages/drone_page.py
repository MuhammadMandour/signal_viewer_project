# pages/drone_page.py

from dash import html, dcc, Input, Output, State
import base64
import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import librosa
import soundfile as sf
import tensorflow as tf
import tensorflow_hub as hub
from tqdm import tqdm

# ----------------- Constants & Configuration -----------------
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODEL_PATH = 'models/yamnet_drone_classifier.h5'
YAMNET_URL = 'https://tfhub.dev/google/yamnet/1'

# Class names
CLASS_NAMES = ['Drone', 'Noise']

# Load models
try:
    yamnet_model = hub.load(YAMNET_URL)
    drone_classifier = tf.keras.models.load_model(MODEL_PATH)
    print("Models loaded successfully")
except Exception as e:
    print(f"Error loading models: {e}")
    yamnet_model = None
    drone_classifier = None

# Class information
CLASS_INFO = {
    'Drone': {
        'description': 'Detected drone acoustic patterns and propeller signatures.',
        'color': '#DC143C'
    },
    'Noise': {
        'description': 'Background environmental sounds without drone signatures.',
        'color': '#2E8B57'
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
                    "Drone Detection - Audio Classification",
                    style={
                        'color': '#1E3A8A',
                        'fontSize': '48px',
                        'fontWeight': 'bold',
                        'marginBottom': '20px',
                        'textShadow': '2px 2px 4px rgba(0,0,0,0.1)'
                    }
                ),
                html.P(
                    "Advanced drone detection using YAMNet-based deep learning with temporal analysis",
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
                html.H2("Upload Audio Data", style={'color': '#1E3A8A', 'marginBottom': '20px'}),

                dcc.Upload(
                    id='upload-drone',
                    children=html.Div([
                        html.Br(),
                        'Drag and Drop or Click to Select Audio Files',
                        html.Br(),
                        html.Small('Supported: .wav, .mp3, .flac', style={'color': '#6B7280'})
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

                html.Div(id='drone-file-status', style={
                    'marginTop': '15px', 
                    'fontSize': '16px', 
                    'color': '#374151',
                    'textAlign': 'center'
                }, children="No file uploaded yet")
            ]
        ),

        # Parameters Section
        html.Div(
            style={
                'backgroundColor': '#FFFFFF',
                'borderRadius': '15px',
                'padding': '25px',
                'boxShadow': '0 4px 15px rgba(0,0,0,0.1)',
                'marginBottom': '30px'
            },
            children=[
                html.H3("Analysis Parameters", style={'color': '#1E3A8A', 'marginBottom': '20px'}),

                html.Div(
                    style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr 1fr', 'gap': '20px'},
                    children=[
                        html.Div([
                            html.Label("Window Size (seconds):", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                            dcc.Input(
                                id='window-size',
                                type='number',
                                value=1.0,
                                min=0.5,
                                max=5,
                                step=0.5,
                                style={
                                    'width': '100%',
                                    'padding': '8px',
                                    'borderRadius': '5px',
                                    'border': '1px solid #D1D5DB'
                                }
                            )
                        ]),
                        html.Div([
                            html.Label("Hop Size (seconds):", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                            dcc.Input(
                                id='hop-size',
                                type='number',
                                value=0.5,
                                min=0.1,
                                max=2,
                                step=0.1,
                                style={
                                    'width': '100%',
                                    'padding': '8px',
                                    'borderRadius': '5px',
                                    'border': '1px solid #D1D5DB'
                                }
                            )
                        ]),
                        html.Div([
                            html.Label("Threshold:", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                            dcc.Input(
                                id='threshold',
                                type='number',
                                value=0.5,
                                min=0,
                                max=1,
                                step=0.1,
                                style={
                                    'width': '100%',
                                    'padding': '8px',
                                    'borderRadius': '5px',
                                    'border': '1px solid #D1D5DB'
                                }
                            )
                        ])
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
                    style={'display': 'flex', 'justifyContent': 'center', 'gap': '15px'},
                    children=[
                        html.Button(
                            "Analyze Audio",
                            id='analyze-drone-btn',
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
                        )
                    ]
                )
            ]
        ),

        # Loading Component
        dcc.Loading(
            id="loading-drone-analysis",
            type="cube",
            color="#3B82F6",
            children=[
                # Audio Player Section
                html.Div(id='drone-player-container'),

                # Results Section
                html.Div(id='drone-results-container'),

                # Visualization Section
                html.Div(id='drone-visualizations')
            ]
        ),

        # Hidden store for uploaded data
        dcc.Store(id='uploaded-drone-data'),

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

# ----------------- Audio Processing Functions -----------------

def load_full_audio(file_path, sr=16000):
    """Load full audio file"""
    audio, _ = librosa.load(file_path, sr=sr, mono=True)
    return audio


def sliding_window(audio, sr=16000, window_sec=1.0, hop_sec=0.5):
    """Split audio into sliding windows"""
    window_size = int(window_sec * sr)
    hop_size = int(hop_sec * sr)
    windows = []

    for start in range(0, len(audio) - window_size + 1, hop_size):
        segment = audio[start:start + window_size]
        windows.append(segment)

    # If audio is shorter than one window
    if not windows:
        pad = np.pad(audio, (0, max(0, window_size - len(audio))))
        windows.append(pad)

    return np.array(windows)


def analyze_audio_temporal(file_path, window_sec=1.0, hop_sec=0.5):
    """Analyze audio using sliding windows"""
    try:
        audio = load_full_audio(file_path)
        windows = sliding_window(audio, window_sec=window_sec, hop_sec=hop_sec)
        times = np.arange(len(windows)) * hop_sec

        preds = []
        for segment in windows:
            segment_tensor = tf.convert_to_tensor(segment, dtype=tf.float32)
            # Extract features from YAMNet
            scores, embeddings, spectrogram = yamnet_model(segment_tensor)
            mean_embedding = tf.reduce_mean(embeddings, axis=0, keepdims=True)
            # Predict using drone classifier
            pred = drone_classifier.predict(mean_embedding, verbose=0)
            preds.append(pred[0])

        preds = np.array(preds)
        return times, preds, audio

    except Exception as e:
        raise Exception(f"Error analyzing audio: {str(e)}")


def create_waveform_plot(audio, sr):
    """Create waveform visualization"""
    time = np.linspace(0, len(audio) / sr, len(audio))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time, 
        y=audio, 
        mode='lines',
        line=dict(color='#0077B6', width=1),
        name='Amplitude'
    ))

    fig.update_layout(
        title="Audio Waveform - Time Domain",
        xaxis_title="Time (seconds)",
        yaxis_title="Amplitude",
        template='plotly_white',
        height=400,
        showlegend=False
    )

    return fig


def create_detection_timeline(times, preds, threshold=0.5):
    """Create drone detection timeline plot"""
    fig = go.Figure()

    # Drone probability
    fig.add_trace(go.Scatter(
        x=times,
        y=preds[:, 0],
        mode='lines',
        line=dict(color='red', width=2),
        name='Drone Probability',
        fill='tozeroy'
    ))

    # Threshold line
    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"Threshold ({threshold})",
        annotation_position="right"
    )

    fig.update_layout(
        title="Drone Detection Timeline",
        xaxis_title="Time (seconds)",
        yaxis_title="Probability",
        template='plotly_white',
        height=400,
        yaxis_range=[0, 1]
    )

    return fig


def create_spectrogram_plot(audio, sr):
    """Create spectrogram plot"""
    from scipy import signal as sp_signal

    f, t, Sxx = sp_signal.spectrogram(audio, sr, nperseg=256)

    fig = go.Figure(data=go.Heatmap(
        z=10 * np.log10(Sxx + 1e-10),
        x=t,
        y=f,
        colorscale='Viridis',
        colorbar=dict(title="Power (dB)")
    ))

    fig.update_layout(
        title="Audio Spectrogram",
        xaxis_title="Time (seconds)",
        yaxis_title="Frequency (Hz)",
        template='plotly_white',
        height=400
    )

    return fig


def create_detection_summary(times, preds, threshold=0.5):
    """Create detection summary statistics"""
    drone_detections = (preds[:, 0] > threshold).sum()
    total_windows = len(preds)
    detection_rate = (drone_detections / total_windows) * 100

    avg_drone_prob = np.mean(preds[:, 0])
    max_drone_prob = np.max(preds[:, 0])

    # Calculate detection periods
    is_drone = preds[:, 0] > threshold
    detection_periods = []
    start_idx = None

    for i, detected in enumerate(is_drone):
        if detected and start_idx is None:
            start_idx = i
        elif not detected and start_idx is not None:
            detection_periods.append((times[start_idx], times[i-1]))
            start_idx = None

    if start_idx is not None:
        detection_periods.append((times[start_idx], times[-1]))

    return html.Div(
        style={
            'backgroundColor': '#FFFFFF',
            'borderRadius': '15px',
            'padding': '25px',
            'boxShadow': '0 4px 15px rgba(0,0,0,0.1)',
            'marginBottom': '20px'
        },
        children=[
            html.H3("Detection Summary", style={'color': '#1E3A8A', 'marginBottom': '20px'}),

            # Statistics Grid
            html.Div(
                style={
                    'display': 'grid',
                    'gridTemplateColumns': '1fr 1fr 1fr 1fr',
                    'gap': '15px',
                    'marginBottom': '20px'
                },
                children=[
                    html.Div(
                        style={
                            'backgroundColor': '#FEF2F2',
                            'borderRadius': '10px',
                            'padding': '15px',
                            'textAlign': 'center',
                            'border': '2px solid #DC143C'
                        },
                        children=[
                            html.H4("Drone Detections", style={'color': '#1E3A8A', 'fontSize': '14px', 'marginBottom': '5px'}),
                            html.H2(f"{drone_detections}", style={'color': '#DC143C', 'margin': '5px 0'}),
                            html.P(f"out of {total_windows} windows", style={'color': '#6B7280', 'fontSize': '12px', 'margin': '0'})
                        ]
                    ),
                    html.Div(
                        style={
                            'backgroundColor': '#F0F9FF',
                            'borderRadius': '10px',
                            'padding': '15px',
                            'textAlign': 'center',
                            'border': '2px solid #3B82F6'
                        },
                        children=[
                            html.H4("Detection Rate", style={'color': '#1E3A8A', 'fontSize': '14px', 'marginBottom': '5px'}),
                            html.H2(f"{detection_rate:.1f}%", style={'color': '#3B82F6', 'margin': '5px 0'}),
                            html.P("of total time", style={'color': '#6B7280', 'fontSize': '12px', 'margin': '0'})
                        ]
                    ),
                    html.Div(
                        style={
                            'backgroundColor': '#F0FDF4',
                            'borderRadius': '10px',
                            'padding': '15px',
                            'textAlign': 'center',
                            'border': '2px solid #10B981'
                        },
                        children=[
                            html.H4("Avg Probability", style={'color': '#1E3A8A', 'fontSize': '14px', 'marginBottom': '5px'}),
                            html.H2(f"{avg_drone_prob:.2f}", style={'color': '#10B981', 'margin': '5px 0'}),
                            html.P("mean confidence", style={'color': '#6B7280', 'fontSize': '12px', 'margin': '0'})
                        ]
                    ),
                    html.Div(
                        style={
                            'backgroundColor': '#FEF3C7',
                            'borderRadius': '10px',
                            'padding': '15px',
                            'textAlign': 'center',
                            'border': '2px solid #F59E0B'
                        },
                        children=[
                            html.H4("Max Probability", style={'color': '#1E3A8A', 'fontSize': '14px', 'marginBottom': '5px'}),
                            html.H2(f"{max_drone_prob:.2f}", style={'color': '#F59E0B', 'margin': '5px 0'}),
                            html.P("peak confidence", style={'color': '#6B7280', 'fontSize': '12px', 'margin': '0'})
                        ]
                    )
                ]
            ),

            # Detection Periods
            html.Div([
                html.H4("Detected Drone Periods:", style={'color': '#1E3A8A', 'marginBottom': '10px'}),
                html.Div([
                    html.P(f"Period {i+1}: {start:.2f}s - {end:.2f}s ({end-start:.2f}s duration)", 
                          style={'margin': '5px 0', 'color': '#374151'})
                    for i, (start, end) in enumerate(detection_periods)
                ]) if detection_periods else html.P("No drone detections above threshold", style={'color': '#6B7280'})
            ])
        ]
    )


# ----------------- Callback Registration Function -----------------
def register_callbacks(app):
    """Register callbacks with the app instance"""

    @app.callback(
        Output('drone-file-status', 'children'),
        Output('uploaded-drone-data', 'data'),
        Input('upload-drone', 'contents'),
        State('upload-drone', 'filename')
    )
    def show_drone_file_status(contents, filename):
        if not contents:
            return "No file uploaded yet", None

        return html.Div([
            f"File uploaded: {filename}",
            html.Br(),
            html.Small(f"Ready for processing", style={'color': '#6B7280'})
        ]), {'contents': contents, 'filename': filename}

    @app.callback(
        Output('drone-player-container', 'children'),
        Output('drone-results-container', 'children'),
        Output('drone-visualizations', 'children'),
        Input('analyze-drone-btn', 'n_clicks'),
        State('uploaded-drone-data', 'data'),
        State('window-size', 'value'),
        State('hop-size', 'value'),
        State('threshold', 'value')
    )
    def process_drone_analysis(analyze_clicks, uploaded_data, window_size, hop_size, threshold):
        if analyze_clicks == 0 or not uploaded_data:
            return "", "", ""

        contents = uploaded_data['contents']
        filename = uploaded_data['filename']

        try:
            # Save uploaded file
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)

            temp_path = os.path.join(UPLOAD_FOLDER, filename)
            with open(temp_path, 'wb') as f:
                f.write(decoded)

            # Check models
            if yamnet_model is None or drone_classifier is None:
                error_msg = html.Div(
                    style={
                        'backgroundColor': '#FEF2F2',
                        'border': '1px solid #FECACA',
                        'borderRadius': '10px',
                        'padding': '20px',
                        'color': '#B91C1C'
                    },
                    children=[
                        html.H4("Model Loading Error"),
                        html.P("Models could not be loaded. Please check model files.")
                    ]
                )
                return "", error_msg, ""

            # Analyze audio
            times, preds, audio = analyze_audio_temporal(temp_path, window_size, hop_size)
            sr = 16000
            duration = len(audio) / sr

            # Audio player
            audio_player = html.Div(
                style={
                    'backgroundColor': '#FFFFFF',
                    'borderRadius': '15px',
                    'padding': '25px',
                    'boxShadow': '0 4px 15px rgba(0,0,0,0.1)',
                    'marginBottom': '20px'
                },
                children=[
                    html.H3("Audio Player", style={'color': '#1E3A8A', 'marginBottom': '15px'}),
                    html.Audio(
                        src=contents,
                        controls=True,
                        style={'width': '100%'}
                    ),
                    html.Div([
                        html.P(f"Filename: {filename}"),
                        html.P(f"Duration: {duration:.2f} seconds"),
                        html.P(f"Sample Rate: {sr} Hz"),
                        html.P(f"Analyzed Windows: {len(times)}")
                    ], style={'marginTop': '15px', 'fontSize': '14px', 'color': '#374151'})
                ]
            )

            # Detection summary
            summary = create_detection_summary(times, preds, threshold)

            # Visualizations
            fig_waveform = create_waveform_plot(audio, sr)
            fig_timeline = create_detection_timeline(times, preds, threshold)
            fig_spectrogram = create_spectrogram_plot(audio, sr)

            visualizations = html.Div(
                style={
                    'backgroundColor': '#FFFFFF',
                    'borderRadius': '15px',
                    'padding': '20px',
                    'boxShadow': '0 4px 15px rgba(0,0,0,0.1)',
                    'marginBottom': '20px'
                },
                children=[
                    html.H3("Analysis Visualizations", style={'color': '#1E3A8A', 'marginBottom': '20px'}),
                    dcc.Graph(figure=fig_timeline),
                    html.Hr(),
                    dcc.Graph(figure=fig_waveform),
                    html.Hr(),
                    dcc.Graph(figure=fig_spectrogram)
                ]
            )

            return audio_player, summary, visualizations

        except Exception as e:
            error_msg = html.Div(
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
            return "", error_msg, ""
