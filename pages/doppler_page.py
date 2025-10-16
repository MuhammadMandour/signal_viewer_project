from dash import html, dcc, Input, Output, State
import dash
import numpy as np
from scipy.io import wavfile
from scipy import signal
import plotly.graph_objects as go
import base64
import io
import os
import pickle
import librosa
from tensorflow import keras

# =========================
# Load Model and Scaler at Module Level
# =========================
# Note: Paths are relative to where app.py is run from (project root)
MODEL_PATH = 'models/doppler_regressor.h5'
SCALER_PATH = 'models/scaler (2).pkl'

print("="*60)
print("DOPPLER PAGE - MODEL LOADING")
print("="*60)
print(f"Current working directory: {os.getcwd()}")
print(f"\nChecking for model files...")
print(f"  Model path: {MODEL_PATH}")
print(f"  Model exists: {os.path.exists(MODEL_PATH)}")
print(f"  Scaler path: {SCALER_PATH}")
print(f"  Scaler exists: {os.path.exists(SCALER_PATH)}")

# List all files in models directory for debugging
if os.path.exists('models'):
    print(f"\nFiles in models directory:")
    for f in os.listdir('models'):
        print(f"  - '{f}'")
else:
    print("\n⚠️ models directory does not exist!")

# Try to load model at module initialization
MODEL_LOADED = False
model = None
scaler = None

try:
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        print(f"\n📦 Loading model from {MODEL_PATH}...")
        
        # Try loading with compile=False to avoid metric deserialization issues
        try:
            model = keras.models.load_model(MODEL_PATH, compile=False)
            print(f"✓ Model loaded successfully (without compilation)!")
        except Exception as e1:
            print(f"⚠️ First attempt failed: {e1}")
            print("Trying with custom objects...")
            # Try with custom objects mapping
            from tensorflow.keras import metrics
            custom_objects = {
                'mse': metrics.MeanSquaredError(),
                'mae': metrics.MeanAbsoluteError()
            }
            model = keras.models.load_model(MODEL_PATH, custom_objects=custom_objects, compile=False)
            print(f"✓ Model loaded successfully with custom objects!")
        
        print(f"📦 Loading scaler from {SCALER_PATH}...")
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        print("✓ Scaler loaded successfully!")
        
        MODEL_LOADED = True
        print("\n" + "="*60)
        print("✓✓✓ DOPPLER MODEL READY FOR PREDICTIONS ✓✓✓")
        print("="*60 + "\n")
    else:
        print(f"\n⚠️ Model files not found:")
        if not os.path.exists(MODEL_PATH):
            print(f"  ✗ Model file missing: {MODEL_PATH}")
        if not os.path.exists(SCALER_PATH):
            print(f"  ✗ Scaler file missing: {SCALER_PATH}")
        print("\n💡 Make sure you're running app.py from the project root directory")
        print("="*60 + "\n")
except Exception as e:
    print(f"\n⚠️ Error loading model: {e}")
    import traceback
    traceback.print_exc()
    print("="*60 + "\n")

# =========================
# Page layout
# =========================
layout = html.Div(
    style={'backgroundColor': '#CAF0F8', 'minHeight': '100vh', 'padding': '20px', 'fontFamily': 'Arial, sans-serif'},
    children=[
        html.H1("Doppler Shift Analysis", style={'color': '#03045E', 'textAlign': 'center', 'marginBottom': '30px'}),
        
        # Back button
        html.Div([
            dcc.Link(html.Button("← Back to Home", style={
                'backgroundColor': '#023E8A',
                'color': 'white',
                'borderRadius': '15px',
                'padding': '10px 20px',
                'fontSize': '16px',
                'border': 'none',
                'cursor': 'pointer',
                'marginBottom': '20px'
            }), href='/')
        ]),
        
        # Main buttons container
        html.Div(id='doppler-main-buttons', style={'textAlign': 'center', 'marginTop': '50px'}, children=[
            html.Button("Generation of Sound", id='btn-generate-sound', n_clicks=0, style={
                'backgroundColor': '#023E8A',
                'color': 'white',
                'borderRadius': '25px',
                'padding': '15px 40px',
                'fontSize': '18px',
                'border': 'none',
                'cursor': 'pointer',
                'margin': '10px'
            }),
            html.Button("Prediction of Freq and Velocity from Sound", id='btn-predict', n_clicks=0, style={
                'backgroundColor': '#023E8A',
                'color': 'white',
                'borderRadius': '25px',
                'padding': '15px 40px',
                'fontSize': '18px',
                'border': 'none',
                'cursor': 'pointer',
                'margin': '10px'
            })
        ]),
        
        # Generation section
        html.Div(id='generation-section', style={'display': 'none', 'marginTop': '30px'}, children=[
            html.Div(style={'backgroundColor': 'white', 'borderRadius': '15px', 'padding': '30px', 'maxWidth': '800px', 'margin': '0 auto'}, children=[
                html.H3("Generate Doppler Shift Sound", style={'color': '#03045E', 'marginBottom': '20px'}),
                
                html.Label("Velocity (m/s):", style={'fontSize': '16px', 'fontWeight': 'bold', 'color': '#023E8A'}),
                dcc.Input(id='input-velocity', type='number', value=60.0, step=1.0,
                          style={'width': '100%', 'padding': '10px', 'marginBottom': '20px', 'borderRadius': '5px', 'border': '2px solid #023E8A'}),
                
                html.Label("Base Frequency (Hz):", style={'fontSize': '16px', 'fontWeight': 'bold', 'color': '#023E8A'}),
                dcc.Input(id='input-frequency', type='number', value=110.0, step=1.0,
                          style={'width': '100%', 'padding': '10px', 'marginBottom': '20px', 'borderRadius': '5px', 'border': '2px solid #023E8A'}),
                
                html.Label("Duration (seconds):", style={'fontSize': '16px', 'fontWeight': 'bold', 'color': '#023E8A'}),
                dcc.Input(id='input-duration', type='number', value=12.0, step=0.5, min=1.0, max=30.0,
                          style={'width': '100%', 'padding': '10px', 'marginBottom': '20px', 'borderRadius': '5px', 'border': '2px solid #023E8A'}),
                
                html.Button("Generate Sound", id='btn-generate', n_clicks=0, style={
                    'backgroundColor': '#0077B6',
                    'color': 'white',
                    'borderRadius': '15px',
                    'padding': '12px 30px',
                    'fontSize': '16px',
                    'border': 'none',
                    'cursor': 'pointer',
                    'marginTop': '10px'
                }),
                
                html.Div(id='audio-output', style={'marginTop': '30px'}),
                html.Div(id='signal-plot', style={'marginTop': '20px'})
            ])
        ]),
        
        # Prediction section
        html.Div(id='prediction-section', style={'display': 'none', 'marginTop': '30px'}, children=[
            html.Div(style={'backgroundColor': 'white', 'borderRadius': '15px', 'padding': '30px', 'maxWidth': '800px', 'margin': '0 auto'}, children=[
                html.H3("Predict Frequency and Velocity from Sound", style={'color': '#03045E', 'marginBottom': '20px'}),
                
                # Model status indicator
                html.Div([
                    html.P(
                        "✓ Model loaded and ready" if MODEL_LOADED else "⚠️ Model not loaded - prediction unavailable",
                        style={
                            'color': '#28a745' if MODEL_LOADED else '#dc3545',
                            'fontWeight': 'bold',
                            'padding': '10px',
                            'backgroundColor': '#d4edda' if MODEL_LOADED else '#f8d7da',
                            'borderRadius': '5px',
                            'marginBottom': '20px'
                        }
                    )
                ]),
                
                html.P("Upload a sound file to analyze:", style={'fontSize': '16px', 'color': '#023E8A'}),
                dcc.Upload(
                    id='upload-audio',
                    children=html.Div([
                        'Drag and Drop or ',
                        html.A('Select a WAV File')
                    ]),
                    style={
                        'width': '100%',
                        'height': '60px',
                        'lineHeight': '60px',
                        'borderWidth': '2px',
                        'borderStyle': 'dashed',
                        'borderRadius': '10px',
                        'borderColor': '#023E8A',
                        'textAlign': 'center',
                        'backgroundColor': '#E8F4F8',
                        'cursor': 'pointer'
                    },
                    disabled=not MODEL_LOADED
                ),
                html.Div(id='prediction-output', style={'marginTop': '30px'})
            ])
        ])
    ]
)

# =========================
# Sound Generation Function
# =========================
def realistic_car_passby(velocity=90.0, base_freq=100.0, duration=12.0, sr=44100):
    c = 343.0
    t = np.linspace(0, duration, int(sr*duration), endpoint=False)
    dt = 1/sr
    pass_time = duration/2.0
    v_rel = np.where(t <= pass_time, velocity, -velocity)
    rpm_curve = base_freq * (1 + 0.4 * (t/duration))
    base_inst_freq = rpm_curve * (c / (c - v_rel))
    jitter = 1.0 + 0.004*np.random.randn(len(t))
    inst_freq = base_inst_freq * jitter
    inst_phase = np.cumsum(2*np.pi*inst_freq*dt)
    orders = [1, 2, 3, 4, 5, 6]
    amps = [1.0, 0.6, 0.4, 0.25, 0.15, 0.1]
    harmonics = sum(a*np.sin(k*inst_phase) for k, a in zip(orders, amps))
    rumble = 0.35*np.sin(2*np.pi*18*t + 0.5*np.sin(2*np.pi*2*t))
    road_noise = np.random.randn(len(t)) * 0.015 * (np.abs(v_rel)/velocity)
    b, a = signal.butter(4, 300/(sr/2), btype="low")
    road_noise = signal.lfilter(b, a, road_noise)
    wind = np.random.randn(len(t)) * 0.01 * (np.abs(v_rel)/velocity)
    b, a = signal.butter(4, 500/(sr/2), btype="highpass")
    wind = signal.lfilter(b, a, wind)
    engine = harmonics + rumble + road_noise + wind
    for f0 in [180, 600, 1200, 2500]:
        b, a = signal.iirpeak(f0/(sr/2), Q=10)
        engine = signal.lfilter(b, a, engine)
    forward_dist = np.where(t <= pass_time, velocity*(pass_time - t), velocity*(t - pass_time))
    lateral_offset = 3.0
    dist = np.sqrt(forward_dist**2 + lateral_offset**2)
    att = 1.0 / (0.2 + dist/4.0)
    engine *= att
    pan = np.tanh((t - pass_time) / (duration/10))
    left = engine * np.sqrt(0.5*(1 - pan))
    right = engine * np.sqrt(0.5*(1 + pan))
    left *= (1 + 0.02*np.random.randn(len(t)))
    right *= (1 + 0.02*np.random.randn(len(t)))
    stereo = np.vstack([left, right]).T.astype(np.float32)
    ir = np.zeros(int(sr*0.25))
    ir[0] = 1
    ir[int(sr*0.05)] = 0.25
    ir[int(sr*0.1)] = 0.15
    ir[int(sr*0.18)] = 0.08
    for ch in range(2):
        reverb = signal.fftconvolve(stereo[:,ch], ir)[:len(stereo)]
        stereo[:,ch] = 0.75*stereo[:,ch] + 0.25*reverb
    stereo /= np.max(np.abs(stereo)) + 1e-6
    return sr, stereo, t, inst_freq

# =========================
# Feature Extraction for Prediction
# =========================
def extract_features(audio_path, sr=22050, n_mfcc=40, n_fft=2048, hop_length=512):
    try:
        y, sr = librosa.load(audio_path, sr=sr)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
        mfccs_mean = np.mean(mfccs, axis=1)
        mfccs_std = np.std(mfccs, axis=1)
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        return np.concatenate([mfccs_mean, mfccs_std, [spectral_centroid, spectral_rolloff, zero_crossing_rate], chroma_mean])
    except Exception as e:
        print(f"⚠️ Error processing audio: {e}")
        return None

def calculate_doppler_frequency(velocity_kmh, source_frequency=5000.0, speed_of_sound=343.0):
    velocity_ms = velocity_kmh * (1000.0 / 3600.0)
    if velocity_ms >= 0:
        return source_frequency * (speed_of_sound / (speed_of_sound - velocity_ms))
    else:
        return source_frequency * (speed_of_sound / (speed_of_sound + abs(velocity_ms)))

# =========================
# Callbacks
# =========================
def register_callbacks(app):
    # -------------------------------
    # Section toggle callback
    # -------------------------------
    @app.callback(
        [Output('generation-section', 'style'), Output('prediction-section', 'style')],
        [Input('btn-generate-sound', 'n_clicks'), Input('btn-predict', 'n_clicks')],
        prevent_initial_call=True
    )
    def toggle_sections(generate_clicks, predict_clicks):
        ctx = dash.callback_context
        if not ctx.triggered:
            return {'display': 'none'}, {'display': 'none'}
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if button_id == 'btn-generate-sound':
            return {'display': 'block'}, {'display': 'none'}
        elif button_id == 'btn-predict':
            return {'display': 'none'}, {'display': 'block'}
        return {'display': 'none'}, {'display': 'none'}

    # -------------------------------
    # Sound generation callback
    # -------------------------------
    @app.callback(
        [Output('audio-output', 'children'), Output('signal-plot', 'children')],
        Input('btn-generate', 'n_clicks'),
        [State('input-velocity', 'value'), State('input-frequency', 'value'), State('input-duration', 'value')],
        prevent_initial_call=True
    )
    def generate_sound(n_clicks, velocity, frequency, duration):
        if n_clicks == 0:
            return "", ""
        sr, stereo, t, inst_freq = realistic_car_passby(velocity=velocity, base_freq=frequency, duration=duration)
        audio_data = (stereo * 32767).astype(np.int16)
        buffer = io.BytesIO()
        wavfile.write(buffer, sr, audio_data)
        buffer.seek(0)
        audio_base64 = base64.b64encode(buffer.read()).decode()
        audio_player = html.Div([html.H4("Generated Audio:", style={'color': '#03045E'}),
                                 html.Audio(src=f'data:audio/wav;base64,{audio_base64}', controls=True, style={'width': '100%'})])
        max_points = 5000
        if len(t) > max_points:
            step = len(t) // max_points
            t_plot = t[::step]
            signal_plot_data = stereo[::step, 0]
            freq_plot_data = inst_freq[::step]
        else:
            t_plot = t
            signal_plot_data = stereo[:, 0]
            freq_plot_data = inst_freq
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t_plot, y=signal_plot_data, mode='lines', name='Signal Amplitude', line=dict(color='#0077B6')))
        fig.update_layout(title='Signal Waveform (Left Channel)', xaxis_title='Time (s)', yaxis_title='Amplitude',
                          plot_bgcolor='white', paper_bgcolor='white', font=dict(color='#03045E'))
        fig_freq = go.Figure()
        fig_freq.add_trace(go.Scatter(x=t_plot, y=freq_plot_data, mode='lines', name='Instantaneous Frequency', line=dict(color='#023E8A')))
        fig_freq.update_layout(title='Doppler-Shifted Frequency Over Time', xaxis_title='Time (s)', yaxis_title='Frequency (Hz)',
                               plot_bgcolor='white', paper_bgcolor='white', font=dict(color='#03045E'))
        signal_viz = html.Div([html.H4("Signal Visualization:", style={'color': '#03045E', 'marginTop': '20px'}),
                               dcc.Graph(figure=fig), dcc.Graph(figure=fig_freq),
                               html.P(f"Velocity: {velocity} m/s | Base Frequency: {frequency} Hz | Duration: {duration} s",
                                      style={'textAlign': 'center', 'color': '#023E8A', 'fontSize': '14px', 'marginTop': '10px'})])
        return audio_player, signal_viz

    # -------------------------------
    # Prediction callback
    # -------------------------------
    @app.callback(
        Output('prediction-output', 'children'),
        Input('upload-audio', 'contents'),
        State('upload-audio', 'filename'),
        prevent_initial_call=True
    )
    def predict_from_audio(contents, filename):
        if contents is None:
            return ""

        # Check if model is loaded
        if not MODEL_LOADED or model is None or scaler is None:
            return html.Div([
                html.P("⚠️ Model not loaded. Cannot predict.", 
                       style={'color': 'red', 'fontWeight': 'bold', 'fontSize': '16px'}),
                html.P(f"Model path: {MODEL_PATH}", style={'color': '#666', 'fontSize': '14px'}),
                html.P(f"Scaler path: {SCALER_PATH}", style={'color': '#666', 'fontSize': '14px'}),
                html.P("Please check that the model files exist in the correct location.", 
                       style={'color': '#666', 'fontSize': '14px'})
            ])

        try:
            # Decode uploaded audio
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)
            buffer = io.BytesIO(decoded)

            # Load audio for plotting and features
            y, sr = librosa.load(buffer, sr=22050)
            buffer.seek(0)
            features = extract_features(buffer)
            
            if features is None:
                return html.Div([html.P("⚠️ Failed to extract features from audio.", style={'color': 'red'})])
            
            features_scaled = scaler.transform(features.reshape(1, -1))
            velocity = float(model.predict(features_scaled, verbose=0)[0][0])
            doppler_freq = calculate_doppler_frequency(velocity)

            # Waveform plot
            max_points = 5000
            if len(y) > max_points:
                step = len(y) // max_points
                t_plot = np.arange(len(y))[::step] / sr
                y_plot = y[::step]
            else:
                t_plot = np.arange(len(y)) / sr
                y_plot = y

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=t_plot, y=y_plot, mode='lines', name='Waveform', line=dict(color='#0077B6')))
            fig.update_layout(title='Uploaded Signal Waveform', xaxis_title='Time (s)', yaxis_title='Amplitude',
                              plot_bgcolor='white', paper_bgcolor='white', font=dict(color='#03045E'))

            return html.Div([
                html.H4("Prediction Results:", style={'color': '#03045E'}),
                html.P(f"File: {filename}", style={'color': '#023E8A'}),
                html.P(f"Predicted Velocity: {velocity:.2f} km/h", style={'color': '#023E8A', 'fontWeight': 'bold', 'fontSize': '18px'}),
                html.P(f"Predicted Doppler Frequency: {doppler_freq:.2f} Hz", style={'color': '#023E8A', 'fontWeight': 'bold', 'fontSize': '18px'}),
                dcc.Graph(figure=fig)
            ])
        
        except Exception as e:
            return html.Div([
                html.P(f"⚠️ Error during prediction: {str(e)}", style={'color': 'red', 'fontWeight': 'bold'}),
                html.P("Please make sure you uploaded a valid WAV file.", style={'color': '#666'})
            ])