from dash import html, dcc, Input, Output, State, callback
import os
import base64
import numpy as np
import plotly.graph_objects as go
import wfdb
from pyts.image import RecurrencePlot
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from scipy.signal import butter, filtfilt, resample

# ----------------- Ensure uploads folder exists -----------------
os.makedirs("uploads", exist_ok=True)

# ----------------- Layout -----------------
layout = html.Div(
    style={'backgroundColor': '#CAF0F8', 'textAlign': 'center', 'fontFamily': 'Arial, sans-serif', 'padding': '50px'},
    children=[
        html.H1("ECG Analysis", style={'color':'#03045E','fontSize':'48px','marginBottom':'40px'}),

        dcc.Upload(
            id='upload-ecg',
            children=html.Div(['Drag and Drop or ', html.A('Select .dat and .hea files')]),
            style={
                'width': '50%', 'height': '100px', 'lineHeight': '100px',
                'borderWidth': '2px', 'borderStyle': 'dashed', 'borderRadius': '10px',
                'textAlign': 'center', 'margin': 'auto', 'backgroundColor': '#E0FBFC',
                'cursor': 'pointer'
            },
            multiple=True
        ),

        html.Div(id='file-list', style={'marginTop': '20px', 'fontSize':'20px','color':'#03045E'}),

        dcc.Loading(
            id="loading-ecg",
            type="circle",
            children=[
                html.Button("Process ECG", id='process-btn', n_clicks=0, style={
                    'marginTop':'20px', 'backgroundColor': '#023E8A', 'color': 'black',
                    'borderRadius': '25px', 'padding': '15px 40px', 'fontSize':'18px',
                    'border':'none', 'cursor':'pointer'
                }),
                html.Div(id='ecg-output', style={'marginTop': '40px','fontSize':'24px','color':'#03045E'}),
                dcc.Graph(id='ecg-waveform', style={'display':'none'}),
                dcc.Graph(id='ecg-polar', style={'display':'none'}),
                dcc.Graph(id='ecg-rp', style={'display':'none'}),
            ]
        ),

        html.Br(),
        dcc.Link(html.Button("Back to Home", style={
            'backgroundColor': '#023E8A', 'color': 'black', 'borderRadius': '25px',
            'padding': '15px 40px', 'fontSize':'18px', 'border': 'none', 'cursor': 'pointer', 'marginTop':'30px'
        }), href='/')
    ]
)

# ----------------- ECG Functions -----------------
def bandpass(sig, fs, low=0.5, high=40, order=4):
    nyq = 0.5*fs
    b,a = butter(order, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, sig)

def process_ecg(record_path, target_fs=250):
    rec = wfdb.rdrecord(record_path)
    signal = rec.p_signal[:, 0]
    fs = rec.fs
    signal = bandpass(signal, fs)
    if fs != target_fs:
        n = int(len(signal) * target_fs / fs)
        signal = resample(signal, n)
        fs = target_fs
    return signal, fs

def generate_rp_image(signal):
    rp = RecurrencePlot()
    X_rp = rp.fit_transform(signal.reshape(1, -1))[0]
    X_rp = ((X_rp - X_rp.min()) / (X_rp.max() - X_rp.min() + 1e-12) * 255).astype('uint8')
    img_rp = Image.fromarray(X_rp).convert('RGB').resize((224,224))
    return img_rp

# ----------------- Callbacks -----------------

# Show uploaded filenames
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

# Process ECG on button click
@callback(
    Output('ecg-waveform', 'figure'),
    Output('ecg-polar', 'figure'),
    Output('ecg-rp', 'figure'),
    Output('ecg-output', 'children'),
    Output('ecg-waveform', 'style'),
    Output('ecg-polar', 'style'),
    Output('ecg-rp', 'style'),
    Input('process-btn', 'n_clicks'),
    State('upload-ecg', 'contents'),
    State('upload-ecg', 'filename')
)
def update_ecg(n_clicks, contents, filenames):
    if n_clicks == 0:
        return {}, {}, {}, "", {'display':'none'}, {'display':'none'}, {'display':'none'}
    
    if not contents or not filenames:
        return {}, {}, {}, "Please upload .dat and .hea files before processing.", {'display':'none'}, {'display':'none'}, {'display':'none'}

    if isinstance(contents, str):
        contents = [contents]
        filenames = [filenames]

    if not any(f.endswith('.dat') for f in filenames) or not any(f.endswith('.hea') for f in filenames):
        return {}, {}, {}, "Please upload both .dat and .hea files.", {'display':'none'}, {'display':'none'}, {'display':'none'}

    # Save uploaded files
    for content, name in zip(contents, filenames):
        data = content.split(',')[1]
        with open(f"uploads/{name}", "wb") as f:
            f.write(base64.b64decode(data))

    # Find .dat record path without extension
    record_name = None
    for f in filenames:
        if f.endswith('.dat'):
            record_name = f"uploads/{f.split('.')[0]}"
            break
    if record_name is None:
        return {}, {}, {}, "Cannot find .dat file.", {'display':'none'}, {'display':'none'}, {'display':'none'}

    try:
        signal, fs = process_ecg(record_name)
    except Exception as e:
        return {}, {}, {}, f"Error reading record: {e}", {'display':'none'}, {'display':'none'}, {'display':'none'}

    # Waveform plot
    fig_wave = go.Figure()
    fig_wave.add_trace(go.Scatter(y=signal, mode='lines', name='ECG Waveform'))
    fig_wave.update_layout(title="ECG Waveform", xaxis_title="Samples", yaxis_title="Amplitude (mV)")

    # Polar plot
    fig_polar = go.Figure()
    fig_polar.add_trace(go.Scatterpolar(r=signal, theta=list(range(len(signal))),
                                        mode='lines', name='Polar ECG'))
    fig_polar.update_layout(title="Polar Plot of ECG Signal")

    # Recurrence plot
    img_rp = generate_rp_image(signal)
    fig_rp = go.Figure()
    fig_rp.add_trace(go.Image(z=np.array(img_rp)))
    fig_rp.update_layout(title="Recurrence Plot (RP)")

    # Prediction
    x = np.expand_dims(np.array(img_rp), axis=0)
    x = preprocess_input(x)
    model = tf.keras.models.load_model('models/best_resnet_ecg.h5')
    pred = model.predict(x)[0][0]
    label = "Healthy" if pred < 0.5 else "Abnormal"

    return (fig_wave, fig_polar, fig_rp,
            f"Prediction: {label} ({pred:.2f})",
            {'display':'block'}, {'display':'block'}, {'display':'block'})
