import os
import base64
import numpy as np
import xarray as xr
from dash import html, dcc, Input, Output, State
import plotly.graph_objects as go

# --------------- CONFIG -----------------
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --------------- Layout -----------------
layout = html.Div(
    style={
        'backgroundColor': '#FFFFFF',  # White background
        'fontFamily': 'Arial, sans-serif',
        'minHeight': '100vh',
        'padding': '30px'
    },
    children=[
        # Header
        html.Div(
            style={'textAlign': 'center', 'marginBottom': '40px'},
            children=[
                html.H1("🛰️ Sentinel-1 InSAR Displacement Viewer",
                        style={'color': '#1E3A8A', 'fontWeight': 'bold', 'fontSize': '42px'}),
                html.P("Upload a NetCDF (.nc) InSAR product to visualize surface displacement",
                       style={'color': '#374151', 'fontSize': '18px', 'fontStyle': 'italic'})
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
                html.H3("Upload InSAR NetCDF File", style={'color': '#1E3A8A', 'marginBottom': '15px'}),
                dcc.Upload(
                    id='upload-insar',
                    children=html.Div([
                        '📁 Drag & Drop or Click to Select .nc File',
                        html.Br(),
                        html.Small('(Sentinel-1 InSAR interferogram data)', style={'color': '#6B7280'})
                    ]),
                    style={
                        'width': '100%',
                        'height': '100px',
                        'lineHeight': '40px',
                        'borderWidth': '3px',
                        'borderStyle': 'dashed',
                        'borderColor': '#3B82F6',
                        'borderRadius': '15px',
                        'textAlign': 'center',
                        'backgroundColor': '#F8FAFF',
                        'cursor': 'pointer'
                    },
                    multiple=False
                ),
                html.Div(id='insar-file-status', style={'marginTop': '15px', 'textAlign': 'center', 'color': '#374151'})
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
                html.H3("Processing Controls", style={'color': '#1E3A8A'}),
                html.Div(
                    style={'textAlign': 'center', 'marginTop': '20px'},
                    children=[
                        html.Button(
                            "Generate Displacement Map",
                            id='process-insar-btn',
                            n_clicks=0,
                            style={
                                'backgroundColor': '#10B981',
                                'color': 'white',
                                'border': 'none',
                                'borderRadius': '10px',
                                'padding': '12px 30px',
                                'fontSize': '16px',
                                'fontWeight': 'bold',
                                'cursor': 'pointer',
                                'boxShadow': '0 2px 8px rgba(16,185,129,0.3)'
                            }
                        )
                    ]
                )
            ]
        ),

        # Output Area
        dcc.Loading(
            id="loading-insar",
            type="cube",
            color="#3B82F6",
            children=[
                html.Div(id='insar-summary'),
                html.Div(id='insar-plot-container', style={'marginTop': '30px'})
            ]
        ),

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
                    ), href='/'
                )
            ]
        )
    ]
)

# --------------- Helpers -----------------
def process_insar_file(file_path):
    ds = xr.open_dataset(file_path, group='science/grids/data')
    unwrapped_phase = ds['unwrappedPhase']
    wavelength = 0.056  # Sentinel-1 C-band
    displacement = (unwrapped_phase * wavelength) / (4 * np.pi)
    return displacement
def create_displacement_plot(displacement):
    disp_np = np.array(displacement.values, dtype=np.float64)

    # Keep NaNs instead of converting them to zeros
    valid_mask = np.isfinite(disp_np)
    valid_disp = disp_np[valid_mask]

    if valid_disp.size == 0:
        max_disp = min_disp = mean_disp = 0.0
    else:
        max_disp = float(np.nanmax(valid_disp))
        min_disp = float(np.nanmin(valid_disp))
        mean_disp = float(np.nanmean(valid_disp))

    # Plot heatmap (hide NaN pixels using masked array)
    fig = go.Figure(data=go.Heatmap(
        z=np.where(valid_mask, disp_np, np.nan),  # keep NaN for no-data
        colorscale='Jet',
        colorbar=dict(title="Displacement (m)"),
        zmin=min_disp,
        zmax=max_disp
    ))

    fig.update_layout(
        title="Surface Displacement Map (m)",
        xaxis_title="Longitude (pixels)",
        yaxis_title="Latitude (pixels)",
        template="plotly_white",
        height=700,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF"
    )

    # Histogram of valid values only
    hist_fig = go.Figure(data=[
        go.Histogram(x=valid_disp, nbinsx=50, marker=dict(color="#0077B6"))
    ])
    hist_fig.update_layout(
        title="Displacement Value Distribution",
        xaxis_title="Displacement (m)",
        yaxis_title="Count",
        template="plotly_white",
        height=400,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF"
    )

    summary = html.Div(
        style={'backgroundColor': "#FFFFFF", 'borderRadius': '15px', 'padding': '20px',
               'boxShadow': '0 4px 10px rgba(0,0,0,0.05)', 'marginBottom': '20px'},
        children=[
            html.H3("Displacement Statistics", style={'color': "#1E3A8A"}),
            html.P(f"Max Displacement: {max_disp:.4f} m"),
            html.P(f"Min Displacement: {min_disp:.4f} m"),
            html.P(f"Mean Displacement: {mean_disp:.4f} m")
        ]
    )

    return summary, hist_fig, fig
def register_callbacks(app):
    # --- Callback 1: Update upload status immediately ---
    @app.callback(
        Output('insar-file-status', 'children'),
        Input('upload-insar', 'filename')
    )
    def update_upload_status(filename):
        if filename:
            return f"✅ File uploaded: {filename}"
        else:
            return "No file uploaded yet."

    # --- Callback 2: Process InSAR when button clicked ---
    @app.callback(
        Output('insar-summary', 'children'),
        Output('insar-plot-container', 'children'),
        Input('process-insar-btn', 'n_clicks'),
        State('upload-insar', 'contents'),
        State('upload-insar', 'filename')
    )
    def handle_insar_upload(n_clicks, contents, filename):
        if n_clicks == 0:
            return "", ""
        if not contents:
            return html.Div("❌ Please upload a .nc file first.", style={'color': 'red'}), ""

        try:
            _, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)

            file_path = os.path.join(UPLOAD_FOLDER, filename)
            with open(file_path, 'wb') as f:
                f.write(decoded)

            displacement = process_insar_file(file_path)
            summary, hist_fig, heatmap_fig = create_displacement_plot(displacement)

            plots = html.Div([
                dcc.Graph(figure=hist_fig, style={'marginBottom': '40px'}),
                dcc.Graph(figure=heatmap_fig)
            ])

            return summary, plots

        except Exception as e:
            return html.Div(f"❌ Error: {str(e)}", style={'color': 'red'}), ""
