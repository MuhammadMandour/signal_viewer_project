from dash import Dash, html, dcc, Input, Output
import dash

app = Dash(__name__, use_pages=True, suppress_callback_exceptions=True)
server = app.server

home_layout = html.Div(
    style={'backgroundColor': '#CAF0F8', 'textAlign': 'center', 'fontFamily': 'Arial, sans-serif', 'padding': '50px'},
    children=[
        html.H1(
            "Welcome to Signal Viewer",
            style={'color': '#03045E', 'fontSize': '64px', 'fontWeight': 'bold', 'marginBottom': '40px'}
        ),
        html.P(
            "Signal Viewer is a professional platform for analyzing a wide range of signals. "
            "It is divided into two main parts: \n"
            "- Medical: Explore physiological signals such as ECG and EEG with advanced visualization and analysis tools. "
            "You can detect abnormalities, study patterns, and monitor vital signals in a user-friendly interface. \n"
            "- Sound: Dive into acoustic analysis through Doppler Shift detection and Radar signal processing. "
            "Analyze frequency shifts, speeds, and signal characteristics with precision and clarity. "
            "Signal Viewer is designed to combine both medical and sound signal analysis in one seamless platform.",
            style={'color': 'black', 'fontSize': '20px', 'lineHeight': '1.8', 'whiteSpace': 'pre-line', 'marginBottom': '50px'}
        ),
        html.Div(
            style={'display': 'flex', 'justifyContent': 'center', 'gap': '30px'},
            children=[
                dcc.Link(html.Button("ECG", style={
                    'backgroundColor': '#023E8A', 'color': 'white', 'borderRadius': '25px',
                    'padding': '15px 40px', 'fontSize': '18px', 'border': 'none', 'cursor': 'pointer'
                }), href='/ecg'),
                dcc.Link(html.Button("EEG", style={
                    'backgroundColor': '#023E8A', 'color': 'white', 'borderRadius': '25px',
                    'padding': '15px 40px', 'fontSize': '18px', 'border': 'none', 'cursor': 'pointer'
                }), href='/eeg'),
                html.Button("Doppler Shift", style={'backgroundColor': '#023E8A', 'color': 'white', 'borderRadius': '25px',
                                                    'padding': '15px 40px', 'fontSize': '18px', 'border': 'none', 'cursor': 'pointer'}),
                html.Button("Radar", style={'backgroundColor': '#023E8A', 'color': 'white', 'borderRadius': '25px',
                                            'padding': '15px 40px', 'fontSize': '18px', 'border': 'none', 'cursor': 'pointer'}),
            ]
        )
    ]
)

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

# Import and register page callbacks
import pages.ecg_page
import pages.eeg_page

# Register EEG callbacks if the module has the registration function
if hasattr(pages.eeg_page, 'register_callbacks'):
    pages.eeg_page.register_callbacks(app)

# Register ECG callbacks - you'll need to add a similar register_callbacks function to ecg_page.py
# if hasattr(pages.ecg_page, 'register_callbacks'):
#     pages.ecg_page.register_callbacks(app)

@app.callback(
    Output('page-content', 'children'),
    Input('url', 'pathname')
)
def display_page(pathname):
    if pathname == '/ecg':
        return pages.ecg_page.layout
    elif pathname == '/eeg':
        return pages.eeg_page.layout
    else:
        return home_layout

if __name__ == '__main__':
    app.run(debug=True)