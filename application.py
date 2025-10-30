import dash
from dash import html as dhtml
from dash import dcc, Input, Output, State
from dash.dependencies import Input, Output, State
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import dash_bootstrap_components as dbc

#non-plotly imports
import numpy as np

#local imports
import kalman_filter as kal

'''
========================================================================================================================
Data
'''
image_component = dhtml.Img(src='/assets/logo.png', style={'width': '100%', 'height': 'auto'})

'''
========================================================================================================================
Dashboard

'''

graph_config = {'modeBarButtonsToRemove' : ['hoverCompareCartesian', 'select2d', 'lasso2d'],
                'doubleClick':  'reset+autosize', 'toImageButtonOptions': { 'height': None, 'width': None, },
                'displaylogo': False}

colors = {'background': '#111111', 'text': '#7FDBFF'}

app = dash.Dash(__name__,
                meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
                external_stylesheets=[dbc.themes.SLATE])
#application = app.server
app.title = 'Kalman filter simulation'


(state, est_state, meas) = kal.eval_filter(20, 0.1, 0.1)
plot = go.Figure(data=go.Scatter(x=state[:, 0], y=state[:, 2], name='True State', mode='lines+markers'))
plot.add_scatter(x=est_state[:, 0], y=est_state[:, 2], name='Filtered State', mode='lines+markers')
plot.add_scatter(x=meas[:, 0], y=meas[:, 1], name='Observed', mode='markers')
plot.update_layout(paper_bgcolor='#515960', plot_bgcolor='#515960',
                 font_color='white',
                 margin=dict(l=5, r=5, t=5, b=5))
plot.update_xaxes(title_text='x[n]')
plot.update_yaxes(title_text='y[n]')

graph_tab = dbc.Card(
    dbc.CardBody(id='graph-card',
        children=[
            dbc.Col([
            ]),
            dbc.Row([dhtml.H2(f'Kalman Filter', id='loc_label')]),
            dbc.Row([
                dcc.Loading(dcc.Graph(id='time_series', figure=plot))
            ])
        ]
    )
)

kal_mag = np.sqrt(np.square(np.abs(est_state[:, 2] + est_state[:, 0])))
obs_mag = np.sqrt(np.square(np.abs(meas[:, 1] + meas[:, 0])))
state_mag = np.sqrt(np.square(np.abs(state[:, 2] + state[:, 0])))

kal_rmse = np.sqrt(np.sum(np.square(kal_mag - state_mag)) / len(state))
obs_rmse = np.sqrt(np.sum(np.square(obs_mag - state_mag)) / len(state))

diffs = make_subplots(rows=3, cols=1)
x = np.array(range(len(state[:, 0])))

diffs.add_trace(go.Scatter(x=x, y=np.abs(state[:, 2] - est_state[:, 2]), name='Filter x Error', mode='markers'), row=1, col=1)
diffs.add_trace(go.Scatter(x=x, y=np.abs(meas[:, 1] - state[:, 2]), name='Meas. x Error', mode='markers'), row=1, col=1)
diffs.add_trace(go.Scatter(x=x, y=np.abs(state[:, 0] - est_state[:, 0]), name='Filter y Error', mode='markers'), row=2, col=1)
diffs.add_trace(go.Scatter(x=x, y=np.abs(meas[:, 0] - state[:, 0]), name='Meas. y Error', mode='markers'), row=2, col=1)
diffs.add_trace(go.Scatter(x=x, y=kal_mag, name='Filter Norm Error', mode='markers'), row=3, col=1)
diffs.add_trace(go.Scatter(x=x, y=obs_mag, name='Meas. Norm Error', mode='markers'), row=3, col=1)
diffs.update_layout(paper_bgcolor='#515960', plot_bgcolor='#515960',
                 font_color='white',
                 margin=dict(l=5, r=5, t=5, b=5))
diffs.update_xaxes(title_text='Sample Number', row=2, col=1)
diffs.update_yaxes(title_text='Difference', row=1, col=1)
diffs.update_yaxes(title_text='Difference', row=2, col=1)

analysis_tab = dbc.Card(
    dbc.CardBody(id='analysis-card',
        children=[
            dbc.Col([
            ]),
            dbc.Row([dhtml.H2(f'Errors')]),
            dbc.Row([
                dcc.Loading(dcc.Graph(id='difference', figure=diffs))
            ])
        ]
    )
)

controls_tab = dbc.Card(
    dbc.CardBody(id='controls-card',
        children=[
            dbc.Col([
                dhtml.H5('Number of Samples:'),
                dcc.Textarea(id='n_samples', value=f'{20}', style={'height': 30, 'width': '100%'}),
                dhtml.H5('x-axis Noise'),
                dcc.Textarea(id='x_sigma', value=f'{0.1}', style={'height': 30, 'width': '100%'}),
                dhtml.H5('y-axis Noise'),
                dcc.Textarea(id='y_sigma', value=f'{0.1}', style={'height': 30, 'width': '100%'}),
                dhtml.H5('Initial condition T'),
                dcc.Textarea(id='T', value=f'{1}', style={'height': 30, 'width': '100%'}),
                dhtml.H5('Initial condition sigma2'),
                dcc.Textarea(id='sig2', value=f'{0.3}', style={'height': 30, 'width': '100%'}),
                dhtml.Button('Refresh', id='redraw', n_clicks=0),
                #dhtml.Spacer,
                dhtml.H5(f'Kalman Err: {kal_rmse:.4f}', id='kal-err'),
                dhtml.H5(f'Obs. Err: {obs_rmse:.4f}', id='obs-err')
            ])
        ]
    )
)

map_page = dbc.Card(
    dbc.CardBody(id='map_container',
        children=[
            dbc.Row([
                dbc.Col(graph_tab, width=5),
                dbc.Col(analysis_tab, width=5),
                dbc.Col(controls_tab, width=2)
            ])
        ]
    )
)

header = dhtml.Div([
    image_component,
    dbc.Collapse(
        dbc.Card(
            dbc.CardBody(id="header",
                         children=[
                             dbc.Row([
                                 dhtml.H6("This dashboard was created by Jewell GeoServices. If you would like your own custom dashboard, we are available to discuss your needs"),
                                 dcc.Link("Jewell GeoServices", href="https://jewellgeo.services"),

                                 dhtml.H6(""),
                                 dcc.Link("Otherwise, feel free to buy me a coffee.", href="https://www.buymeacoffee.com/shjewell")
                         ])
                     ])
        ),
        id="collapse",
        is_open=True
    ),
    dbc.Button(
        "Hide",
        id="hide-button",
        color="primary",
        n_clicks=0
    )
])


app.layout = dhtml.Div([
    dbc.CardBody(
        id='main_card',
        children=[header,
                  dbc.Card(map_page),
                  dcc.Link('By SHJewell', href=f'https://shjewell.com'),
                  dhtml.H6(f'Built using Python and Plotly Dash'),
                  dcc.Link(f'Based on Kalman filter implementation by Marko Cotra',
                           href=f'https://medium.com/towards-data-science/wtf-is-sensor-fusion-part-2-the-good-old-kalman-filter-3642f321440')
                  ]
    )
])

'''
========================================================================================================================
Callbacks
'''

@app.callback(
    [Output('time_series', 'figure'),
     Output('difference', 'figure'),
     Output('kal-err', 'children'),
     Output('obs-err', 'children')],
    Input('redraw', 'n_clicks'),
    [State('n_samples', 'value'),
     State('x_sigma', 'value'),
     State('y_sigma', 'value'),
     State('T', 'value'),
     State('sig2', 'value')]
)

def refilter(_, N, sig_x, sig_y, T, sig2):

    try:
        (state, est_state, meas) = kal.eval_filter(int(N), float(sig_x), float(sig_y), float(T), float(sig2))
    except ValueError:
        return dash.no_update

    plot = go.Figure(data=go.Scatter(x=state[:, 0], y=state[:, 2], name='True State', mode='lines+markers'))
    plot.add_scatter(x=est_state[:, 0], y=est_state[:, 2], name='Filtered State', mode='lines+markers')
    plot.add_scatter(x=meas[:, 0], y=meas[:, 1], name='Observed', mode='markers')
    plot.update_layout(paper_bgcolor='#515960', plot_bgcolor='#515960',
                     font_color='white',
                     margin=dict(l=5, r=5, t=5, b=5))
    plot.update_xaxes(title_text='x[n]')
    plot.update_yaxes(title_text='y[n]')

    kal_mag = np.sqrt(np.square(np.abs(est_state[:, 2] + est_state[:, 0])))
    obs_mag = np.sqrt(np.square(np.abs(meas[:, 1] + meas[:, 0])))
    state_mag = np.sqrt(np.square(np.abs(state[:, 2] + state[:, 0])))

    kal_rmse = np.sqrt(np.sum(np.square(kal_mag - state_mag)) / len(state))
    obs_rmse = np.sqrt(np.sum(np.square(obs_mag - state_mag)) / len(state))

    diffs = make_subplots(rows=3, cols=1)
    x = np.array(range(len(state[:, 0])))
    kal_mag = np.sqrt(
        np.square(np.abs(state[:, 2] - est_state[:, 2])) + np.square(np.abs(state[:, 0] - est_state[:, 0])))
    obs_mag = np.sqrt(np.square(np.abs(meas[:, 1] - state[:, 2])) + np.square(np.abs(meas[:, 0] - state[:, 0])))
    diffs.add_trace(go.Scatter(x=x, y=np.abs(state[:, 2] - est_state[:, 2]), name='Filter x Error', mode='markers'),
                    row=1, col=1)
    diffs.add_trace(go.Scatter(x=x, y=np.abs(meas[:, 1] - state[:, 2]), name='Meas. x Error', mode='markers'), row=1,
                    col=1)
    diffs.add_trace(go.Scatter(x=x, y=np.abs(state[:, 0] - est_state[:, 0]), name='Filter y Error', mode='markers'),
                    row=2, col=1)
    diffs.add_trace(go.Scatter(x=x, y=np.abs(meas[:, 0] - state[:, 0]), name='Meas. y Error', mode='markers'), row=2,
                    col=1)
    diffs.add_trace(go.Scatter(x=x, y=kal_mag, name='Filter Norm Error', mode='markers'), row=3, col=1)
    diffs.add_trace(go.Scatter(x=x, y=obs_mag, name='Meas. Norm Error', mode='markers'), row=3, col=1)
    diffs.update_layout(paper_bgcolor='#515960', plot_bgcolor='#515960',
                        font_color='white',
                        margin=dict(l=5, r=5, t=5, b=5))
    diffs.update_xaxes(title_text='Sample Number [n]', row=3, col=1)
    diffs.update_yaxes(title_text='x[n] Difference', row=1, col=1)
    diffs.update_yaxes(title_text='y[n] Difference', row=2, col=1)
    diffs.update_yaxes(title_text='Magnitude[n] diff', row=3, col=1)

    kal_err = f'Filter error: {kal_rmse:.3f}'
    obs_err = f'Obs. Error {obs_rmse:.3f}'


    return plot, diffs, kal_err, obs_err


@app.callback(
    Output("collapse", "is_open"),
    [Input("hide-button", "n_clicks")],
    [State("collapse", "is_open")],
)
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open


if __name__ == '__main__':
    #app.run_server(debug=True, port=8080)
    app.run(debug=True, port=8080)
    #application.run(port=8080)

