from dash import Dash, html, dcc
import dash
import dash_bootstrap_components as dbc

app = Dash(__name__, use_pages=True)
server = app.server

app.layout = html.Div(
    [
        # main app framework
        html.Div("Data Visualization", style={'fontSize': 50, 'textAlign': 'center'}),
        html.Div([
            dcc.Link(page['name'].title() + "  |  ", href=page['path'])
            for page in dash.page_registry.values()
        ]),
        html.Hr(),

        # content of each page
        dash.page_container
    ]
)

# Change to True for Testing
if __name__ == '__main__':
    app.run_server(debug=False)
