from dash import Dash, html, dcc
import dash
import dash_bootstrap_components as dbc

app = Dash(__name__, use_pages=True)
# server = app.server
# navbar code if needed
# nav_bar = dbc.Nav(
#             [
#
#                 dbc.NavLink(
#                     [
#                         html.Div(page["name"], className="ms-2"),
#                     ],
#                     href=page["path"],
#                     active="exact",
#                 )
#                 for page in dash.page_registry.values()
#             ],
#             vertical=False,
#             pills=True,
#             className="bg-light",
#             style={'display':'block'},
# )

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

if __name__ == '__main__':
    app.run_server(debug=True)
