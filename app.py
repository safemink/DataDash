from dash import Dash, html, dcc
import dash
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

### When pushing to github, make sure to make all changes locally and then push ###
# dcc.Link(page['name'].title() + "  |  ", href=page['path'])
# for page in dash.page_registry.values()

app = Dash(__name__, use_pages=True, assets_ignore=r"custom\.css")
# Keep the original Bootstrap appearance independent of the retired redesign stylesheet.
app.index_string = app.index_string.replace("{%css%}", "{%css%}<style>" + '/* Changing max size of container for big monitor */\n\n@media (min-width: 1400px) {\n  .container-xxl, .container-xl, .container-lg, .container-md, .container-sm, .container {\n    max-width: 1800px;\n  }\n}\n\n/* Preserve the original light Bootstrap appearance in every system theme. */\n:root { color-scheme: light; }\nbody { background: #fff; color: #212529; }\n.dash-table-container { overflow-x: auto; max-width: 100%; }\n.dash-table-container .dash-spreadsheet-container { max-width: 100%; }\n@media (max-width: 700px) {\n  .original-home > div > div[style*="display: flex"] { display: block !important; margin: 20px !important; }\n  .original-home > div > div > div[style*="width: 50%"] { width: 100% !important; }\n  .container, .container-fluid { padding-left: 12px; padding-right: 12px; }\n  .container-fluid > div[style*="width: 15%"], .container-fluid > div[style*="width: 25%"] { width: 100% !important; margin-bottom: .75rem; }\n  .container-fluid > div[style*="margin: 100px"] { margin: 1rem 0 !important; }\n  .dash-graph { width: 100% !important; display: block !important; }\n  .row > .col-2 { width: 100%; margin-left: 0; }\n}\n' + "</style>")
server = app.server

nav_drop = dbc.DropdownMenu(id='nav',
                            children=[
                                dbc.DropdownMenuItem(page['name'].title(), href=page['path'])
                                for page in dash.page_registry.values()
                            ],
                            in_navbar=True,
                            menu_variant='dark',
                            color='primary',
                            size='lg',
                            className='mb-3'
                            )

app.layout = html.Div(
    [
        # main app framework
        dcc.Location(id='url'),
        html.Div([dbc.NavbarSimple([nav_drop], links_left=True, color="light", dark=False)]),

        # content of each page
        dash.page_container
    ]
)


@app.callback(
    Output('nav', 'label'),
    Input('url', 'pathname'),
    # prevent_initial_call=True
)
def update_nav_label(pathname):
    if pathname == '/':
        label = 'Home'
    else:
        label = pathname.title()[1:].replace('-', ' ')
    return label


# Change to True for Testing
if __name__ == '__main__':
    app.run(debug=False)
