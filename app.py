from dash import Dash, html, dcc
import dash
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

### When pushing to github, make sure to make all changes locally and then push ###
# dcc.Link(page['name'].title() + "  |  ", href=page['path'])
# for page in dash.page_registry.values()

app = Dash(__name__, use_pages=True)
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
