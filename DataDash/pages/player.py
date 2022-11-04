import dash
from dash import Dash, html, dash_table, Input, Output, dcc, callback
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

### To link pages, use dash.register page and remove the 'app' from app.layout/@app.callback

# external_stylesheets = [{'src': 'static/css/styles.css'}] (https://www.youtube.com/watch?v=vqVwpL4bGKY)
dash.register_page(__name__)

df = pd.read_csv('master_data_cleaned.csv')
df.drop(df.columns[0], axis=1, inplace=True)

# columns to include
df = df[(df.stat_type == 'Player')]  # filter by stat type
df = df[['Last', 'Minutes', 'Seconds', 'Goals', 'Shots', 'Opponent', 'Outcome', 'Team']]

df.rename(columns={'Last': 'Name'}, inplace=True)  # change last name col into name

# sort the teams in alphabetical order
teams = sorted(df.Team.unique().tolist())

layout = html.Div(
    children=[
        html.H1(children='Individual Player Data'),
        html.Div(children='Filter by Team and Player to view relevant statistics'),
        html.Br(),

        html.Div([
            dcc.Dropdown(
                id='filter_dropdown1',
                options=[{'label': t, 'value': t} for t in teams],
                value='USA',
            ),
        ], style={'width': '25%', 'display': 'inline-block'}),
        html.Div([
            dcc.Dropdown(
                id='filter_dropdown2',
                value='WOODHEAD',
                options=[],
            ),
        ], style={'width': '25%', 'display': 'inline-block'}),
        dash_table.DataTable(id='table-container',  # initiate table
                             css=[{'selector': 'table', 'rule': 'table-layout: fixed'}],
                             style_cell={
                                 'width': '{}%'.format(len(df.columns))
                             }
                             ),
        html.Div(children=[
            dcc.Graph(id='bar-chart',
                      style={'width': '50%', 'display': 'inline-block'}),
            dcc.Graph(id='pie-chart',
                      style={'width': '50%', 'display': 'inline-block'})
        ])

    ], style={'width': '80%'}  # adjust the size of the table

)


@callback(
    Output('filter_dropdown2', 'options'),
    Input('filter_dropdown1', 'value'))
def players_on_team(team):
    dff = df[df.Team == team]
    players = sorted(dff.Name.unique().tolist())
    return [{'label': p, 'value': p} for p in players]


@callback(
    [Output('table-container', 'data'),
     Output('bar-chart', 'figure'),
     Output('pie-chart', 'figure')],
    [Input('filter_dropdown1', 'value'),
     Input('filter_dropdown2', 'value')])
def display_info(team, name):
    dff = df[(df.Name == name) & (df.Team == team)]
    dash_table.DataTable(id='table-container',
                         columns=[{'id': c, 'name': c} for c in df.columns.values])  # apply to table
    # Could use a Go object to add traces and show goals/shots per opponent
    bar = px.bar(dff, x='Opponent', y='Goals', color='Opponent')
    bar.update_xaxes(categoryorder='total descending')
    pie = px.pie(dff, names='Opponent', values='Goals', color='Opponent')

    return dff.to_dict('records'), bar, pie




