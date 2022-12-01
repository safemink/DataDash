import dash
from dash import Dash, html, dash_table, Input, Output, dcc, callback
import plotly.express as px
import pandas as pd

dash.register_page(__name__, order=1)


# read in csv and filter by stat type, drop index col, add shooting percentage and reorder shot %
# return df
def initiate_df(stat_type):
    df = pd.read_csv('data/master_data_cleaned.csv')
    df = df.rename(columns={'Last': 'Name'}).drop(df.columns[0], axis=1)
    df['Shooting %'] = (df.Goals / df.Shots * 100).round(2)
    cols = list(df)
    cols = cols[:7] + ['Shooting %'] + cols[7:-1]
    return df[(df.stat_type == stat_type)][cols]


# takes in table dfs and returns data and cols
def table_data_cols(*tbl_df):
    table_list = []
    for table in tbl_df:
        table_list.append(table.to_dict('records'))
        table_list.append([{'id': c, 'name': c, 'hideable': True} for c in table.columns])
    return table_list


def filter_team_name(team, player, df):
    return df[(df.Name == player) & (df.Team == team)]


# creates and returns list of UNIQUE games (some games are repeat opponents)
def game_col(df):
    opp_list = list(df.Opponent)
    return [f"{opp_list[i]}:{i + 1}" for i in range(0, df.Opponent.shape[0])]


# gets play time as an int (using seconds)
def playtime_flt(df):
    df['Play Time'] = (df.Minutes + df.Seconds / 60).round(2)
    return df


# style shooting percentage in table
def shot_per_style():
    bounds = [100, 80, 60, 40, 20, 0]
    colors = ['#00ff00', '#99ff66', '#ffff00', '#ffbf00', '#f74040']
    temp = []
    for i in range(len(bounds) - 1):
        s = f'{{Shooting %}} <= {bounds[i]} && {{Shooting %}} > {bounds[i + 1] - 1}'
        style = {'if':
                     {'filter_query': s,
                      'column_id': 'Shooting %'},
                 'backgroundColor': colors[i],
                 }
        temp.append(style)
    return temp


# customized table
def custom_table(df):
    dff = df[FULL_COLS]
    data = dff.to_dict('records')
    columns = [{'id': c, 'name': c, 'hideable': True} for c in dff.columns]
    return data, columns


# df = pd.read_csv('master_data_cleaned copy.csv')
df = initiate_df('Player')

# Dropdown 4 columns
exclude = ['Minutes', 'Seconds']  # probably better way to do it
filter_cols = df.select_dtypes(include=['float64']).columns.tolist()
filter_cols = ['Play Time'] + filter_cols

# Hard coded columns to include in each table
GEN_COLS = ['Opponent', 'Play Time', 'Goals', 'Shots', 'Shooting %', 'TF', 'ST', 'BL', 'SP Won', 'SP Attempts',
            'Outcome']
SHOT_COLS = ['Opponent', 'Action Goals', 'Action Shots', 'Center Goals', 'Center Shots', 'Drive Goals',
             'Drive Shots', 'Extra Goals', 'Extra Shots', 'Foul Goals', 'Foul Shots', '6MF Goals', '6MF Shots',
             'PS Goals', 'PS Shots', 'CA Goals', 'CA Shots', 'Outcome']  # 7:23 in df if slicing
EX_COLS = ['Opponent', 'CP EX', 'FP EX', 'DS EX', 'M6 EX',
           'CS EX', 'DE', 'P EX', 'Total EX', 'Outcome']
FULL_COLS = GEN_COLS[:-1] + SHOT_COLS[1:-1] + EX_COLS[1:]

# table style (uniform height)
tbl_style = {'height': '10%'}

# Style to show wins/losses in green/red
outcome_style = [
    {'if':
         {'filter_query': '{Outcome} contains "W"',
          'column_id': 'Outcome'},
     'backgroundColor': 'ForestGreen',
     },
    {'if':
         {'filter_query': '{Outcome} contains "L"',
          'column_id': 'Outcome'},
     'backgroundColor': 'Red',
     }
]

grayNA_style = [
    {
        'if': {
            'filter_query': '{{{}}} is blank'.format(col),
            'column_id': col
        },
        'backgroundColor': '#f2f2f2',
        'color': 'white'
    } for col in df.columns
]

shotP_style = shot_per_style()

# sort the teams in alphabetical order
teams = sorted(df.Team.unique().tolist())

layout = html.Div([
    html.H1(children='Individual Player Data'),
    html.H4(children='Filter by Team and Player to view relevant statistics'),
    html.Br(),

    # Team Dropdown
    html.Div([
        dcc.Dropdown(
            id='filter_dropdown1',
            options=[{'label': t, 'value': t} for t in teams],
            value='USA',
            placeholder='Select a Team'
        ),
    ], style={'width': '25%', 'display': 'inline-block'}),

    # Player Dropdown
    html.Div([  # Player dropdown
        dcc.Dropdown(
            id='filter_dropdown2',
            options=[],
            placeholder='Select a Player'
        ),
    ], style={'width': '25%', 'display': 'inline-block'}),

    # show table checklist
    html.Div([
        html.H3('Select the tables to show: '),
        dcc.Checklist(
            id='checklist',
            options=[
                {'label': 'General Stats', 'value': 'gen'},
                {'label': 'Goal/Shot Stats', 'value': 'shots'},
                {'label': 'Exclusion Stats', 'value': 'ex'},
                {'label': 'Custom Table', 'value': 'cust'}
            ],
            value=['gen']
        ),
    ]),

    # Container with 3 tables
    html.Div(id='graph-container1', children=[

        html.Div(id='show_gen', children=[

            html.H2('General Stats Data'),

            dash_table.DataTable(id='general-table',
                                 sort_action='native',
                                 css=[{'selector': 'table', 'rule': 'table-layout: fixed'},
                                      {'selector': '.dash-spreadsheet th div',
                                       'rule': 'display:flex; flex-wrap:wrap; justify-content:center'}],
                                 style_cell={
                                     'width': '{}%'.format(len(df.columns)),
                                     'whiteSpace': 'normal',
                                     'textAlign': 'center'},
                                 style_table=tbl_style,
                                 style_cell_conditional=[
                                     {'if': {'column_id': 'Opponent'},
                                      'width': '75px'}  # fix width of opponent col
                                 ],
                                 style_data_conditional=outcome_style + shotP_style + grayNA_style
                                 ),
        ]),

        html.Div(id='show_shot', children=[

            html.H3('Goal and Shot Data'),

            dash_table.DataTable(id='shot-table',
                                 sort_action='native',
                                 css=[{'selector': 'table', 'rule': 'table-layout: fixed'}],
                                 style_cell={
                                     'width': '{}%'.format(len(df.columns)),
                                     'whiteSpace': 'normal',
                                     'textAlign': 'center'},
                                 style_table=tbl_style,
                                 style_cell_conditional=[
                                     {'if': {'column_id': 'Opponent'},
                                      'width': '75px'}  # fix width of opponent col
                                 ],
                                 style_data_conditional=outcome_style + [{'if': {
                                     'column_id': SHOT_COLS[1:-1], },
                                     'backgroundColor': 'Beige', }] + grayNA_style
                                 ),
        ]),

        html.Div(id='show_ex', children=[

            html.H3('Exclusion Data'),

            dash_table.DataTable(id='exclusion-table',
                                 sort_action='native',
                                 css=[{'selector': 'table', 'rule': 'table-layout: fixed'}],
                                 style_cell={
                                     'width': '{}%'.format(len(df.columns)),
                                     'whiteSpace': 'normal',
                                     'textAlign': 'center'},
                                 style_table=tbl_style,
                                 style_cell_conditional=[
                                     {'if': {'column_id': 'Opponent'},
                                      'width': '75px'}  # fix width of opponent col
                                 ],
                                 style_data_conditional=outcome_style + [{'if': {
                                     'column_id': EX_COLS[1:-1], },
                                     'backgroundColor': 'DarkSalmon', }] + grayNA_style
                                 ),
        ]),

        html.Div(id='show_cust', children=[

            html.H3('Custom Table:'),

            dash_table.DataTable(id='custom-table',
                                 sort_action='native',

                                 css=[{'selector': 'table', 'rule': 'table-layout: fixed'},
                                      {'selector': '.dash-spreadsheet th div',
                                       'rule': '''
                                                  display: flex; flex-wrap:wrap;
                                                  align-items: center; justify-content: center;                                                  
                                              '''
                                       }],

                                 style_cell={
                                     'width': '{}%'.format(len(df.columns)),
                                     'whiteSpace': 'normal',
                                     'textAlign': 'center'},
                                 style_cell_conditional=[
                                     {'if': {'column_id': 'Opponent'},
                                      'width': '75px'}  # fix width of opponent col
                                 ],
                                 style_data_conditional=outcome_style + [{'if': {
                                     'column_id': EX_COLS[1:-1], },
                                     'backgroundColor': 'DarkSalmon', }] + grayNA_style,

                                 tooltip_header={i: i for i in FULL_COLS},
                                 tooltip_delay=0,
                                 style_header={
                                     'textDecoration': 'underline',
                                     'textDecorationStyle': 'dotted',
                                 },
                                 ),
        ]),

    ]),
    html.Br(),

    # Container with Lower dropdowns
    html.Div(id='graph-container2', children=[
        html.Hr(),
        html.H3('Use the dropdown to select a stat to visualize'),
        html.H4('Stats displayed for each Opponent by Game'),

        # Copy of Player Dropdown
        html.Div([
            dcc.Dropdown(
                id='filter_dropdown3',
                value=None,
                options=[],
                placeholder='Select a Player'
            ),
        ], style={'width': '25%', 'display': 'inline-block'}),

        # Stat Selection Dropdown
        html.Div([
            dcc.Dropdown(
                id='filter_dropdown4',
                value='Goals',
                options=[{'label': c, 'value': c} for c in filter_cols if c not in exclude],
                clearable=False,
                placeholder='Select a Stat'
            ),
        ], style={'width': '25%', 'display': 'inline-block'}),

    ]),

    # Hide graphs if no stat is selected
    html.Div(id='graph-container3', children=[
        dcc.Graph(id='player-graph1',
                  style={'width': '50%', 'display': 'inline-block'}),
        dcc.Graph(id='player-graph2',  # total goals by type
                  style={'width': '50%', 'display': 'inline-block'})
    ])

], style={'width': '100%'})


# filters the player dropdown options by Team
@callback(
    [Output('filter_dropdown2', 'options'),
     Output('filter_dropdown3', 'options')],
    [Input('filter_dropdown1', 'value')]
)
def players_on_team(team):
    dff = df[df.Team == team]
    players = sorted(dff.Name.unique().tolist())
    options = [{'label': p, 'value': p} for p in players]
    return options, options


# Links the 2 player dropdown menus
# use value from drop2 (upper) if that is selected, drop3 if lower is selected
# also resets the values if new team is selected
@callback(
    [Output('filter_dropdown2', 'value'),
     Output('filter_dropdown3', 'value')],
    [Input('filter_dropdown2', 'value'),
     Input('filter_dropdown3', 'value'),
     Input('filter_dropdown1', 'value')]
)
def link_dropdowns(drop2, drop3, team):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    value = drop2 if trigger_id == 'filter_dropdown2' else drop3 if trigger_id == 'filter_dropdown3' else ''
    return value, value


# show tables based on user input (checklist)
@callback(
    [Output('show_gen', 'style'),
     Output('show_shot', 'style'),
     Output('show_ex', 'style'),
     Output('show_cust', 'style')],
    [Input('checklist', 'value')])
def show_tables(checked):
    dic = {'gen': {'display': 'none'},
           'shots': {'display': 'none'},
           'ex': {'display': 'none'},
           'cust': {'display': 'none'}}
    if checked:
        for item in checked:
            dic[item] = {'display': 'block'}
    return list(dic.values())


# Inputs the team and player name, outputs the 4 tables (data and cols)
@callback(
    [Output('general-table', 'data'),
     Output('general-table', 'columns'),
     Output('shot-table', 'data'),
     Output('shot-table', 'columns'),
     Output('exclusion-table', 'data'),
     Output('exclusion-table', 'columns'),
     Output('custom-table', 'data'),
     Output('custom-table', 'columns')
     ],
    [Input('filter_dropdown1', 'value'),
     Input('filter_dropdown2', 'value'),
     Input('filter_dropdown3', 'value')])
def display_table(team, name, name_copy):
    # if name_copy:  # uses the 3rd dropdown if a player name is chosen
    #     player = name_copy
    # else:
    #     player = name

    # Name Filtered DF
    dff = df[(df.Name == name) & (df.Team == team)]

    # DataFrame for Each Table
    df_gen = dff[GEN_COLS]
    df_shots = dff[SHOT_COLS]
    df_ex = dff[EX_COLS]

    # List of data, columns for each df
    table_list = table_data_cols(df_gen, df_shots, df_ex)

    # Full Table
    full = custom_table(dff)

    return tuple(table_list) + tuple(full)


# hides all outputs until team and player are selected
@callback(
    [Output('graph-container1', 'style'),
     Output('graph-container2', 'style'),
     Output('graph-container3', 'style')],
    [Input('filter_dropdown1', 'value'),
     Input('filter_dropdown3', 'value'),
     Input('filter_dropdown4', 'value')])
def hide_containers(team, drop3, stat):
    dic = {'team': {'display': 'none'},
           'name': {'display': 'none'},
           'stat': {'display': 'none'}}
    if team and drop3 and stat:
        dic['team'] = {'display': 'block'}
        dic['name'] = {'display': 'block'}
        dic['stat'] = {'display': 'block'}
    elif team and drop3:
        dic['team'] = {'display': 'block'}
        dic['name'] = {'display': 'block'}

    return list(dic.values())


# should hide graphs if stat is unselected
@callback(
    [Output('player-graph1', 'figure'),
     Output('player-graph2', 'figure')],
    [Input('filter_dropdown1', 'value'),
     Input('filter_dropdown3', 'value'),
     Input('filter_dropdown4', 'value')])
def data_vis(team, player, col):
    dff = filter_team_name(team, player, df)

    # adds "Game" col in form Opponent:Game# (JPN:1) and suppresses copy error
    pd.options.mode.chained_assignment = None
    dff['Game'] = game_col(dff)
    # adds lowercase "play time" as minutes + secs (for graphing)
    dff = playtime_flt(dff)
    pd.options.mode.chained_assignment = 'warn'  # re-adds warning

    bar = px.bar(dff, x='Game',
                 y=col,
                 color='Opponent',
                 title=f'{col} by Game',
                 text=col)
    bar.update_xaxes(categoryorder='array', categoryarray=list(dff.Game))
    bar.update_layout(title={'x': 0.5})

    pie = px.pie(dff, names='Opponent',
                 values=col,
                 color='Opponent',
                 title=f'{col} by Opponent')
    pie.update_layout(title={'x': 0.5})

    return bar, pie
