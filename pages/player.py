import dash
from dash import html, dash_table, Input, Output, dcc, callback
from dash.exceptions import PreventUpdate
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


# BEFORE CALLBACKS: Creates a df with totals for each player
def create_total_df(df):
    cols = df.select_dtypes('float').columns.to_list()
    d = {col: 'sum' if col != 'Shooting %' else 'mean' for col in cols}
    d.update({'Outcome': 'count', 'Opponent': 'count'})
    df = df.fillna(0)  # may not need this
    total_df = df.groupby(['Team', 'Name']).agg(d)
    total_df = playtime_flt(total_df)[FULL_COLS]
    total_df['Opponent'] = 'TOTALS'
    return total_df.reset_index()


# add total row to table data frame (don't need blank row)
# returns full df and list of strings to filter as null
def create_full_df(table_df, total_row):
    wins, losses = (table_df.Outcome.values == 'W').sum(), (table_df.Outcome.values == 'L').sum()
    total_row = total_row.astype(str)
    total_row.Outcome = f'{wins}-{losses}'
    full_df = pd.concat([table_df, total_row])
    return full_df, full_df.iloc[-1].values


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


# function to set data conditional, css, and cell style
def unique_style(id):
    totals_style = [
        {'if':
             {'filter_query': '{Opponent} contains "TOTALS"'},
         'backgroundColor': '#82888a',
         'color': 'white',
         'borderTop': '2px solid black'
         },
    ]
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
    cell_style = {
        'width': '{}%'.format(len(FULL_COLS)),
        'whiteSpace': 'normal',
        'textAlign': 'center'}
    css_style = [{'selector': 'table', 'rule': 'table-layout: fixed'}]  # base style
    if id == 'general-table':
        style = shot_per_style()
        css_style = css_style + [{'selector': '.dash-spreadsheet th div',
                                  'rule': 'display:flex; flex-wrap:wrap; justify-content:center'}]
    elif id == 'shot-table':
        style = [{'if': {
            'column_id': SHOT_COLS[1:-1], },
            'backgroundColor': 'Beige'}]
    elif id == 'exclusion-table':
        style = [{'if': {
            'column_id': EX_COLS[1:-1], },
            'backgroundColor': 'DarkSalmon'}]
    else:
        cell_style.update({'minWidth': '50px', 'width': '100px', 'maxWidth': '180px'})
        style = [{'if': {
            'column_id': EX_COLS[1:-1], },
            'backgroundColor': 'DarkSalmon'}]
        css_style = [{'selector': 'table', 'rule': 'table-layout: scroll'},
                     {'selector': '.dash-spreadsheet th div',
                      'rule': 'display:flex; flex-wrap:wrap; justify-content:center'}]

    style = outcome_style + style + grayNA_style + totals_style
    return style, css_style, cell_style


# takes in the column filtered df, section title, table id, and list of null strings
def create_table(table_df, title, id, null_list):
    data = [html.H2(title)]

    style, css_style, cell_style = unique_style(id)

    table = dash_table.DataTable(id=id,
                                 data=table_df.to_dict('records'),
                                 columns=[{'id': c, 'name': c, 'hideable': True} for c in table_df.columns],
                                 sort_action='native',
                                 sort_as_null=null_list,
                                 css=css_style,
                                 style_cell=cell_style,
                                 style_table={'height': '10%', 'overflowX': 'auto'},
                                 style_cell_conditional=[
                                     {'if': {'column_id': 'Opponent'},
                                      'width': '115px'}  # fix width of opponent col
                                 ],
                                 style_data_conditional=style,
                                 tooltip_header={i: i for i in FULL_COLS},
                                 tooltip_delay=0,
                                 style_header={
                                     'textDecoration': 'underline',
                                     'textDecorationStyle': 'dotted',
                                 },
                                 )
    data.append(table)
    return data


# Hard coded columns to include in each table
GEN_COLS = ['Opponent', 'Play Time', 'Goals', 'Shots', 'Shooting %', 'Total EX', 'TF', 'ST', 'BL', 'SP Won',
            'SP Attempts',
            'Outcome']
SHOT_COLS = ['Opponent', 'Action Goals', 'Action Shots', 'Center Goals', 'Center Shots', 'Drive Goals',
             'Drive Shots', 'Extra Goals', 'Extra Shots', 'Foul Goals', 'Foul Shots', '6MF Goals', '6MF Shots',
             'PS Goals', 'PS Shots', 'CA Goals', 'CA Shots', 'Outcome']  # 7:23 in df if slicing
EX_COLS = ['Opponent', 'CP EX', 'FP EX', 'DS EX', 'M6 EX',
           'CS EX', 'DE', 'P EX', 'Total EX', 'Outcome']
# FULL_COLS = GEN_COLS[:-1] + SHOT_COLS[1:-1] + EX_COLS[1:]
# Hard full cols so no duplicates
FULL_COLS = ['Opponent', 'Play Time', 'Goals', 'Shots', 'Shooting %', 'TF', 'ST', 'BL', 'SP Won',
             'SP Attempts', 'Action Goals', 'Action Shots', 'Center Goals', 'Center Shots', 'Drive Goals',
             'Drive Shots', 'Extra Goals', 'Extra Shots', 'Foul Goals', 'Foul Shots', '6MF Goals', '6MF Shots',
             'PS Goals', 'PS Shots', 'CA Goals', 'CA Shots', 'CP EX', 'FP EX', 'DS EX', 'M6 EX', 'CS EX', 'DE', 'P EX',
             'Total EX', 'Outcome']

# df = pd.read_csv('master_data_cleaned copy.csv')
df = initiate_df('Player')
total_df = create_total_df(df)

# Dropdown 4 columns
exclude = ['Minutes', 'Seconds']  # probably better way to do it
filter_cols = df.select_dtypes(include=['float64']).columns.tolist()
filter_cols = ['Play Time'] + filter_cols

# Hard coded tooltip header/legend
# Reads in dictionary saved in text file
with open('data/legend.txt') as f:
    legend_dic = eval(f.read())

# sort the teams in alphabetical order
teams = sorted(df.Team.unique().tolist())

layout = html.Div([
    dcc.Store(id='filtered-mem'),
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

    # Container with 4 tables
    html.Div(id='graph-container1', children=[

        html.Div(id='show_gen'),
        html.Div(id='show_shot'),
        html.Div(id='show_ex'),
        html.Div(id='show_cust'),

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
def show_tables(checked):  # shows divs with tables in them if checked
    dic = {'gen': {'display': 'none'},
           'shots': {'display': 'none'},
           'ex': {'display': 'none'},
           'cust': {'display': 'none'}}
    if checked:
        for item in checked:
            dic[item] = {'display': 'block'}
    return list(dic.values())


# stores filtered dfs as JSON for speed
@callback(
    [Output('filtered-mem', 'data')],
    [Input('filter_dropdown1', 'value'),
     Input('filter_dropdown2', 'value')],
)
def filter_df(team, name):
    if name is None:
        raise PreventUpdate
    else:
        dff = df[(df.Name == name) & (df.Team == team)]
        total_row = total_df[(total_df.Name == name) & (total_df.Team == team)]
        temp_dic = {'dff': dff.to_dict('records'), 'trow': total_row.to_dict('records')}
        return [temp_dic]


@callback(
    [Output('show_gen', 'children'),
     Output('show_shot', 'children'),
     Output('show_ex', 'children'),
     Output('show_cust', 'children')],
    [Input('filtered-mem', 'data'),
     Input('filter_dropdown2', 'value')]
)
def display_table(dic, name):
    # Name and Team Filtered DF
    # dff = filter_team_name(team, name, df)
    # total_row = filter_team_name(team, name, total_df)

    # DF with totals row added, values of that row as strings
    # strings = ['']
    # if name:
    #     dff, strings = create_full_df(dff, total_row)

    # DataFrame for Each Table
    # df_gen = dff[GEN_COLS]
    # df_shots = dff[SHOT_COLS]
    # df_ex = dff[EX_COLS]
    if not name:
        raise PreventUpdate

    dff = pd.DataFrame(dic['dff'])
    total_row = pd.DataFrame(dic['trow'])

    dff, strings = create_full_df(dff, total_row)

    gen_table = create_table(dff[GEN_COLS], 'General Table', 'general-table', strings)
    shots_table = create_table(dff[SHOT_COLS], 'Shots Table', 'shot-table', strings)
    ex_table = create_table(dff[EX_COLS], 'EX Table', 'exclusion-table', strings)
    full_table = create_table(dff[FULL_COLS], 'Custom Table', 'custom-table', strings)

    return gen_table, shots_table, ex_table, full_table


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
    # pd.options.mode.chained_assignment = 'warn'  # re-adds warning

    bar = px.bar(dff, x='Game', y=col, color='Opponent',
                 title=f'{col} by Game', text=col)
    bar.update_xaxes(categoryorder='array', categoryarray=list(dff.Game))
    bar.update_layout(title={'x': 0.5})

    pie = px.pie(dff, names='Opponent', values=col, color='Opponent',
                 title=f'{col} by Opponent')
    pie.update_layout(title={'x': 0.5})

    return bar, pie
