import dash
from dash import Dash, html, dash_table, Input, Output, dcc, callback
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

dash.register_page(__name__, order=3)


def initiate_df(stat_type):
    df = pd.read_csv('data/master_data_cleaned.csv')
    df = df.rename(columns={'Last': 'Name'}).drop(df.columns[0], axis=1)
    df['Shooting %'] = (df.Goals / df.Shots * 100).round(2)
    cols = list(df)
    cols = cols[:7] + ['Shooting %'] + cols[7:-1]
    df = df.fillna(0)  # set NA vals to 0
    return df[(df.stat_type == stat_type)][cols]


# renames the df cols that are grouped by Opponent (so they are unique for merging)
def rename_against_cols(cols):
    return {name: f"{name.split(' ')[0]} Against" for name in cols}


# groups the df by Team and Opponent using mean or sum (func_name).
# only keeps relevant cols (using hard coded columns)
def groups(df, func_name, cols):
    if func_name == 'Average':
        df1 = df.groupby('Team').mean(numeric_only=True)[cols]
        df2 = df.groupby('Opponent').mean(numeric_only=True)[cols]
    else:
        df1 = df.groupby('Team').sum(numeric_only=True)[cols]
        df2 = df.groupby('Opponent').sum(numeric_only=True)[cols]

    # Use mean shooting percentage in both cases
    if len(cols) == 4:
        per_for = df.groupby('Team').mean(numeric_only=True)['Shooting %']
        per_against = df.groupby('Opponent').mean(numeric_only=True)['Shooting %']
        df1['Shooting %'], df2['Shooting %'] = per_for, per_against
    # Change ex col names in main
    df2 = df2.rename(columns=rename_against_cols(cols))
    return df1, df2


# merges the two dfs on the index
def merge_dfs(df1, df2):
    return pd.concat([df1, df2], axis=1)


# swaps exclusion columns (only used for 1st table)
def swap_ex_cols(df):
    swap = list(df)
    swap = swap[:3] + ['EX Earned'] + swap[4:-1] + ['EX Taken']
    return df[swap]


# puts the shot/goal df in a form that can be used for creating chart
# all goal cols + all shot cols
def reorder_shot_df(df):
    df = df.set_index('Team')
    gcols = df.columns[df.columns.str.contains('Goals')]
    scols = df.columns[~df.columns.str.contains('Goals')]
    return df[list(gcols) + list(scols)].reset_index()


# Create Standings dic and add to df, returns table ready df
def add_standings(df):
    # final standings list could be initialized globally
    final_standings = ['SRB', 'GRE', 'HUN', 'ESP', 'CRO', 'USA', 'ITA', 'MNE', 'AUS', 'JPN', 'KAZ', 'RSA']
    dic = {final_standings[i]: i + 1 for i in range(0, len(final_standings))}
    df['Standings'] = df.index.map(dic)

    # sort by standings
    df.sort_values('Standings', ascending=True, inplace=True)

    # returns df with 'Team' in columns
    return df.rename_axis('Team').reset_index().round(2)


# label = Average or Total
# data_label = list
def bar_plot(teams, col1, col2, func_name, data_label):
    """
  This function creates the two bar plot traces.

  It creates the traces usings the two columns and adds
  a title based on the data used.

  Parameters:
    teams (list): list of team names from index of df
    col1 (Series): col for 1st trace
    col2 (Series): col for 2nd trace
    func_name (string): Type of calculation done on data (mean or sum)
    data_label (list): Type of grouping done (For or Against, or Goal and Shot)
    v_space (float): Vertical Spacing between subplots (.15 or less)

  Returns:
    go.Figure: complete figure with all traces added
    string: Title to be used (ex: {Average} {Goals} {For} and {Against})

  """
    t1 = data_label[0]
    t2 = data_label[1]

    # changes title if using Goal and Shot grouping
    if t1 == 'For':
        title = f'{func_name} {col1.name}: {t1} and {t2}'
        name1 = f'{col1.name} {t1}' # For
        name2 = f'{col1.name} {t2}' # Against
    elif t1 == 'Against':
        title = f'{func_name} {col2.name} Earned and Taken'
        name1 = f'{col2.name} {t1}' # Against
        name2 = f'{col2.name} {t2}' # By
    else:
        title = f'{func_name} {col1.name} and {t2}'  # Most are For and Against
        name1 = f'{col1.name} {t1}'
        name2 = f'{col1.name} {t2}'

    trace1 = go.Bar(name=name1, x=teams, y=col1)
    trace2 = go.Bar(name=name2, x=teams, y=col2)
    data = [trace1, trace2]

    return go.Figure(data=data), title


# Adds all traces to subplots and returns completed figure
# takes in the table_df, 'Average or Total', grid, and vertical spacing
def create_chart(df, func_name, data_label, grid, v_space):
    df = df.set_index('Team')[list(df)[1:-1]]
    # df = df[df.columns[0:-1]]

    x = len(grid)  # range to be used

    temp_titles = list('abcdefghijklmnopqrstuvwxyz')[:x]
    title_dic = dict.fromkeys(temp_titles)
    fig = make_subplots(rows=int(x / 2), cols=2,
                        subplot_titles=tuple(temp_titles),
                        vertical_spacing=v_space,
                        horizontal_spacing=0.05)
    for i in range(x):
        r, c, cols = grid[i][0], grid[i][1], list(df)  # row,col, and column list variables

        c1, c2 = df[cols[i]], df[cols[x + i]]  # two columns to use

        bar, title = bar_plot(df.index, c1, c2, func_name, data_label)

        fig.add_trace(bar.data[0], row=r, col=c)
        fig.add_trace(bar.data[1], row=r, col=c)

        title_dic[list(title_dic.keys())[i]] = title  # dic of titles to add to subplots

    fig.for_each_annotation(lambda a: a.update(text=title_dic[a.text]))  # adds each subplot title
    fig.update_layout(height=800)
    return fig


def create_legend(df, legend):
    table_legend = {col: legend.get(col) for col in list(df) if legend.get(col)}
    leg_df = pd.DataFrame(table_legend, columns=table_legend.keys(), index=['Column Meaning'])
    leg_df = leg_df.rename_axis('Column Name').reset_index()
    dash_table.DataTable(id='legend',
                         columns=[{'id': c, 'name': c} for c in leg_df.columns])
    return leg_df.to_dict('records')


def table_style_data(df, stat):
    if stat != 'shots':
        num = int(len(list(df)[1:-1]) / 2)
        cols1, cols2 = list(df)[1:num + 1], list(df)[num + 2:-1]
        style = [
            {'if':
                 {'column_id': cols1},
             'backgroundColor': '#70bf62',
             },
            {'if':
                 {'column_id': cols2},
             'backgroundColor': '#FFB5C5',
             },
            {'if':
                 {'column_id': ['Team', 'Standings']},
             'backgroundColor': '#FFF5EE',
             }
        ]
        return style
    else:
        return [
            {'if':
                 {'column_id': ['Team', 'Standings']},
             'backgroundColor': '#FFF5EE',
             },
            {'if':
                 {'column_id': SHOT_COLS},
             'backgroundColor': 'Beige',
             }
        ]


# Reads in dictionary saved in text file
with open('data/legend.txt') as f:
    legend_dic = eval(f.read())

# Hard Code Columns for each of the 4 tables
GEN_COLS = ['Goals', 'Shots', 'Shooting %', 'Total EX']
GOAL_COLS = ['Action Goals', 'Center Goals', 'Drive Goals',  # For and against
             'Extra Goals', 'Foul Goals', '6MF Goals',
             'PS Goals', 'CA Goals']
SHOT_COLS = ['Action Shots', 'Center Shots', 'Drive Shots', 'Extra Shots',
             'Foul Shots', '6MF Shots', 'PS Shots', 'CA Shots']
EX_COLS = ['CP EX', 'FP EX', 'DS EX', 'M6 EX',
           'CS EX', 'DE', 'P EX', 'Total EX']

# includes goal and shot columns
SHOT_COLS_FULL = ['Action Goals', 'Action Shots', 'Center Goals',
                  'Center Shots', 'Drive Goals', 'Drive Shots', 'Extra Goals',
                  'Extra Shots', 'Foul Goals', 'Foul Shots', '6MF Goals', '6MF Shots',
                  'PS Goals', 'PS Shots', 'CA Goals', 'CA Shots']

# dictionary for ease of use
STAT_DIC = {'gen': {'cols': ['Goals', 'Shots', 'Shooting %', 'Total EX'], 'data_label': ['For', 'Against']}, 'goals': {
    'cols': ['Action Goals', 'Center Goals', 'Drive Goals', 'Extra Goals', 'Foul Goals', '6MF Goals', 'PS Goals',
             'CA Goals'], 'data_label': ['For', 'Against']},
            'ex': {'cols': ['CP EX', 'FP EX', 'DS EX', 'M6 EX', 'CS EX', 'DE', 'P EX', 'Total EX'],
                   'data_label': ['Against', 'By']}, 'shots': {
        'cols': ['Action Goals', 'Action Shots', 'Center Goals', 'Center Shots', 'Drive Goals', 'Drive Shots',
                 'Extra Goals', 'Extra Shots', 'Foul Goals', 'Foul Shots', '6MF Goals', '6MF Shots', 'PS Goals',
                 'PS Shots', 'CA Goals', 'CA Shots'], 'data_label': ['Goals', 'Shots']}}

GRID1 = [(1, 1), (1, 2), (2, 1), (2, 2)]
GRID2 = [(1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2), (4, 1), (4, 2)]

df = initiate_df('Team')

# General Style for each table
table_style = [
    {'if':
         {'column_id': ['Team', 'Standings']},
     'backgroundColor': '#FFF5EE',
     },
    {'if':
         {'header_index': SHOT_COLS},
     'backgroundColor': 'Beige',
     }
]

layout = html.Div([
    html.Div([
        dcc.Dropdown(
            id='filter_dropdown',
            options=[
                {'label': 'General Stats For and Against', 'value': 'gen'},
                {'label': 'Goal and Shot Specific', 'value': 'shots'},
                {'label': 'Goal Specific For and Against', 'value': 'goals'},
                {'label': 'Exclusion Specific For and Against', 'value': 'ex'}],
            value='gen',
        ),
    ], style={'width': '25%', 'display': 'inline-block'}),

    html.Div([
        dcc.Dropdown(
            id='function_dropdown',
            options=[
                {'label': 'Per Games Stats', 'value': 'Average'},
                {'label': 'Totals', 'value': 'Total'}],
            value='Average',
        ),
    ], style={'width': '25%', 'display': 'inline-block'}),

    dcc.Checklist(id='show_legend',
                  options=[
                      {'label': 'Show Legend', 'value': 'True'}
                  ],
                  ),

    html.Div(id='output_container1', children=[

        html.H2(id='title1', children='Team Stats For and Against Each Team (sorted by final standing)'),

        html.H3('Data in table form'),

        # Data Table
        dash_table.DataTable(id='teams_table',
                             sort_action='native',
                             css=[{'selector': 'table', 'rule': 'overflow-x: auto'}],
                             style_cell={
                                 'whiteSpace': 'normal',
                                 'textAlign': 'center'},
                             style_table={'height': '40%'},
                             style_header={'border': '1px solid purple',
                                           'textDecoration': 'underline',
                                           'textDecorationStyle': 'dotted',
                                           },
                             style_data={'backgroundColor': '#FFB5C5'},
                             style_cell_conditional=[
                                 {'if': {'column_id': 'Team'},
                                  'width': '75px'}],
                             tooltip_header=legend_dic,
                             tooltip_delay=0,
                             ),

        html.Br(),

        # Legend Table (don't need this)
        dash_table.DataTable(id='legend',
                             css=[{'selector': 'table', 'rule': 'table-layout: fixed'},
                                  {'selector': '.dash-spreadsheet th div',
                                   'rule': '''
                                              height:50px; 
                                              display: flex; flex-wrap:wrap;
                                              align-items: center; justify-content: center;                                                  
                                          '''
                                   }
                                  ],
                             style_cell={
                                 # 'width': '{}%'.format(len(df.columns)),
                                 'whiteSpace': 'normal',
                                 'textAlign': 'center'},
                             style_table={'height': '50%', 'width': '100%'},
                             style_header={'textAlign': 'center'},
                             style_cell_conditional=[
                                 {'if': {'column_id': 'Column Name'},
                                  'backgroundColor': 'gray',
                                  'color': 'black'}  # fix width of index col
                             ]
                             ),

        html.Br(),
        html.H3('Bar Plots displaying the table data'),
        html.Div([dcc.Graph(id='bar_chart1')], style={'width': '100%'}),

    ])
])


@callback(
    [Output('title1', 'children'),
     Output('bar_chart1', 'figure'),
     Output('teams_table', 'data'),
     Output('teams_table', 'columns'),
     Output('teams_table', 'style_data_conditional'),
     Output('legend', 'data')],
    [Input('filter_dropdown', 'value'),
     Input('function_dropdown', 'value'),
     Input('show_legend', 'value')]
)
def show_table(table_type, func_name, legend):  # takes in a string of gen, shots, goals, ex and either mean or sum
    dff = df
    table_df, fig_df, grid, v_space = None, None, GRID2, .1
    # titles = ['Team Stats For and Against Each Team','Goals and Shots for each Team']
    title = 'Team Stats For and Against Each Team'
    table_legend = []  # don't need

    filter_cols = STAT_DIC.get(table_type)['cols']
    df1, df2 = groups(dff, func_name, filter_cols)

    if table_type == 'shots':
        # For shot cols, only use df1 (for one team) for now
        table_df = add_standings(df1)
        fig_df = reorder_shot_df(table_df)  # for graphing only
        title = 'Goals and Shots for each Team'
    elif table_type == 'gen':
        df1 = df1.rename(columns={'Total EX': 'EX Taken'})
        df2 = df2.rename(columns={'Total Against': 'EX Earned'})
        table_df = add_standings(swap_ex_cols(merge_dfs(df1, df2)))  # merge, swap, add standings
        grid, v_space = GRID1, .15
    elif table_type == 'goals':
        table_df = add_standings(merge_dfs(df1, df2))
    else:
        table_df = add_standings(merge_dfs(df2, df1))
        table_df = table_df.rename(columns={'Total EX': 'EX Taken', 'Total Against': 'EX Earned'})

    # data conditional style, and label
    style = table_style_data(table_df, table_type)
    data_label = STAT_DIC.get(table_type)['data_label']
    fig_df = fig_df if table_type == 'shots' else table_df

    # Graph
    fig = create_chart(fig_df, func_name, data_label=data_label, grid=grid, v_space=v_space)

    # table data and columns
    data = table_df.to_dict('records')
    columns = [{'id': c, 'name': c, 'hideable': True} for c in table_df.columns]

    return title, fig, data, columns, style, table_legend  # don't need
