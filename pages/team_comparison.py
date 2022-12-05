import dash
from dash import html, dash_table, Input, Output, dcc, callback
from dash.exceptions import PreventUpdate
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash_bootstrap_components as dbc

### https://stackoverflow.com/questions/70020608/plotly-dash-datatable-how-create-multi-headers-table-from-pandas-multi-headers
# function that converts multi header df to dash format:


dash.register_page(__name__, order=3)


def initiate_df(stat_type):
    df = pd.read_csv('data/master_data_cleaned.csv')
    df = df.rename(columns={'Last': 'Name'}).drop(df.columns[0], axis=1)
    df['Shooting %'] = (df.Goals / df.Shots * 100).round(2)
    cols = list(df)
    cols = cols[:7] + ['Shooting %'] + cols[7:-1]
    df = df.fillna(0)  # set NA vals to 0
    return df[(df.stat_type == stat_type)][cols]


# returns dic of av or tots dfs
def mean_sum_dfs(df):
    av_team = df.groupby('Team').mean(numeric_only=True)
    av_opp = df.groupby('Opponent').mean(numeric_only=True)

    tot_team = df.groupby('Team').sum(numeric_only=True)
    tot_opp = df.groupby('Opponent').sum(numeric_only=True)

    tot_team['Shooting %'] = av_team['Shooting %']
    tot_opp['Shooting %'] = av_opp['Shooting %']

    dic = {
        'Average': {'df1': av_team, 'df2': av_opp},
        'Total': {'df1': tot_team, 'df2': tot_opp},
    }

    return dic


# Creating Multi Level DF from For and Against DFs (Team and Opp)
# also joins them at the end
def make_multiheader_table(df1, df2):
    df2.index.name == 'Team'

    header1 = pd.MultiIndex.from_product([['For'], list(df1)])
    header2 = pd.MultiIndex.from_product([['Against'], list(df2)])

    df_for = pd.DataFrame(df1.values, index=df1.index, columns=header1)
    df_ag = pd.DataFrame(df2.values, index=df2.index, columns=header2)

    return df_for.join(df_ag)


# takes in multiheader df
def add_standings(df):
    # final standings list could be initialized globally
    final_standings = ['SRB', 'GRE', 'HUN', 'ESP', 'CRO', 'USA', 'ITA', 'MNE', 'AUS', 'JPN', 'KAZ', 'RSA']
    pd.options.mode.chained_assignment = None
    dic = {final_standings[i]: i + 1 for i in range(0, len(final_standings))}
    df['', 'Standings'] = df.index.map(dic)  # adds

    # sort by standings
    df.sort_values(('', 'Standings'), ascending=True, inplace=True)

    return df.round(2)


# stack overflow function
# create multi pandas df first, then use this function to graph (feed into cols and data of table creation functions)
def convert_df_to_dash(df):
    """
    Converts a pandas data frame to a format accepted by dash
    Returns columns and data in the format dash requires
    """
    # adds team to columns for table output
    df['', 'Team'] = df.index
    col_order = list(df)
    col_order = [('', 'Team')] + col_order[:-1]
    df = df[col_order]

    # create ids for multi indexes (single index stays unchanged)
    # [('', 'A'), ('B', 'C'), ('D', 'E')] -> ['A', 'B_C', 'D_E']
    ids = ["".join([col for col in multi_col if col]) for multi_col in list(df.columns)]

    # build columns list from ids and columns of the dataframe
    cols = [{"name": list(col), "id": id_, 'hideable': True} for col, id_ in zip(list(df.columns), ids)]

    # build data list from ids and rows of the dataframe
    data = [{k: v for k, v in zip(ids, row)} for row in df.values]

    return cols, data


def data_style(cols):
    ids = [col['id'] for col in cols]
    index = int(len(ids) / 2)
    f_ids, ag_ids = ids[0:index], ids[index:]
    style = [
        {'if': {'column_id': f_ids}, 'backgroundColor': '#FFFAFA'}, # maybe grays would look better
        {'if': {'column_id': ag_ids}, 'backgroundColor': '#F5F5F5'}
    ]
    return style


# Creates the dash table
def team_comp_table(table_df, data_style, id, columns, data):
    table = dash_table.DataTable(id=id,
                                 data=data,
                                 columns=columns,
                                 sort_action='native',
                                 css=[{'selector': 'table', 'rule': 'overflow-x: auto'}],
                                 style_cell={
                                     'whiteSpace': 'normal',
                                     'textAlign': 'center'},
                                 style_table={'height': '40%'},
                                 style_header={'backgroundColor': 'rgb(220, 220, 220)',
                                               'color': 'black',
                                               'fontWeight': '500',
                                               'border': '1.5px solid black',
                                               'textDecoration': 'underline',
                                               'textDecorationStyle': 'dotted',
                                               },
                                 style_header_conditional=[
                                     {'if': {'header_index': 0}, 'backgroundColor': 'rgb(240, 240, 240)'},
                                     {'if': {'column_id': ['Team', 'Standings'], 'header_index': 0},
                                      'backgroundColor': 'white'},
                                 ],
                                 style_data=None,
                                 style_data_conditional=data_style,
                                 style_cell_conditional=[
                                     {'if': {'column_id': ['Team', 'Standings']},
                                      'width': '100px'}],
                                 tooltip_header=legend_dic,
                                 tooltip_delay=0,
                                 merge_duplicate_headers=True,
                                 )
    return table


def bar_plot(teams, col1, col2, func_name):
    name = col1.name  # name of column used
    title = f'{func_name} {name}: For and Against'  # subplot title

    trace1 = go.Bar(name=f'{name} For', x=teams, y=col1)
    trace2 = go.Bar(name=f'{name} Against', x=teams, y=col2)
    data = [trace1, trace2]

    return go.Figure(data=data), title


# Adds all traces to subplots and returns completed figure
# takes in the table_df, 'Average or Total', grid, and vertical spacing
def create_chart(df1, df2, func_name, grid, v_space, sort):
    x = len(grid)  # range to be used

    temp_titles = list('abcdefghijklmnopqrstuvwxyz')[:x]
    title_dic = dict.fromkeys(temp_titles)
    fig = make_subplots(rows=int(x / 2), cols=2,
                        subplot_titles=tuple(temp_titles),
                        vertical_spacing=v_space,
                        horizontal_spacing=0.05)
    for i in range(x):
        r, c = grid[i][0], grid[i][1]

        # cols to be used for chart
        col1, col2 = df1[df1.columns[i]], df2[df2.columns[i]]

        bar, title = bar_plot(sort, col1, col2, func_name)

        fig.add_trace(bar.data[0], row=r, col=c)
        fig.add_trace(bar.data[1], row=r, col=c)
        # fig.update_layout(xaxis={'categoryorder': 'array', 'categoryarray': sort})

        title_dic[list(title_dic.keys())[i]] = title

    fig.for_each_annotation(lambda a: a.update(text=title_dic[a.text]))
    # fig.update_layout(height=800, width=1400, legend_tracegroupgap=180)
    return fig


# Reads in dictionary saved in text file
with open('data/legend.txt') as f:
    legend_dic = eval(f.read())

# Hard Code Columns for each of the 4 tables
GEN_COLS = ['Goals', 'Shots', 'Shooting %', 'Total EX']
GOAL_COLS = ['Action Goals', 'Center Goals', 'Drive Goals',  # For and against
             'Extra Goals', 'Foul Goals', '6MF Goals',
             'PS Goals', 'CA Goals']
SHOT_COLS = ['Action Shots', 'Center Shots', 'Drive Shots',  # For and against
             'Extra Shots', 'Foul Shots', '6MF Shots',
             'PS Shots', 'CA Shots']
EX_COLS = ['CP EX', 'FP EX', 'DS EX', 'M6 EX',
           'CS EX', 'DE', 'P EX', 'Total EX']

# includes goal and shot columns
SHOT_COLS_FULL = ['Action Goals', 'Action Shots', 'Center Goals',
                  'Center Shots', 'Drive Goals', 'Drive Shots', 'Extra Goals',
                  'Extra Shots', 'Foul Goals', 'Foul Shots', '6MF Goals', '6MF Shots',
                  'PS Goals', 'PS Shots', 'CA Goals', 'CA Shots']

GRID1 = [(1, 1), (1, 2), (2, 1), (2, 2)]
GRID2 = [(1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2), (4, 1), (4, 2)]

STAT_DIC = {
    'gen': {'cols': GEN_COLS, 'grid': GRID1, 'vspace': .15, 'height': 800},
    'goals': {'cols': GOAL_COLS, 'grid': GRID2, 'vspace': .1, 'height': 1200},
    'shots': {'cols': SHOT_COLS, 'grid': GRID2, 'vspace': .1, 'height': 1200},
    'ex': {'cols': EX_COLS, 'grid': GRID2, 'vspace': .1, 'height': 1200}
}

df = initiate_df('Team')
df_dic = mean_sum_dfs(df)

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

layout = dbc.Container([
    dcc.Store(id='table-mem'),
    html.Div([
        html.Label('Stats to Display'),
        dcc.Dropdown(
            id='filter_dropdown',
            options=[
                {'label': 'General Stats', 'value': 'gen'},
                {'label': 'Goal Stats', 'value': 'goals'},
                {'label': 'Shooting Stats', 'value': 'shots'},
                {'label': 'Exclusion Stats', 'value': 'ex'}],
            value='gen',
        ),
    ], style={'width': '15%', 'display': 'inline-block', 'margin-right': '15px'}),

    html.Div([
        html.Label('Per Game or Total Stats'),
        dcc.Dropdown(
            id='function_dropdown',
            options=[
                {'label': 'Per Games Stats', 'value': 'Average'},
                {'label': 'Totals', 'value': 'Total'}],
            value='Average',
        ),
    ], style={'width': '15%', 'display': 'inline-block'}),

    dcc.Checklist(id='show_legend',
                  options=[
                      {'label': 'Show Legend', 'value': 'True'}
                  ],
                  ),

    html.Div(id='output_container1', children=[

        html.H2(id='title1', children='Team Stats For and Against Each Team (sorted by final standing)'),

        html.H3('Data in table form'),

        html.Div(id='table-div'),

        html.Br(),
        html.Br(),
        html.Hr(),

        html.H3('Bar Plots displaying the table data'),
        html.Div([dcc.Graph(id='bar_chart1')], style={'width': '100%'}),

    ], style={'margin-top': '25px'})
], class_name='container-fluid')


@callback(
    [Output('title1', 'children'),
     Output('table-mem', 'data'),
     Output('table-div', 'children')],
    [Input('filter_dropdown', 'value'),
     Input('function_dropdown', 'value'),
     Input('show_legend', 'value')]
)
def show_table(table_type, func_name, legend):
    if table_type is None or func_name is None:
        raise PreventUpdate

    # table specifications from dictionary
    t_dic = STAT_DIC.get(table_type)

    filter_cols = t_dic['cols']
    df1 = df_dic[func_name]['df1'][filter_cols]
    df2 = df_dic[func_name]['df2'][filter_cols]

    # creates multiheader table
    table_df = make_multiheader_table(df1, df2)
    table_df = add_standings(table_df)  # adds standings to table

    # converts pandas table to dash table form
    table_cols, table_data = convert_df_to_dash(table_df)

    # creates the dash table
    style = data_style(table_cols[1:-1])
    table = team_comp_table(table_df, style, f'{table_type}-{func_name}', table_cols, table_data)

    # stores for and against tables to use in graphing function
    df1 = df1.reset_index()
    store_dic = dict(df1=df1.to_dict('records'), df2=df2.to_dict('records'),
                     func=func_name, grid=t_dic['grid'], v_space=t_dic['vspace'],
                     sort=table_df.index, table_type=table_type, height=t_dic['height'])
    title = 'Team Stats For and Against Each Team'

    return title, [store_dic], table


## Need to rename Total EX cols ###
@callback(
    [Output('bar_chart1', 'figure')],
    [Input('table-mem', 'data')]
)
def create_figures(dic):  # df1, df2, func, grid, v_space, sort_array, height
    dic = dic[0]
    dList = list(dic.values())
    df1 = pd.DataFrame(dList[0])
    df2 = pd.DataFrame(dList[1])

    # rename Total EX cols so it looks better
    if dic['table_type'] == 'gen' or 'ex':
        df1.rename(columns={'Total EX': 'EX'}, inplace=True)
        df2.rename(columns={'Total EX': 'EX'}, inplace=True)

    # uses df1 index (Team) to make charts
    df1 = df1.set_index('Team')
    fig = create_chart(df1, df2, func_name=dList[2],
                       grid=dList[3], v_space=dList[4], sort=dList[5])
    fig.update_layout(height=dic['height'], legend_tracegroupgap=180)

    return [fig]
