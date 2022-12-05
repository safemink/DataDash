import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
from dash import dash_table, callback, dcc, html
from dash.dependencies import Output, Input

dash.register_page(__name__, order=4)

# Import Data
df = pd.read_csv('data/master_data_refs.csv')
df = df.fillna(0)
df.drop(df.columns[0], axis=1, inplace=True)

refs = pd.concat([df['Ref 1'], df['Ref 2']])
reflist = sorted(refs.unique().tolist())

# Lists
TOP_8 = ['ITA', 'GRE', 'SRB', 'USA', 'HUN', 'ESP', 'CRO', 'MNE']
GEN_COLS = ['Goals', 'Total EX', 'TF']
KNOCK_GAMES = [31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42]

# Styles
tableheaderstyle = {'backgroundColor': 'rgb(220, 220, 220)',
                    'color': 'black',
                    'fontWeight': 'bold'}
tablestyle = {'whiteSpace': 'normal',
             'textAlign': 'center',
              'font_size': '14px',
              }
pietitlestyle = {
    'y': 0.95,  # new
    'x': 0.5,
    'xanchor': 'center',
    'yanchor': 'top'  # new
}
bartitlestyle = {
    'y': 0.95,  # new
    'x': 0.5,
    'xanchor': 'center',
    'yanchor': 'top'  # new
}

def mergedf(data):
    data = data.drop(data[data.stat_type != "Team"].index)
    data = data.drop(['Last', 'First', 'Play Time', 'Minutes', 'Seconds', 'SP Won', 'SP Attempts', 'stat_type'], axis=1)

    ref1df = data.drop(['Ref 2'], axis=1).rename(columns={'Ref 1': 'Ref'})
    ref2df = data.drop(['Ref 1'], axis=1).rename(columns={'Ref 2': 'Ref'})

    master = pd.concat([ref1df, ref2df])

    return master
def winlossdf(master):
    winners = master.drop(master[master.Outcome != "W"].index)
    losers = master.drop(master[master.Outcome != "L"].index)

    winAvg = winners.groupby('Ref').mean(numeric_only=True)[GEN_COLS].rename(
        columns={'Goals': 'Winner Goals pg', 'Total EX': 'Winner EX pg', 'TF': 'Winner TF pg'})
    loserAvg = losers.groupby('Ref').mean(numeric_only=True)[GEN_COLS].rename(
        columns={'Goals': 'Loser Goals pg', 'Total EX': 'Loser EX pg', 'TF': 'Loser TF pg'})

    genTable = pd.concat([winAvg, loserAvg], axis=1)
    genTable['Total Goals pg'] = winAvg['Winner Goals pg'] + loserAvg['Loser Goals pg']
    genTable['Total EX pg'] = winAvg['Winner EX pg'] + loserAvg['Loser EX pg']
    genTable['Total TF pg'] = winAvg['Winner TF pg'] + loserAvg['Loser TF pg']
    genTable['Avg Goal Diff'] = winAvg['Winner Goals pg'] - loserAvg['Loser Goals pg']
    genTable = genTable.reset_index()
    genTable = genTable[
        ['Ref', 'Total Goals pg', 'Winner Goals pg', 'Loser Goals pg', 'Avg Goal Diff', 'Total EX pg', 'Winner EX pg',
         'Loser EX pg', 'Total TF pg', 'Winner TF pg', 'Loser TF pg']]

    return genTable.round(2)
def refavgdf(master):
    masterAvg = master.groupby('Ref').mean(numeric_only=True).reset_index()
    masterAvg = masterAvg.drop(['Match Number'], axis=1)

    gamesRef = master.groupby('Ref').count()['Outcome']
    gamesRef = gamesRef / 2
    masterAvg['Games'] = gamesRef.tolist()

    masterAvg.loc[len(masterAvg.index)] = masterAvg.mean(numeric_only=True)
    masterAvg.at[masterAvg.index[-1], 'Ref'] = "Avg Ref"

    return masterAvg
def reftables(dff):

    goaldf = dff.filter(['Ref', 'Goals', 'Action Goals', 'Center Goals', 'Drive Goals', 'Extra Goals', '6MF Goals', 'Counter Goals'])
    exdf = dff.filter(['Ref', 'Total EX', 'CP EX', 'FP EX', 'CS EX', 'M6 EX', 'P EX'])
    gendf = dff.filter(['Ref', 'BL', 'ST', 'Shots'])

    return goaldf.round(2), exdf.round(2), gendf.round(2)
def buildbar(dff, ref, titlet):
    data = dff.drop(['Ref'], axis=1)
    headers = list(data.columns.values)
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=headers,
        y=data.iloc[0],
        name=ref,
    ))

    fig.add_trace(go.Bar(
        x=headers,
        y=data.iloc[1],
        name='Avg Ref',
    ))

    fig.update_layout(legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.73
    ))

    fig.update_layout(
        title={
            'text': titlet,
        }
    )

    fig.update_layout(
        title=bartitlestyle
    )
    return fig

df = mergedf(df)

layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Referee Analysis",
                        className='text-center, mb-4'),
                width=12)
    ]),  # General Title
    dbc.Row([
        dbc.Col([
            "Tournament Filter",
            dcc.Dropdown(
                id='tournament_dropdown_ref',
                options=['All', 'OLY2020', 'WC2022', 'EURO2022'],
                value='All',
            )], width={'size': 2, 'offset': 3}, id='tournament_output_ref', className='mb-4'),
        dbc.Col([
            "Game Type Filter",
            dcc.Dropdown(
                id='opponent_dropdown_ref',
                options=['All', 'TOP 8', 'Knockout Rounds'],
                value='All',
            )], width={'size': 2}, id='opponent_output_ref', className='mb-4'),
        dbc.Col([
            "Sort by",
            dcc.Dropdown(
                id='rank_dropdown_ref',
                options=['Ref', 'Total Goals pg', 'Avg Goal Diff', 'Total EX pg', 'Total TF pg', 'Games'],
                value='Ref',
            )], width={'size': 2}, id='rank_output_ref', className='mb-4')
    ]),  # Dropdowns
    dbc.Row([
        dbc.Col(html.H4("General Stats", className='text-center, mb-4'),
                width=12),

        dbc.Col(
            dash_table.DataTable(id='refTable',  # initiate table
                                 css=[{'selector': 'table', 'rule': 'table-layout: fixed'}],
                                 style_cell=tablestyle,
                                 style_header=tableheaderstyle,
                                 ), className='mb-4')
    ]),  # Ref Table
    dbc.Row([
        dbc.Col(html.H3("Referee Individual Stats",
                        className='text-center, mb-4'),
                width=12)
    ]),  # Ref Title
    dbc.Row([
        dbc.Col([
            "Select a Referee",
            dcc.Dropdown(
                id='ref_dropdown',
                options=[{'label': r, 'value': r} for r in reflist],
                value='ALEXANDRESCU',
            )], width={'size': 2, 'offset': 2}, id='ref_output', className='mb-4'),
        dbc.Col([
            "Game Type Filter",
            dcc.Dropdown(
                id='opponent_dropdown_copy_ref',
                options=[],
                value='All'
            )], width={'size': 2}, id='opponent_output_copy_ref', className='mb-4'),
    ]),  # 2nd Dropdowns
    dbc.Row([
        dbc.Col([
            dcc.Graph(id="goalGraph_ref", style={'width': '48%', 'display': 'inline-block'}),
            dcc.Graph(id="exGraph_ref", style={'width': '48%', 'display': 'inline-block'})
        ], width=12, className='mb-4')
    ]),  # Player Charts
    dbc.Row([
        dbc.Col(html.H4("Referee Analysis Limitations",
                        className='text-center, mb-4'),
                width=12)
    ]),  # Ref Limitations Title
    dbc.Row([
        dbc.Col(html.Div("The referee analysis is based on game statistics where the referee was one of two referees on"
                         "the game. Thus, referee values have less relation with the specific ref the less games (pop"
                         "size) they have documented.",
                        className='text-center, mb-4'),
                width=12)
    ])  # Ref Limitations Text
])

@callback(
    Output('refTable', 'data'),
    Input('tournament_dropdown_ref', 'value'),
    Input('opponent_dropdown_ref', 'value'),
    Input('rank_dropdown_ref', 'value'))
def update_table(tournament, opponent, rank):
    dff = df

    if tournament != "All":
        dff = dff.drop(dff[dff.Tournament != tournament].index)
    if opponent != "All":
        if opponent == "TOP 8":
            dff = dff[dff['Opponent'].isin(TOP_8)]
            dff = dff[dff['Team'].isin(TOP_8)]
        if opponent == "Knockout Rounds":
            dff = dff[dff['Match Number'].isin(KNOCK_GAMES)]

    refwinloss = winlossdf(dff)
    refavg = refavgdf(dff)
    refwinloss['Games'] = refavg['Games']

    if rank == 'Ref':
        refwinloss = refwinloss.sort_values(by=rank, ascending=True)
    else:
        refwinloss = refwinloss.sort_values(by=rank, ascending=False)

    # Tables
    dash_table.DataTable(id='player-table',
                         columns=[{'id': c, 'name': c} for c in refwinloss.columns.values])  # apply to table
    refTable = refwinloss.to_dict('records')

    return refTable

@callback(
    Output('goalGraph_ref', 'figure'),
    Output('exGraph_ref', 'figure'),
    Input('tournament_dropdown_ref', 'value'),
    Input('opponent_dropdown_copy_ref', 'value'),
    Input('ref_dropdown', 'value'))
def update_charts(tournament, opponent, ref):
    dff = df

    if tournament != "All":
        dff = dff.drop(dff[dff.Tournament != tournament].index)
    if opponent != "All":
        if opponent == "TOP 8":
            dff = dff[dff['Opponent'].isin(TOP_8)]
            dff = dff[dff['Team'].isin(TOP_8)]
        if opponent == "Knockout Rounds":
            dff = dff[dff['Match Number'].isin(KNOCK_GAMES)]

    refAvg = refavgdf(dff)
    dfff = refAvg.drop(refAvg[(refAvg['Ref'] != ref) & (refAvg['Ref'] != 'Avg Ref')].index)

    goaldf, exdf, gendf = reftables(dfff)

    return buildbar(goaldf, ref, 'Avg Goals'), buildbar(exdf, ref, 'Avg Exclusions')
@callback(
    [Output('opponent_dropdown_copy_ref', 'options'),
     Output('ref_dropdown', 'value')],
    [Input('ref_dropdown', 'value'),
        Input('opponent_dropdown_copy_ref', 'value')])
def updatedropdowns(ref, opponent):
    refdff = pd.DataFrame()
    refdf = df.filter(['Ref', 'Team', 'Match Number'])

    for index, row in refdf.iterrows():
        if ref == row['Ref']:
            refdff = pd.concat([refdff, row], axis=1)
    refdff = refdff.T

    teamlist = sorted(refdff.Team.unique().tolist())
    gamelist = sorted(refdff['Match Number'].unique().tolist())

    topcheck = any(item in teamlist for item in TOP_8)
    gamecheck = any(item in gamelist for item in KNOCK_GAMES)

    retlist = ['All']

    if topcheck is True:
        retlist.insert(1, "Top 8")
    else:
        if opponent == "Top 8":
            ref = "ALEXANDRESCU"
    if gamecheck is True:
        retlist.insert(1, "Knockout Rounds")
    else:
        if opponent == "Knockout Rounds":
            ref = "ALEXANDRESCU"

    if ref == "":
        ref = "ALEXANDRESCU"

    return [{'label': t, 'value': t} for t in retlist], ref