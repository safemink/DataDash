import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import dash_table, callback, dcc, html, callback_context
from dash.dependencies import Output, Input

dash.register_page(__name__, order=2)

# Import Data
df = pd.read_csv('data/master_data_refs.csv')
df = df.fillna(0)
df.drop(df.columns[0], axis=1, inplace=True)
df['Outcome'] = df['Outcome'].replace('W', 1)
df['Outcome'] = df['Outcome'].replace('L', 0)
df['Last'] = df['Last'].astype(str) + ' ' + df['First'].astype(str).str[0] + '.'

# get list of teams
teamlist = sorted(df.Team.unique().tolist())
oplistdf = df.drop(df[df.Team != "USA"].index)
oplist = sorted(oplistdf.Opponent.unique().tolist())
oplist.insert(0, "Top 8")
oplist.insert(0, "All")

# Lists
TOP_8 = ['ITA', 'GRE', 'SRB', 'USA', 'HUN', 'ESP', 'CRO', 'MNE']
OpNames = {'Opponent': 'TeamT', 'Team': 'OpponentO',
            'Goals': 'Goals Ag', 'Shots': 'Shots Ag',
            'Action Goals': 'Action Goals Ag', 'Action Shots': 'Action Shots Ag',
            'Center Goals': 'Center Goals Ag', 'Center Shots': 'Center Shots Ag',
            'Drive Goals': 'Drive Goals Ag', 'Drive Shots': 'Drive Shots Ag',
            'Extra Goals': 'Extra Goals Ag', 'Extra Shots': 'Extra Shots Ag',
            'Foul Goals': 'Foul Goals Ag', 'Foul Shots': 'Foul Shots Ag',
            '6MF Goals': '6MF Goals Ag', '6MF Shots': '6MF Shots Ag',
            'CA Goals': 'CA Goals Ag', 'CA Shots': 'CA Shots Ag',
            'PS Goals': 'PS Goals Ag', 'PS Shots': 'PS Shots Ag',
            'TF': 'TF Ag', 'ST': 'ST Ag', 'RB': 'RB Ag', 'BL': 'BL Ag',
            'CP EX': 'CP DEX', 'FP EX': 'FP DEX', 'DS EX': 'DS DEX',
            'M6 EX': 'M6 DEX', 'CS EX': 'CS DEX', 'DE':'DE DEX', 'P EX': 'P DEX',
            'Total EX': 'Total DEX'}
PAVG_NAMES = {'Goals': 'Goals pg', 'Shots': 'Shots pg',
             'Action Goals': 'Action Goals pg', 'Action Shots': 'Action Shots pg',
             'Extra Goals': 'Extra Goals pg', 'Extra Shots': 'Extra Shots pg',
             'Center Goals': 'Center Goals pg', 'Center Shots': 'Center Shots pg',
             'Foul Goals': 'Foul Goals pg', 'Foul Shots': 'Foul Shots pg',
             '6MF Goals': '6MF Goals pg', '6MF Shots': '6MF Shots pg',
             'PS Goals': 'PS Goals pg', 'PS Shots': 'PS Shots pg',
             'CA Goals': 'CA Goals pg', 'CA Shots': 'CA Shots pg',
             'TF': 'TF pg', 'ST': 'ST pg', 'RB': 'RB pg', 'BL': 'BL pg',
             'SP Won': 'SP Won pg', 'SP Attempts': 'SP Attempts pg',
             'CP EX': 'CP EX pg', 'FP EX': 'FP EX pg', 'DS EX': 'DS EX pg',
             'M6 EX': 'M6 EX pg', 'CS EX': 'CS EX pg', 'DE': 'DE pg', 'P EX': 'P EX pg',
             'Total EX': 'Total EX pg', 'Outcome': 'Games'}
PAVG_COLS = {'Goals': 'mean', 'Shots': 'mean',
             'Action Goals': 'mean', 'Action Shots': 'mean',
             'Extra Goals': 'mean', 'Extra Shots': 'mean',
             'Center Goals': 'mean', 'Center Shots': 'mean',
             'Foul Goals': 'mean', 'Foul Shots': 'mean',
             '6MF Goals': 'mean', '6MF Shots': 'mean',
             'PS Goals': 'mean', 'PS Shots': 'mean',
             'CA Goals': 'mean', 'CA Shots': 'mean',
             'TF': 'mean', 'ST': 'mean', 'RB': 'mean', 'BL': 'mean',
             'SP Won': 'mean', 'SP Attempts': 'mean',
             'CP EX': 'mean', 'FP EX': 'mean', 'DS EX': 'mean', 'M6 EX': 'mean', 'CS EX': 'mean', 'DE': 'mean', 'P EX': 'mean',
             'Total EX': 'mean', 'Outcome': 'count'}
TAVG_NAMES = {'Goals': 'Goals pg', 'Shots': 'Shots pg',
             'Action Goals': 'Action Goals pg', 'Action Shots': 'Action Shots pg',
             'Extra Goals': 'Extra Goals pg', 'Extra Shots': 'Extra Shots pg',
             'Center Goals': 'Center Goals pg', 'Center Shots': 'Center Shots pg',
             'Foul Goals': 'Foul Goals pg', 'Foul Shots': 'Foul Shots pg',
             '6MF Goals': '6MF Goals pg', '6MF Shots': '6MF Shots pg',
             'PS Goals': 'PS Goals pg', 'PS Shots': 'PS Shots pg',
             'CA Goals': 'CA Goals pg', 'CA Shots': 'CA Shots pg',
             'TF': 'TF pg', 'ST': 'ST pg', 'RB': 'RB pg', 'BL': 'BL pg',
             'SP Won': 'SP Won pg', 'SP Attempts': 'SP Attempts pg',
             'CP EX': 'CP EX pg', 'FP EX': 'FP EX pg', 'DS EX': 'DS EX pg',
             'M6 EX': 'M6 EX pg', 'CS EX': 'CS EX pg', 'DE': 'DE pg', 'P EX': 'P EX pg',
             'Total EX': 'Total EX pg',
             'Goals Ag': 'Goals Ag pg', 'Shots Ag': 'Shots Ag pg',
             'Action Goals Ag': 'Action Goals Ag pg', 'Action Shots Ag': 'Action Shots Ag pg',
             'Extra Goals Ag': 'Extra Goals Ag pg', 'Extra Shots Ag': 'Extra Shots Ag pg',
             'Center Goals Ag': 'Center Goals Ag pg', 'Center Shots Ag': 'Center Shots Ag pg',
             'Foul Goals Ag': 'Foul Goals Ag pg', 'Foul Shots Ag': 'Foul Shots Ag pg',
             '6MF Goals Ag': '6MF Goals Ag pg', '6MF Shots Ag': '6MF Shots Ag pg',
             'PS Goals Ag': 'PS Goals Ag pg', 'PS Shots Ag': 'PS Shots Ag pg',
             'CA Goals Ag': 'CA Goals Ag pg', 'CA Shots Ag': 'CA Shots Ag pg',
             'TF Ag': 'TF Ag pg', 'ST Ag': 'ST Ag pg', 'RB Ag': 'RB Ag pg', 'BL Ag': 'BL Ag pg',
             'CP DEX': 'CP DEX pg', 'FP DEX': 'FP DEX pg', 'DS DEX': 'DS DEX pg',
             'M6 DEX': 'M6 DEX pg', 'CS DEX': 'CS DEX pg', 'DE DEX': 'DEX pg', 'P DEX': 'P DEX pg',
             'Total DEX': 'Total DEX pg', 'Outcome': 'Games'}
TAVG_COLS = {'Goals': 'mean', 'Shots': 'mean',
             'Action Goals': 'mean', 'Action Shots': 'mean',
             'Extra Goals': 'mean', 'Extra Shots': 'mean',
             'Center Goals': 'mean', 'Center Shots': 'mean',
             'Foul Goals': 'mean', 'Foul Shots': 'mean',
             '6MF Goals': 'mean', '6MF Shots': 'mean',
             'PS Goals': 'mean', 'PS Shots': 'mean',
             'CA Goals': 'mean', 'CA Shots': 'mean',
             'TF': 'mean', 'ST': 'mean', 'RB': 'mean', 'BL': 'mean',
             'SP Won': 'mean', 'SP Attempts': 'mean',
             'CP EX': 'mean', 'FP EX': 'mean', 'DS EX': 'mean', 'M6 EX': 'mean',
             'CS EX': 'mean', 'DE': 'mean', 'P EX': 'mean', 'Total EX': 'mean',
             'Goals Ag': 'mean', 'Shots Ag': 'mean',
             'Action Goals Ag': 'mean', 'Action Shots Ag': 'mean',
             'Extra Goals Ag': 'mean', 'Extra Shots Ag': 'mean',
             'Center Goals Ag': 'mean', 'Center Shots Ag': 'mean',
             'Foul Goals Ag': 'mean', 'Foul Shots Ag': 'mean',
             '6MF Goals Ag': 'mean', '6MF Shots Ag': 'mean',
             'PS Goals Ag': 'mean', 'PS Shots Ag': 'mean',
             'CA Goals Ag': 'mean', 'CA Shots Ag': 'mean',
             'TF Ag': 'mean', 'ST Ag': 'mean', 'RB Ag': 'mean', 'BL Ag': 'mean',
             'CP DEX': 'mean', 'FP DEX': 'mean', 'DS DEX': 'mean', 'M6 DEX': 'mean',
             'CS DEX': 'mean', 'DE DEX': 'mean', 'P DEX': 'mean', 'Total DEX': 'mean', 'Outcome': 'count'}

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

# groups by Team and Opponent, merges two df, swaps ex cols
def mergeteamdf(data):

    dff = data.drop(data[data.stat_type != "Team"].index)
    dff = dff.drop(['Last', 'First', 'Play Time', 'Minutes', 'Seconds', 'Match Number', 'Ref 1', 'Ref 2'], axis=1)

    team = dff
    op = dff
    op = op.drop(['SP Won', 'SP Attempts', 'Outcome'], axis=1)

    op = op.rename(columns=OpNames)
    op.set_index('TeamT')
    team.set_index('Team')

    for i in range(len(op)):
        num = i % 2
        if num == 0:
            temp = op.iloc[i]
            op.iloc[i] = op.iloc[i + 1]
            op.iloc[i + 1] = temp

    master = pd.concat([team, op], axis=1)
    master = master.drop(['TeamT', 'OpponentO', 'stat_type'], axis=1).set_index('Team')

    return master
def filterteamdf(team, tournament, result, opponent, data):
    dff = data
    if tournament != "All":
        dff = dff.drop(dff[dff.Tournament != tournament].index)
    if result != "All":
        if result == 'W':
            dff = dff[~(dff['Outcome'] != 1)]
        else:
            dff = dff[~(dff['Outcome'] != 0)]
    if opponent != "All":
        if opponent == "TOP 8":
            dff = dff[dff['Opponent'].isin(TOP_8)]
        if opponent != "TOP 8":
            dff = dff[~(dff['Opponent'] != opponent)]
    return dff
def filterplayerdf(team, tournament, result, opponent, df2):
    dff = df2
    dff = dff.drop(dff[dff.Team != team].index)
    if tournament != "All":
        dff = dff.drop(dff[dff.Tournament != tournament].index)
    if result != "All":
        if result == 'W':
            dff = dff.drop(dff[dff.Outcome != 1].index)
        else:
            dff = dff.drop(dff[dff.Outcome != 0].index)
    if opponent != "All":
        if opponent == "TOP 8":
            dff = dff[dff['Opponent'].isin(TOP_8)]
        if opponent != "TOP 8":
            dff = dff[~(dff['Opponent'] != opponent)]

    return dff
def playercompile(df2):
    playerComp = df2.groupby('Last').sum(numeric_only=True).reset_index()
    playerAvg = df2.groupby('Last').agg(PAVG_COLS).rename(columns=PAVG_NAMES).reset_index()

    playerAvg['Minutes pg'] = (((playerComp['Minutes'] * 60) + playerComp['Seconds']) / 60) / playerAvg['Games']
    playerAvg['Shooting %'] = playerComp['Goals'] * 100 / playerComp['Shots']
    playerAvg['Extra Shooting %'] = playerComp['Extra Goals'] * 100 / playerComp['Extra Shots']
    playerAvg['Center Shooting %'] = playerComp['Center Goals'] * 100 / playerComp['Center Shots']
    playerAvg['Action Shooting %'] = playerComp['Action Goals'] * 100 / playerComp['Action Shots']
    playerAvg['Foul Shooting %'] = playerComp['Foul Goals'] * 100 / playerComp['Foul Shots']
    playerAvg['6MF Shooting %'] = playerComp['6MF Goals'] * 100 / playerComp['6MF Shots']
    playerAvg['PS Shooting %'] = playerComp['PS Goals'] * 100 / playerComp['PS Shots']
    playerAvg['CA Shooting %'] = playerComp['CA Goals'] * 100 / playerComp['CA Shots']

    playerComp['Shooting %'] = playerComp['Goals'] * 100 / playerComp['Shots']
    playerComp['Extra Shooting %'] = playerComp['Extra Goals'] * 100 / playerComp['Extra Shots']
    playerComp['Center Shooting %'] = playerComp['Center Goals'] * 100 / playerComp['Center Shots']
    playerComp['Action Shooting %'] = playerComp['Action Goals'] * 100 / playerComp['Action Shots']
    playerComp['Foul Shooting %'] = playerComp['Foul Goals'] * 100 / playerComp['Foul Shots']
    playerComp['6MF Shooting %'] = playerComp['6MF Goals'] * 100 / playerComp['6MF Shots']
    playerComp['PS Shooting %'] = playerComp['PS Goals'] * 100 / playerComp['PS Shots']
    playerComp['CA Shooting %'] = playerComp['CA Goals'] * 100 / playerComp['CA Shots']

    return playerComp.round(2), playerAvg.round(2)
def teamcompile(dff):
    teamComp = dff.groupby('Team').sum(numeric_only=True).reset_index()
    teamAvg = dff.groupby('Team').agg(TAVG_COLS).rename(columns=TAVG_NAMES).reset_index()
    teamComp = teamComp.rename(columns={'Outcome': 'Wins'})

    teamComp['Games'] = teamAvg['Games']
    teamComp['Shooting %'] = teamComp['Goals'] * 100 / teamComp['Shots']
    teamComp['Extra Shooting %'] = teamComp['Extra Goals'] * 100 / teamComp['Extra Shots']
    teamComp['Center Shooting %'] = teamComp['Center Goals'] * 100 / teamComp['Center Shots']
    teamComp['Action Shooting %'] = teamComp['Action Goals'] * 100 / teamComp['Action Shots']
    teamComp['Foul Shooting %'] = teamComp['Foul Goals'] * 100 / teamComp['Foul Shots']
    teamComp['6MF Shooting %'] = teamComp['6MF Goals'] * 100 / teamComp['6MF Shots']
    teamComp['PS Shooting %'] = teamComp['PS Goals'] * 100 / teamComp['PS Shots']
    teamComp['CA Shooting %'] = teamComp['CA Goals'] * 100 / teamComp['CA Shots']

    teamComp['Op Shooting %'] = teamComp['Goals Ag'] * 100 / teamComp['Shots Ag']
    teamComp['Op Extra Shooting %'] = teamComp['Extra Goals Ag'] * 100 / teamComp['Extra Shots Ag']
    teamComp['Op Center Shooting %'] = teamComp['Center Goals Ag'] * 100 / teamComp['Center Shots Ag']
    teamComp['Op Action Shooting %'] = teamComp['Action Goals Ag'] * 100 / teamComp['Action Shots Ag']
    teamComp['Op Foul Shooting %'] = teamComp['Foul Goals Ag'] * 100 / teamComp['Foul Shots Ag']
    teamComp['Op 6MF Shooting %'] = teamComp['6MF Goals Ag'] * 100 / teamComp['6MF Shots Ag']
    teamComp['Op PS Shooting %'] = teamComp['PS Goals Ag'] * 100 / teamComp['PS Shots Ag']
    teamComp['Op CA Shooting %'] = teamComp['CA Goals Ag'] * 100 / teamComp['CA Shots Ag']

    teamAvg['Shooting %'] = teamComp['Goals'] * 100 / teamComp['Shots']
    teamAvg['Extra Shooting %'] = teamComp['Extra Goals'] * 100 / teamComp['Extra Shots']
    teamAvg['Center Shooting %'] = teamComp['Center Goals'] * 100 / teamComp['Center Shots']
    teamAvg['Action Shooting %'] = teamComp['Action Goals'] * 100 / teamComp['Action Shots']
    teamAvg['Foul Shooting %'] = teamComp['Foul Goals'] * 100 / teamComp['Foul Shots']
    teamAvg['6MF Shooting %'] = teamComp['6MF Goals'] * 100 / teamComp['6MF Shots']
    teamAvg['PS Shooting %'] = teamComp['PS Goals'] * 100 / teamComp['PS Shots']
    teamAvg['CA Shooting %'] = teamComp['CA Goals'] * 100 / teamComp['CA Shots']
    teamAvg['Win %'] = teamComp['Wins'] / teamComp['Games']

    teamAvg['Op Shooting %'] = teamComp['Goals Ag'] * 100 / teamComp['Shots Ag']
    teamAvg['Op Extra Shooting %'] = teamComp['Extra Goals Ag'] * 100 / teamComp['Extra Shots Ag']
    teamAvg['Op Center Shooting %'] = teamComp['Center Goals Ag'] * 100 / teamComp['Center Shots Ag']
    teamAvg['Op Action Shooting %'] = teamComp['Action Goals Ag'] * 100 / teamComp['Action Shots Ag']
    teamAvg['Op Foul Shooting %'] = teamComp['Foul Goals Ag'] * 100 / teamComp['Foul Shots Ag']
    teamAvg['Op 6MF Shooting %'] = teamComp['6MF Goals Ag'] * 100 / teamComp['6MF Shots Ag']
    teamAvg['Op PS Shooting %'] = teamComp['PS Goals Ag'] * 100 / teamComp['PS Shots Ag']
    teamAvg['Op CA Shooting %'] = teamComp['CA Goals Ag'] * 100 / teamComp['CA Shots Ag']

    return teamComp.round(2), teamAvg.round(2)
def addaverages(dff):
    dff.loc[len(dff.index)] = dff.mean(numeric_only=True)
    dff.at[dff.index[-1], 'Team'] = 'All Teams Avg'

    dff.loc[len(dff.index)] = dff[dff['Team'].isin(TOP_8)].mean(numeric_only=True)
    dff.at[dff.index[-1], 'Team'] = 'Top 8 Avg'

    return dff.round(2)
def gettables(playerAvg, stat, teamAvg, team):
    teamtable = teamAvg[~(teamAvg['Team'] != team) | ~(teamAvg['Team'] != 'All Teams Avg') | ~(teamAvg['Team'] != 'Top 8 Avg')]

    table1 = playerAvg.filter(
            ['Last', 'Games', 'Goals pg', 'Shots pg', 'Shooting %', 'Extra Shooting %', 'Total EX pg', stat], axis=1)
    table2 = teamtable.filter(['Team', 'Win %', 'Goals pg', 'Shots pg', 'Extra Shooting %', 'Total DEX pg'], axis=1)
    table3 = teamtable.filter(['Team', 'Games', 'Goals Ag pg', 'Shots Ag pg', 'Op Extra Shooting %', 'Total EX pg'], axis=1)

    return table1, table2, table3
def getchartdfs(dff, team):
    dff = dff[~(dff['Team'] != team) | ~(dff['Team'] != "Top 8 Avg") | ~(dff['Team'] != "All Teams Avg")]

    t_shootp = dff.filter(['Action Shooting %', 'Center Shooting %', 'Drive Shooting %', 'Extra Shooting %', 'Foul Shooting %', 'Penalty Shooting %'], axis=1)
    o_shootp = dff.filter(['Op Action Shooting %', 'Op Center Shooting %', 'Op Drive Shooting %', 'Op Extra Shooting %', 'Op Foul Shooting %', 'Op Penalty Shooting %'], axis=1)

    t_ex = dff.filter(['CP DEX pg', 'FP DEX pg', 'DS DEX pg', 'M6 DEX', 'CS DEX pg', 'DE DEX pg', 'P EX pg'], axis=1)
    o_ex = dff.filter(['CP EX pg', 'FP EX pg', 'DS EX pg', 'M6 EX pg', 'CS EX pg', 'DE EX pg', 'P EX pg'], axis=1)
    avg_ex = dff.filter(['CP EX pg', 'FP EX pg', 'DS EX pg', 'M6 EX pg', 'CS EX pg', 'DE EX pg', 'P EX pg'], axis=1)
    avg_ex = avg_ex.iloc[2:]

    t_goals = dff.filter(['Action Goals pg', 'Extra Goals pg', 'Center Goals pg', 'CA Goals pg', 'Drive Goals pg', 'Foul Goals pg', 'PS Goals pg'], axis=1)
    o_goals = dff.filter(['Action Goals Ag pg', 'Extra Goals Ag pg', 'Center Goals Ag pg', 'CA Goals Ag pg', 'Drive Goals Ag pg', '6MF Goals Ag pg', 'PS Goals Ag pg'], axis=1)

    # t_shootp.drop('Team', inplace=True, axis=1)
    # o_shootp.drop('Team', inplace=True, axis=1)
    # t_ex.drop('Team', inplace=True, axis=1)
    # o_ex.drop('Team', inplace=True, axis=1)

    return t_shootp.reindex(), o_shootp.reindex(), t_ex.reindex(), o_ex.reindex(), avg_ex.reindex(), t_goals.reindex(), o_goals.reindex()
def buildbar(dff, team, titlet, opponent):
    data = dff
    headers = list(data.columns.values)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=headers,
        y=data.iloc[0],
        name=team
    ))

    fig.add_trace(go.Bar(
        x=headers,
        y=data.iloc[1],
        name='All Teams Avg'
    ))
    fig.add_trace(go.Bar(
        x=headers,
        y=data.iloc[2],
        name='Top 8 Avg'
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
def buildpie(dff, team, titlet):
    data = dff

    data = data.T
    data['Values'] = data.iloc[:, 0]
    data['Names'] = list(data.index)

    fig = px.pie(data, values='Values', names='Names')
    fig.update_traces(textposition='inside', textinfo='value+label')

    fig.update_layout(
        title={
            'text': titlet
        })

    fig.update_layout(
        title=pietitlestyle
    )

    return fig
def buildppie(dff, team, titlet):
    data = dff

    data['Values'] = data.iloc[:, 1]
    data['Names'] = data.iloc[:, 0]

    fig = px.pie(data, values='Values', names='Names')
    fig.update_traces(textposition='inside', textinfo='value+label')

    fig.update_layout(
        title={
            'text': titlet
        }
    )
    fig.update_layout(
        title=pietitlestyle
    )

    return fig

players_master = df.drop(df[df.stat_type != 'Player'].index)
teams_master = mergeteamdf(df).reset_index()

layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Team Analysis",
                        className='text-center, mb-4'),
                width=12)
    ]),     # Title
    dbc.Row([
        dbc.Col([
            "Select a Team",
            dcc.Dropdown(
                id='team_dropdown_tpg',
                options=[{'label': t, 'value': t} for t in teamlist],
                value='USA',
            )], width={'size': 2, 'offset': 2}, id='team_output_tpg', className='mb-4'),
        dbc.Col([
            "Tournament Filter",
            dcc.Dropdown(
                id='tournament_dropdown_tpg',
                options=['All', 'OLY2020'],
                value='All',
            )], width={'size': 2}, id='tournament_output_tpg', className='mb-4'),
        dbc.Col([
            "Result Filter",
            dcc.Dropdown(
                id='result_dropdown_tpg',
                options=['All', 'W', 'L'],
                value='All',
            )], width={'size': 2}, id='result_output_tpg', className='mb-4'),
        dbc.Col([
            "Opponent Filter",
            dcc.Dropdown(
                id='opponent_dropdown_tpg',
                options=[],
                value='All',
            )], width={'size': 2}, id='opponent_output_tpg', className='mb-4')
    ]),     # Dropdowns
    dbc.Row([
        dbc.Col(html.H4("Offensive Team Stats", className='text-center, mb-4'),
                width=12),

        dbc.Col(
            dash_table.DataTable(id='teamStats_table',  # initiate table
                                 css=[{'selector': 'table', 'rule': 'table-layout: fixed'}],
                                 style_cell=tablestyle,
                                 style_header=tableheaderstyle,
                                 ), className='mb-4')
    ]),     # Team Table
    dbc.Row([
        dbc.Col(html.H4("Defensive Team Stats", className='text-center, mb-4'),
                width=12),

        dbc.Col(
            dash_table.DataTable(id='opStats_table',  # initiate table
                                 css=[{'selector': 'table', 'rule': 'table-layout: fixed'}],
                                 style_cell=tablestyle,
                                 style_header=tableheaderstyle,
                                 ), className='mb-4')
    ]),     # Opponent Table
    dbc.Row([
        dbc.Col(html.H4("Team Scoring Statisitcs", className='text-center, mb-4'),
                width=12),

        dbc.Col([
            dcc.Graph(id="TeamGoalbar", style={'width': '48%', 'display': 'inline-block'}),
            dcc.Graph(id="OpGoalbar", style={'width': '48%', 'display': 'inline-block'})
        ], width=12, className='mb-4')
    ]),     # Scoring Charts
    dbc.Row([
        dbc.Col(html.H4("Shooting Percentage", className='text-center, mb-4'),
                width=12),

        dbc.Col([
            dcc.Graph(id="TeamShootbar", style={'width': '48%', 'display': 'inline-block'}),
            dcc.Graph(id="OpShootbar", style={'width': '48%', 'display': 'inline-block'})
        ], width=12, className='mb-4')
    ]),     # Shooting Percentage Charts
    dbc.Row([
        dbc.Col(html.H4("Team Exclusion Statisitcs", className='text-center, mb-4'),
                width=12),

        dbc.Col([
            dcc.Graph(id="DEXpie", style={'width': '33%', 'display': 'inline-block'}),
            dcc.Graph(id="EXpie", style={'width': '33%', 'display': 'inline-block'}),
            dcc.Graph(id="AVGEXpie", style={'width': '33%', 'display': 'inline-block'})
        ], width=12, className='mb-4')
    ]),     # EX Charts
    dbc.Row([
        dbc.Col(html.H3("Team Player Stats", className='text-center, mb-4'),
                width=12),
    ]),     # Player Title
    dbc.Row([
        dbc.Col([
            "Result Filter",
            dcc.Dropdown(id='result_dropdown_copy_tpg',
                         options=['All', 'W', 'L'],
                         value='All')], width={'size': 2, 'offset': 1}, className='mb-4'),
        dbc.Col([
            "Opponent Filter",
            dcc.Dropdown(id='opponent_dropdown_copy_tpg',
                         options=[],
                         value='All')], width={'size': 2}, className='mb-4'),
        dbc.Col([
            "Custom Stat",
            dcc.Dropdown(id='stat_dropdown_tpg',
                         options=['Action Goals pg', 'Action Shots pg', 'Extra Goals pg', 'Extra Shots pg',
                                  'Center Goals pg', 'Center Shots pg', 'Foul Goals pg', 'Foul Shots pg',
                                  '6MF Goals pg', '6MF Shots pg', 'PS Goals pg', 'PS Shots pg', 'CA Goals pg',
                                  'CA Shots pg', 'TF pg', 'ST pg', 'RB pg', 'BL pg', 'SP Won pg', 'SP Attempts pg',
                                  'CP EX pg', 'FP EX pg', 'DS EX pg', 'M6 EX pg', 'CS EX pg', 'DE pg', 'P EX pg',
                                  'Minutes pg', 'Center Shooting %', 'Action Shooting %', 'Foul Shooting %',
                                  '6MF Shooting %', 'PS Shooting %', 'CA Shooting %'],
                         value='Minutes pg')], width={'size': 2}, className='mb-4')
    ]),     # Secondary Dropdowns
    dbc.Row([
        dbc.Col(
            dash_table.DataTable(id='playerStats_table',  # initiate table
                                css=[{'selector': 'table', 'rule': 'table-layout: fixed'}],
                                 style_cell=tablestyle,
                                 style_header=tableheaderstyle,
                                 ), className='mb-4')
    ]),     # Player Table
    dbc.Row([
        dbc.Col(html.H3("Player/Team Statistics", className='text-center, mb-4'),
                width=12),
        dbc.Col([
            dcc.Graph(id="pGoalPie", style={'width': '48%', 'display': 'inline-block'}),
            dcc.Graph(id="pShotPie", style={'width': '48%', 'display': 'inline-block'})
        ], width=12, className='mb-4')
    ]),     # Player Charts
])

@callback(
    Output('playerStats_table', 'data'),
    Output('teamStats_table', 'data'),
    Output('opStats_table', 'data'),
    Output('pGoalPie', 'figure'),
    Output('pShotPie', 'figure'),
    Input('team_dropdown_tpg', 'value'),
    Input('tournament_dropdown_tpg', 'value'),
    Input('result_dropdown_tpg', 'value'),
    Input('opponent_dropdown_tpg', 'value'),
    Input('stat_dropdown_tpg', 'value'))
def update_tables(team, tournament, result, opponent, stat):
    teamstats = teams_master
    playerstats = players_master

    teamstats = filterteamdf(team, tournament, result, opponent, teamstats)
    playerstats = filterplayerdf(team, tournament, result, opponent, playerstats)

    playerComp, playerAvg = playercompile(playerstats)

    teamComp, teamAvg = teamcompile(teamstats)
    teamAvg = addaverages(teamAvg)

    table1, table2, table3 = gettables(playerAvg, stat, teamAvg, team)

    # Tables
    dash_table.DataTable(id='player-table',
                         columns=[{'id': c, 'name': c} for c in table1.columns.values])  # apply to table
    player_table = table1.to_dict('records')
    dash_table.DataTable(id='team-table',
                         columns=[{'id': c, 'name': c} for c in table2.columns.values])  # apply to table
    team_table = table2.to_dict('records')

    dash_table.DataTable(id='op-table',
                         columns=[{'id': c, 'name': c} for c in table3.columns.values])  # apply to table
    op_table = table3.to_dict('records')

    pgoal_fig = playerAvg.filter(['Last', 'Goals pg'], axis=1)
    pshot_fig = playerAvg.filter(['Last', 'Shots pg'], axis=1)

    return player_table, team_table, op_table, buildppie(pgoal_fig, team, "Goals by Player pg"), \
           buildppie(pshot_fig, team, "Shots by Player pg")

@callback(
    Output('TeamShootbar', 'figure'),
    Output('OpShootbar', 'figure'),
    Output('DEXpie', 'figure'),
    Output('EXpie', 'figure'),
    Output('AVGEXpie', 'figure'),
    Output('TeamGoalbar', 'figure'),
    Output('OpGoalbar', 'figure'),
    Input('team_dropdown_tpg', 'value'),
    Input('tournament_dropdown_tpg', 'value'),
    Input('result_dropdown_tpg', 'value'),
    Input('opponent_dropdown_tpg', 'value'))
def update_charts(team, tournament, result, opponent):
    teamstats = teams_master
    teamstats = filterteamdf(team, tournament, result, opponent, teamstats)

    teamComp, teamAvg = teamcompile(teamstats)
    teamAvg = addaverages(teamAvg)

    tshoot_fig, opshoot_fig, tdex_fig, opex_fig, avgex_fig, tgoals_fig, ogoals_fig = getchartdfs(teamAvg, team)

    return buildbar(tshoot_fig, team, "Team Shooting %", opponent), buildbar(opshoot_fig, team, "Opponent Shooting %", opponent), \
           buildpie(tdex_fig, team, "Team DEX pg"), buildpie(opex_fig, team, "Team EX pg"), \
            buildpie(avgex_fig, team, "Top 8 Avg EX"), \
           buildbar(tgoals_fig, team, "Goals pg", opponent), buildbar(ogoals_fig, team, "Opponent Goals pg", opponent)

@callback(
    Output('opponent_dropdown_tpg', 'options'),
    Output('opponent_dropdown_copy_tpg', 'options'),
    Input('team_dropdown_tpg', 'value'),
    Input('tournament_dropdown_tpg', 'value'),
    Input('result_dropdown_tpg', 'value'))
def updatedropdowns(team, tournament, result):
    teamstats = filterteamdf(team, tournament, result, 'All', teams_master)
    oplistdf = teamstats.drop(teamstats[teamstats.Team != team].index)
    oplist = sorted(oplistdf.Opponent.unique().tolist())

    check = any(item in oplist for item in TOP_8)

    if check is True:
        oplist.insert(0, "TOP 8")

    oplist.insert(0, "All")
    return [{'label': t, 'value': t} for t in oplist], [{'label': t, 'value': t} for t in oplist]

@callback(
    [Output('result_dropdown_tpg', 'value'),
     Output('result_dropdown_copy_tpg', 'value'),
     Output('opponent_dropdown_tpg', 'value'),
     Output('opponent_dropdown_copy_tpg', 'value')],
    [Input('result_dropdown_tpg', 'value'),
     Input('result_dropdown_copy_tpg', 'value'),
     Input('opponent_dropdown_tpg', 'value'),
     Input('opponent_dropdown_copy_tpg', 'value')])
def linkdropdowns(result, result_copy, opponent, opponent_copy):
    ctx = callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    result_value = result if trigger_id == 'result_dropdown_tpg' else result_copy
    opponent_value = opponent if trigger_id == 'opponent_dropdown_tpg' else opponent_copy

    return result_value, result_value, opponent_value, opponent_value