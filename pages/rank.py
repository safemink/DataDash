import dash
import dash_bootstrap_components as dbc
import pandas as pd
from dash import dash_table, callback, dcc, html
from dash.dependencies import Output, Input

dash.register_page(__name__, order=5)

df = pd.read_csv('data/master_data_refs.csv')
df = df.fillna(0)
df.drop(df.columns[0], axis=1, inplace=True)
df['Outcome'] = df['Outcome'].replace('W', 1)
df['Outcome'] = df['Outcome'].replace('L', 0)
df['Last'] = df['Last'].astype(str) + ' ' + df['First'].astype(str).str[0] + '.'

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
def filterplayerdf(tournament, gamet, df):
    dff = df

    if tournament != "All":
        dff = dff.drop(dff[dff.Tournament != tournament].index)
    if gamet != "All":
        if gamet == "TOP 8":
            dff = dff[dff['Opponent'].isin(TOP_8)]
        if gamet != "TOP 8":
            dff = dff[dff['Match Number'].isin(KNOCK_GAMES)]
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
def gettable(playerComp, playerAvg, stat, sort):
    table1 = playerComp.filter(
            ['Last', stat], axis=1)

    if stat != 'Shooting %' and stat != 'Extra Shooting %' and stat != 'Center Shooting %' and \
            stat != 'Action Shooting %' and stat != 'Foul Shooting %' and stat != '6MF Shooting %' and \
            stat != 'PS Shooting %' and stat != 'CA Shooting %':
        table1[stat + ' pg'] = playerComp[stat] / playerAvg['Games']
        table1['Minutes per ' + stat] = playerAvg['Minutes pg'] / table1[stat + ' pg']

    table1['Games'] = playerAvg['Games']
    table1['Minutes pg'] = playerAvg['Minutes pg']

    if sort == 'Total':
        table1 = table1.sort_values(by=[stat], ascending=False)
    if sort == 'Per Game':
        table1 = table1.sort_values(by=[stat + ' pg'], ascending=False)
    if sort == 'Per Minute':
        table1 = table1.sort_values(by=['Minutes per ' + stat], ascending=True)

    return table1.round(2)

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
            "Select a Stat",
            dcc.Dropdown(
                id='stat_dropdown_rpg',
                options=['Goals', 'Shots', 'Total EX', 'Minutes',
                         'Action Goals', 'Action Shots', 'Extra Goals', 'Extra Shots',
                         'Center Goals', 'Center Shots', 'Foul Goals', 'Foul Shots',
                         '6MF Goals', '6MF Shots', 'PS Goals', 'PS Shots', 'CA Goals',
                         'CA Shots', 'TF', 'ST', 'RB', 'BL', 'SP Won', 'SP Attempts',
                         'CP EX', 'FP EX', 'DS EX', 'M6 EX', 'CS EX', 'DE', 'P EX', 'Shooting %',
                         'Extra Shooting %', 'Center Shooting %', 'Action Shooting %', 'Foul Shooting %',
                         '6MF Shooting %', 'PS Shooting %', 'CA Shooting %'],
                value='Goals',
            )], width={'size': 2, 'offset': 1}, id='stat_output_rpg', className='mb-4'),
        dbc.Col([
            "Tournament Filter",
            dcc.Dropdown(
                id='tournament_dropdown_rpg',
                options=['All', 'OLY2020'],
                value='All',
            )], width={'size': 2}, id='tournament_output_rpg', className='mb-4'),
        dbc.Col([
            "Game Type Filter",
            dcc.Dropdown(
                id='gamet_dropdown_rpg',
                options=['All', 'Top 8', 'Knockout Round'],
                value='All'
            )], width={'size': 2}, id='gamet_output_rpg', className='mb-4'),
        dbc.Col([
            "Sort by",
            dcc.Dropdown(
                id='sort_dropdown_rpg',
                options=['Total', 'Per Game', 'Per Minute'],
                value='Total'
            )], width={'size': 2}, id='sort_output_rpg', className='mb-4')
    ]),     # Dropdowns
    dbc.Row([
        dbc.Col(html.H4("Player Stats", className='text-center, mb-4'),
                width=12),

        dbc.Col(
            dash_table.DataTable(id='playerStats_table_rpg',  # initiate table
                                 css=[{'selector': 'table', 'rule': 'table-layout: fixed'}],
                                 style_cell=tablestyle,
                                 style_header=tableheaderstyle,
                                 ), className='mb-4')
    ]),     # Team Table
])

@callback(
    Output('playerStats_table_rpg', 'data'),
    Input('stat_dropdown_rpg', 'value'),
    Input('tournament_dropdown_rpg', 'value'),
    Input('gamet_dropdown_rpg', 'value'),
    Input('sort_dropdown_rpg', 'value'))
def update_tables(stat, tournament, gamet, sort):
    playerstats = players_master

    playerstats = filterplayerdf(tournament, gamet, playerstats)

    playerComp, playerAvg = playercompile(playerstats)

    table1 = gettable(playerComp, playerAvg, stat, sort)

    # Tables
    dash_table.DataTable(id='player-table',
                         columns=[{'id': c, 'name': c} for c in table1.columns.values])  # apply to table
    player_table = table1.to_dict('records')

    return player_table