"""Restoration regression checks; run from the repository root."""
import hashlib
import importlib
import json
from pathlib import Path
import pandas as pd
from app import app

checks = 0
failures = []

def check(label, fn):
    global checks
    try:
        fn()
        checks += 1
    except Exception as exc:
        failures.append({'check':label,'error':f'{type(exc).__name__}: {exc}'})

def require(condition):
    assert condition

client = app.server.test_client()
for route in ['/', '/player', '/team-analysis', '/team-comparison', '/ref-analysis', '/rank', '/_dash-layout', '/_dash-dependencies']:
    check(route, lambda route=route: require(client.get(route).status_code == 200))
pdf = Path('assets/tokyo-2020-match-reports.pdf').read_bytes()
check('source PDF served byte-for-byte', lambda: require(client.get('/assets/tokyo-2020-match-reports.pdf').data == pdf))
mods = {n:importlib.import_module('pages.' + n) for n in ['player','Team_Analysis','team_comparison','Ref_Analysis','rank']}
t = mods['Team_Analysis']
for team in sorted(t.players_master.Team.unique()):
    for result in ['All','W','L']:
        for opponent in ['All','TOP 8',None]:
            check(f'team table {team}/{result}/{opponent}', lambda team=team,result=result,opponent=opponent: t.update_tables(team,'All',result,opponent,'Minutes pg'))
            check(f'team charts {team}/{result}/{opponent}', lambda team=team,result=result,opponent=opponent: t.update_charts(team,'All',result,opponent))
for table in mods['team_comparison'].STAT_DIC:
    for mode in ['Average','Total']:
        def compare(table=table,mode=mode):
            _, data, _ = mods['team_comparison'].show_table(table,mode,[])
            mods['team_comparison'].create_figures(data)
        check(f'comparison {table}/{mode}', compare)
p = mods['player']
for team,name in p.df[['Team','Name']].drop_duplicates().itertuples(index=False,name=None):
    def player(team=team,name=name):
        data = p.filter_df(team,name)[0]
        p.make_tables(data,name)
        p.data_vis(team,name,'Goals')
    check(f'player {team}/{name}', player)
r = mods['Ref_Analysis']
for opponent in ['All','TOP 8','Knockout Rounds']:
    check(f'ref table {opponent}', lambda opponent=opponent:r.update_table('All',opponent,'Ref'))
    for ref in r.reflist:
        check(f'ref chart {ref}/{opponent}', lambda ref=ref,opponent=opponent:r.update_charts('All',opponent,ref))
for stat in ['Goals','Shots']:
    for mode in ['Total','Average']:
        for games in ['All','TOP 8','Knockout Rounds']:
            check(f'rank {stat}/{mode}/{games}', lambda stat=stat,mode=mode,games=games:mods['rank'].update_tables(stat,'All',games,mode))
data = pd.read_csv('data/master_data_refs.csv')
check('42 matches and 12 teams', lambda:require(data['Match Number'].nunique()==42 and data.Team.nunique()==12))
bowen = data[(data.Last=='BOWEN') & (data.Team=='USA') & (data['Match Number']==3)].iloc[0]
check('PDF example fields', lambda:require((bowen.Goals,bowen.Shots,bowen.Minutes,bowen.Seconds)==(5,6,21,13)))
players=data[data.stat_type=='Player'].groupby(['Match Number','Team']).Goals.sum().sort_index()
teams=data[data.stat_type=='Team'].set_index(['Match Number','Team']).Goals.sort_index()
check('player goals reconcile to all 84 team totals', lambda:require(players.equals(teams)))
print(json.dumps({'passed':checks,'failed':len(failures),'failures':failures},indent=2))
raise SystemExit(bool(failures))
