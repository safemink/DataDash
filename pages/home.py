import dash
from dash import html

dash.register_page(__name__, path='/', name='Home', order=0)
PDF = '/assets/tokyo-2020-match-reports.pdf'

def link(label, href, button=False):
    return html.A(label, href=href, className='project-button' if button else None,
                  target='_blank' if href.startswith('https:') or '.pdf' in href else None,
                  rel='noopener noreferrer')

layout = html.Main([
    html.P('PERSONAL PROJECT · 2022–2023 · RESTORED NOVEMBER 2026', className='project-eyebrow'),
    html.H1("Andrew and Dylan’s Water Polo Dashboard"),
    html.P('Turning tournament PDF reports into interactive player and team comparisons.', className='project-lead'),
    html.P('Co-built by Andrew Chun and U.S. Olympian Dylan Woodhead. This Python project brings together '
           'data extraction, cleaning and visualization for the Tokyo 2020 men’s water polo tournament, played in 2021.'),
    html.Div([
        link('Explore team statistics', '/team-analysis', True),
        link('View source PDF · 84 pages', PDF, True),
        link('View project code', 'https://github.com/safemink/DataDash/tree/master'),
    ], className='project-actions'),
    html.P('42 matches · 12 national teams · Historical dataset', className='project-meta'),
    html.Section([
        html.H2('From a PDF report to usable data'),
        html.P('The starting point was a set of dense match-report tables, with match details in page headers '
               'and several statistics packed into single cells. Andrew worked on Python extraction and cleaning '
               'scripts to turn these reports into structured records for the shared dashboard.'),
        html.Ol([
            html.Li([html.Strong('Extract: '), 'Tabula reads table regions using page and column coordinates; '
                     'PyPDF2 and regular expressions recover match, team, score and referee details.']),
            html.Li([html.Strong('Normalize: '), 'pandas and NumPy separate combined fields, distinguish player '
                     'rows from team totals, and attach opponent and result information.']),
            html.Li([html.Strong('Explore: '), 'Cleaned CSV records feed Dash and Plotly filters, tables and charts.']),
        ]),
        html.H3('One example: USA vs Japan, July 25, 2021'),
        html.P(['USA won 15–13. ', link('Open the original match report on PDF page 5', PDF + '#page=5'),
                ' and look for Alex Bowen’s row.']),
        html.Div(html.Table([
            html.Thead(html.Tr([html.Th('In the report'), html.Th('Structured fields'), html.Th('What this enables')])),
            html.Tbody([
                html.Tr([html.Td('BOWEN Alex'), html.Td('Last: BOWEN · First: Alex'), html.Td('Filter by player')]),
                html.Tr([html.Td('Goals / shots: 5/6'), html.Td('Goals: 5 · Shots: 6'), html.Td('Aggregate attempts and scoring')]),
                html.Tr([html.Td('Playing time: 21:13'), html.Td('Minutes: 21 · Seconds: 13'), html.Td('Calculate time-based measures')]),
                html.Tr([html.Td('Header: USA 15–13 JPN · Match 3'), html.Td('USA · JPN · Win · Match 3'), html.Td('Join game context to each player row')]),
            ]),
        ]), className='project-table-wrap'),
        html.P('The linked PDF is the saved 84-page collection used by the project, containing official Tokyo 2020 '
               'match reports. It is a prepared collection, not the complete Olympic results book. Report content '
               'is credited to the Tokyo 2020 Organising Committee. This selected-field example was checked against '
               'the saved PDF and CSV records during the 2026 restoration.', className='project-note'),
    ], id='source-data'),
    html.Section([
        html.H2('Explore the dashboard'),
        html.Div([
            html.Div([html.H3(link(title, href)), html.P(description)], className='project-card')
            for title, href, description in [
                ('Player', '/player', 'Choose a team and player to explore game-by-game statistics.'),
                ('Team Analysis', '/team-analysis', 'Filter results and opponents, then compare team and player measures.'),
                ('Team Comparison', '/team-comparison', 'Compare national teams using totals or per-game averages.'),
                ('Referee Analysis', '/ref-analysis', 'Explore statistics from games associated with each referee.'),
                ('Player Rankings', '/rank', 'Sort players by a selected statistic within this historical dataset.'),
            ]
        ], className='project-grid'),
    ]),
    html.Section([
        html.H2('Scope and limitations'),
        html.P('This is an independent personal-project prototype. It uses saved Tokyo 2020 data and does not '
               'update with live results. PDF extraction and historical calculations can contain errors; use the '
               'source reports to check a statistic before relying on it. Referee comparisons describe games '
               'they officiated and do not establish referee quality, bias or causal impact.'),
        html.P('Separate historical scripts also handled World Championship, European Championship and World League '
               'report layouts. Those tournament outputs are not included in this dashboard. The original pipeline '
               'used manual stages and cached intermediates; this restoration runs the saved dashboard CSVs.'),
        html.P('The project was built in 2022–2023. The November 2026 restoration updates presentation, compatibility '
               'and selected interaction bugs while preserving the historical dataset.'),
    ]),
], className='project-home')
