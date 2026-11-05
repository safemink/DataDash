import dash
from dash import html, dcc

dash.register_page(__name__, path='/', order=0)

layout = html.Div(
    className='original-home',
    children=[
        html.Div(
            children=[
                html.Div(
                    children=[
                        html.Strong(
                            'Welcome to Andrew and Dylan\'s Water Polo Dashboard! ',
                            style={'font-size': '20px', 'font-weight': 'bold'}
                        ),
                        'This site, hosted by ',
                        html.A('Render,', href='https://render.com/'),
                        '''
                        is meant to aid in viewing and analyzing Water Polo statistics, specifically in international play. 
                        This site in its current form is more of a proof of concept and only includes data from the 
                        2020 Tokyo Olympics, but as more and better data is added to the database, and more features are incorporated, 
                        it will hopefully serve some use in the future. Generally speaking, the pages of the site are the following: 
                        '''
                    ],
                    style={'margin': '20px'}
                ),
                html.Div(
                    children=[
                        html.Div(children=[
                            html.Div([html.Strong('Home:'),
                                      ' A comprehensive hub for water polo enthusiasts providing up-to-the-minute water polo statistics, game analysis, and player information. A home for the community to engage, learn, and discuss all things water polo.'],
                                     style={'margin': '20px'}),
                            html.Div([html.Strong('Player:'),
                                      ' A detailed exploration of individual player statistics, performance, and history. Compare stats, track progress, and analyze performance trends of your favorite water polo players over the course of their careers.'],
                                     style={'margin': '20px'}),
                            html.Div([html.Strong('Team Analysis:'),
                                      ' Delve into in-depth statistics and data-driven insights of various water polo teams. This page features team performance analysis, strategic evaluations, and data visualization of team patterns over time.'],
                                     style={'margin': '20px'}),
                        ], style={'width': '50%', 'display': 'inline-block'}),

                        html.Div(children=[
                            html.Div([html.Strong('Team Comparison:'),
                                      ' A head-to-head comparison of different water polo teams. This interactive tool allows you to juxtapose team stats, compare their strengths and weaknesses, and predict potential outcomes based on historical data.'],
                                     style={'margin': '20px'}),
                            html.Div([html.Strong('Ref Analysis:'),
                                      ' An objective analysis of referees and their impact on the game. This page provides insight into ref performance, controversial calls, and their influence on the game outcomes. A unique perspective to understand the game beyond just the players and teams.'],
                                     style={'margin': '20px'}),
                            html.Div([html.Strong('Rank:'),
                                      ' A live ranking of water polo teams and players based on their performances. This page continuously updates to reflect the current standings, offering a clear picture of who leads the sport at any given time.'],
                                     style={'margin': '20px'}),
                        ], style={'width': '50%', 'display': 'inline-block'}),
                    ], style={'display': 'flex', 'margin': '40px'}),
                html.Div(
                    children=[
                        html.Strong(
                            'Important Limitations:',
                            style={'font-size': '20px', 'font-weight': 'bold'}
                        ),
                        '''Great effort was invested in ensuring the acquisition of high-quality and consistent data. 
                        However, a significant portion of the data was obtained through the challenging task of 
                        scraping PDF game logs, which presented various difficulties. It is conceivable that certain 
                        letters or numbers may have been illegible or misinterpreted during the data compilation 
                        process, potentially resulting in inaccuracies or errors. Nonetheless, we have diligently 
                        conducted thorough testing and retesting to address these issues. Therefore, 
                        we are optimistic that the majority of them have already been identified and rectified. '''
                    ],
                    style={'margin': '20px'}
                ),
                html.P([
                    'Archived Tokyo 2020 dataset · ',
                    html.A('View the original match reports (PDF)',
                           href='/assets/tokyo-2020-match-reports.pdf#page=5',
                           target='_blank', rel='noopener noreferrer'),
                    '.'
                ], style={'margin': '20px', 'font-size': '14px'})
            ]
        )
    ]
)


