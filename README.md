# DataDash — Water Polo Analytics

A personal Python project co-built by **Andrew Chun and U.S. Olympian Dylan Woodhead** in 2022–2023, restored in November 2026.

**[Explore the dashboard](https://datadash-e4i8.onrender.com/)** · **[View the source PDF](https://datadash-e4i8.onrender.com/assets/tokyo-2020-match-reports.pdf#page=5)**

## What we built

Published water polo match reports contain dense tables, combined statistics such as goals/shots, and match context outside the player rows. Andrew worked on Python extraction and cleaning scripts using Tabula, PyPDF2, pandas, NumPy and regular expressions. The shared Dash/Plotly dashboard turns the resulting CSV records into player, team and referee views.

The running dashboard contains **42 Tokyo 2020 men's matches and 12 national teams**, played in 2021. Separate historical extraction scripts handled other tournament formats; their outputs are not integrated here. This is a personal prototype, not an official team platform or a live results service.

## See the starting material

[The saved 84-page input PDF](assets/tokyo-2020-match-reports.pdf) is a prepared collection of official Tokyo 2020 match reports, credited to the Tokyo 2020 Organising Committee. It is not the complete publisher results book. On physical page 5, USA–Japan, Alex Bowen's row shows:

| PDF field | Structured record |
| --- | --- |
| BOWEN Alex | Last = BOWEN; First = Alex |
| Goals/shots = 5/6 | Goals = 5; Shots = 6 |
| Playing time = 21:13 | Minutes = 21; Seconds = 13 |
| Header: USA 15–13 JPN; Match 3 | Team = USA; Opponent = JPN; Outcome = W; Match Number = 3 |

These selected fields were checked against the saved PDF and CSV records in 2026. The historical extraction scripts were a separate local project with manual stages and cached intermediates. This repository runs saved dashboard data; it does not claim a fully reproducible ingestion pipeline.

## Run locally

Use Python 3.11 and an isolated environment:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

For hosting, run `gunicorn app:server`. Render builds with `pip install -r requirements.txt`; `.python-version` selects Python 3.11. The existing live service tracks `master`. Free hosting can take a minute to wake after inactivity.

## Pages

- Home: project context, source PDF and an extraction example.
- Player: team/player filters and game-level statistics.
- Team Analysis: results/opponent filters and team/player comparisons.
- Team Comparison: national-team totals and averages.
- Referee Analysis: descriptive game statistics, not evidence of referee quality or causality.
- Player Rankings: sortable statistics within the saved dataset.

## Restoration notes

The public `master` branch originally held a December 2022 deployment. A fuller local version survived with homepage and navigation changes that had not been committed. The November 2026 restoration recovers that version, updates runtime compatibility, replaces literal-dictionary `eval` calls, repairs selected empty-filter behavior, clarifies labels, and adds source context. Historical CSV files are preserved.

This remains an archival prototype. Extraction errors, incomplete names and historical calculation limitations may remain. Verify any statistic against the source reports before relying on it. No live data feed, predictive model, official adoption or business outcome is claimed.
