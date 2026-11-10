# Data sources

All game results and the current-season schedule used in this project come from the **NBA official statistics API** at [https://stats.nba.com](https://stats.nba.com), accessed with the open-source Python package [`nba_api`](https://github.com/swar/nba_api) (version 1.12.0). No paid API was used. No games, scores, or probabilities were fabricated.

## Endpoints

| Endpoint | Package class | What was downloaded |
|---|---|---|
| League game log, regular season | `nba_api.stats.endpoints.leaguegamelog.LeagueGameLog` | One row per team per completed (or live) regular-season game: date, matchup, win/loss, points, shooting, rebounds, assists, turnovers, plus-minus, and related box-score totals. |
| League schedule | `nba_api.stats.endpoints.scheduleleaguev2.ScheduleLeagueV2` | Official designated home and away teams, tip times, arena, game status (scheduled / in progress / final), and labels such as NBA Cup or Mexico City. Used to list upcoming games and to assign home/away on neutral-site games where both teams appear as `@` in the game log. |

Season parameter values: `2021-22`, `2022-23`, `2023-24`, `2024-25`, `2025-26`, `2026-27`. Game logs used `season_type_all_star="Regular Season"` only. Playoff and preseason games are not in the training data.

## Seasons collected

| Season | Completed regular-season games in the processed file | Date range |
|---|---:|---|
| 2021-22 | 1,230 | 19 Oct 2021 – 10 Apr 2022 |
| 2022-23 | 1,230 | 18 Oct 2022 – 9 Apr 2023 |
| 2023-24 | 1,230 | 24 Oct 2023 – 14 Apr 2024 |
| 2024-25 | 1,230 | 22 Oct 2024 – 13 Apr 2025 |
| 2025-26 | 1,230 | 21 Oct 2025 – 12 Apr 2026 |
| 2026-27 | 152 (season in progress) | 20 Oct 2026 – 9 Nov 2026 |

Two 2026-27 games that were still in progress at collection time (partial box scores, `WL` missing, team minutes below regulation) were excluded from the processed completed-game file and from team form snapshots. They remain on the website as in-progress games with **pre-game** estimates only.

The official 2026-27 regular-season schedule feed returned 1,206 entries rather than the usual 1,230. Upcoming-game lists follow the published schedule as returned.

## Raw files

Raw CSVs are stored under `data/raw/`. Collection metadata, including the exact as-of timestamp, is in `data/raw/collection_meta.json`. The pipeline is rerunnable with `python scripts/collect_data.py`.

## Citation (suggested)

NBA Statistics. *League game log and schedule*, regular seasons 2021-22 to 2026-27. stats.nba.com, accessed 10 November 2026 via `nba_api` 1.12.0.
