#!/usr/bin/env python3
"""Download NBA regular-season game logs and the current-season schedule.

Source: NBA official stats API (https://stats.nba.com) via the open-source
`nba_api` Python package. No scores or games are invented.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import leaguegamelog, scheduleleaguev2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from team_meta import SEASONS, CURRENT_SEASON  # noqa: E402

RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

HKT = timezone(timedelta(hours=8))


def _fetch_with_retries(fn, attempts: int = 4, pause: float = 1.5):
    last = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 — network/API can fail in many ways
            last = exc
            wait = pause * (i + 1)
            print(f"  retry {i + 1}/{attempts} after error: {exc} (sleep {wait:.1f}s)")
            time.sleep(wait)
    raise RuntimeError(f"Failed after {attempts} attempts: {last}") from last


def fetch_season_log(season: str) -> pd.DataFrame:
    print(f"Fetching regular-season game log: {season}")

    def _call():
        ep = leaguegamelog.LeagueGameLog(
            season=season,
            season_type_all_star="Regular Season",
            timeout=60,
        )
        return ep.get_data_frames()[0]

    df = _fetch_with_retries(_call)
    df = df.copy()
    df["SEASON"] = season
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    out = RAW / f"gamelog_{season.replace('-', '_')}.csv"
    df.to_csv(out, index=False)
    n_games = df["GAME_ID"].nunique()
    n_teams = df["TEAM_ABBREVIATION"].nunique()
    print(
        f"  saved {out.name}: {len(df)} team-games, {n_games} games, "
        f"{n_teams} teams, {df['GAME_DATE'].min().date()} → {df['GAME_DATE'].max().date()}"
    )
    time.sleep(0.7)
    return df


def fetch_schedule(season: str) -> pd.DataFrame:
    print(f"Fetching schedule: {season}")

    def _call():
        ep = scheduleleaguev2.ScheduleLeagueV2(season=season, timeout=60)
        return ep.get_data_frames()[0]

    raw = _fetch_with_retries(_call)
    keep = [
        "seasonYear",
        "gameDate",
        "gameId",
        "gameStatus",
        "gameStatusText",
        "gameDateTimeUTC",
        "gameLabel",
        "gameSubLabel",
        "gameSubtype",
        "isNeutral",
        "arenaName",
        "arenaCity",
        "homeTeam_teamId",
        "homeTeam_teamCity",
        "homeTeam_teamName",
        "homeTeam_teamTricode",
        "homeTeam_wins",
        "homeTeam_losses",
        "homeTeam_score",
        "awayTeam_teamId",
        "awayTeam_teamCity",
        "awayTeam_teamName",
        "awayTeam_teamTricode",
        "awayTeam_wins",
        "awayTeam_losses",
        "awayTeam_score",
    ]
    df = raw[[c for c in keep if c in raw.columns]].copy()
    df["game_date"] = pd.to_datetime(df["gameDate"], format="mixed")
    df["gid_prefix"] = df["gameId"].astype(str).str[:3]
    out = RAW / f"schedule_{season.replace('-', '_')}.csv"
    df.to_csv(out, index=False)
    print(f"  saved {out.name}: {len(df)} rows")
    # compact regular-season view
    reg = df[df["gid_prefix"] == "002"].copy()
    compact = RAW / f"schedule_{season.replace('-', '_')}_regular.csv"
    reg.to_csv(compact, index=False)
    print(
        f"  regular-season entries: {len(reg)}; "
        f"status counts: {reg['gameStatus'].value_counts().to_dict()}"
    )
    return df


def main() -> None:
    as_of = datetime.now(HKT)
    frames = []
    for season in SEASONS:
        frames.append(fetch_season_log(season))
    all_games = pd.concat(frames, ignore_index=True)
    all_path = RAW / "gamelog_all_regular_season.csv"
    all_games.to_csv(all_path, index=False)

    schedules = {}
    for season in SEASONS:
        schedules[season] = fetch_schedule(season)

    summary = {
        "source": "NBA official stats API (stats.nba.com) via nba_api",
        "package": "nba_api",
        "as_of_hkt": as_of.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "as_of_iso": as_of.isoformat(),
        "seasons": {},
        "current_schedule_season": CURRENT_SEASON,
        "notes": (
            "Regular-season LeagueGameLog only (season type = Regular Season). "
            "Schedules are used to flag official home/away for neutral-site games "
            "(NBA Cup knockout, Mexico City, Paris, etc.) where both teams appear as '@' in the log. "
            "No scores or games were fabricated."
        ),
    }
    for season, g in all_games.groupby("SEASON"):
        summary["seasons"][season] = {
            "team_games": int(len(g)),
            "games": int(g["GAME_ID"].nunique()),
            "teams": int(g["TEAM_ABBREVIATION"].nunique()),
            "min_date": str(g["GAME_DATE"].min().date()),
            "max_date": str(g["GAME_DATE"].max().date()),
        }
    summary["schedules"] = {}
    for season, schedule in schedules.items():
        if "gid_prefix" in schedule.columns:
            reg = schedule[schedule["gid_prefix"] == "002"]
            summary["schedules"][season] = {
                "n": int(len(reg)),
                "status_counts": {int(k): int(v) for k, v in reg["gameStatus"].value_counts().items()},
                "min_date": str(reg["game_date"].min().date()) if len(reg) else None,
                "max_date": str(reg["game_date"].max().date()) if len(reg) else None,
            }
    schedule = schedules[CURRENT_SEASON]

    meta_path = RAW / "collection_meta.json"
    meta_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {all_path} ({len(all_games)} rows) and {meta_path}")
    print(json.dumps(summary["seasons"], indent=2))


if __name__ == "__main__":
    main()
