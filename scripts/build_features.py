#!/usr/bin/env python3
"""Build leakage-free pre-game features from raw NBA game logs.

Every feature for game t uses only information from games strictly before t
for the two participating teams. Season statistics reset each season.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from team_meta import CURRENT_SEASON  # noqa: E402

RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

# Estimate possessions (Dean Oliver-style, no official tracking required).
POSS_FT_COEF = 0.44
OPENER_REST_DAYS = 7.0  # first game of a season has no prior NBA game in-season


def load_schedule_home_map() -> pd.DataFrame:
    """Official designated home team and neutral flag from season schedules."""
    frames = []
    for path in sorted(RAW.glob("schedule_*_regular.csv")):
        frames.append(pd.read_csv(path, dtype={"gameId": str}))
    if not frames:
        return pd.DataFrame(columns=["GAME_ID", "sched_home", "sched_away", "is_neutral", "arena"])
    sch = pd.concat(frames, ignore_index=True)
    sch["GAME_ID"] = sch["gameId"].astype(str).str.zfill(10)
    out = pd.DataFrame({
        "GAME_ID": sch["GAME_ID"],
        "sched_home": sch["homeTeam_teamTricode"],
        "sched_away": sch["awayTeam_teamTricode"],
        "is_neutral": sch["isNeutral"].fillna(False).astype(bool) if "isNeutral" in sch.columns else False,
        "arena": sch["arenaName"] if "arenaName" in sch.columns else "",
        "game_label": sch["gameLabel"] if "gameLabel" in sch.columns else "",
    }).drop_duplicates("GAME_ID")
    return out


def _prior_expanding_mean(group: pd.Series) -> pd.Series:
    return group.shift(1).expanding(min_periods=1).mean()


def _prior_rolling_mean(group: pd.Series, window: int) -> pd.Series:
    return group.shift(1).rolling(window, min_periods=1).mean()


def add_team_game_priors(tg: pd.DataFrame) -> pd.DataFrame:
    """Add pre-game and through-game rolling/season stats on team-game rows."""
    df = tg.sort_values(["TEAM_ABBREVIATION", "SEASON", "GAME_DATE", "GAME_ID"]).copy()

    df["win"] = (df["WL"] == "W").astype(int)
    df["is_home_matchup"] = df["MATCHUP"].str.contains(r"vs\.", regex=True).astype(int)
    df["GAME_ID"] = df["GAME_ID"].astype(str).str.zfill(10)
    sched = load_schedule_home_map()
    df = df.merge(sched, on="GAME_ID", how="left")
    df["is_home"] = np.where(
        df["sched_home"].notna(),
        (df["TEAM_ABBREVIATION"] == df["sched_home"]).astype(int),
        df["is_home_matchup"],
    )
    df["is_neutral"] = df["is_neutral"].fillna(False).astype(bool)
    df["opp_abbr"] = (
        df["MATCHUP"]
        .str.replace(r"^[A-Z]{3}\s+(vs\.|@)\s+", "", regex=True)
        .str.strip()
    )
    # For double-@ neutral games, opponent is the other team in the pair, already in opp_abbr.
    df["opp_pts"] = df["PTS"] - df["PLUS_MINUS"]
    df["point_diff"] = df["PLUS_MINUS"].astype(float)
    df["poss"] = df["FGA"] - df["OREB"] + df["TOV"] + POSS_FT_COEF * df["FTA"]
    # Avoid divide-by-zero on bizarre box scores.
    df["off_rtg"] = np.where(df["poss"] > 0, df["PTS"] / df["poss"] * 100.0, np.nan)
    df["def_rtg"] = np.where(df["poss"] > 0, df["opp_pts"] / df["poss"] * 100.0, np.nan)
    df["net_rtg"] = df["off_rtg"] - df["def_rtg"]

    g = df.groupby(["TEAM_ABBREVIATION", "SEASON"], sort=False)

    df["prev_date"] = g["GAME_DATE"].shift(1)
    df["rest_days"] = (df["GAME_DATE"] - df["prev_date"]).dt.days.astype(float)
    df["rest_days"] = df["rest_days"].fillna(OPENER_REST_DAYS)
    df["b2b"] = (df["rest_days"] == 1).astype(int)
    df["prior_games"] = g.cumcount()  # 0 on a team's first game of the season

    def add_prior(col: str, name: str) -> None:
        df[name] = g[col].transform(_prior_expanding_mean)

    def add_roll(col: str, window: int, name: str) -> None:
        df[name] = g[col].transform(lambda s, w=window: _prior_rolling_mean(s, w))

    add_prior("win", "win_pct")
    add_prior("point_diff", "pd")
    add_prior("PTS", "pts")
    add_prior("opp_pts", "opp_pts_avg")
    add_prior("off_rtg", "off_rtg")
    add_prior("def_rtg", "def_rtg")
    add_prior("FG_PCT", "fg_pct")
    add_prior("FG3_PCT", "fg3_pct")
    add_prior("FT_PCT", "ft_pct")
    add_prior("REB", "reb")
    add_prior("AST", "ast")
    add_prior("TOV", "tov")
    add_prior("poss", "poss_avg")

    for w in (5, 10):
        add_roll("win", w, f"l{w}_win_pct")
        add_roll("point_diff", w, f"l{w}_pd")
        add_roll("off_rtg", w, f"l{w}_off_rtg")
        add_roll("def_rtg", w, f"l{w}_def_rtg")
        add_roll("net_rtg", w, f"l{w}_net_rtg")
        add_roll("PTS", w, f"l{w}_pts")
        add_roll("opp_pts", w, f"l{w}_opp_pts")
        add_roll("FG_PCT", w, f"l{w}_fg_pct")
        add_roll("TOV", w, f"l{w}_tov")
        add_roll("REB", w, f"l{w}_reb")

    # Home / away splits from prior games of the same type this season.
    home_mask = df["is_home"] == 1
    df["home_win_pct"] = np.nan
    df["away_win_pct"] = np.nan
    df.loc[home_mask, "home_win_pct"] = (
        df.loc[home_mask]
        .groupby(["TEAM_ABBREVIATION", "SEASON"])["win"]
        .transform(_prior_expanding_mean)
    )
    df.loc[~home_mask, "away_win_pct"] = (
        df.loc[~home_mask]
        .groupby(["TEAM_ABBREVIATION", "SEASON"])["win"]
        .transform(_prior_expanding_mean)
    )
    # Carry last known split forward so an away game still has a home-split feature, and vice versa.
    df["home_win_pct"] = g["home_win_pct"].ffill()
    df["away_win_pct"] = g["away_win_pct"].ffill()
    # ffill includes the current game's outcome if we fill after assignment?
    # Careful: for a home game, home_win_pct is already the PRIOR mean (shift in expanding).
    # ffill then copies that prior value to later away games — good, it does not include the
    # current game because the value stored on the home row is already pre-game.
    # But wait: after the home game is played, the NEXT game should include this result in
    # the home split. Currently the stored home_win_pct on this row is pre-game, and ffill
    # would carry the stale pre-game split to the next away game, skipping this result.
    #
    # Fix: also store through-game home/away win% and ffill THAT for future games, while
    # keeping the pre-game value on the current row.
    df["home_win_pct_incl"] = np.nan
    df["away_win_pct_incl"] = np.nan
    df.loc[home_mask, "home_win_pct_incl"] = (
        df.loc[home_mask]
        .groupby(["TEAM_ABBREVIATION", "SEASON"])["win"]
        .transform(lambda s: s.expanding(min_periods=1).mean())
    )
    df.loc[~home_mask, "away_win_pct_incl"] = (
        df.loc[~home_mask]
        .groupby(["TEAM_ABBREVIATION", "SEASON"])["win"]
        .transform(lambda s: s.expanding(min_periods=1).mean())
    )
    # For pre-game feature on a home game, use prior home split (already in home_win_pct).
    # For pre-game feature on an away game, use last included home split from a previous home game.
    df["prev_home_split"] = g["home_win_pct_incl"].transform(lambda s: s.shift(1).ffill())
    df["prev_away_split"] = g["away_win_pct_incl"].transform(lambda s: s.shift(1).ffill())
    df["home_split"] = np.where(home_mask, df["home_win_pct"], df["prev_home_split"])
    df["away_split"] = np.where(~home_mask, df["away_win_pct"], df["prev_away_split"])

    # Through-this-game season/form stats (for snapshots and trajectory charts, not as
    # pre-game model features on the SAME game).
    df["incl_games"] = df["prior_games"] + 1
    df["incl_wins"] = g["win"].cumsum()
    df["incl_win_pct"] = df["incl_wins"] / df["incl_games"]
    df["incl_pd"] = g["point_diff"].transform(lambda s: s.expanding(min_periods=1).mean())
    df["incl_l10_win_pct"] = g["win"].transform(lambda s: s.rolling(10, min_periods=1).mean())
    df["incl_l10_pd"] = g["point_diff"].transform(lambda s: s.rolling(10, min_periods=1).mean())
    df["incl_l10_net_rtg"] = g["net_rtg"].transform(lambda s: s.rolling(10, min_periods=1).mean())
    df["incl_l10_off_rtg"] = g["off_rtg"].transform(lambda s: s.rolling(10, min_periods=1).mean())
    df["incl_l10_def_rtg"] = g["def_rtg"].transform(lambda s: s.rolling(10, min_periods=1).mean())
    df["incl_l5_win_pct"] = g["win"].transform(lambda s: s.rolling(5, min_periods=1).mean())
    df["incl_off_rtg"] = g["off_rtg"].transform(lambda s: s.expanding(min_periods=1).mean())
    df["incl_def_rtg"] = g["def_rtg"].transform(lambda s: s.expanding(min_periods=1).mean())
    df["incl_fg_pct"] = g["FG_PCT"].transform(lambda s: s.expanding(min_periods=1).mean())
    df["incl_pts"] = g["PTS"].transform(lambda s: s.expanding(min_periods=1).mean())
    df["incl_opp_pts"] = g["opp_pts"].transform(lambda s: s.expanding(min_periods=1).mean())
    df["incl_reb"] = g["REB"].transform(lambda s: s.expanding(min_periods=1).mean())
    df["incl_ast"] = g["AST"].transform(lambda s: s.expanding(min_periods=1).mean())
    df["incl_tov"] = g["TOV"].transform(lambda s: s.expanding(min_periods=1).mean())
    df["incl_l10_fg_pct"] = g["FG_PCT"].transform(lambda s: s.rolling(10, min_periods=1).mean())
    df["incl_l10_pts"] = g["PTS"].transform(lambda s: s.rolling(10, min_periods=1).mean())
    df["incl_l10_opp_pts"] = g["opp_pts"].transform(lambda s: s.rolling(10, min_periods=1).mean())
    df["incl_l10_tov"] = g["TOV"].transform(lambda s: s.rolling(10, min_periods=1).mean())
    df["incl_l10_reb"] = g["REB"].transform(lambda s: s.rolling(10, min_periods=1).mean())

    # Neutral priors when a team has no prior games this season.
    rate_defaults = {
        "win_pct": 0.5,
        "l5_win_pct": 0.5,
        "l10_win_pct": 0.5,
        "home_split": 0.5,
        "away_split": 0.5,
    }
    zero_defaults = [
        "pd", "pts", "opp_pts_avg", "off_rtg", "def_rtg", "fg_pct", "fg3_pct",
        "ft_pct", "reb", "ast", "tov", "poss_avg",
        "l5_pd", "l10_pd", "l5_off_rtg", "l10_off_rtg", "l5_def_rtg", "l10_def_rtg",
        "l5_net_rtg", "l10_net_rtg", "l5_pts", "l10_pts", "l5_opp_pts", "l10_opp_pts",
        "l5_fg_pct", "l10_fg_pct", "l5_tov", "l10_tov", "l5_reb", "l10_reb",
    ]
    # For first games, 0 ratings are unrealistic; fill rating-like fields with league
    # prior later at the game-table stage. Here mark them NaN and fill win% with 0.5.
    for col, val in rate_defaults.items():
        df[col] = df[col].fillna(val)
    for col in zero_defaults:
        if col in df.columns:
            # leave NaN for first game; filled with training-set means later if needed
            pass

    return df


def pair_games(team_games: pd.DataFrame) -> pd.DataFrame:
    """One row per game, home vs away, with pre-game features only."""
    home = team_games[team_games["is_home"] == 1].copy()
    away = team_games[team_games["is_home"] == 0].copy()

    # Sanity: every game should have exactly one home and one away row.
    home_ids = set(home["GAME_ID"])
    away_ids = set(away["GAME_ID"])
    both = home_ids & away_ids
    dropped = (home_ids | away_ids) - both
    if dropped:
        print(f"  dropping {len(dropped)} games without a clear home/away pair")

    cols_feat = [
        "GAME_ID", "SEASON", "GAME_DATE", "TEAM_ABBREVIATION", "TEAM_NAME", "TEAM_ID",
        "WL", "PTS", "opp_pts", "PLUS_MINUS", "win", "opp_abbr",
        "rest_days", "b2b", "prior_games", "is_neutral",
        "win_pct", "pd", "pts", "opp_pts_avg", "off_rtg", "def_rtg",
        "fg_pct", "fg3_pct", "ft_pct", "reb", "ast", "tov", "poss_avg",
        "l5_win_pct", "l5_pd", "l5_off_rtg", "l5_def_rtg", "l5_net_rtg",
        "l5_pts", "l5_opp_pts", "l5_fg_pct",
        "l10_win_pct", "l10_pd", "l10_off_rtg", "l10_def_rtg", "l10_net_rtg",
        "l10_pts", "l10_opp_pts", "l10_fg_pct", "l10_tov", "l10_reb",
        "home_split", "away_split",
        "MIN", "FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT", "FTM", "FTA", "FT_PCT",
        "OREB", "DREB", "REB", "AST", "STL", "BLK", "TOV", "PF",
    ]
    cols_feat = [c for c in cols_feat if c in team_games.columns]

    h = home[cols_feat].rename(columns={c: f"home_{c}" if c not in {"GAME_ID", "SEASON", "GAME_DATE"} else c for c in cols_feat})
    a = away[cols_feat].rename(columns={c: f"away_{c}" if c not in {"GAME_ID", "SEASON", "GAME_DATE"} else c for c in cols_feat})
    # avoid duplicate SEASON/GAME_DATE from away
    a = a.drop(columns=[c for c in ("SEASON", "GAME_DATE") if c in a.columns])

    games = h.merge(a, on="GAME_ID", how="inner")
    games["home_win"] = (games["home_WL"] == "W").astype(int)
    games["home_abbr"] = games["home_TEAM_ABBREVIATION"]
    games["away_abbr"] = games["away_TEAM_ABBREVIATION"]
    games["home_name"] = games["home_TEAM_NAME"]
    games["away_name"] = games["away_TEAM_NAME"]

    # Differentials (home − away), useful for a linear model.
    diff_pairs = [
        ("rest_days", "rest_days_diff"),
        ("win_pct", "win_pct_diff"),
        ("pd", "pd_diff"),
        ("l5_win_pct", "l5_win_pct_diff"),
        ("l5_pd", "l5_pd_diff"),
        ("l10_win_pct", "l10_win_pct_diff"),
        ("l10_pd", "l10_pd_diff"),
        ("l10_off_rtg", "l10_off_rtg_diff"),
        ("l10_def_rtg", "l10_def_rtg_diff"),
        ("l10_net_rtg", "l10_net_rtg_diff"),
        ("off_rtg", "off_rtg_diff"),
        ("def_rtg", "def_rtg_diff"),
        ("prior_games", "prior_games_diff"),
    ]
    for src, dest in diff_pairs:
        games[dest] = games[f"home_{src}"] - games[f"away_{src}"]

    if "home_is_neutral" in games.columns:
        games["is_neutral"] = games["home_is_neutral"].fillna(False).astype(int)
    else:
        games["is_neutral"] = 0
    games = games.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)
    return games


def fill_first_game_means(games: pd.DataFrame, train_seasons: list[str]) -> pd.DataFrame:
    """Fill NaN pre-game rating stats (season openers) with training-season means."""
    train = games[games["SEASON"].isin(train_seasons)]
    rating_cols = [
        c for c in games.columns
        if c.startswith(("home_", "away_")) and any(
            key in c for key in (
                "_pd", "_pts", "_off_rtg", "_def_rtg", "_net_rtg", "_fg_pct",
                "_fg3_pct", "_ft_pct", "_reb", "_ast", "_tov", "_poss", "_opp_pts",
            )
        )
        and not c.startswith(("home_PTS", "away_PTS", "home_PLUS", "away_PLUS",
                              "home_FGM", "away_FGM", "home_FGA", "away_FGA",
                              "home_FG3M", "away_FG3M", "home_FG3A", "away_FG3A",
                              "home_FTM", "away_FTM", "home_FTA", "away_FTA",
                              "home_OREB", "away_OREB", "home_DREB", "away_DREB",
                              "home_REB", "away_REB", "home_AST", "away_AST",
                              "home_STL", "away_STL", "home_BLK", "away_BLK",
                              "home_TOV", "away_TOV", "home_PF", "away_PF",
                              "home_MIN", "away_MIN", "home_FG_PCT", "away_FG_PCT",
                              "home_FG3_PCT", "away_FG3_PCT", "home_FT_PCT", "away_FT_PCT",
                              "home_opp_pts", "away_opp_pts"))
        or c.endswith("_diff") and any(k in c for k in ("pd", "rtg", "pts"))
    ]
    # Simpler: fill all feature-like float columns that are NaN only on openers.
    feature_like = [
        c for c in games.columns
        if c.startswith(("home_", "away_")) or c.endswith("_diff")
    ]
    exclude_prefixes = (
        "home_WL", "away_WL", "home_TEAM", "away_TEAM", "home_abbr", "away_abbr",
        "home_name", "away_name", "home_win", "home_PTS", "away_PTS", "home_PLUS",
        "away_PLUS", "home_FGM", "away_FGM", "home_FGA", "away_FGA", "home_FG3M",
        "away_FG3M", "home_FG3A", "away_FG3A", "home_FTM", "away_FTM", "home_FTA",
        "away_FTA", "home_OREB", "away_OREB", "home_DREB", "away_DREB", "home_REB",
        "away_REB", "home_AST", "away_AST", "home_STL", "away_STL", "home_BLK",
        "away_BLK", "home_TOV", "away_TOV", "home_PF", "away_PF", "home_MIN",
        "away_MIN", "home_FG_PCT", "away_FG_PCT", "home_FG3_PCT", "away_FG3_PCT",
        "home_FT_PCT", "away_FT_PCT", "home_opp_pts", "away_opp_pts", "home_win",
    )
    fill_cols = []
    for c in feature_like:
        if games[c].dtype == object:
            continue
        if any(c == p or c.startswith(p) for p in exclude_prefixes) and c not in {
            "home_pd", "away_pd", "home_pts", "away_pts", "home_opp_pts_avg",
            "away_opp_pts_avg", "home_off_rtg", "away_off_rtg", "home_def_rtg",
            "away_def_rtg", "home_fg_pct", "away_fg_pct", "home_reb", "away_reb",
            "home_ast", "away_ast", "home_tov", "away_tov",
        }:
            # still allow explicit pre-game feature names
            if not any(
                c.endswith(suf) or c.endswith(suf + "_diff") or f"_{suf}" in f"_{c}"
                for suf in (
                    "pd", "pts", "off_rtg", "def_rtg", "net_rtg", "fg_pct",
                    "fg3_pct", "ft_pct", "reb", "ast", "tov", "poss_avg",
                    "opp_pts_avg", "l5_pts", "l10_pts", "l5_opp_pts", "l10_opp_pts",
                    "l5_pd", "l10_pd", "l5_fg_pct", "l10_fg_pct", "l10_tov", "l10_reb",
                )
            ):
                continue
        fill_cols.append(c)

    means = train[fill_cols].mean(numeric_only=True)
    games[fill_cols] = games[fill_cols].fillna(means)
    # Any remaining diffs that are NaN because both sides were NaN → 0
    for c in games.columns:
        if c.endswith("_diff"):
            games[c] = games[c].fillna(0.0)
    return games, {k: float(v) for k, v in means.items() if pd.notna(v)}


def current_snapshots(team_games: pd.DataFrame) -> pd.DataFrame:
    """State of each team after its most recent completed game (current season preferred)."""
    cur = team_games[team_games["SEASON"] == CURRENT_SEASON]
    if cur.empty:
        cur = team_games
    # last game per team
    last = (
        cur.sort_values(["TEAM_ABBREVIATION", "GAME_DATE", "GAME_ID"])
        .groupby("TEAM_ABBREVIATION", as_index=False)
        .tail(1)
        .copy()
    )
    return last


def leakage_audit(team_games: pd.DataFrame, games: pd.DataFrame, n_checks: int = 40) -> dict:
    """Recompute a few pre-game stats from raw rows and compare to engineered features."""
    rng = np.random.default_rng(2233)
    completed = games.dropna(subset=["home_win"])
    sample = completed.sample(min(n_checks, len(completed)), random_state=2233)
    mismatches = []
    for _, row in sample.iterrows():
        team = row["home_abbr"]
        season = row["SEASON"]
        date = row["GAME_DATE"]
        gid = row["GAME_ID"]
        hist = team_games[
            (team_games["TEAM_ABBREVIATION"] == team)
            & (team_games["SEASON"] == season)
            & (
                (team_games["GAME_DATE"] < date)
                | ((team_games["GAME_DATE"] == date) & (team_games["GAME_ID"] < gid))
            )
        ]
        if len(hist) == 0:
            expected = 0.5
        else:
            expected = float(hist["win"].mean())
        got = float(row["home_win_pct"])
        if abs(expected - got) > 1e-9:
            mismatches.append(
                {
                    "game_id": str(gid),
                    "team": team,
                    "expected_win_pct": expected,
                    "got": got,
                    "prior_games": int(len(hist)),
                }
            )
    return {
        "checks": int(len(sample)),
        "win_pct_mismatches": mismatches,
        "ok": len(mismatches) == 0,
    }


def main() -> None:
    src = RAW / "gamelog_all_regular_season.csv"
    if not src.exists():
        raise SystemExit(f"Missing {src}; run scripts/collect_data.py first.")
    raw = pd.read_csv(src, parse_dates=["GAME_DATE"], dtype={"GAME_ID": str, "SEASON_ID": str})
    print(f"Loaded {len(raw)} team-games from {src}")

    # Drop in-progress or otherwise unofficial rows (WL is null; team minutes < regulation).
    raw["MIN"] = pd.to_numeric(raw["MIN"], errors="coerce")
    unfinished_ids = set(raw.loc[raw["WL"].isna() | (raw["MIN"] < 230), "GAME_ID"].astype(str))
    if unfinished_ids:
        print(f"Excluding {len(unfinished_ids)} unofficial/in-progress game(s): {sorted(unfinished_ids)}")
        raw = raw[~raw["GAME_ID"].astype(str).isin(unfinished_ids)].copy()

    team_games = add_team_game_priors(raw)
    team_path = PROCESSED / "team_games.csv"
    team_games.to_csv(team_path, index=False)
    print(f"Wrote {team_path} ({len(team_games)} rows)")

    games = pair_games(team_games)
    from team_meta import TRAIN_SEASONS

    games, fill_means = fill_first_game_means(games, TRAIN_SEASONS)
    games_path = PROCESSED / "games.csv"
    games.to_csv(games_path, index=False)
    print(f"Wrote {games_path} ({len(games)} games)")

    snaps = current_snapshots(team_games)
    snap_path = PROCESSED / "team_snapshots.csv"
    snaps.to_csv(snap_path, index=False)
    print(f"Wrote {snap_path} ({len(snaps)} teams)")

    # Form history for current season charts (plus a tail of the prior season if useful).
    form = team_games[team_games["SEASON"].isin([CURRENT_SEASON, "2025-26"])].copy()
    form_path = PROCESSED / "team_form_history.csv"
    form.to_csv(form_path, index=False)

    audit = leakage_audit(team_games, games)
    audit_path = PROCESSED / "leakage_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2))
    print(f"Leakage audit: {audit['checks']} checks, mismatches={len(audit['win_pct_mismatches'])}")

    (PROCESSED / "feature_fill_means.json").write_text(json.dumps(fill_means, indent=2))

    print("Games by season:")
    print(games.groupby("SEASON").size())
    print("Home win rate by season:")
    print(games.groupby("SEASON")["home_win"].mean())


if __name__ == "__main__":
    main()
