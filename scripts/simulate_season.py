#!/usr/bin/env python3
"""Monte Carlo forecast of the rest of the 2026-27 season and the playoffs.

Every remaining regular-season game is simulated with the fitted logistic
model. After each simulated game the team's season record, point differential,
last-10 / last-5 form and home/road splits are updated, so later games are
scored with features that reflect the simulated season so far (the same
"pre-game information only" rule used for training).

The result of one simulation is a full 82-game table for both conferences.
The play-in tournament and a best-of-seven playoff bracket are then simulated
on top of it. Repeating this many times turns the game-level probabilities
into probabilities of making the playoffs, winning the conference and winning
the championship.

Simplifications (also listed in RESULTS.md):
  * Last-10 offensive / defensive ratings are held at their current values.
  * Standings ties are broken by season point differential, not the NBA's
    head-to-head rules.
  * The schedule endpoint lists 80 games per team; the last two depend on the
    NBA Cup. They are filled with random same-conference opponents (fixed seed).
  * Injuries, trades and playoff-specific effects are not modelled.
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from team_meta import CURRENT_SEASON, TEAMS  # noqa: E402

N_SIMS = 20000
SEED = 2233
PLAYOFF_REST_DAYS = 2.0
# Higher seed hosts games 1, 2, 5 and 7 of a best-of-seven series.
SERIES_HOME_PATTERN = (1, 1, 0, 0, 1, 0, 1)


def margin_sigma(p_home: np.ndarray, home_margin: np.ndarray) -> float:
    """Scale that links a win probability to a point margin.

    Model: margin = sigma * (z + e), z = Phi^-1(p), e ~ N(0, 1), so that
    P(margin > 0) = p. sigma is the least-squares slope of the observed home
    margin on z (through the origin) in the training seasons.
    """
    z = norm.ppf(np.clip(p_home, 1e-4, 1 - 1e-4))
    return float((z * home_margin).sum() / (z * z).sum())


def remaining_schedule(sched: pd.DataFrame, played_ids: set, rng: np.random.Generator) -> pd.DataFrame:
    """Unplayed regular-season games, padded to 82 games per team."""
    s = sched.copy()
    s["gameId"] = s["gameId"].astype(str).str.zfill(10)
    s["game_date"] = pd.to_datetime(s["game_date"]).dt.date
    known = s.dropna(subset=["homeTeam_teamTricode", "awayTeam_teamTricode"])
    todo = known[~known["gameId"].isin(played_ids)]
    rows = [
        {
            "date": r["game_date"],
            "home": r["homeTeam_teamTricode"],
            "away": r["awayTeam_teamTricode"],
            "neutral": int(bool(r.get("isNeutral", False))),
            "filler": 0,
        }
        for _, r in todo.iterrows()
    ]

    # Games per team in the published schedule; pad up to 82.
    counts = pd.concat([known["homeTeam_teamTricode"], known["awayTeam_teamTricode"]]).value_counts()
    need = {t: 82 - int(counts.get(t, 0)) for t in TEAMS}
    tbd = s[s["homeTeam_teamTricode"].isna()]
    base = min(tbd["game_date"]) if len(tbd) else date(2026, 12, 4)
    rnd = 0
    while any(v > 0 for v in need.values()):
        east = [t for t in TEAMS if TEAMS[t]["conference"] == "East" and need[t] > 0]
        west = [t for t in TEAMS if TEAMS[t]["conference"] == "West" and need[t] > 0]
        rng.shuffle(east)
        rng.shuffle(west)
        pairs = []
        for grp in (east, west):
            while len(grp) >= 2:
                pairs.append((grp.pop(), grp.pop()))
        left = east + west
        if len(left) == 2:
            pairs.append((left[0], left[1]))
        if not pairs:
            break
        d = base + timedelta(days=4 * rnd)
        for a, b in pairs:
            home, away = (a, b) if rnd % 2 == 0 else (b, a)
            rows.append({"date": d, "home": home, "away": away, "neutral": 0, "filler": 1})
            need[a] -= 1
            need[b] -= 1
        rnd += 1
    out = pd.DataFrame(rows).sort_values(["date", "filler", "home"]).reset_index(drop=True)
    return out


def simulate(
    lr_pipe,
    features: list,
    team_games: pd.DataFrame,
    sched: pd.DataFrame,
    sigma: float,
    n_sims: int = N_SIMS,
    seed: int = SEED,
) -> dict:
    rng = np.random.default_rng(seed)
    abbrs = sorted(TEAMS.keys())
    idx = {a: i for i, a in enumerate(abbrs)}
    T, N = len(abbrs), n_sims

    scaler = lr_pipe.named_steps["scaler"]
    clf = lr_pipe.named_steps["clf"]
    mean, scale = scaler.mean_, scaler.scale_
    coef, intercept = clf.coef_[0], float(clf.intercept_[0])

    # ---- current state from completed games --------------------------------
    cur = team_games[team_games["SEASON"] == CURRENT_SEASON].sort_values(["GAME_DATE", "GAME_ID"])
    played_ids = set(cur["GAME_ID"].astype(str).str.zfill(10))
    games = np.zeros(T, dtype=int)
    hg = np.zeros(T, dtype=int)
    ag = np.zeros(T, dtype=int)
    wins0 = np.zeros(T)
    sumpd0 = np.zeros(T)
    hw0 = np.zeros(T)
    aw0 = np.zeros(T)
    bufw0 = np.zeros((T, 10))
    bufm0 = np.zeros((T, 10))
    off = np.zeros(T)
    deff = np.zeros(T)
    last_date = [None] * T
    for abbr, g in cur.groupby("TEAM_ABBREVIATION"):
        t = idx[abbr]
        for _, r in g.iterrows():
            slot = games[t] % 10
            bufw0[t, slot] = r["win"]
            bufm0[t, slot] = r["point_diff"]
            games[t] += 1
            wins0[t] += r["win"]
            sumpd0[t] += r["point_diff"]
            if int(r["is_home"]) == 1:
                hg[t] += 1
                hw0[t] += r["win"]
            else:
                ag[t] += 1
                aw0[t] += r["win"]
        last = g.iloc[-1]
        off[t] = last["incl_l10_off_rtg"]
        deff[t] = last["incl_l10_def_rtg"]
        last_date[t] = pd.Timestamp(last["GAME_DATE"]).date()
    current_w = wins0.astype(int).copy()
    current_g = games.copy()

    wins = np.tile(wins0, (N, 1))
    sumpd = np.tile(sumpd0, (N, 1))
    hw = np.tile(hw0, (N, 1))
    aw = np.tile(aw0, (N, 1))
    bufw = np.tile(bufw0, (N, 1, 1))
    bufm = np.tile(bufm0, (N, 1, 1))

    def team_feats(t: int) -> dict:
        """Pre-game features of team t in every simulation (arrays of length N)."""
        g = games[t]
        k10 = max(min(g, 10), 1)
        k5 = max(min(g, 5), 1)
        last5 = [(g - 1 - j) % 10 for j in range(min(g, 5))]
        return {
            "win_pct": wins[:, t] / g if g else np.full(N, 0.5),
            "pd": sumpd[:, t] / g if g else np.zeros(N),
            "l10_win_pct": bufw[:, t, :].sum(axis=1) / k10,
            "l10_pd": bufm[:, t, :].sum(axis=1) / k10,
            "l5_win_pct": bufw[:, t, last5].sum(axis=1) / k5 if last5 else np.full(N, 0.5),
            "l10_off_rtg": np.full(N, off[t]),
            "l10_def_rtg": np.full(N, deff[t]),
            "home_split": hw[:, t] / hg[t] if hg[t] else np.full(N, 0.5),
            "away_split": aw[:, t] / ag[t] if ag[t] else np.full(N, 0.5),
            "prior_games": np.full(N, float(g)),
        }

    def p_home(fh: dict, fa: dict, rest_h, rest_a, b2b_h, b2b_a, neutral) -> np.ndarray:
        vals = {
            "is_neutral": neutral,
            "home_rest_days": rest_h,
            "away_rest_days": rest_a,
            "home_b2b": b2b_h,
            "away_b2b": b2b_a,
            "home_home_split": fh["home_split"],
            "away_away_split": fa["away_split"],
        }
        for k in ("win_pct", "pd", "l10_win_pct", "l10_pd", "l5_win_pct", "l10_off_rtg", "l10_def_rtg", "prior_games"):
            vals[f"home_{k}"] = fh[k]
            vals[f"away_{k}"] = fa[k]
        z = np.full(N, intercept)
        for j, name in enumerate(features):
            if coef[j] != 0.0:
                z = z + coef[j] * (np.asarray(vals[name], dtype=float) - mean[j]) / scale[j]
        return 1.0 / (1.0 + np.exp(-z))

    # ---- rest of the regular season ----------------------------------------
    todo = remaining_schedule(sched, played_ids, rng)
    for r in todo.itertuples(index=False):
        h, a = idx[r.home], idx[r.away]
        rest_h = max((r.date - last_date[h]).days, 1)
        rest_a = max((r.date - last_date[a]).days, 1)
        p = p_home(team_feats(h), team_feats(a), rest_h, rest_a, int(rest_h == 1), int(rest_a == 1), r.neutral)
        margin = sigma * (norm.ppf(np.clip(p, 1e-6, 1 - 1e-6)) + rng.standard_normal(N))
        hwin = (margin > 0).astype(float)
        for t, w, m, home_side in ((h, hwin, margin, True), (a, 1.0 - hwin, -margin, False)):
            slot = games[t] % 10
            bufw[:, t, slot] = w
            bufm[:, t, slot] = m
            wins[:, t] += w
            sumpd[:, t] += m
            if home_side:
                hw[:, t] += w
                hg[t] += 1
            else:
                aw[:, t] += w
                ag[t] += 1
            games[t] += 1
            last_date[t] = r.date

    # ---- playoffs -----------------------------------------------------------
    final = {t: team_feats(t) for t in range(T)}
    stack = {k: np.stack([final[t][k] for t in range(T)], axis=1) for k in final[0]}  # (N, T)
    rows = np.arange(N)

    def pick(team_idx: np.ndarray) -> dict:
        return {k: v[rows, team_idx] for k, v in stack.items()}

    def game(home_idx: np.ndarray, away_idx: np.ndarray) -> np.ndarray:
        """True where the home team wins a single playoff game."""
        p = p_home(pick(home_idx), pick(away_idx), PLAYOFF_REST_DAYS, PLAYOFF_REST_DAYS, 0, 0, 0)
        return rng.random(N) < p

    def series(hi: np.ndarray, lo: np.ndarray) -> np.ndarray:
        """Winner of a best-of-seven; `hi` has home-court advantage."""
        hi_wins = np.zeros(N, dtype=int)
        for at_hi in SERIES_HOME_PATTERN:
            if at_hi:
                hi_wins += game(hi, lo)
            else:
                hi_wins += ~game(lo, hi)
        return np.where(hi_wins >= 4, hi, lo)

    strength = wins + sumpd / 1e5  # wins, then point differential as tiebreak
    reach = {k: np.zeros(T) for k in ("top6", "playoffs", "round2", "conf_finals", "finals", "champion")}
    seed_sum = np.zeros(T)
    conf_champ = {}
    for conf in ("East", "West"):
        members = np.array([idx[a] for a in abbrs if TEAMS[a]["conference"] == conf])
        order = np.argsort(-strength[:, members], axis=1)
        seeds = members[order]  # (N, 15): seeds[:, 0] is the 1 seed
        for pos in range(seeds.shape[1]):
            np.add.at(seed_sum, seeds[:, pos], pos + 1)
        # Play-in: 7 hosts 8 (winner = 7 seed); 9 hosts 10; loser(7/8) hosts winner(9/10) for the 8 seed.
        s7, s8, s9, s10 = (seeds[:, i] for i in (6, 7, 8, 9))
        g78 = game(s7, s8)
        seed7 = np.where(g78, s7, s8)
        loser78 = np.where(g78, s8, s7)
        win910 = np.where(game(s9, s10), s9, s10)
        seed8 = np.where(game(loser78, win910), loser78, win910)
        bracket = [seeds[:, i] for i in range(6)] + [seed7, seed8]
        for i in range(6):
            np.add.at(reach["top6"], bracket[i], 1)
        for b in bracket:
            np.add.at(reach["playoffs"], b, 1)
        rank = {}  # team -> seed number, per simulation, for home court in later rounds
        seed_of = np.full((N, T), 99)
        for i, b in enumerate(bracket):
            seed_of[rows, b] = i + 1

        def ordered(x, y):
            x_hi = seed_of[rows, x] < seed_of[rows, y]
            return np.where(x_hi, x, y), np.where(x_hi, y, x)

        r1 = [series(bracket[0], bracket[7]), series(bracket[3], bracket[4]),
              series(bracket[1], bracket[6]), series(bracket[2], bracket[5])]
        for w in r1:
            np.add.at(reach["round2"], w, 1)
        r2 = [series(*ordered(r1[0], r1[1])), series(*ordered(r1[2], r1[3]))]
        for w in r2:
            np.add.at(reach["conf_finals"], w, 1)
        champ = series(*ordered(r2[0], r2[1]))
        np.add.at(reach["finals"], champ, 1)
        conf_champ[conf] = champ

    e, w = conf_champ["East"], conf_champ["West"]
    e_hi = strength[rows, e] >= strength[rows, w]
    nba_champ = series(np.where(e_hi, e, w), np.where(e_hi, w, e))
    np.add.at(reach["champion"], nba_champ, 1)

    # ---- summary ------------------------------------------------------------
    teams = []
    for a in abbrs:
        t = idx[a]
        wv = wins[:, t]
        teams.append(
            {
                "abbr": a,
                "name": TEAMS[a]["name"],
                "conference": TEAMS[a]["conference"],
                "current_w": int(current_w[t]),
                "current_l": int(current_g[t] - current_w[t]),
                "proj_w": float(wv.mean()),
                "proj_l": float(82 - wv.mean()),
                "w_p10": float(np.percentile(wv, 10)),
                "w_p90": float(np.percentile(wv, 90)),
                "avg_seed": float(seed_sum[t] / N),
                "p_top6": float(reach["top6"][t] / N),
                "p_playoffs": float(reach["playoffs"][t] / N),
                "p_round2": float(reach["round2"][t] / N),
                "p_conf_finals": float(reach["conf_finals"][t] / N),
                "p_conf_champ": float(reach["finals"][t] / N),
                "p_champion": float(reach["champion"][t] / N),
            }
        )
    tables = {}
    for conf in ("East", "West"):
        rows_c = sorted((r for r in teams if r["conference"] == conf), key=lambda r: -r["proj_w"])
        for i, r in enumerate(rows_c, start=1):
            r["proj_rank"] = i
        tables[conf] = [r["abbr"] for r in rows_c]

    def fav(key, conf=None):
        pool = [r for r in teams if conf is None or r["conference"] == conf]
        best = max(pool, key=lambda r: r[key])
        return {"abbr": best["abbr"], "name": best["name"], "p": best[key]}

    # Most common Finals pairing
    pair_codes = e.astype(int) * 100 + w.astype(int)
    vals, cnts = np.unique(pair_codes, return_counts=True)
    top = vals[np.argmax(cnts)]
    return {
        "season": CURRENT_SEASON,
        "n_sims": int(N),
        "seed": int(seed),
        "margin_sigma": float(sigma),
        "n_games_simulated": int(len(todo)),
        "n_filler_games": int(todo["filler"].sum()),
        "teams": teams,
        "tables": tables,
        "favorites": {
            "east": fav("p_conf_champ", "East"),
            "west": fav("p_conf_champ", "West"),
            "champion": fav("p_champion"),
            "finals_pair": {
                "east": abbrs[int(top // 100)],
                "west": abbrs[int(top % 100)],
                "p": float(cnts.max() / N),
            },
        },
    }


def forecast_markdown(fc: dict) -> list:
    by = {r["abbr"]: r for r in fc["teams"]}
    f = fc["favorites"]
    lines = [
        f"## {fc['season']} season forecast (Monte Carlo, {fc['n_sims']:,} simulated seasons)",
        "",
        f"- Eastern Conference favourite: **{f['east']['name']}** ({f['east']['p']:.1%} to win the East).",
        f"- Western Conference favourite: **{f['west']['name']}** ({f['west']['p']:.1%} to win the West).",
        f"- Championship favourite: **{f['champion']['name']}** ({f['champion']['p']:.1%}).",
        f"- Most common Finals: {f['finals_pair']['east']} vs {f['finals_pair']['west']} ({f['finals_pair']['p']:.1%} of simulations).",
        "",
    ]
    for conf in ("East", "West"):
        lines += [
            f"### {conf}ern Conference: projected final table",
            "",
            "| # | Team | Now | Projected W-L | 10-90% wins | Playoffs | Conf. finals | Win conf. | Champion |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for i, a in enumerate(fc["tables"][conf], start=1):
            r = by[a]
            lines.append(
                f"| {i} | {r['name']} | {r['current_w']}-{r['current_l']} | {r['proj_w']:.1f}-{r['proj_l']:.1f} | "
                f"{r['w_p10']:.0f}-{r['w_p90']:.0f} | {r['p_playoffs']:.1%} | {r['p_conf_finals']:.1%} | "
                f"{r['p_conf_champ']:.1%} | {r['p_champion']:.1%} |"
            )
        lines.append("")
    return lines


if __name__ == "__main__":
    import json

    import joblib

    sys.path.insert(0, str(ROOT / "scripts"))
    from train_and_export import FEATURES, TRAIN_SEASONS

    lr = joblib.load(ROOT / "models" / "logistic_regression.joblib")
    games = pd.read_csv(ROOT / "data" / "processed" / "games.csv", dtype={"GAME_ID": str})
    tg = pd.read_csv(ROOT / "data" / "processed" / "team_games.csv", parse_dates=["GAME_DATE"], dtype={"GAME_ID": str})
    sched = pd.read_csv(ROOT / "data" / "raw" / "schedule_2026_27_regular.csv", dtype={"gameId": str})
    tr = games[games["SEASON"].isin(TRAIN_SEASONS)]
    sig = margin_sigma(lr.predict_proba(tr[FEATURES])[:, 1], tr["home_PLUS_MINUS"].to_numpy(dtype=float))
    fc = simulate(lr, FEATURES, tg, sched, sig)
    (ROOT / "web" / "data" / "season_forecast.json").write_text(json.dumps(fc, indent=1))
    print("\n".join(forecast_markdown(fc)))
