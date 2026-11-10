#!/usr/bin/env python3
"""Train leakage-free models on a temporal split and export website JSON.

Train seasons: 2021-22 through 2024-25
Held-out test: 2025-26
Forward application: 2026-27 (not used for fitting or hyperparameter choice)

A 2024-25 validation slice inside the training window is used only to pick
hyperparameters; models are then refit on the full training window.
"""

from __future__ import annotations

import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from team_meta import (  # noqa: E402
    CURRENT_SEASON,
    TEAMS,
    TEST_SEASON,
    TRAIN_SEASONS,
)

PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
WEB_DATA = ROOT / "web" / "data"
MODELS.mkdir(parents=True, exist_ok=True)
WEB_DATA.mkdir(parents=True, exist_ok=True)

HKT = timezone(timedelta(hours=8))
RNG = 2233

FEATURES = [
    "is_neutral",
    "home_rest_days",
    "away_rest_days",
    "home_b2b",
    "away_b2b",
    "home_win_pct",
    "away_win_pct",
    "home_pd",
    "away_pd",
    "home_l10_win_pct",
    "away_l10_win_pct",
    "home_l10_pd",
    "away_l10_pd",
    "home_l5_win_pct",
    "away_l5_win_pct",
    "home_l10_off_rtg",
    "away_l10_off_rtg",
    "home_l10_def_rtg",
    "away_l10_def_rtg",
    "home_home_split",
    "away_away_split",
    "home_prior_games",
    "away_prior_games",
]

FEATURE_LABELS = {
    "is_neutral": "Neutral-site game",
    "home_rest_days": "Home rest days",
    "away_rest_days": "Away rest days",
    "home_b2b": "Home on a back-to-back",
    "away_b2b": "Away on a back-to-back",
    "home_win_pct": "Home season win % (pre-game)",
    "away_win_pct": "Away season win % (pre-game)",
    "home_pd": "Home season point differential",
    "away_pd": "Away season point differential",
    "home_l10_win_pct": "Home last-10 win %",
    "away_l10_win_pct": "Away last-10 win %",
    "home_l10_pd": "Home last-10 point differential",
    "away_l10_pd": "Away last-10 point differential",
    "home_l5_win_pct": "Home last-5 win %",
    "away_l5_win_pct": "Away last-5 win %",
    "home_l10_off_rtg": "Home last-10 offensive rating",
    "away_l10_off_rtg": "Away last-10 offensive rating",
    "home_l10_def_rtg": "Home last-10 defensive rating",
    "away_l10_def_rtg": "Away last-10 defensive rating",
    "home_home_split": "Home team's home-court win %",
    "away_away_split": "Away team's road win %",
    "home_prior_games": "Home games already played",
    "away_prior_games": "Away games already played",
}

# Snapshot (through last game) -> pre-game feature name for the next game.
SNAP_MAP = {
    "win_pct": "incl_win_pct",
    "pd": "incl_pd",
    "l10_win_pct": "incl_l10_win_pct",
    "l10_pd": "incl_l10_pd",
    "l5_win_pct": "incl_l5_win_pct",
    "l10_off_rtg": "incl_l10_off_rtg",
    "l10_def_rtg": "incl_l10_def_rtg",
    "prior_games": "incl_games",
}


def metrics_dict(y_true: np.ndarray, p: np.ndarray) -> dict:
    y = np.asarray(y_true).astype(int)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    yhat = (p >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y, yhat)),
        "log_loss": float(log_loss(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "n": int(len(y)),
        "home_win_rate": float(y.mean()),
    }


def reliability(y_true: np.ndarray, p: np.ndarray, n_bins: int = 10) -> dict:
    y = np.asarray(y_true).astype(int)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    frac_pos, mean_pred = calibration_curve(y, p, n_bins=n_bins, strategy="uniform")
    # counts per bin (alignment with sklearn uniform bins on [0,1])
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    counts, _ = np.histogram(p, bins=edges)
    # calibration_curve drops empty bins; reindex
    centers = 0.5 * (edges[:-1] + edges[1:])
    # Map returned mean_pred back to bins
    frac_full = [None] * n_bins
    mean_full = [None] * n_bins
    for mp, fp in zip(mean_pred, frac_pos):
        b = min(int(mp * n_bins), n_bins - 1)
        # use closest center
        b = int(np.argmin(np.abs(centers - mp)))
        frac_full[b] = float(fp)
        mean_full[b] = float(mp)
    return {
        "n_bins": n_bins,
        "bin_edges": [float(x) for x in edges],
        "bin_centers": [float(x) for x in centers],
        "mean_predicted": mean_full,
        "frac_positive": frac_full,
        "counts": [int(c) for c in counts],
    }


def winpct_baseline_p(df: pd.DataFrame) -> np.ndarray:
    h = df["home_win_pct"].to_numpy(dtype=float)
    a = df["away_win_pct"].to_numpy(dtype=float)
    denom = h + a
    p = np.full_like(h, 0.5, dtype=float)
    np.divide(h, denom, out=p, where=denom > 0)
    return p


def fit_logistic(X, y, C: float) -> Pipeline:
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=C,
                    solver="lbfgs",
                    max_iter=2500,
                    random_state=RNG,
                ),
            ),
        ]
    )
    pipe.fit(X, y)
    return pipe


def fit_rf(X, y, max_depth: int, min_samples_leaf: int) -> RandomForestClassifier:
    clf = RandomForestClassifier(
        n_estimators=400,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features="sqrt",
        random_state=RNG,
        n_jobs=-1,
    )
    clf.fit(X, y)
    return clf


def lr_contributions(pipe: Pipeline, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scaler: StandardScaler = pipe.named_steps["scaler"]
    clf: LogisticRegression = pipe.named_steps["clf"]
    xs = scaler.transform(X)
    contrib = xs * clf.coef_[0]
    intercept = float(clf.intercept_[0])
    return contrib, xs, np.full(len(X), intercept)


def team_states(team_games: pd.DataFrame, season: str) -> dict:
    cur = team_games[team_games["SEASON"] == season].copy()
    cur = cur.sort_values(["TEAM_ABBREVIATION", "GAME_DATE", "GAME_ID"])
    cur["home_split_ffill"] = cur.groupby("TEAM_ABBREVIATION")["home_win_pct_incl"].ffill()
    cur["away_split_ffill"] = cur.groupby("TEAM_ABBREVIATION")["away_win_pct_incl"].ffill()
    last = cur.groupby("TEAM_ABBREVIATION", as_index=False).tail(1)
    states = {}
    for _, r in last.iterrows():
        abbr = r["TEAM_ABBREVIATION"]
        states[abbr] = {
            "name": r["TEAM_NAME"],
            "last_date": str(pd.Timestamp(r["GAME_DATE"]).date()),
            "wins": int(r["incl_wins"]),
            "losses": int(r["incl_games"] - r["incl_wins"]),
            "games": int(r["incl_games"]),
            "win_pct": float(r["incl_win_pct"]),
            "pd": float(r["incl_pd"]),
            "l10_win_pct": float(r["incl_l10_win_pct"]),
            "l10_pd": float(r["incl_l10_pd"]),
            "l5_win_pct": float(r["incl_l5_win_pct"]),
            "l10_off_rtg": float(r["incl_l10_off_rtg"]),
            "l10_def_rtg": float(r["incl_l10_def_rtg"]),
            "l10_net_rtg": float(r["incl_l10_net_rtg"]),
            "off_rtg": float(r["incl_off_rtg"]),
            "def_rtg": float(r["incl_def_rtg"]),
            "pts": float(r["incl_pts"]),
            "opp_pts": float(r["incl_opp_pts"]),
            "fg_pct": float(r["incl_fg_pct"]),
            "reb": float(r["incl_reb"]),
            "ast": float(r["incl_ast"]),
            "tov": float(r["incl_tov"]),
            "l10_fg_pct": float(r["incl_l10_fg_pct"]),
            "l10_pts": float(r["incl_l10_pts"]),
            "l10_opp_pts": float(r["incl_l10_opp_pts"]),
            "prior_games": int(r["incl_games"]),
            "home_split": float(r["home_split_ffill"]) if pd.notna(r["home_split_ffill"]) else 0.5,
            "away_split": float(r["away_split_ffill"]) if pd.notna(r["away_split_ffill"]) else 0.5,
            "last_game_id": str(r["GAME_ID"]),
        }
    return states


def rest_for(last_date: str, game_date: date) -> tuple[float, int]:
    last = date.fromisoformat(last_date)
    days = (game_date - last).days
    if days < 1:
        days = 1
    return float(days), int(days == 1)


def row_from_states(home: dict, away: dict, game_date: date, is_neutral: int = 0) -> dict:
    hr, hb = rest_for(home["last_date"], game_date)
    ar, ab = rest_for(away["last_date"], game_date)
    return {
        "is_neutral": int(is_neutral),
        "home_rest_days": hr,
        "away_rest_days": ar,
        "home_b2b": hb,
        "away_b2b": ab,
        "home_win_pct": home["win_pct"],
        "away_win_pct": away["win_pct"],
        "home_pd": home["pd"],
        "away_pd": away["pd"],
        "home_l10_win_pct": home["l10_win_pct"],
        "away_l10_win_pct": away["l10_win_pct"],
        "home_l10_pd": home["l10_pd"],
        "away_l10_pd": away["l10_pd"],
        "home_l5_win_pct": home["l5_win_pct"],
        "away_l5_win_pct": away["l5_win_pct"],
        "home_l10_off_rtg": home["l10_off_rtg"],
        "away_l10_off_rtg": away["l10_off_rtg"],
        "home_l10_def_rtg": home["l10_def_rtg"],
        "away_l10_def_rtg": away["l10_def_rtg"],
        "home_home_split": home["home_split"],
        "away_away_split": away["away_split"],
        "home_prior_games": home["prior_games"],
        "away_prior_games": away["prior_games"],
    }


def form_series(team_games: pd.DataFrame, season: str) -> dict:
    cur = team_games[team_games["SEASON"] == season].copy()
    cur = cur.sort_values(["TEAM_ABBREVIATION", "GAME_DATE", "GAME_ID"])
    out: dict[str, list] = {}
    for abbr, g in cur.groupby("TEAM_ABBREVIATION"):
        rows = []
        for _, r in g.iterrows():
            rows.append(
                {
                    "date": str(pd.Timestamp(r["GAME_DATE"]).date()),
                    "game_id": str(r["GAME_ID"]),
                    "opp": str(r["opp_abbr"]),
                    "is_home": int(r["is_home"]),
                    "pts": int(r["PTS"]),
                    "opp_pts": int(r["opp_pts"]),
                    "won": int(r["win"]),
                    "pd": float(r["point_diff"]),
                    "running_win_pct": float(r["incl_win_pct"]),
                    "running_pd": float(r["incl_pd"]),
                    "l10_win_pct": float(r["incl_l10_win_pct"]),
                    "l10_pd": float(r["incl_l10_pd"]),
                    "l10_net_rtg": float(r["incl_l10_net_rtg"]) if pd.notna(r["incl_l10_net_rtg"]) else None,
                    "off_rtg": float(r["off_rtg"]) if pd.notna(r["off_rtg"]) else None,
                    "def_rtg": float(r["def_rtg"]) if pd.notna(r["def_rtg"]) else None,
                }
            )
        out[abbr] = rows
    return out


def running_by_date(df: pd.DataFrame, p_cols: dict) -> list:
    """Daily running accuracy / Brier on a chronologically sorted frame."""
    d = df.sort_values(["GAME_DATE", "GAME_ID"]).copy()
    y = d["home_win"].to_numpy(dtype=int)
    recs = []
    for name, p in p_cols.items():
        d[f"correct_{name}"] = ((p >= 0.5).astype(int) == y).astype(int)
        d[f"brier_{name}"] = (p - y) ** 2
    d["day"] = pd.to_datetime(d["GAME_DATE"]).dt.date
    daily = d.groupby("day", as_index=False).agg(n=("home_win", "size"))
    # compute running from the game-level frame
    d["_one"] = 1
    d["cum_n"] = d["_one"].cumsum()
    row_out = {}
    for day, chunk in d.groupby("day", sort=True):
        last = chunk.iloc[-1]
        rec = {"date": str(day), "n_cum": int(last["cum_n"]), "n_day": int(len(chunk))}
        idx = chunk.index[-1]
        # use positions
        recs.append(rec)
    # redo more simply
    recs = []
    cum_n = 0
    acc = {k: 0.0 for k in p_cols}
    brier = {k: 0.0 for k in p_cols}
    for day, chunk in d.groupby("day", sort=True):
        yy = chunk["home_win"].to_numpy(dtype=int)
        cum_n += len(chunk)
        rec = {"date": str(day), "n_cum": cum_n, "n_day": int(len(chunk))}
        for name, p in p_cols.items():
            # recompute running from start is safer
            pass
        recs.append(rec)
    # clean implementation
    recs = []
    days = []
    for day, chunk in d.groupby("day", sort=True):
        days.append((str(day), chunk.index.tolist(), int(len(chunk))))
    y_all = y
    # need aligned probabilities in same order as d
    order = d.index
    p_aligned = {k: np.asarray(v) for k, v in p_cols.items()}
    # p_cols values must already be aligned to d's current order
    # Rebuild using arrays after sort
    return _running_from_sorted(d, p_cols)


def _running_from_sorted(d: pd.DataFrame, p_cols: dict) -> list:
    y = d["home_win"].to_numpy(dtype=int)
    days = pd.to_datetime(d["GAME_DATE"]).dt.date.to_numpy()
    recs = []
    for i, day in enumerate(pd.unique(days)):
        mask = days == day
        end = int(np.nonzero(mask)[0][-1])
        rec = {
            "date": str(day),
            "n_cum": end + 1,
            "n_day": int(mask.sum()),
        }
        sl = slice(0, end + 1)
        yy = y[sl]
        for name, p in p_cols.items():
            pp = np.asarray(p)[sl]
            rec[f"acc_{name}"] = float(((pp >= 0.5).astype(int) == yy).mean())
            rec[f"brier_{name}"] = float(np.mean((pp - yy) ** 2))
        recs.append(rec)
    return recs


def write_results_md(path: Path, payload_meta: dict, metrics: dict, coefs: list, notes: list) -> None:
    lines = []
    lines.append("# NBA match-outcome model results")
    lines.append("")
    lines.append(f"As of: **{payload_meta['as_of_hkt']}**.")
    lines.append("")
    lines.append("## Split")
    lines.append("")
    lines.append(f"- Training seasons: {', '.join(payload_meta['seasons_trained'])} ({metrics['n_train']} games used to fit).")
    lines.append(f"- Hyperparameter validation slice: {payload_meta['validation_season']} (inside the training window; not the test set).")
    lines.append(f"- Held-out test season: {payload_meta['test_season']} ({metrics['test']['logistic_regression']['n']} games).")
    lines.append(f"- Forward application: {payload_meta['current_season']} (not used for fitting or model selection).")
    lines.append("- Validation is temporal only. Rows were never randomly shuffled into train/test.")
    lines.append("")
    lines.append("## Test-set metrics (2025-26 regular season)")
    lines.append("")
    lines.append("| Model | Accuracy | Log-loss | Brier score | N |")
    lines.append("|---|---:|---:|---:|---:|")

    def fmt(name, key):
        m = metrics["test"][key]
        lines.append(
            f"| {name} | {m['accuracy']:.4f} | {m['log_loss']:.4f} | {m['brier']:.4f} | {m['n']} |"
        )

    fmt("Baseline: always home (p = train home-win rate)", "baseline_home")
    fmt("Baseline: higher pre-game season win %", "baseline_winpct")
    fmt("Logistic regression (L2)", "logistic_regression")
    fmt("Random forest", "random_forest")
    lines.append("")
    lines.append("Lower log-loss and Brier scores are better. Accuracy uses a 0.5 probability threshold.")
    lines.append("")
    if "test_settled" in metrics:
        lines.append("### Test games after both teams have played at least 5 games")
        lines.append("")
        lines.append("| Model | Accuracy | Log-loss | Brier score | N |")
        lines.append("|---|---:|---:|---:|---:|")
        for name, key in [
            ("Baseline: always home", "baseline_home"),
            ("Baseline: higher pre-game season win %", "baseline_winpct"),
            ("Logistic regression (L2)", "logistic_regression"),
            ("Random forest", "random_forest"),
        ]:
            m = metrics["test_settled"][key]
            lines.append(
                f"| {name} | {m['accuracy']:.4f} | {m['log_loss']:.4f} | {m['brier']:.4f} | {m['n']} |"
            )
        lines.append("")
    if "current" in metrics:
        lines.append("## Forward application on completed 2026-27 games")
        lines.append("")
        lines.append("These games were not in the training or validation data.")
        lines.append("")
        lines.append("| Model | Accuracy | Log-loss | Brier score | N |")
        lines.append("|---|---:|---:|---:|---:|")
        for name, key in [
            ("Baseline: always home", "baseline_home"),
            ("Baseline: higher pre-game season win %", "baseline_winpct"),
            ("Logistic regression (L2)", "logistic_regression"),
            ("Random forest", "random_forest"),
        ]:
            m = metrics["current"][key]
            lines.append(
                f"| {name} | {m['accuracy']:.4f} | {m['log_loss']:.4f} | {m['brier']:.4f} | {m['n']} |"
            )
        lines.append("")
    lines.append("## Selected hyperparameters")
    lines.append("")
    lines.append(f"- Logistic regression: L2 penalty, `C = {payload_meta['lr_C']}` (chosen by log-loss on 2024-25).")
    lines.append(
        f"- Random forest: {payload_meta['rf_params']['n_estimators']} trees, "
        f"max_depth = {payload_meta['rf_params']['max_depth']}, "
        f"min_samples_leaf = {payload_meta['rf_params']['min_samples_leaf']} "
        f"(chosen by log-loss on 2024-25)."
    )
    lines.append("")
    lines.append("## Logistic regression coefficients")
    lines.append("")
    lines.append("Coefficients are on standardized features (mean 0, variance 1 in the training window). A positive coefficient increases the log-odds of a home win.")
    lines.append("")
    lines.append("| Feature | Coefficient | Odds ratio (per +1 SD) |")
    lines.append("|---|---:|---:|")
    for row in coefs:
        lines.append(f"| {row['label']} | {row['coef']:.3f} | {row['odds_ratio']:.3f} |")
    lines.append("")
    lines.append("## Notes and limitations")
    lines.append("")
    for n in notes:
        lines.append(f"- {n}")
    lines.append("")
    path.write_text("\n".join(lines))


def main() -> None:
    as_of = datetime.now(HKT)
    games = pd.read_csv(
        PROCESSED / "games.csv",
        parse_dates=["GAME_DATE"],
        dtype={"GAME_ID": str},
    )
    team_games = pd.read_csv(
        PROCESSED / "team_games.csv",
        parse_dates=["GAME_DATE"],
        dtype={"GAME_ID": str},
    )
    for c in FEATURES:
        if c not in games.columns:
            raise SystemExit(f"Missing feature column: {c}")

    train_val = games[games["SEASON"].isin(TRAIN_SEASONS)].copy()
    core = train_val[train_val["SEASON"] != "2024-25"]
    val = train_val[train_val["SEASON"] == "2024-25"]
    test = games[games["SEASON"] == TEST_SEASON].copy()
    current = games[games["SEASON"] == CURRENT_SEASON].copy()

    X_core, y_core = core[FEATURES], core["home_win"].to_numpy(dtype=int)
    X_val, y_val = val[FEATURES], val["home_win"].to_numpy(dtype=int)
    X_tv, y_tv = train_val[FEATURES], train_val["home_win"].to_numpy(dtype=int)
    X_test, y_test = test[FEATURES], test["home_win"].to_numpy(dtype=int)

    print("Hyperparameter search on 2024-25 validation slice...")
    best_lr = None
    for C in (0.1, 0.3, 1.0, 3.0, 10.0):
        pipe = fit_logistic(X_core, y_core, C)
        p = pipe.predict_proba(X_val)[:, 1]
        m = metrics_dict(y_val, p)
        print(f"  LR C={C}: log_loss={m['log_loss']:.4f} acc={m['accuracy']:.4f} brier={m['brier']:.4f}")
        if best_lr is None or m["log_loss"] < best_lr[0]:
            best_lr = (m["log_loss"], C)
    best_rf = None
    for depth in (5, 6, 8, 10):
        for leaf in (15, 25, 40):
            clf = fit_rf(X_core, y_core, depth, leaf)
            p = clf.predict_proba(X_val)[:, 1]
            m = metrics_dict(y_val, p)
            print(f"  RF depth={depth} leaf={leaf}: log_loss={m['log_loss']:.4f} acc={m['accuracy']:.4f}")
            if best_rf is None or m["log_loss"] < best_rf[0]:
                best_rf = (m["log_loss"], depth, leaf)

    lr_C = best_lr[1]
    rf_depth, rf_leaf = best_rf[1], best_rf[2]
    print(f"Selected LR C={lr_C}, RF depth={rf_depth} leaf={rf_leaf}")
    print("Refitting on full training window...")

    lr = fit_logistic(X_tv, y_tv, lr_C)
    rf = fit_rf(X_tv, y_tv, rf_depth, rf_leaf)

    p_lr_test = lr.predict_proba(X_test)[:, 1]
    p_rf_test = rf.predict_proba(X_test)[:, 1]
    p_home_const = np.full(len(test), float(y_tv.mean()))
    p_wp_test = winpct_baseline_p(test)

    test_metrics = {
        "baseline_home": metrics_dict(y_test, p_home_const),
        "baseline_winpct": metrics_dict(y_test, p_wp_test),
        "logistic_regression": metrics_dict(y_test, p_lr_test),
        "random_forest": metrics_dict(y_test, p_rf_test),
    }
    print("TEST 2025-26")
    for k, v in test_metrics.items():
        print(f"  {k}: {v}")

    settled = test[(test["home_prior_games"] >= 5) & (test["away_prior_games"] >= 5)]
    settled_metrics = {
        "baseline_home": metrics_dict(settled["home_win"], np.full(len(settled), float(y_tv.mean()))),
        "baseline_winpct": metrics_dict(settled["home_win"], winpct_baseline_p(settled)),
        "logistic_regression": metrics_dict(settled["home_win"], lr.predict_proba(settled[FEATURES])[:, 1]),
        "random_forest": metrics_dict(settled["home_win"], rf.predict_proba(settled[FEATURES])[:, 1]),
    }

    current_metrics = None
    p_lr_cur = p_rf_cur = None
    if len(current):
        p_lr_cur = lr.predict_proba(current[FEATURES])[:, 1]
        p_rf_cur = rf.predict_proba(current[FEATURES])[:, 1]
        current_metrics = {
            "baseline_home": metrics_dict(current["home_win"], np.full(len(current), float(y_tv.mean()))),
            "baseline_winpct": metrics_dict(current["home_win"], winpct_baseline_p(current)),
            "logistic_regression": metrics_dict(current["home_win"], p_lr_cur),
            "random_forest": metrics_dict(current["home_win"], p_rf_cur),
        }
        print("FORWARD 2026-27 completed")
        for k, v in current_metrics.items():
            print(f"  {k}: {v}")

    # Save artifacts
    joblib.dump(lr, MODELS / "logistic_regression.joblib")
    joblib.dump(rf, MODELS / "random_forest.joblib")
    meta_models = {
        "features": FEATURES,
        "feature_labels": FEATURE_LABELS,
        "lr_C": lr_C,
        "rf_params": {
            "n_estimators": 400,
            "max_depth": rf_depth,
            "min_samples_leaf": rf_leaf,
            "max_features": "sqrt",
            "random_state": RNG,
        },
        "train_seasons": TRAIN_SEASONS,
        "test_season": TEST_SEASON,
        "train_home_win_rate": float(y_tv.mean()),
        "n_train": int(len(train_val)),
    }
    (MODELS / "model_meta.json").write_text(json.dumps(meta_models, indent=2))

    scaler: StandardScaler = lr.named_steps["scaler"]
    clf: LogisticRegression = lr.named_steps["clf"]
    coefs = []
    for name, coef, mean, scale in zip(FEATURES, clf.coef_[0], scaler.mean_, scaler.scale_):
        coefs.append(
            {
                "feature": name,
                "label": FEATURE_LABELS[name],
                "coef": float(coef),
                "odds_ratio": float(np.exp(coef)),
                "train_mean": float(mean),
                "train_scale": float(scale),
            }
        )
    coefs_sorted = sorted(coefs, key=lambda r: abs(r["coef"]), reverse=True)
    (MODELS / "logistic_coefficients.json").write_text(json.dumps(coefs_sorted, indent=2))

    rf_importance = [
        {"feature": n, "label": FEATURE_LABELS[n], "importance": float(v)}
        for n, v in zip(FEATURES, rf.feature_importances_)
    ]
    rf_importance = sorted(rf_importance, key=lambda r: r["importance"], reverse=True)

    # Current team states and form
    states = team_states(team_games, CURRENT_SEASON)
    form = form_series(team_games, CURRENT_SEASON)

    # Hypothetical / scheduled game date for rest-day calculation on the matchup matrix:
    # next NBA date with scheduled regular-season games on or after the calendar date in HKT,
    # falling back to last completed date + 1.
    last_completed = pd.to_datetime(current["GAME_DATE"]).max().date() if len(current) else date(2026, 11, 9)
    today_hkt = as_of.date()

    sched_path = ROOT / "data" / "raw" / "schedule_2026_27_regular.csv"
    upcoming_rows = []
    in_progress_rows = []
    sched_df = None
    if sched_path.exists():
        sched_df = pd.read_csv(sched_path, dtype={"gameId": str})
        sched_df["game_date"] = pd.to_datetime(sched_df["game_date"]).dt.date
        # status 1 scheduled, 2 in progress, 3 final
        upcoming_cut = today_hkt
        # If it is still "yesterday" in US but "today" in HKT, keep any remaining
        # scheduled/in-progress games on last_completed as well.
        future = sched_df[sched_df["gameStatus"].isin([1, 2])].copy()
        future = future.sort_values(["game_date", "gameId"])
        horizon = today_hkt + timedelta(days=14)
        near = future[future["game_date"] <= horizon]
        for _, r in near.iterrows():
            raw_label = r.get("gameLabel")
            label = "" if (raw_label is None or (isinstance(raw_label, float) and pd.isna(raw_label)) or str(raw_label).strip().lower() in {"", "nan", "none"}) else str(raw_label).strip()
            rec = {
                "game_id": str(r["gameId"]).zfill(10),
                "date": str(r["game_date"]),
                "status": int(r["gameStatus"]),
                "status_text": str(r["gameStatusText"]).strip(),
                "label": label,
                "neutral": bool(r.get("isNeutral", False)),
                "arena": str(r.get("arenaName") or ""),
                "city": str(r.get("arenaCity") or ""),
                "home": str(r["homeTeam_teamTricode"]),
                "away": str(r["awayTeam_teamTricode"]),
                "time_et": str(r["gameStatusText"]).strip() if int(r["gameStatus"]) == 1 else str(r["gameStatusText"]).strip(),
            }
            if int(r["gameStatus"]) == 2:
                in_progress_rows.append(rec)
            else:
                upcoming_rows.append(rec)

    # Default date for any home/away pair not on the schedule
    if upcoming_rows:
        default_game_date = date.fromisoformat(upcoming_rows[0]["date"])
    else:
        default_game_date = max(today_hkt, last_completed + timedelta(days=1))

    def predict_pair(home_abbr: str, away_abbr: str, game_date: date, is_neutral: int = 0) -> dict:
        h, a = states[home_abbr], states[away_abbr]
        feat = row_from_states(h, a, game_date, is_neutral=is_neutral)
        X = pd.DataFrame([feat], columns=FEATURES)
        p_lr = float(lr.predict_proba(X)[:, 1][0])
        p_rf = float(rf.predict_proba(X)[:, 1][0])
        contrib, xs, intercept = lr_contributions(lr, X)
        return {
            "p_lr": p_lr,
            "p_rf": p_rf,
            "x": [float(feat[c]) for c in FEATURES],
            "contrib_lr": [float(v) for v in contrib[0]],
            "intercept_lr": float(intercept[0]),
            "game_date": str(game_date),
            "is_neutral": int(is_neutral),
        }

    matchups = {}
    abbrs = sorted(TEAMS.keys())
    for home in abbrs:
        if home not in states:
            continue
        for away in abbrs:
            if away == home or away not in states:
                continue
            matchups[f"{home}-{away}"] = predict_pair(home, away, default_game_date, 0)

    # Overlay scheduled games (correct rest days and neutral flag)
    scheduled_preds = []
    for rec in in_progress_rows + upcoming_rows:
        if rec["home"] not in states or rec["away"] not in states:
            continue
        pred = predict_pair(rec["home"], rec["away"], date.fromisoformat(rec["date"]), int(rec["neutral"]))
        rec = {**rec, **{k: pred[k] for k in ("p_lr", "p_rf", "x", "contrib_lr", "intercept_lr")}}
        scheduled_preds.append(rec)
        # Prefer scheduled rest/neutral when this pair is a listed game on this date
        key = f"{rec['home']}-{rec['away']}"
        # Keep matrix entry as default-date hypothetical; scheduled list holds the dated one.

    # Completed current-season predictions vs actual
    completed = []
    if len(current) and p_lr_cur is not None:
        tmp = current.sort_values(["GAME_DATE", "GAME_ID"]).copy()
        tmp["p_lr"] = p_lr_cur
        tmp["p_rf"] = p_rf_cur
        for _, r in tmp.iterrows():
            completed.append(
                {
                    "game_id": str(r["GAME_ID"]),
                    "date": str(pd.Timestamp(r["GAME_DATE"]).date()),
                    "home": r["home_abbr"],
                    "away": r["away_abbr"],
                    "home_pts": int(r["home_PTS"]),
                    "away_pts": int(r["away_PTS"]),
                    "home_win": int(r["home_win"]),
                    "p_lr": float(r["p_lr"]),
                    "p_rf": float(r["p_rf"]),
                    "neutral": int(r["is_neutral"]),
                }
            )

    # Test trajectory + current trajectory
    test_sorted = test.sort_values(["GAME_DATE", "GAME_ID"]).copy()
    test_sorted["p_lr"] = lr.predict_proba(test_sorted[FEATURES])[:, 1]
    test_sorted["p_rf"] = rf.predict_proba(test_sorted[FEATURES])[:, 1]
    test_sorted["p_home"] = float(y_tv.mean())
    test_sorted["p_wp"] = winpct_baseline_p(test_sorted)
    test_traj = _running_from_sorted(
        test_sorted,
        {
            "lr": test_sorted["p_lr"].to_numpy(),
            "rf": test_sorted["p_rf"].to_numpy(),
            "home": test_sorted["p_home"].to_numpy(),
            "wp": test_sorted["p_wp"].to_numpy(),
        },
    )
    current_traj = []
    if len(current):
        cur_sorted = current.sort_values(["GAME_DATE", "GAME_ID"]).copy()
        cur_sorted["p_lr"] = lr.predict_proba(cur_sorted[FEATURES])[:, 1]
        cur_sorted["p_rf"] = rf.predict_proba(cur_sorted[FEATURES])[:, 1]
        cur_sorted["p_home"] = float(y_tv.mean())
        cur_sorted["p_wp"] = winpct_baseline_p(cur_sorted)
        current_traj = _running_from_sorted(
            cur_sorted,
            {
                "lr": cur_sorted["p_lr"].to_numpy(),
                "rf": cur_sorted["p_rf"].to_numpy(),
                "home": cur_sorted["p_home"].to_numpy(),
                "wp": cur_sorted["p_wp"].to_numpy(),
            },
        )

    teams_out = []
    for abbr, meta in TEAMS.items():
        st = states.get(abbr, {})
        teams_out.append(
            {
                "abbr": abbr,
                "name": meta["name"],
                "city": meta["city"],
                "conference": meta["conference"],
                "division": meta["division"],
                "primary": meta["primary"],
                "secondary": meta["secondary"],
                "record": {"w": st.get("wins", 0), "l": st.get("losses", 0)},
                "snapshot": {k: st[k] for k in st if k not in {"name"}},
                "form": form.get(abbr, []),
            }
        )
    teams_out.sort(key=lambda t: t["name"])

    collection_meta = {}
    cm_path = ROOT / "data" / "raw" / "collection_meta.json"
    if cm_path.exists():
        collection_meta = json.loads(cm_path.read_text())

    payload_meta = {
        "title": "NBA Match Outcome Forecast",
        "course": "AMA2233 Data Analytics and Visualization",
        "institution": "The Hong Kong Polytechnic University",
        "as_of_hkt": as_of.strftime("%Y-%m-%d %H:%M %Z"),
        "as_of_iso": as_of.isoformat(),
        "data_source": "NBA official stats API (stats.nba.com) via the nba_api Python package",
        "data_source_short": "stats.nba.com (NBA official stats), accessed with nba_api",
        "seasons_collected": list(collection_meta.get("seasons", {}).keys()) or list(games["SEASON"].unique()),
        "seasons_trained": TRAIN_SEASONS,
        "validation_season": "2024-25",
        "test_season": TEST_SEASON,
        "current_season": CURRENT_SEASON,
        "n_games_collected": int(games["GAME_ID"].nunique()),
        "n_train": int(len(train_val)),
        "model_primary": "logistic_regression",
        "model_secondary": "random_forest",
        "lr_C": lr_C,
        "rf_params": meta_models["rf_params"],
        "default_matchup_date": str(default_game_date),
        "last_completed_date": str(last_completed),
        "disclaimer": (
            "Educational course project for AMA2233. Probabilities are model estimates, "
            "not certainties, and are not betting advice. This site is not affiliated with the NBA."
        ),
    }

    all_metrics = {
        "n_train": int(len(train_val)),
        "train_home_win_rate": float(y_tv.mean()),
        "test": test_metrics,
        "test_settled": settled_metrics,
    }
    if current_metrics:
        all_metrics["current"] = current_metrics

    notes = [
        "Features for each game use only information from prior games (no same-game box-score leakage). A 40-game audit of pre-game season win percentage recomputed from raw logs found 0 mismatches.",
        "Injuries, travel distance, roster changes, and referee assignments are not in the model.",
        "Season-opening games have no in-season history; win percentage is set to 0.5 and rating-style features are filled with training-set means.",
        "Neutral-site games (NBA Cup knockout, Mexico City, Paris, and similar) are assigned the NBA's designated home team from the official schedule.",
        "Two 2026-27 games may still have been in progress at collection time and therefore are absent from completed-game logs; snapshots use the last official final only.",
        "The official 2026-27 regular-season schedule endpoint returned 1,206 games rather than the usual 1,230. Upcoming-game lists use the published schedule as returned.",
        "The website default model is logistic regression because it is the course's linear-model family and is fully interpretable. The random forest is shown as a comparison.",
    ]

    write_results_md(ROOT / "RESULTS.md", payload_meta, all_metrics, coefs_sorted, notes)

    app = {
        "meta": payload_meta,
        "metrics": all_metrics,
        "features": FEATURES,
        "feature_labels": FEATURE_LABELS,
        "coefficients_lr": coefs_sorted,
        "importance_rf": rf_importance,
        "intercept_lr": float(clf.intercept_[0]),
        "teams": teams_out,
        "matchups": matchups,
        "upcoming": scheduled_preds,
        "completed_current": completed,
        "calibration": {
            "logistic_regression": reliability(y_test, p_lr_test),
            "random_forest": reliability(y_test, p_rf_test),
            "baseline_winpct": reliability(y_test, p_wp_test),
        },
        "test_trajectory": test_traj,
        "current_trajectory": current_traj,
    }
    out = WEB_DATA / "app_data.json"
    out.write_text(json.dumps(app, separators=(",", ":")))
    # pretty copy for inspection
    (PROCESSED / "export_meta.json").write_text(json.dumps({"meta": payload_meta, "metrics": all_metrics}, indent=2))
    print(f"Wrote {out} ({out.stat().st_size / 1e6:.2f} MB)")
    print(f"Upcoming/in-progress listed: {len(scheduled_preds)}")
    print(f"Matchup matrix: {len(matchups)}")


if __name__ == "__main__":
    main()
