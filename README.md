# NBA Match Outcome Forecast

Course project for **AMA2233 Data Analytics and Visualization**, The Hong Kong Polytechnic University. The site estimates the probability that the home team wins an NBA game from leakage-free, pre-tip-off features, and presents the results in a small dashboard of charts.

This is an educational project. The estimates are not betting advice, and the site is not affiliated with the NBA.

## Open the website

Clone this repository and serve the static files in `web/`:

```bash
git clone https://github.com/Desaklo/AMA2233-NBA-Forecast.git
cd AMA2233-NBA-Forecast
python3 -m http.server 8765 --directory web
```

Then open [http://127.0.0.1:8765/](http://127.0.0.1:8765/) in a browser.

The same page is served by `python3 scripts/serve.py` (default port 8765). After the environment in the next section is installed, `.venv/bin/python scripts/serve.py` does the same thing. The site is a static page (`web/index.html`) that reads `web/data/app_data.json`. No login is required.

GitHub Actions can publish that same `web/` folder. The workflow is `.github/workflows/pages.yml`. After this branch is on `main`, turn on Pages under Settings → Pages → Source: GitHub Actions if it is not already enabled. The site will be at `https://desaklo.github.io/AMA2233-NBA-Forecast/`. Enabling Pages is optional; the local server above is enough to view the project.

## What you can do on the site

- Pick any home and away team, or click an upcoming (or in-progress) scheduled game.
- See the predicted winner and both win probabilities. Switch between the logistic regression (default, interpretable) and a random forest comparison.
- Switch among six charts: win probability, recent form, why this prediction (logistic contributions), calibration, season trajectory, and team statistical comparison.

## Reproduce the data and models

Requires Python 3.11+ (developed with 3.13) and a network connection to stats.nba.com.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/collect_data.py      # NBA official stats via nba_api
python scripts/build_features.py    # leakage-free pre-game features
python scripts/train_and_export.py  # train, evaluate, write web/data/app_data.json
python scripts/serve.py
```

Or run `scripts/run_all.sh`.

## Project layout

```
README.md
DATA_SOURCES.md            # citations and season coverage
RESULTS.md                 # metrics, split, coefficients, limitations
requirements.txt
scripts/
  collect_data.py
  build_features.py
  train_and_export.py
  serve.py
  run_all.sh
  team_meta.py
data/raw/                  # original API extracts
data/processed/            # feature tables
models/                    # joblib artifacts and coefficients
src/
web/                       # the website
previews/                  # static screenshots of the site
```

## Modeling in brief

- **Features** (known before tip-off only): rest days and back-to-backs, pre-game season win percentage and point differential, last-5 / last-10 form, estimated offensive and defensive ratings, home/road splits, a neutral-site flag.
- **Split**: train on 2021-22 through 2024-25; hyperparameters chosen on 2024-25 inside that window; test on 2025-26; apply forward to completed 2026-27 games.
- **Baselines**: always-home (probability = training home-win rate); higher pre-game season win percentage (renormalized into a probability).
- **Models**: L2 logistic regression and a random forest. Metrics: accuracy, log-loss, Brier score, plus a reliability diagram.

See `RESULTS.md` for the numbers and `DATA_SOURCES.md` for the source.
