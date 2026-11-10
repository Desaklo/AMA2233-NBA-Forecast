# NBA match-outcome model results

As of: **2026-11-10 12:21 UTC+08:00**.

## Split

- Training seasons: 2021-22, 2022-23, 2023-24, 2024-25 (4920 games used to fit).
- Hyperparameter validation slice: 2024-25 (inside the training window; not the test set).
- Held-out test season: 2025-26 (1230 games).
- Forward application: 2026-27 (not used for fitting or model selection).
- Validation is temporal only. Rows were never randomly shuffled into train/test.

## Test-set metrics (2025-26 regular season)

| Model | Accuracy | Log-loss | Brier score | N |
|---|---:|---:|---:|---:|
| Baseline: always home (p = train home-win rate) | 0.5545 | 0.6872 | 0.2470 | 1230 |
| Baseline: higher pre-game season win % | 0.6699 | 0.7372 | 0.2224 | 1230 |
| Logistic regression (L2) | 0.6683 | 0.6104 | 0.2106 | 1230 |
| Random forest | 0.6764 | 0.5998 | 0.2060 | 1230 |

Lower log-loss and Brier scores are better. Accuracy uses a 0.5 probability threshold.

### Test games after both teams have played at least 5 games

| Model | Accuracy | Log-loss | Brier score | N |
|---|---:|---:|---:|---:|
| Baseline: always home | 0.5517 | 0.6878 | 0.2473 | 1151 |
| Baseline: higher pre-game season win % | 0.6768 | 0.6572 | 0.2184 | 1151 |
| Logistic regression (L2) | 0.6733 | 0.6055 | 0.2087 | 1151 |
| Random forest | 0.6846 | 0.5961 | 0.2043 | 1151 |

## Forward application on completed 2026-27 games

These games were not in the training or validation data.

| Model | Accuracy | Log-loss | Brier score | N |
|---|---:|---:|---:|---:|
| Baseline: always home | 0.5197 | 0.6946 | 0.2507 | 152 |
| Baseline: higher pre-game season win % | 0.6711 | 1.2456 | 0.2324 | 152 |
| Logistic regression (L2) | 0.6382 | 0.6526 | 0.2260 | 152 |
| Random forest | 0.6579 | 0.6269 | 0.2185 | 152 |

## Selected hyperparameters

- Logistic regression: L2 penalty, `C = 0.1` (chosen by log-loss on 2024-25).
- Random forest: 400 trees, max_depth = 10, min_samples_leaf = 25 (chosen by log-loss on 2024-25).

## Logistic regression coefficients

Coefficients are on standardized features (mean 0, variance 1 in the training window). A positive coefficient increases the log-odds of a home win.

| Feature | Coefficient | Odds ratio (per +1 SD) |
|---|---:|---:|
| Home last-10 point differential | 0.254 | 1.290 |
| Away last-10 point differential | -0.245 | 0.783 |
| Home last-10 win % | -0.200 | 0.819 |
| Away last-10 offensive rating | -0.187 | 0.830 |
| Away last-10 defensive rating | 0.168 | 1.183 |
| Away last-10 win % | 0.162 | 1.175 |
| Home last-10 offensive rating | 0.153 | 1.165 |
| Home last-10 defensive rating | -0.136 | 0.873 |
| Home on a back-to-back | -0.127 | 0.881 |
| Home season win % (pre-game) | 0.122 | 1.130 |
| Home team's home-court win % | 0.112 | 1.118 |
| Away on a back-to-back | 0.102 | 1.107 |
| Away team's road win % | -0.096 | 0.909 |
| Home season point differential | 0.081 | 1.084 |
| Away last-5 win % | -0.071 | 0.932 |
| Home games already played | -0.068 | 0.934 |
| Home last-5 win % | 0.058 | 1.059 |
| Away rest days | -0.049 | 0.953 |
| Away season win % (pre-game) | -0.048 | 0.953 |
| Home rest days | 0.038 | 1.038 |
| Neutral-site game | -0.025 | 0.975 |
| Away games already played | -0.022 | 0.978 |
| Away season point differential | 0.013 | 1.013 |

## Notes and limitations

- Features for each game use only information from prior games (no same-game box-score leakage). A 40-game audit of pre-game season win percentage recomputed from raw logs found 0 mismatches.
- Injuries, travel distance, roster changes, and referee assignments are not in the model.
- Season-opening games have no in-season history; win percentage is set to 0.5 and rating-style features are filled with training-set means.
- Neutral-site games (NBA Cup knockout, Mexico City, Paris, and similar) are assigned the NBA's designated home team from the official schedule.
- Two 2026-27 games may still have been in progress at collection time and therefore are absent from completed-game logs; snapshots use the last official final only.
- The official 2026-27 regular-season schedule endpoint returned 1,206 games rather than the usual 1,230. Upcoming-game lists use the published schedule as returned.
- The website default model is logistic regression because it is the course's linear-model family and is fully interpretable. The random forest is shown as a comparison.
