# NBA match-outcome model results

As of: **2026-11-10 15:19 UTC+08:00**.

## Split

- Training seasons: 2021-22, 2022-23, 2023-24, 2024-25 (4920 games used to fit).
- Hyperparameter selection: rolling-origin cross-validation inside the training window (train 2021-22 -> validate 2022-23; train 2021-23 -> validate 2023-24; train 2021-24 -> validate 2024-25), mean validation log-loss.
- Held-out test season: 2025-26 (1230 games).
- Forward application: 2026-27 (not used for fitting or model selection).
- Validation is temporal only. Rows were never randomly shuffled into train/test.

## Test-set metrics (2025-26 regular season)

| Model | Accuracy | Log-loss | Brier score | N |
|---|---:|---:|---:|---:|
| Baseline: always home (p = train home-win rate) | 0.5545 | 0.6872 | 0.2470 | 1230 |
| Baseline: higher pre-game season win % | 0.6699 | 0.7372 | 0.2224 | 1230 |
| Logistic regression (Lasso, L1) | 0.6748 | 0.6103 | 0.2105 | 1230 |
| Logistic regression (L2, all features) | 0.6650 | 0.6101 | 0.2105 | 1230 |
| PCA + logistic regression | 0.6642 | 0.6156 | 0.2129 | 1230 |
| Decision tree | 0.6488 | 0.6129 | 0.2126 | 1230 |
| k-nearest neighbours | 0.6683 | 0.6113 | 0.2114 | 1230 |
| Gaussian Naive Bayes | 0.6764 | 0.9741 | 0.2506 | 1230 |
| Random forest | 0.6764 | 0.5992 | 0.2058 | 1230 |

Lower log-loss and Brier scores are better. Accuracy uses a 0.5 probability threshold.

### Confusion matrix of the default model (test season, threshold 0.5)

| | Predicted away win | Predicted home win |
|---|---:|---:|
| Actual away win | 317 | 231 |
| Actual home win | 169 | 513 |

Precision 0.690, recall 0.752, specificity 0.578.

### Test games after both teams have played at least 5 games

| Model | Accuracy | Log-loss | Brier score | N |
|---|---:|---:|---:|---:|
| Baseline: always home (p = train home-win rate) | 0.5517 | 0.6878 | 0.2473 | 1151 |
| Baseline: higher pre-game season win % | 0.6768 | 0.6572 | 0.2184 | 1151 |
| Logistic regression (Lasso, L1) | 0.6768 | 0.6062 | 0.2088 | 1151 |
| Logistic regression (L2, all features) | 0.6699 | 0.6052 | 0.2085 | 1151 |
| PCA + logistic regression | 0.6699 | 0.6098 | 0.2106 | 1151 |
| Decision tree | 0.6542 | 0.6085 | 0.2106 | 1151 |
| k-nearest neighbours | 0.6725 | 0.6079 | 0.2097 | 1151 |
| Gaussian Naive Bayes | 0.6829 | 0.9332 | 0.2472 | 1151 |
| Random forest | 0.6838 | 0.5957 | 0.2041 | 1151 |

## Forward application on completed 2026-27 games

These games were not in the training or validation data.

| Model | Accuracy | Log-loss | Brier score | N |
|---|---:|---:|---:|---:|
| Baseline: always home (p = train home-win rate) | 0.5197 | 0.6946 | 0.2507 | 152 |
| Baseline: higher pre-game season win % | 0.6711 | 1.2456 | 0.2324 | 152 |
| Logistic regression (Lasso, L1) | 0.6447 | 0.6455 | 0.2254 | 152 |
| Logistic regression (L2, all features) | 0.6447 | 0.6512 | 0.2256 | 152 |
| PCA + logistic regression | 0.6513 | 0.6412 | 0.2211 | 152 |
| Decision tree | 0.6382 | 0.6297 | 0.2207 | 152 |
| k-nearest neighbours | 0.6382 | 0.6288 | 0.2190 | 152 |
| Gaussian Naive Bayes | 0.6250 | 1.3879 | 0.2836 | 152 |
| Random forest | 0.6513 | 0.6260 | 0.2182 | 152 |

## Selected hyperparameters

Scheme: rolling-origin cross-validation inside the training window (train 2021-22 -> validate 2022-23; train 2021-23 -> validate 2023-24; train 2021-24 -> validate 2024-25), mean validation log-loss.

| Model | Selected | Mean CV log-loss | Fold log-losses |
|---|---|---:|---|
| Logistic regression (Lasso, L1) | C = 0.03 | 0.6386 | 0.6647, 0.6303, 0.6207 |
| Logistic regression (L2, all features) | C = 0.03 | 0.6312 | 0.6563, 0.6219, 0.6155 |
| PCA + logistic regression | n_components = 10 | 0.6310 | 0.6546, 0.6201, 0.6183 |
| Decision tree | max_depth = 4, min_samples_leaf = 200 | 0.6468 | 0.6646, 0.6407, 0.6350 |
| k-nearest neighbours | k = 151 | 0.6347 | 0.6536, 0.6252, 0.6251 |
| Gaussian Naive Bayes | no hyperparameters | 0.9407 | 0.9689, 0.8753, 0.9779 |
| Random forest | max_depth = 8, min_samples_leaf = 25 | 0.6333 | 0.6520, 0.6259, 0.6218 |

The Lasso `C` uses the one-standard-error rule: the most regularised value whose mean CV log-loss is within one standard error of the minimum (minimum at C = 0.3). The grid now extends down to 0.001, so no selected value sits on the edge of its grid.

## Multicollinearity and why the default model is a Lasso

Season win %, point differential and their last-10 / last-5 versions measure the same thing (team strength), so they are strongly correlated. Variance inflation factors in the training window:

| Feature | VIF | L2 coefficient | Lasso coefficient |
|---|---:|---:|---:|
| Home games already played | 346.8 | -0.056 | -0.050 |
| Away games already played | 346.7 | -0.033 | +0.000 |
| Home season point differential | 16.8 | +0.123 | +0.287 |
| Away season point differential | 16.0 | -0.040 | -0.119 |
| Away season win % (pre-game) | 13.0 | -0.023 | +0.000 |
| Home season win % (pre-game) | 12.5 | +0.093 | +0.000 |
| Home last-10 win % | 11.0 | -0.145 | +0.000 |
| Away last-10 win % | 11.0 | +0.112 | +0.000 |
| Home last-10 point differential | 9.4 | +0.209 | +0.092 |
| Away last-10 point differential | 9.3 | -0.199 | -0.132 |
| Home last-10 offensive rating | 3.9 | +0.141 | +0.050 |
| Away last-10 offensive rating | 3.8 | -0.166 | -0.100 |
| Home last-10 defensive rating | 3.5 | -0.125 | -0.047 |
| Away last-5 win % | 3.3 | -0.058 | +0.000 |
| Home last-5 win % | 3.3 | +0.045 | +0.000 |
| Away last-10 defensive rating | 3.3 | +0.149 | +0.077 |
| Home team's home-court win % | 3.0 | +0.108 | +0.101 |
| Away rest days | 2.7 | -0.050 | +0.000 |
| Away team's road win % | 2.6 | -0.094 | -0.076 |
| Home rest days | 2.5 | +0.039 | +0.000 |
| Away on a back-to-back | 1.6 | +0.098 | +0.088 |
| Home on a back-to-back | 1.5 | -0.123 | -0.108 |
| Neutral-site game | 1.0 | -0.024 | +0.000 |

With every feature kept (L2), correlated features share one effect and individual signs become unstable: last-10 win % gets a negative coefficient next to a positive one on last-10 point differential. That does not mean winning hurts; it means the two columns cannot be separated. The L1 penalty sets redundant coefficients exactly to zero. It keeps 13 of 23 features and drops the win-percentage columns in favour of point differential, so every remaining team-strength effect has the expected direction, with essentially the same test log-loss as the L2 model. PCA tells the same story from the other side: the first 11 principal components carry 90% of the variance of the 23 standardised features (PC1 alone: 23.7%).

## Decision tree (course model, for interpretation)

```
|--- Home season point differential <= -3.69
|   |--- Away last-10 point differential <= -2.95
|   |   |--- class: 1
|   |--- Away last-10 point differential >  -2.95
|   |   |--- Home games already played <= 50.50
|   |   |   |--- Home last-10 defensive rating <= 113.88
|   |   |   |   |--- class: 0
|   |   |   |--- Home last-10 defensive rating >  113.88
|   |   |   |   |--- class: 0
|   |   |--- Home games already played >  50.50
|   |   |   |--- class: 0
|--- Home season point differential >  -3.69
|   |--- Away season point differential <= -1.45
|   |   |--- Away season point differential <= -6.01
|   |   |   |--- Home last-5 win % <= 0.55
|   |   |   |   |--- class: 1
|   |   |   |--- Home last-5 win % >  0.55
|   |   |   |   |--- class: 1
|   |   |--- Away season point differential >  -6.01
|   |   |   |--- Home last-10 point differential <= 3.37
|   |   |   |   |--- class: 1
|   |   |   |--- Home last-10 point differential >  3.37
|   |   |   |   |--- class: 1
|   |--- Away season point differential >  -1.45
|   |   |--- Home season point differential <= 1.29
|   |   |   |--- Away last-10 point differential <= 7.95
|   |   |   |   |--- class: 1
|   |   |   |--- Away last-10 point differential >  7.95
|   |   |   |   |--- class: 0
|   |   |--- Home season point differential >  1.29
|   |   |   |--- Away team's road win % <= 0.45
|   |   |   |   |--- class: 1
|   |   |   |--- Away team's road win % >  0.45
|   |   |   |   |--- class: 1
```

## 2026-27 season forecast (Monte Carlo, 20,000 simulated seasons)

- Eastern Conference favourite: **Detroit Pistons** (24.1% to win the East).
- Western Conference favourite: **Minnesota Timberwolves** (22.1% to win the West).
- Championship favourite: **Minnesota Timberwolves** (14.0%).
- Most common Finals: DET vs MIN (5.4% of simulations).

### Eastern Conference: projected final table

| # | Team | Now | Projected W-L | 10-90% wins | Playoffs | Conf. finals | Win conf. | Champion |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | Detroit Pistons | 7-3 | 52.3-29.7 | 45-60 | 98.6% | 41.3% | 24.1% | 11.5% |
| 2 | Boston Celtics | 8-2 | 52.0-30.0 | 44-59 | 98.2% | 35.4% | 18.7% | 8.2% |
| 3 | Philadelphia 76ers | 9-2 | 49.7-32.3 | 42-57 | 95.8% | 24.2% | 11.0% | 4.0% |
| 4 | Atlanta Hawks | 7-5 | 48.3-33.7 | 41-56 | 94.8% | 28.0% | 13.8% | 5.5% |
| 5 | Toronto Raptors | 5-5 | 47.4-34.6 | 39-55 | 92.7% | 32.9% | 18.4% | 8.3% |
| 6 | New York Knicks | 5-4 | 42.9-39.1 | 35-51 | 77.3% | 11.6% | 4.4% | 1.4% |
| 7 | Cleveland Cavaliers | 5-6 | 42.5-39.5 | 35-50 | 77.0% | 12.6% | 5.0% | 1.6% |
| 8 | Washington Wizards | 4-7 | 37.7-44.3 | 30-46 | 48.8% | 6.0% | 2.3% | 0.6% |
| 9 | Indiana Pacers | 3-8 | 35.3-46.7 | 28-43 | 35.6% | 3.5% | 1.1% | 0.3% |
| 10 | Miami Heat | 4-6 | 34.8-47.2 | 27-43 | 29.7% | 1.8% | 0.4% | 0.1% |
| 11 | Orlando Magic | 2-7 | 32.2-49.8 | 24-40 | 19.5% | 1.2% | 0.3% | 0.1% |
| 12 | Charlotte Hornets | 4-7 | 31.5-50.5 | 24-39 | 14.3% | 0.6% | 0.2% | 0.0% |
| 13 | Brooklyn Nets | 2-8 | 30.4-51.6 | 23-38 | 12.6% | 0.7% | 0.2% | 0.0% |
| 14 | Milwaukee Bucks | 1-9 | 26.9-55.1 | 20-35 | 4.6% | 0.3% | 0.1% | 0.0% |
| 15 | Chicago Bulls | 1-8 | 22.0-60.0 | 15-29 | 0.7% | 0.0% | 0.0% | 0.0% |

### Western Conference: projected final table

| # | Team | Now | Projected W-L | 10-90% wins | Playoffs | Conf. finals | Win conf. | Champion |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | Minnesota Timberwolves | 8-3 | 54.2-27.8 | 47-61 | 96.4% | 38.2% | 22.1% | 14.0% |
| 2 | Houston Rockets | 8-3 | 53.2-28.8 | 46-60 | 94.5% | 31.6% | 16.5% | 10.2% |
| 3 | Los Angeles Lakers | 10-2 | 53.2-28.8 | 46-60 | 93.9% | 24.6% | 11.8% | 6.6% |
| 4 | Oklahoma City Thunder | 8-2 | 51.4-30.6 | 44-59 | 89.2% | 20.3% | 9.2% | 5.0% |
| 5 | Portland Trail Blazers | 6-3 | 51.3-30.7 | 43-59 | 90.1% | 31.0% | 17.2% | 10.4% |
| 6 | San Antonio Spurs | 8-2 | 51.0-31.0 | 43-59 | 88.1% | 18.9% | 8.1% | 4.1% |
| 7 | Memphis Grizzlies | 7-3 | 49.9-32.1 | 42-58 | 85.8% | 22.1% | 10.9% | 6.4% |
| 8 | Golden State Warriors | 7-3 | 43.1-38.9 | 35-51 | 48.1% | 3.9% | 1.1% | 0.4% |
| 9 | Dallas Mavericks | 7-4 | 41.9-40.1 | 34-50 | 41.6% | 3.9% | 1.4% | 0.5% |
| 10 | Denver Nuggets | 5-4 | 40.3-41.7 | 32-48 | 32.6% | 2.5% | 0.8% | 0.3% |
| 11 | New Orleans Pelicans | 4-6 | 37.8-44.2 | 30-46 | 21.4% | 1.8% | 0.6% | 0.2% |
| 12 | LA Clippers | 2-7 | 35.6-46.4 | 28-44 | 14.1% | 1.1% | 0.4% | 0.1% |
| 13 | Phoenix Suns | 3-5 | 31.2-50.8 | 23-39 | 3.5% | 0.1% | 0.0% | 0.0% |
| 14 | Utah Jazz | 1-9 | 25.9-56.1 | 19-33 | 0.5% | 0.0% | 0.0% | 0.0% |
| 15 | Sacramento Kings | 1-9 | 24.0-58.0 | 17-31 | 0.2% | 0.0% | 0.0% | 0.0% |

## Logistic regression coefficients

Default model (Lasso). Coefficients are on standardized features (mean 0, variance 1 in the training window). A positive coefficient increases the log-odds of a home win. A coefficient of 0.000 means the Lasso removed the feature.

| Feature | Coefficient | Odds ratio (per +1 SD) |
|---|---:|---:|
| Home season point differential | 0.287 | 1.333 |
| Away last-10 point differential | -0.132 | 0.876 |
| Away season point differential | -0.119 | 0.888 |
| Home on a back-to-back | -0.108 | 0.898 |
| Home team's home-court win % | 0.101 | 1.106 |
| Away last-10 offensive rating | -0.100 | 0.905 |
| Home last-10 point differential | 0.092 | 1.096 |
| Away on a back-to-back | 0.088 | 1.092 |
| Away last-10 defensive rating | 0.077 | 1.080 |
| Away team's road win % | -0.076 | 0.927 |
| Home games already played | -0.050 | 0.951 |
| Home last-10 offensive rating | 0.050 | 1.051 |
| Home last-10 defensive rating | -0.047 | 0.954 |
| Neutral-site game | 0.000 | 1.000 |
| Home rest days | 0.000 | 1.000 |
| Away rest days | 0.000 | 1.000 |
| Home season win % (pre-game) | 0.000 | 1.000 |
| Away season win % (pre-game) | 0.000 | 1.000 |
| Home last-10 win % | 0.000 | 1.000 |
| Away last-10 win % | 0.000 | 1.000 |
| Home last-5 win % | 0.000 | 1.000 |
| Away last-5 win % | 0.000 | 1.000 |
| Away games already played | 0.000 | 1.000 |

## Notes and limitations

- Features for each game use only information from prior games (no same-game box-score leakage). A 40-game audit of pre-game season win percentage recomputed from raw logs found 0 mismatches.
- Injuries, travel distance, roster changes, and referee assignments are not in the model.
- Season-opening games have no in-season history; win percentage is set to 0.5 and rating-style features are filled with training-set means.
- Neutral-site games (NBA Cup knockout, Mexico City, Paris, and similar) are assigned the NBA's designated home team from the official schedule.
- Two 2026-27 games may still have been in progress at collection time and therefore are absent from completed-game logs; snapshots use the last official final only.
- The official 2026-27 regular-season schedule endpoint returned 1,206 games rather than the usual 1,230. Upcoming-game lists use the published schedule as returned.
- The website default model is the Lasso logistic regression because it is the course's linear-model family, is fully interpretable, and removes the redundant features. The random forest is not course material and is shown only as a comparison.
- Season forecast: last-10 offensive/defensive ratings are held at their current values during the simulation; standings ties are broken by point differential rather than the NBA's head-to-head rules; the two NBA Cup-dependent games per team are filled with random same-conference opponents; injuries and trades are not modelled. Early in the season the forecast leans on a small number of games, so the percentages will move a lot.
