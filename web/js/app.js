/* NBA Match Outcome Forecast — AMA2233 */
const HOME = "#0072B2";
const AWAY = "#E69F00";
const TEAL = "#009E73";
const VERM = "#D55E00";
const SKY = "#56B4E9";
const PURPLE = "#CC79A7";
const INK = "#1C1916";
const MUTED = "#6B6459";
const PAPER = "#F3EFE6";
const SURFACE = "#FFFDF8";

const PLOT_LAYOUT = {
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(0,0,0,0)",
  font: { family: "Source Sans 3, sans-serif", color: INK, size: 13 },
  margin: { l: 64, r: 28, t: 36, b: 56 },
  hovermode: "closest",
  legend: { orientation: "h", y: 1.12, x: 0, font: { size: 12 } },
};

const PLOT_CONFIG = { displayModeBar: false, responsive: true };

let DATA = null;
let activeChart = "prob";
let trajMode = "teams";
let selectedGameId = null;

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => [...document.querySelectorAll(sel)];

function teamByAbbr(abbr) {
  return DATA.teams.find((t) => t.abbr === abbr);
}

function modelKey() {
  const el = document.querySelector('input[name="model"]:checked');
  return el ? el.value : "lr";
}

function pHomeOf(pred) {
  return modelKey() === "rf" ? pred.p_rf : pred.p_lr;
}

function fmtPct(p) {
  return `${(p * 100).toFixed(1)}%`;
}

function fmtDate(iso) {
  const [y, m, d] = iso.split("-").map(Number);
  const dt = new Date(Date.UTC(y, m - 1, d));
  return dt.toLocaleDateString("en-US", { month: "short", day: "numeric", timeZone: "UTC" });
}

function wpBaseline(homeSnap, awaySnap) {
  const h = homeSnap.win_pct;
  const a = awaySnap.win_pct;
  const denom = h + a;
  return denom > 0 ? h / denom : 0.5;
}

function scheduledPred(home, away) {
  const list = DATA.upcoming || [];
  const matches = list.filter((g) => g.home === home && g.away === away);
  if (!matches.length) return null;
  if (selectedGameId) {
    const hit = matches.find((g) => g.game_id === selectedGameId);
    if (hit) return hit;
  }
  return matches[0];
}

function matrixPred(home, away) {
  return DATA.matchups[`${home}-${away}`] || null;
}

function currentPred(home, away) {
  return scheduledPred(home, away) || matrixPred(home, away);
}

function populateTeams() {
  const teams = DATA.teams.slice().sort((a, b) => a.name.localeCompare(b.name));
  for (const id of ["home-select", "away-select"]) {
    const sel = $(`#${id}`);
    sel.innerHTML = teams
      .map((t) => `<option value="${t.abbr}">${t.city === "LA" || t.city === "Los Angeles" || t.city === "Golden State" ? t.name : `${t.city} ${t.name.split(" ").slice(-1)}`} — ${t.abbr}</option>`)
      .join("");
  }
  // Cleaner option labels: official name
  for (const id of ["home-select", "away-select"]) {
    const sel = $(`#${id}`);
    sel.innerHTML = teams.map((t) => `<option value="${t.abbr}">${t.name}</option>`).join("");
  }
}

function setChip(id, abbr) {
  const t = teamByAbbr(abbr);
  const el = $(id);
  if (!t) return;
  el.style.background = t.primary;
}

function recordText(abbr) {
  const t = teamByAbbr(abbr);
  if (!t) return "";
  const s = t.snapshot;
  const pd = s.pd;
  const pdStr = `${pd > 0 ? "+" : ""}${pd.toFixed(1)} PD`;
  return `${t.conference} · ${s.wins}–${s.losses} · ${pdStr}`;
}

function renderUpcoming() {
  const strip = $("#game-strip");
  const games = DATA.upcoming || [];
  if (!games.length) {
    strip.innerHTML = `<p class="muted">No published upcoming games were available. Any home/away pair can still be scored from current form.</p>`;
    return;
  }
  strip.innerHTML = games
    .map((g) => {
      const p = g.p_lr;
      const home = teamByAbbr(g.home);
      const away = teamByAbbr(g.away);
      const status =
        g.status === 2
          ? `<span class="badge">${escapeHtml(g.status_text || "In progress")}</span>`
          : `<span>${escapeHtml(g.time_et || "")}</span>`;
      const label = g.label ? ` · ${escapeHtml(g.label)}` : "";
      return `<button type="button" class="game-card" role="listitem" data-game-id="${g.game_id}" data-home="${g.home}" data-away="${g.away}">
        <div class="when"><span>${fmtDate(g.date)}${label}</span>${status}</div>
        <div class="match">${away ? away.abbr : g.away} @ ${home ? home.abbr : g.home}</div>
        <div class="mini-bar" title="Logistic pre-game estimate"><span style="width:${(1 - p) * 100}%"></span><span style="width:${p * 100}%"></span></div>
      </button>`;
    })
    .join("");
  strip.querySelectorAll(".game-card").forEach((btn) => {
    btn.addEventListener("click", () => {
      selectedGameId = btn.dataset.gameId;
      $("#home-select").value = btn.dataset.home;
      $("#away-select").value = btn.dataset.away;
      refresh();
    });
  });
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function highlightStrip() {
  $$(".game-card").forEach((btn) => {
    const on =
      btn.dataset.home === $("#home-select").value &&
      btn.dataset.away === $("#away-select").value &&
      (!selectedGameId || btn.dataset.gameId === selectedGameId);
    btn.classList.toggle("is-active", on);
  });
}

function refresh() {
  const home = $("#home-select").value;
  const away = $("#away-select").value;
  if (home === away) {
    $("#winner-line").textContent = "Choose two different teams.";
    $("#hero-note").textContent = "";
    $("#away-pct").textContent = "—";
    $("#home-pct").textContent = "—";
    return;
  }
  setChip("#home-chip", home);
  setChip("#away-chip", away);
  $("#home-record").textContent = recordText(home);
  $("#away-record").textContent = recordText(away);
  $("#home-who").textContent = teamByAbbr(home).name;
  $("#away-who").textContent = teamByAbbr(away).name;

  const pred = currentPred(home, away);
  if (!pred) {
    $("#winner-line").textContent = "No estimate is available for this pair.";
    return;
  }
  const p = pHomeOf(pred);
  const homeFav = p >= 0.5;
  const fav = homeFav ? teamByAbbr(home) : teamByAbbr(away);
  const favP = homeFav ? p : 1 - p;
  $("#winner-line").innerHTML = `<span style="color:${homeFav ? HOME : AWAY}">${fav.name}</span> are favoured at ${fmtPct(favP)}`;

  $("#prob-home-fill").style.width = `${p * 100}%`;
  $("#prob-away-fill").style.width = `${(1 - p) * 100}%`;
  $("#home-pct").textContent = fmtPct(p);
  $("#away-pct").textContent = fmtPct(1 - p);

  const sched = scheduledPred(home, away);
  const modelName = modelKey() === "rf" ? "random forest" : "logistic regression";
  let context = "Model estimate — not a certainty";
  let note = `Default game date for unscheduled pairs is ${fmtDate(DATA.meta.default_matchup_date)}. Rest days are computed from each team's last completed game.`;
  if (sched) {
    if (sched.status === 2) {
      context = `In progress · pre-game ${modelName} estimate`;
      note = `${teamByAbbr(away).abbr} @ ${teamByAbbr(home).abbr} at ${sched.arena || sched.city} (${sched.status_text}). The live score is not used; features come from games completed before tip-off.`;
    } else {
      context = `Scheduled ${fmtDate(sched.date)} · ${sched.time_et || ""} · ${modelName}`.replace(/ · $/, "");
      const bits = [`${teamByAbbr(away).abbr} @ ${teamByAbbr(home).abbr}`];
      if (sched.arena) bits.push(sched.arena);
      if (sched.label) bits.push(sched.label);
      if (sched.neutral) bits.push("neutral site");
      note = `${bits.join(" · ")}. Probabilities are pre-game model estimates, not certainties.`;
    }
  } else {
    context = `Hypothetical matchup · ${modelName}`;
  }
  $("#hero-context").textContent = context;
  $("#hero-note").textContent = note;
  highlightStrip();
  drawChart();
}

function chartCaption(text) {
  $("#chart-caption").textContent = text;
}

function layoutExtras(extra) {
  return Object.assign({}, PLOT_LAYOUT, extra);
}

function drawChart() {
  const home = $("#home-select").value;
  const away = $("#away-select").value;
  const pred = currentPred(home, away);
  const sub = $("#chart-sub");
  const strip = $("#form-strip");
  sub.hidden = true;
  sub.innerHTML = "";
  strip.hidden = true;
  strip.innerHTML = "";
  if (!pred || home === away) {
    Plotly.purge("chart");
    return;
  }
  const chartEl = document.getElementById("chart");
  if (activeChart !== "cmp") chartEl.style.height = "";
  const drawers = { prob: drawProb, form: drawForm, why: drawWhy, cal: drawCal, traj: drawTraj, cmp: drawCmp };
  drawers[activeChart](home, away, pred);
}

function drawProb(home, away, pred) {
  const ht = teamByAbbr(home);
  const at = teamByAbbr(away);
  const pHomeConst = DATA.metrics.train_home_win_rate;
  const pWp = wpBaseline(ht.snapshot, at.snapshot);
  const models = [
    { name: "Logistic regression", p: pred.p_lr, color: HOME },
    { name: "Random forest", p: pred.p_rf, color: TEAL },
    { name: "Win% baseline", p: pWp, color: AWAY },
    { name: "Home-court baseline", p: pHomeConst, color: MUTED },
  ];
  const traces = [
    {
      type: "bar",
      name: at.abbr,
      x: models.map((m) => m.name),
      y: models.map((m) => 1 - m.p),
      marker: { color: AWAY },
      hovertemplate: `${at.abbr} %{y:.1%}<extra></extra>`,
    },
    {
      type: "bar",
      name: ht.abbr,
      x: models.map((m) => m.name),
      y: models.map((m) => m.p),
      marker: { color: HOME },
      hovertemplate: `${ht.abbr} %{y:.1%}<extra></extra>`,
    },
  ];
  Plotly.newPlot(
    "chart",
    traces,
    layoutExtras({
      barmode: "stack",
      barnorm: "fraction",
      yaxis: { title: "Win probability", tickformat: ".0%", range: [0, 1] },
      xaxis: { tickangle: 0 },
      margin: { l: 64, r: 20, t: 28, b: 64 },
      annotations: models.map((m, i) => ({
        x: m.name,
        y: 1.02,
        text: "",
        showarrow: false,
      })),
    }),
    PLOT_CONFIG
  );
  chartCaption(
    `${at.abbr} (orange, bottom) versus ${ht.abbr} (blue, top). The win-percentage baseline renormalizes the two teams' current season win percentages. The home-court baseline is the training-set home win rate (${fmtPct(pHomeConst)}).`
  );
}

function drawForm(home, away) {
  const ht = teamByAbbr(home);
  const at = teamByAbbr(away);
  const traces = [
    {
      type: "scatter",
      mode: "lines+markers",
      name: at.abbr,
      x: at.form.map((r) => r.date),
      y: at.form.map((r) => r.l10_net_rtg),
      line: { color: AWAY, width: 2.4 },
      marker: { size: 7, color: AWAY },
      hovertemplate: `${at.abbr} %{x}<br>Last-10 net rating: %{y:.1f}<extra></extra>`,
    },
    {
      type: "scatter",
      mode: "lines+markers",
      name: ht.abbr,
      x: ht.form.map((r) => r.date),
      y: ht.form.map((r) => r.l10_net_rtg),
      line: { color: HOME, width: 2.4 },
      marker: { size: 7, color: HOME },
      hovertemplate: `${ht.abbr} %{x}<br>Last-10 net rating: %{y:.1f}<extra></extra>`,
    },
  ];
  Plotly.newPlot(
    "chart",
    traces,
    layoutExtras({
      yaxis: { title: "Last-10 net rating (points per 100 possessions)", zeroline: true, zerolinecolor: "#bbb" },
      xaxis: { title: "Game date, 2026–27", tickformat: "%b %d" },
      shapes: [
        { type: "line", xref: "paper", x0: 0, x1: 1, y0: 0, y1: 0, line: { color: "#b7af9f", width: 1, dash: "dot" } },
      ],
    }),
    PLOT_CONFIG
  );
  renderWlStrip(home, away);
  chartCaption(
    "Rolling net rating over each team's last 10 completed games this season (points scored minus points allowed, per 100 possessions). The dotted line is even. Wins and losses are shown under the chart, oldest to newest."
  );
}

function renderWlStrip(home, away) {
  const strip = $("#form-strip");
  strip.hidden = false;
  const mk = (abbr, color) => {
    const t = teamByAbbr(abbr);
    const boxes = t.form
      .map((r) => {
        const letter = r.won ? "W" : "L";
        const loc = r.is_home ? "vs" : "@";
        const title = `${r.date} ${loc} ${r.opp} ${r.pts}–${r.opp_pts}`;
        return `<span class="wl ${letter}" title="${title}">${letter}</span>`;
      })
      .join("");
    return `<div><div class="wl-row"><span class="lab" style="color:${color}">${t.abbr}</span>${boxes}</div>
      <p class="wl-caption">Season: ${t.snapshot.wins}–${t.snapshot.losses} · last 10: ${t.form
        .slice(-10)
        .filter((r) => r.won).length}–${t.form.slice(-10).filter((r) => !r.won).length}</p></div>`;
  };
  strip.innerHTML = mk(away, AWAY) + mk(home, HOME);
}

function drawWhy(home, away, pred) {
  const contrib = pred.contrib_lr;
  const feats = DATA.features;
  const labels = DATA.feature_labels;
  const items = feats
    .map((f, i) => ({ f, label: shortLabel(labels[f] || f), v: contrib[i] }))
    .sort((a, b) => Math.abs(b.v) - Math.abs(a.v));
  const topN = 10;
  const top = items.slice(0, topN);
  const rest = items.slice(topN).reduce((s, x) => s + x.v, 0);
  const intercept = pred.intercept_lr;
  const rows = [...top, { label: "Other features", v: rest }];
  const logOdds = intercept + contrib.reduce((s, v) => s + v, 0);
  const p = 1 / (1 + Math.exp(-logOdds));
  Plotly.newPlot(
    "chart",
    [
      {
        type: "bar",
        orientation: "h",
        y: rows.map((r) => r.label).reverse(),
        x: rows.map((r) => r.v).reverse(),
        marker: {
          color: rows.map((r) => (r.v >= 0 ? TEAL : VERM)).reverse(),
        },
        hovertemplate: "%{y}: %{x:.3f}<extra></extra>",
      },
    ],
    layoutExtras({
      xaxis: { title: "Contribution to log-odds of a home win", zeroline: true, zerolinecolor: INK, zerolinewidth: 1 },
      yaxis: { automargin: true },
      margin: { l: 180, r: 36, t: 20, b: 52 },
      showlegend: false,
      shapes: [
        { type: "line", yref: "paper", y0: 0, y1: 1, x0: 0, x1: 0, line: { color: INK, width: 1 } },
      ],
    }),
    PLOT_CONFIG
  );
  const modelNote =
    modelKey() === "rf"
      ? " The headline probability above uses the random forest; this breakdown is always from the logistic model, which is fully interpretable."
      : "";
  chartCaption(
    `Each bar is a logistic-regression contribution (coefficient × standardized feature) to the log-odds that ${teamByAbbr(home).abbr} wins at home. Teal raises the home-win log-odds; vermillion lowers them. Intercept (baseline when features are at their training means): ${intercept >= 0 ? "+" : ""}${intercept.toFixed(2)}. Predicted log-odds: ${logOdds.toFixed(2)}, or P(home win) = ${fmtPct(p)}.${modelNote}`
  );
}

function shortLabel(s) {
  return s
    .replace(" (pre-game)", "")
    .replace("point differential", "PD")
    .replace("win %", "win%")
    .replace("offensive rating", "off. rating")
    .replace("defensive rating", "def. rating");
}

function drawCal() {
  const series = [
    { key: "logistic_regression", name: "Logistic regression", color: HOME },
    { key: "random_forest", name: "Random forest", color: TEAL },
    { key: "baseline_winpct", name: "Win% baseline", color: AWAY },
  ];
  const traces = series.map((s) => {
    const c = DATA.calibration[s.key];
    const x = [];
    const y = [];
    const text = [];
    const size = [];
    c.mean_predicted.forEach((mp, i) => {
      if (mp === null || c.frac_positive[i] === null) return;
      if ((c.counts[i] || 0) < 10) return; // unstable frequencies
      x.push(mp);
      y.push(c.frac_positive[i]);
      text.push(`${c.counts[i]} games`);
      size.push(Math.max(8, Math.sqrt(c.counts[i]) * 2.2));
    });
    return {
      type: "scatter",
      mode: "lines+markers",
      name: s.name,
      x,
      y,
      text,
      marker: { color: s.color, size, line: { width: 0.5, color: SURFACE } },
      line: { color: s.color, width: 2 },
      hovertemplate: "%{x:.2f} predicted<br>%{y:.2f} observed<br>%{text}<extra>%{fullData.name}</extra>",
    };
  });
  traces.push({
    type: "scatter",
    mode: "lines",
    name: "Perfect calibration",
    x: [0, 1],
    y: [0, 1],
    line: { color: MUTED, width: 1, dash: "dot" },
    hoverinfo: "skip",
  });
  Plotly.newPlot(
    "chart",
    traces,
    layoutExtras({
      xaxis: { title: "Predicted P(home win)", range: [0, 1], tickformat: ".0%", dtick: 0.2, zeroline: false },
      yaxis: { title: "Observed home-win rate", range: [0, 1], tickformat: ".0%", dtick: 0.2, zeroline: false },
      margin: { l: 64, r: 20, t: 28, b: 56 },
    }),
    PLOT_CONFIG
  );
  const m = DATA.metrics.test;
  chartCaption(
    `Reliability diagram on the held-out 2025–26 season (${m.logistic_regression.n} games). Marker size reflects the number of games in each predicted-probability bin. Bins with fewer than 10 games are omitted because the observed rate is too noisy. A well-calibrated model follows the dotted diagonal: when it says 70%, home teams should win about 70% of the time.`
  );
}

function drawTraj(home, away) {
  const sub = $("#chart-sub");
  sub.hidden = false;
  sub.innerHTML = `
    <button type="button" data-mode="teams" class="${trajMode === "teams" ? "is-active" : ""}">Team win % this season</button>
    <button type="button" data-mode="model" class="${trajMode === "model" ? "is-active" : ""}">Model accuracy over time</button>`;
  sub.querySelectorAll("button").forEach((b) =>
    b.addEventListener("click", () => {
      trajMode = b.dataset.mode;
      drawChart();
    })
  );
  if (trajMode === "teams") drawTeamTraj(home, away);
  else drawModelTraj();
}

function drawTeamTraj(home, away) {
  const ht = teamByAbbr(home);
  const at = teamByAbbr(away);
  const traces = [
    {
      type: "scatter",
      mode: "lines+markers",
      name: at.abbr,
      x: at.form.map((r) => r.date),
      y: at.form.map((r) => r.running_win_pct),
      line: { color: AWAY, width: 2.4 },
      marker: { size: 7, color: at.form.map((r) => (r.won ? AWAY : "#b47c00")), symbol: at.form.map((r) => (r.won ? "circle" : "x")) },
      hovertemplate: `${at.abbr} %{x}<br>Win %: %{y:.1%}<extra></extra>`,
    },
    {
      type: "scatter",
      mode: "lines+markers",
      name: ht.abbr,
      x: ht.form.map((r) => r.date),
      y: ht.form.map((r) => r.running_win_pct),
      line: { color: HOME, width: 2.4 },
      marker: { size: 7, color: HOME, symbol: ht.form.map((r) => (r.won ? "circle" : "x")) },
      hovertemplate: `${ht.abbr} %{x}<br>Win %: %{y:.1%}<extra></extra>`,
    },
  ];
  Plotly.newPlot(
    "chart",
    traces,
    layoutExtras({
      yaxis: { title: "Season win percentage so far", tickformat: ".0%", range: [0, 1] },
      xaxis: { title: "Game date, 2026–27", tickformat: "%b %d" },
      shapes: [
        {
          type: "line",
          xref: "paper",
          x0: 0,
          x1: 1,
          y0: 0.5,
          y1: 0.5,
          line: { color: "#b7af9f", width: 1, dash: "dot" },
        },
      ],
    }),
    PLOT_CONFIG
  );
  chartCaption(
    `Cumulative win percentage after each completed 2026–27 game. Circles are wins; crosses are losses. The dotted line is .500. Last completed games in the file are dated ${fmtDate(DATA.meta.last_completed_date)}.`
  );
}

function drawModelTraj() {
  const test = DATA.test_trajectory;
  const traces = [
    { name: "Logistic regression", x: test.map((r) => r.date), y: test.map((r) => r.acc_lr), color: HOME, dash: "solid" },
    { name: "Random forest", x: test.map((r) => r.date), y: test.map((r) => r.acc_rf), color: TEAL, dash: "solid" },
    { name: "Win% baseline", x: test.map((r) => r.date), y: test.map((r) => r.acc_wp), color: AWAY, dash: "dot" },
    { name: "Always home", x: test.map((r) => r.date), y: test.map((r) => r.acc_home), color: MUTED, dash: "dot" },
  ];
  Plotly.newPlot(
    "chart",
    traces.map((t) => ({
      type: "scatter",
      mode: "lines",
      name: t.name,
      x: t.x,
      y: t.y,
      line: { color: t.color, width: 2, dash: t.dash },
      hovertemplate: "%{x}<br>%{fullData.name}: %{y:.1%}<extra></extra>",
    })),
    layoutExtras({
      yaxis: { title: "Running accuracy (threshold 0.5)", tickformat: ".0%", range: [0.45, 0.75] },
      xaxis: { title: "Date, 2025–26 test season", tickformat: "%b %d" },
      legend: { orientation: "h", y: 1.16, x: 0, font: { size: 11 } },
      margin: { l: 64, r: 20, t: 64, b: 56 },
    }),
    PLOT_CONFIG
  );
  const t = DATA.metrics.test.logistic_regression;
  const c = DATA.metrics.current && DATA.metrics.current.logistic_regression;
  chartCaption(
    `Running accuracy through the 2025–26 test season (logistic ${fmtPct(t.accuracy)}, random forest ${fmtPct(DATA.metrics.test.random_forest.accuracy)}, ${t.n} games).` +
      (c ? ` 2026–27 completed games are a forward application (logistic ${fmtPct(c.accuracy)}, random forest ${fmtPct(DATA.metrics.current.random_forest.accuracy)}, ${c.n} games) and were not used in training.` : "")
  );
}

function fmtMetric(v, fmt) {
  if (fmt === "pct") return (v * 100).toFixed(1) + "%";
  if (fmt === "pd") return (v > 0 ? "+" : "") + v.toFixed(1);
  return v.toFixed(1);
}

function drawCmp(home, away) {
  const ht = teamByAbbr(home);
  const at = teamByAbbr(away);
  Plotly.purge("chart");
  const rows = [
    { label: "Season win %", h: ht.snapshot.win_pct, a: at.snapshot.win_pct, fmt: "pct", lo: 0, hi: 1 },
    { label: "Last-10 win %", h: ht.snapshot.l10_win_pct, a: at.snapshot.l10_win_pct, fmt: "pct", lo: 0, hi: 1 },
    { label: "Season point differential", h: ht.snapshot.pd, a: at.snapshot.pd, fmt: "pd", diverge: true },
    { label: "Last-10 point differential", h: ht.snapshot.l10_pd, a: at.snapshot.l10_pd, fmt: "pd", diverge: true },
    { label: "Last-10 offensive rating", h: ht.snapshot.l10_off_rtg, a: at.snapshot.l10_off_rtg, fmt: "num", lo: 90 },
    { label: "Last-10 defensive rating", h: ht.snapshot.l10_def_rtg, a: at.snapshot.l10_def_rtg, fmt: "num", lo: 90, hint: "lower is better" },
    { label: "Points per game", h: ht.snapshot.pts, a: at.snapshot.pts, fmt: "num", lo: 90 },
    { label: "Opponent points per game", h: ht.snapshot.opp_pts, a: at.snapshot.opp_pts, fmt: "num", lo: 90, hint: "lower is better" },
  ];
  const barRow = (r) => {
    let lo, hi;
    if (r.diverge) {
      const m = Math.max(12, Math.abs(r.h), Math.abs(r.a)) * 1.15;
      lo = -m;
      hi = m;
    } else {
      lo = r.lo ?? 0;
      hi = r.hi ?? Math.max(r.h, r.a) * 1.08;
      if (hi <= lo) hi = lo + 1;
    }
    const span = hi - lo;
    const zeroPct = ((0 - lo) / span) * 100;
    const barStyle = (v) => {
      if (r.diverge) {
        if (v >= 0) return `left:${zeroPct}%;width:${(v / span) * 100}%`;
        return `left:${((v - lo) / span) * 100}%;width:${(-v / span) * 100}%`;
      }
      return `left:0;width:${(Math.max(0, v - lo) / span) * 100}%`;
    };
    const hint = r.hint ? ` <span class="hint">(${r.hint})</span>` : "";
    const track = (cls, v) =>
      `<div class="diverge-track" style="--zero:${zeroPct}%">${r.diverge ? '<span class="tick"></span>' : ""}<i class="${cls}" style="${barStyle(v)}"></i></div>`;
    return `<tr>
      <td class="metric">${r.label}${hint}</td>
      <td class="num away">${fmtMetric(r.a, r.fmt)}</td>
      <td class="bars"><div class="cmp-bars">
        ${track("a", r.a)}
        ${track("h", r.h)}
      </div></td>
      <td class="num home">${fmtMetric(r.h, r.fmt)}</td>
    </tr>`;
  };
  document.getElementById("chart").innerHTML = `
    <table class="cmp-table">
      <thead><tr><th>Metric</th><th class="num">${at.abbr}</th><th></th><th class="num">${ht.abbr}</th></tr></thead>
      <tbody>${rows.map(barRow).join("")}</tbody>
    </table>`;
  document.getElementById("chart").style.height = "auto";
  chartCaption(
    `Current-season snapshot after each team's last completed game. Each row has its own scale, so win percentages are not plotted against ratings. Offensive and defensive ratings are estimated points scored or allowed per 100 possessions. These are descriptive statistics, not the model output.`
  );
}

function renderHeader() {
  const m = DATA.meta;
  $("#header-meta").innerHTML = `<div>As of <strong>${escapeHtml(m.as_of_hkt)}</strong></div>
    <div>Last completed games: ${fmtDate(m.last_completed_date)}</div>
    <div>${m.n_games_collected.toLocaleString()} regular-season games · 2021–22 to 2026–27</div>`;
}

function renderMetrics() {
  const test = DATA.metrics.test;
  const cur = DATA.metrics.current;
  const row = (name, m) =>
    `<tr><th scope="row">${name}</th><td>${m.accuracy.toFixed(3)}</td><td>${m.log_loss.toFixed(3)}</td><td>${m.brier.toFixed(3)}</td></tr>`;
  let html = `<table><caption>Held-out test season, 2025–26 (${test.logistic_regression.n} games)</caption>
    <thead><tr><th>Model</th><th>Accuracy</th><th>Log-loss</th><th>Brier</th></tr></thead><tbody>
    ${row("Always home", test.baseline_home)}
    ${row("Higher pre-game win %", test.baseline_winpct)}
    ${row("Logistic regression (L2)", test.logistic_regression)}
    ${row("Random forest", test.random_forest)}
    </tbody></table>`;
  if (cur) {
    html += `<table style="margin-top:16px"><caption>Forward application, completed 2026–27 games (${cur.logistic_regression.n} games)</caption>
      <thead><tr><th>Model</th><th>Accuracy</th><th>Log-loss</th><th>Brier</th></tr></thead><tbody>
      ${row("Always home", cur.baseline_home)}
      ${row("Higher pre-game win %", cur.baseline_winpct)}
      ${row("Logistic regression (L2)", cur.logistic_regression)}
      ${row("Random forest", cur.random_forest)}
      </tbody></table>`;
  }
  $("#metrics-inline").innerHTML = html;
}

function renderFooter() {
  const m = DATA.meta;
  $("#footer").innerHTML = `
    <p><strong>Data:</strong> ${escapeHtml(m.data_source)}. Regular-season game logs for 2021–22 through 2026–27, plus the official 2026–27 schedule.</p>
    <p><strong>Model:</strong> Default is L2 logistic regression (C = ${m.lr_C}), trained on ${m.seasons_trained[0]} to ${m.seasons_trained[m.seasons_trained.length - 1]} (${m.n_train.toLocaleString()} games). A random forest (${m.rf_params.n_estimators} trees, max depth ${m.rf_params.max_depth}) is available as a comparison. Test season: ${m.test_season}.</p>
    <p>${escapeHtml(m.disclaimer)}</p>
    <p>${escapeHtml(m.course)} · ${escapeHtml(m.institution)}.</p>`;
}

function bind() {
  $("#home-select").addEventListener("change", () => {
    selectedGameId = null;
    refresh();
  });
  $("#away-select").addEventListener("change", () => {
    selectedGameId = null;
    refresh();
  });
  $("#swap-btn").addEventListener("click", () => {
    const h = $("#home-select").value;
    $("#home-select").value = $("#away-select").value;
    $("#away-select").value = h;
    selectedGameId = null;
    refresh();
  });
  $$('input[name="model"]').forEach((el) => el.addEventListener("change", refresh));
  $$(".tab").forEach((tab) =>
    tab.addEventListener("click", () => {
      $$(".tab").forEach((t) => {
        t.classList.toggle("is-active", t === tab);
        t.setAttribute("aria-selected", t === tab ? "true" : "false");
      });
      activeChart = tab.dataset.chart;
      drawChart();
    })
  );
  window.addEventListener("resize", () => {
    const el = document.getElementById("chart");
    if (el && el.data) Plotly.Plots.resize(el);
  });
}

async function init() {
  const res = await fetch("data/app_data.json");
  if (!res.ok) {
    document.body.insertAdjacentHTML("afterbegin", `<p style="padding:20px">Could not load data/app_data.json (${res.status}). Serve the site from the web/ folder.</p>`);
    return;
  }
  DATA = await res.json();
  populateTeams();
  renderHeader();
  renderUpcoming();
  renderMetrics();
  renderFooter();
  bind();
  const firstSched = (DATA.upcoming || []).find((g) => g.status === 1) || (DATA.upcoming || [])[0];
  if (firstSched) {
    selectedGameId = firstSched.game_id;
    $("#home-select").value = firstSched.home;
    $("#away-select").value = firstSched.away;
  } else {
    $("#home-select").value = "NYK";
    $("#away-select").value = "BOS";
  }
  refresh();
}

init();
