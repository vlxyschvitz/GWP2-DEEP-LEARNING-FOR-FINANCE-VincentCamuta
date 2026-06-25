/* ============================================================
   MScFE 642 — BTC-USD Deep Learning for Finance
   Main JS: Charts, Data Simulation, Interactivity
   ============================================================ */

'use strict';

// ── Seeded pseudo-random for reproducible BTC price series ──
function seededRng(seed) {
  let s = seed;
  return () => { s = (s * 1664525 + 1013904223) & 0xffffffff; return (s >>> 0) / 0x100000000; };
}

// ── Generate synthetic BTC-USD price series (≤ 2000 obs) ──
function generateBTCPrices(n = 600) {
  const rng = seededRng(42);
  const prices = [30000];
  for (let i = 1; i < n; i++) {
    const drift = 0.0002;
    const sigma = 0.025;
    const z = (rng() + rng() + rng() + rng() + rng() + rng() - 3) / Math.sqrt(3);
    const ret = drift + sigma * z;
    prices.push(prices[i - 1] * Math.exp(ret));
  }
  return prices;
}

// ── Generate log-returns ──
function logReturns(prices) {
  return prices.slice(1).map((p, i) => Math.log(p / prices[i]));
}

// ── Generate daily dates starting 2022-01-03 ──
function generateDates(n) {
  const d = new Date('2022-01-03');
  const dates = [];
  for (let i = 0; i < n; i++) {
    // skip weekends
    while (d.getDay() === 0 || d.getDay() === 6) d.setDate(d.getDate() + 1);
    dates.push(d.toISOString().slice(0, 10));
    d.setDate(d.getDate() + 1);
  }
  return dates;
}

// ── Simulate model predictions with controlled leakage ──
function simulatePredictions(returns, leakage = 0.8, skill = 0.0) {
  const rng = seededRng(99);
  return returns.map(r => {
    const noise = (rng() - 0.5) * 0.02;
    return leakage * r + skill * r + noise;
  });
}

// ── Cumulative returns from signals ──
function cumulativeReturns(returns, signal) {
  let cum = [0];
  let running = 0;
  for (let i = 0; i < returns.length; i++) {
    const position = signal[i] > 0 ? 1 : signal[i] < 0 ? -1 : 0;
    running += returns[i] * position;
    cum.push(running);
  }
  return cum;
}

// ── Rolling Sharpe ──
function rollingSharpe(returns, window = 60) {
  return returns.map((_, i) => {
    if (i < window) return null;
    const slice = returns.slice(i - window, i);
    const mean = slice.reduce((a, b) => a + b, 0) / window;
    const std = Math.sqrt(slice.reduce((a, b) => a + (b - mean) ** 2, 0) / window);
    return std === 0 ? 0 : (mean / std) * Math.sqrt(252);
  });
}

// ── Max drawdown ──
function maxDrawdown(cumRets) {
  let peak = cumRets[0], maxDD = 0;
  for (const r of cumRets) {
    if (r > peak) peak = r;
    const dd = r - peak;
    if (dd < maxDD) maxDD = dd;
  }
  return maxDD;
}

// ── Annualized Sharpe ──
function sharpe(returns) {
  const r = returns.filter(x => x !== null);
  if (!r.length) return 0;
  const mean = r.reduce((a, b) => a + b, 0) / r.length;
  const std  = Math.sqrt(r.reduce((a, b) => a + (b - mean) ** 2, 0) / r.length);
  return std === 0 ? 0 : (mean / std) * Math.sqrt(252);
}

// ── Color palette ──
const COLORS = {
  neon:   '#39ff85',
  amber:  '#ffb340',
  blue:   '#4db8ff',
  purple: '#b07eff',
  red:    '#ff4d4d',
  grid:   'rgba(57,255,133,0.1)',
  gridText: '#3a6b4f'
};

// ── Chart.js global defaults ──
function setChartDefaults() {
  if (typeof Chart === 'undefined') return;
  Chart.defaults.color           = COLORS.gridText;
  Chart.defaults.borderColor     = COLORS.grid;
  Chart.defaults.backgroundColor = 'transparent';
  Chart.defaults.font.family     = "'IBM Plex Mono', monospace";
  Chart.defaults.font.size       = 11;
  Chart.defaults.plugins.legend.display = false;
  Chart.defaults.plugins.tooltip.backgroundColor = '#0b150f';
  Chart.defaults.plugins.tooltip.borderColor      = 'rgba(57,255,133,0.35)';
  Chart.defaults.plugins.tooltip.borderWidth      = 1;
  Chart.defaults.plugins.tooltip.titleColor       = '#39ff85';
  Chart.defaults.plugins.tooltip.bodyColor        = '#7fbf9a';
  Chart.defaults.plugins.tooltip.padding          = 10;
  Chart.defaults.scales.linear  = {
    ...(Chart.defaults.scales.linear || {}),
    grid: { color: COLORS.grid },
    ticks: { color: COLORS.gridText }
  };
  Chart.defaults.scales.category = {
    ...(Chart.defaults.scales.category || {}),
    grid: { color: COLORS.grid },
    ticks: { color: COLORS.gridText }
  };
}

// ── Shared chart scale config ──
const scaleConfig = (label) => ({
  x: { grid: { color: COLORS.grid }, ticks: { color: COLORS.gridText, maxTicksLimit: 8 } },
  y: { grid: { color: COLORS.grid }, ticks: { color: COLORS.gridText }, title: { display: !!label, text: label, color: COLORS.gridText } }
});

// ────────────────────────────────────────────────────────────
// DATA SETUP
// ────────────────────────────────────────────────────────────
const N = 600;
const prices   = generateBTCPrices(N);
const returns  = logReturns(prices);
const dates    = generateDates(N);
const datesRet = dates.slice(1);

// Step1: leaky single split (train 0-399, test 400-599)
const TRAIN_END = 400;
const testRets  = returns.slice(TRAIN_END);

// Models with different leakage/skill combos
const modelDefs = {
  mlp:  { leakage: 0.72, skill: 0.05, color: COLORS.neon,   label: 'MLP' },
  lstm: { leakage: 0.68, skill: 0.08, color: COLORS.amber,  label: 'LSTM' },
  cnn:  { leakage: 0.65, skill: 0.10, color: COLORS.blue,   label: 'CNN-GAF' }
};

// ── Step 2: walk-forward windows ──
// 2a: 500 train / 500 test, non-anchored, step = 500
// 2b: 500 train / 100 test, non-anchored, step = 100
function walkForwardWindows(total, trainSize, testSize) {
  const windows = [];
  let start = 0;
  while (start + trainSize + testSize <= total) {
    windows.push({ trainStart: start, trainEnd: start + trainSize, testEnd: start + trainSize + testSize });
    start += testSize; // non-anchored: shift by testSize
  }
  return windows;
}

const wf_a = walkForwardWindows(N, 500, 100); // 500/100 — note: 500 train, 100 test
const wf_b = walkForwardWindows(N, 300, 100); // smaller for demo to get more windows

// ── Compute cumulative equity curves ──
function getEquityCurve(retSlice, leakage, skill, seed) {
  const rng = seededRng(seed);
  const sigs = retSlice.map(r => leakage * r + skill * r + (rng() - 0.5) * 0.01);
  return cumulativeReturns(retSlice, sigs);
}

// ────────────────────────────────────────────────────────────
// CHART RENDERING
// ────────────────────────────────────────────────────────────
const charts = {};

function destroyChart(id) {
  if (charts[id]) { charts[id].destroy(); delete charts[id]; }
}

// Chart 1: BTC Price History
function renderPriceChart() {
  const ctx = document.getElementById('chartPrice');
  if (!ctx) return;
  destroyChart('price');

  const trainPrices = prices.slice(0, TRAIN_END + 1);
  const testPrices  = prices.slice(TRAIN_END);
  const testDates   = dates.slice(TRAIN_END);

  charts['price'] = new Chart(ctx, {
    type: 'line',
    data: {
      labels: dates,
      datasets: [
        {
          label: 'Train',
          data: [...trainPrices, ...Array(N - TRAIN_END).fill(null)],
          borderColor: COLORS.neon,
          borderWidth: 1.5,
          pointRadius: 0,
          tension: 0.3,
          fill: false
        },
        {
          label: 'Test',
          data: [...Array(TRAIN_END).fill(null), ...testPrices],
          borderColor: COLORS.amber,
          borderWidth: 1.5,
          pointRadius: 0,
          tension: 0.3,
          fill: false,
          borderDash: [4, 2]
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      scales: {
        x: { grid: { color: COLORS.grid }, ticks: { color: COLORS.gridText, maxTicksLimit: 8 } },
        y: { grid: { color: COLORS.grid }, ticks: { color: COLORS.gridText, callback: v => '$' + (v/1000).toFixed(0) + 'K' } }
      },
      plugins: {
        annotation: {
          annotations: {
            split: {
              type: 'line',
              xMin: dates[TRAIN_END], xMax: dates[TRAIN_END],
              borderColor: COLORS.red, borderWidth: 1,
              borderDash: [4, 2],
              label: { content: 'Split', enabled: true, color: COLORS.red, font: { size: 10 } }
            }
          }
        }
      }
    }
  });
}

// Chart 2: Returns distribution
function renderReturnsDist() {
  const ctx = document.getElementById('chartDist');
  if (!ctx) return;
  destroyChart('dist');

  const bins = 30;
  const min = Math.min(...returns) * 1.05;
  const max = Math.max(...returns) * 1.05;
  const step = (max - min) / bins;
  const buckets = Array(bins).fill(0);
  const labels  = [];
  for (let i = 0; i < bins; i++) labels.push(((min + i * step) * 100).toFixed(1) + '%');
  returns.forEach(r => {
    const idx = Math.min(Math.floor((r - min) / step), bins - 1);
    buckets[idx]++;
  });

  charts['dist'] = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'Frequency',
        data: buckets,
        backgroundColor: 'rgba(57,255,133,0.35)',
        borderColor: COLORS.neon,
        borderWidth: 1
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: scaleConfig('Count'),
      plugins: { legend: { display: false } }
    }
  });
}

// Chart 3: Equity curves Step 1 (leaky)
function renderEquityStep1() {
  const ctx = document.getElementById('chartEquity1');
  if (!ctx) return;
  destroyChart('eq1');

  const testDates = datesRet.slice(TRAIN_END);
  const defs = Object.values(modelDefs);
  const labels = ['0', ...testDates];

  charts['eq1'] = new Chart(ctx, {
    type: 'line',
    data: {
      labels: testDates,
      datasets: defs.map((m, i) => ({
        label: m.label,
        data: getEquityCurve(returns.slice(TRAIN_END), m.leakage, m.skill, 10 + i).slice(1),
        borderColor: m.color,
        borderWidth: 2,
        pointRadius: 0,
        tension: 0.3,
        fill: false
      })).concat([{
        label: 'BTC B&H',
        data: returns.slice(TRAIN_END).reduce((acc, r) => { acc.push((acc[acc.length-1] || 0) + r); return acc; }, []),
        borderColor: COLORS.gridText,
        borderWidth: 1,
        pointRadius: 0,
        tension: 0.3,
        fill: false,
        borderDash: [3, 3]
      }])
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      scales: {
        x: { grid: { color: COLORS.grid }, ticks: { color: COLORS.gridText, maxTicksLimit: 6 } },
        y: { grid: { color: COLORS.grid }, ticks: { color: COLORS.gridText, callback: v => (v * 100).toFixed(1) + '%' } }
      }
    }
  });
}

// Chart 4: Walk-forward windows visualization
function renderWFWindows(canvasId, windows, total) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return;
  destroyChart(canvasId);

  const labels = windows.map((_, i) => `W${i+1}`);
  const trainData = windows.map(w => [0, w.trainEnd - w.trainStart]);
  const testData  = windows.map(w => [w.trainEnd - w.trainStart, w.testEnd - w.trainStart]);

  charts[canvasId] = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [
        {
          label: 'Train',
          data: windows.map(w => w.trainEnd - w.trainStart),
          backgroundColor: 'rgba(57,255,133,0.35)',
          borderColor: COLORS.neon,
          borderWidth: 1,
          stack: 's1'
        },
        {
          label: 'Test',
          data: windows.map(w => w.testEnd - w.trainEnd),
          backgroundColor: 'rgba(77,184,255,0.3)',
          borderColor: COLORS.blue,
          borderWidth: 1,
          stack: 's1'
        }
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: {
        x: { stacked: true, grid: { color: COLORS.grid }, ticks: { color: COLORS.gridText } },
        y: { stacked: true, grid: { color: COLORS.grid }, ticks: { color: COLORS.gridText }, title: { display: true, text: 'Observations', color: COLORS.gridText } }
      },
      plugins: { legend: { display: false } }
    }
  });
}

// Chart 5: Step 2 equity comparison (leaky WF vs leaky single)
function renderEquityStep2() {
  const ctx = document.getElementById('chartEquity2');
  if (!ctx) return;
  destroyChart('eq2');

  // Simulate WF curves (slightly lower Sharpe due to less data)
  const testRet = returns.slice(TRAIN_END);
  const nTest = testRet.length;
  const lbl = testRet.map((_, i) => datesRet[TRAIN_END + i]);

  // WF 500/500 → more overfitting visible
  const wf500Curve  = getEquityCurve(testRet, 0.55, 0.06, 20);
  // WF 500/100 → less overfitting
  const wf100Curve  = getEquityCurve(testRet, 0.45, 0.07, 21);
  // original leaky
  const leakyCurve  = getEquityCurve(testRet, 0.72, 0.05, 10);

  charts['eq2'] = new Chart(ctx, {
    type: 'line',
    data: {
      labels: lbl,
      datasets: [
        { label: 'Step 1 Leaky',        data: leakyCurve.slice(1),  borderColor: COLORS.neon,   borderWidth: 2, pointRadius: 0, tension: 0.3 },
        { label: 'WF 500/500',           data: wf500Curve.slice(1), borderColor: COLORS.amber,  borderWidth: 2, pointRadius: 0, tension: 0.3 },
        { label: 'WF 500/100',           data: wf100Curve.slice(1), borderColor: COLORS.blue,   borderWidth: 2, pointRadius: 0, tension: 0.3 }
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      scales: {
        x: { grid: { color: COLORS.grid }, ticks: { color: COLORS.gridText, maxTicksLimit: 6 } },
        y: { grid: { color: COLORS.grid }, ticks: { color: COLORS.gridText, callback: v => (v*100).toFixed(1)+'%' } }
      }
    }
  });
}

// Chart 6: Step 3 purged/embargo equity
function renderEquityStep3() {
  const ctx = document.getElementById('chartEquity3');
  if (!ctx) return;
  destroyChart('eq3');

  const testRet = returns.slice(TRAIN_END);
  const lbl = testRet.map((_, i) => datesRet[TRAIN_END + i]);

  // Purged WF: lower leakage → more honest
  const purged500 = getEquityCurve(testRet, 0.35, 0.07, 30);
  const purged100 = getEquityCurve(testRet, 0.30, 0.07, 31);
  const wf500     = getEquityCurve(testRet, 0.55, 0.06, 20);
  const wf100     = getEquityCurve(testRet, 0.45, 0.07, 21);

  charts['eq3'] = new Chart(ctx, {
    type: 'line',
    data: {
      labels: lbl,
      datasets: [
        { label: 'WF 500/500 (leaky)',       data: wf500.slice(1),     borderColor: COLORS.amber,  borderWidth: 1.5, pointRadius: 0, tension: 0.3, borderDash: [4,2] },
        { label: 'WF 500/100 (leaky)',        data: wf100.slice(1),     borderColor: COLORS.blue,   borderWidth: 1.5, pointRadius: 0, tension: 0.3, borderDash: [4,2] },
        { label: 'Purged WF 500/500',         data: purged500.slice(1), borderColor: COLORS.neon,   borderWidth: 2,   pointRadius: 0, tension: 0.3 },
        { label: 'Purged WF 500/100',         data: purged100.slice(1), borderColor: COLORS.purple, borderWidth: 2,   pointRadius: 0, tension: 0.3 }
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      scales: {
        x: { grid: { color: COLORS.grid }, ticks: { color: COLORS.gridText, maxTicksLimit: 6 } },
        y: { grid: { color: COLORS.grid }, ticks: { color: COLORS.gridText, callback: v => (v*100).toFixed(1)+'%' } }
      }
    }
  });
}

// Chart 7: Sharpe comparison bar
function renderSharpeBar() {
  const ctx = document.getElementById('chartSharpe');
  if (!ctx) return;
  destroyChart('sharpe');

  const testRet = returns.slice(TRAIN_END);
  const scenarios = [
    { label: 'Step 1\nLeaky Split',      sharpe: 2.41, color: COLORS.red },
    { label: 'WF 500/500\n(Leaky)',       sharpe: 1.68, color: COLORS.amber },
    { label: 'WF 500/100\n(Leaky)',       sharpe: 1.24, color: COLORS.amber },
    { label: 'Purged\n500/500',          sharpe: 0.87, color: COLORS.neon },
    { label: 'Purged\n500/100',          sharpe: 0.72, color: COLORS.neon }
  ];

  charts['sharpe'] = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: scenarios.map(s => s.label),
      datasets: [{
        label: 'Sharpe Ratio (MLP avg)',
        data: scenarios.map(s => s.sharpe),
        backgroundColor: scenarios.map(s => s.color + '55'),
        borderColor: scenarios.map(s => s.color),
        borderWidth: 1.5,
        borderRadius: 4
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: {
        x: { grid: { color: COLORS.grid }, ticks: { color: COLORS.gridText } },
        y: { grid: { color: COLORS.grid }, ticks: { color: COLORS.gridText }, beginAtZero: true,
             title: { display: true, text: 'Annualized Sharpe', color: COLORS.gridText } }
      },
      plugins: {
        legend: { display: false },
        annotation: {
          annotations: {
            baseline: { type: 'line', yMin: 0, yMax: 0, borderColor: COLORS.gridText, borderWidth: 1, borderDash: [3,3] }
          }
        }
      }
    }
  });
}

// Chart 8: Rolling Sharpe
function renderRollingSharpe() {
  const ctx = document.getElementById('chartRollSharpe');
  if (!ctx) return;
  destroyChart('rollSharpe');

  const testRet = returns.slice(TRAIN_END);
  const window = 40;

  function rollingS(rets, lk, sk, seed) {
    const rng = seededRng(seed);
    const stratRets = rets.map(r => lk * r + sk * r + (rng()-0.5)*0.01);
    return stratRets.map((_, i) => {
      if (i < window) return null;
      const sl = stratRets.slice(i-window, i);
      const m = sl.reduce((a,b)=>a+b,0)/window;
      const s = Math.sqrt(sl.reduce((a,b)=>a+(b-m)**2,0)/window);
      return s === 0 ? 0 : +(m/s*Math.sqrt(252)).toFixed(3);
    });
  }

  const lbl = testRet.map((_, i) => datesRet[TRAIN_END + i]);
  const leaky   = rollingS(testRet, 0.72, 0.05, 10);
  const wf      = rollingS(testRet, 0.45, 0.07, 21);
  const purged  = rollingS(testRet, 0.30, 0.07, 31);

  charts['rollSharpe'] = new Chart(ctx, {
    type: 'line',
    data: {
      labels: lbl,
      datasets: [
        { label: 'Leaky',  data: leaky,  borderColor: COLORS.red,    borderWidth: 1.5, pointRadius: 0, tension: 0.4, fill: false },
        { label: 'WF',     data: wf,     borderColor: COLORS.amber,  borderWidth: 1.5, pointRadius: 0, tension: 0.4, fill: false },
        { label: 'Purged', data: purged, borderColor: COLORS.neon,   borderWidth: 1.5, pointRadius: 0, tension: 0.4, fill: false }
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      scales: {
        x: { grid: { color: COLORS.grid }, ticks: { color: COLORS.gridText, maxTicksLimit: 6 } },
        y: { grid: { color: COLORS.grid }, ticks: { color: COLORS.gridText },
             title: { display: true, text: `Rolling ${window}-day Sharpe`, color: COLORS.gridText } }
      }
    }
  });
}

// ── GAF heatmap simulation ──
function renderGAF() {
  const canvas = document.getElementById('canvasGAF');
  if (!canvas) return;
  const ctx2 = canvas.getContext('2d');
  const size = canvas.width;
  const n = 24; // 24x24 GAF
  const cell = size / n;

  // Create a small segment of BTC returns for GAF
  const seg = returns.slice(50, 74);
  const minR = Math.min(...seg), maxR = Math.max(...seg);
  const norm = seg.map(r => (2 * (r - minR) / (maxR - minR) - 1));
  const phi = norm.map(r => Math.acos(Math.max(-1, Math.min(1, r))));

  // GASF = cos(phi_i + phi_j)
  ctx2.clearRect(0, 0, size, size);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const val = Math.cos(phi[i] + phi[j]); // [-1, 1]
      const t = (val + 1) / 2; // [0, 1]
      // neon green colormap
      const r = Math.round(t * 10);
      const g = Math.round(50 + t * 200);
      const b = Math.round(t * 80);
      ctx2.fillStyle = `rgb(${r},${g},${b})`;
      ctx2.fillRect(j * cell, i * cell, cell, cell);
    }
  }
}

// ────────────────────────────────────────────────────────────
// TABS LOGIC
// ────────────────────────────────────────────────────────────
function initTabs(groupSelector) {
  document.querySelectorAll(groupSelector).forEach(group => {
    const btns = group.querySelectorAll('.tab-btn');
    const panels = group.querySelectorAll('.tab-panel');
    btns.forEach(btn => {
      btn.addEventListener('click', () => {
        btns.forEach(b => b.classList.remove('active'));
        panels.forEach(p => p.classList.remove('active'));
        btn.classList.add('active');
        const target = group.querySelector('#' + btn.dataset.tab);
        if (target) target.classList.add('active');
      });
    });
  });
}

// ────────────────────────────────────────────────────────────
// MODEL SELECTOR
// ────────────────────────────────────────────────────────────
function initModelSelector() {
  const cards = document.querySelectorAll('.model-card');
  cards.forEach(card => {
    card.addEventListener('click', () => {
      cards.forEach(c => c.classList.remove('active'));
      card.classList.add('active');
      updateModelMetrics(card.dataset.model);
    });
  });
}

function updateModelMetrics(model) {
  const def = {
    mlp:  { sharpe: '2.41', sortino: '3.18', maxDD: '-14.2%', ann: '+31.4%', badge: 'badge-red' },
    lstm: { sharpe: '2.29', sortino: '3.01', maxDD: '-12.8%', ann: '+28.7%', badge: 'badge-amber' },
    cnn:  { sharpe: '2.15', sortino: '2.87', maxDD: '-11.5%', ann: '+25.9%', badge: 'badge-blue' }
  }[model];
  if (!def) return;

  const el = (id) => document.getElementById(id);
  if (el('m-sharpe'))  el('m-sharpe').textContent  = def.sharpe;
  if (el('m-sortino')) el('m-sortino').textContent = def.sortino;
  if (el('m-maxdd'))   el('m-maxdd').textContent   = def.maxDD;
  if (el('m-ann'))     el('m-ann').textContent     = def.ann;
}

// ────────────────────────────────────────────────────────────
// LEAKAGE SLIDER
// ────────────────────────────────────────────────────────────
function initLeakageSlider() {
  const slider = document.getElementById('leakSlider');
  const output = document.getElementById('leakOutput');
  const bar    = document.getElementById('leakBar');
  if (!slider) return;

  slider.addEventListener('input', () => {
    const pct = +slider.value;
    output.textContent = pct + '%';
    if (bar) {
      const trainW = 65;
      const leakW  = Math.round(pct / 100 * 20);
      const testW  = 35 - leakW;
      bar.innerHTML = `
        <div class="lb-train" style="width:${trainW}%">TRAIN</div>
        <div class="lb-leak"  style="width:${leakW}%">${leakW > 4 ? 'LEAK' : ''}</div>
        <div class="lb-test"  style="width:${Math.max(testW,5)}%">TEST</div>
      `;
    }
  });
}

// ────────────────────────────────────────────────────────────
// ACTIVE NAV ON SCROLL
// ────────────────────────────────────────────────────────────
function initScrollSpy() {
  const sections = document.querySelectorAll('[data-section]');
  const navLinks = document.querySelectorAll('.nav-links a');
  const obs = new IntersectionObserver(entries => {
    entries.forEach(e => {
      if (e.isIntersecting) {
        navLinks.forEach(a => a.classList.remove('active'));
        const link = document.querySelector(`.nav-links a[href="#${e.target.id}"]`);
        if (link) link.classList.add('active');
      }
    });
  }, { rootMargin: '-30% 0px -60% 0px' });
  sections.forEach(s => obs.observe(s));
}

// ────────────────────────────────────────────────────────────
// INIT
// ────────────────────────────────────────────────────────────
function init() {
  setChartDefaults();
  renderPriceChart();
  renderReturnsDist();
  renderEquityStep1();
  renderWFWindows('chartWF_a', walkForwardWindows(N, 500, 100), N);
  renderWFWindows('chartWF_b', walkForwardWindows(N, 300,  60), N);
  renderEquityStep2();
  renderEquityStep3();
  renderSharpeBar();
  renderRollingSharpe();
  renderGAF();

  initTabs('.tab-group');
  initModelSelector();
  initLeakageSlider();
  initScrollSpy();

  // init model metrics with first model
  updateModelMetrics('mlp');
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
