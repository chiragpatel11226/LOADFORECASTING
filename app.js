/* ============================================================
   Power Demand Forecast -- Dashboard JS

   Fetches pre-computed chart data + metrics from the Flask API and
   renders everything with Chart.js. Also wires up the live /api/predict
   call for the "try it yourself" panel.
   ============================================================ */

const PALETTE = {
  amber: "#e8a33d",
  amberFill: "#e8a33d26",
  rust: "#c1440e",
  rustFill: "#c1440e26",
  sage: "#8fa662",
  sageFill: "#8fa66226",
  grid: "#37312680",
  text: "#a89e8c",
};

Chart.defaults.font.family = "'IBM Plex Sans', sans-serif";
Chart.defaults.color = PALETTE.text;
Chart.defaults.borderColor = PALETTE.grid;

function fmtMW(value) {
  return `${Math.round(value).toLocaleString()} MW`;
}

async function fetchJSON(url, options) {
  const res = await fetch(url, options);
  if (!res.ok) throw new Error(`Request failed: ${url}`);
  return res.json();
}

function baseLineOptions(extra = {}) {
  return {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: "index", intersect: false },
    scales: {
      x: { grid: { display: false }, ticks: { maxTicksLimit: 8 } },
      y: { grid: { color: PALETTE.grid }, ticks: { callback: (v) => `${(v / 1000).toFixed(0)}k` } },
    },
    plugins: { legend: { display: false } },
    ...extra,
  };
}

async function renderCharts() {
  const data = await fetchJSON("/api/dashboard-data");

  // ---- Stat cards ----
  document.getElementById("stat-peak").textContent = fmtMW(data.summary.peak_mw);
  document.getElementById("stat-avg").textContent = fmtMW(data.summary.avg_mw);
  document.getElementById("stat-min").textContent = fmtMW(data.summary.min_mw);

  // ---- Daily trend (full history) ----
  new Chart(document.getElementById("chart-daily"), {
    type: "line",
    data: {
      labels: data.daily_series.labels,
      datasets: [{
        data: data.daily_series.values,
        borderColor: PALETTE.amber,
        backgroundColor: PALETTE.amberFill,
        fill: true,
        pointRadius: 0,
        borderWidth: 2,
        tension: 0.15,
      }],
    },
    options: baseLineOptions(),
  });

  // ---- Hourly profile ----
  new Chart(document.getElementById("chart-hourly"), {
    type: "line",
    data: {
      labels: data.hourly_profile.labels,
      datasets: [{
        data: data.hourly_profile.values,
        borderColor: PALETTE.amber,
        backgroundColor: PALETTE.amberFill,
        fill: true,
        pointRadius: 0,
        borderWidth: 2,
        tension: 0.35,
      }],
    },
    options: baseLineOptions(),
  });

  // ---- Monthly profile ----
  new Chart(document.getElementById("chart-monthly"), {
    type: "bar",
    data: {
      labels: data.monthly_profile.labels,
      datasets: [{
        data: data.monthly_profile.values,
        backgroundColor: PALETTE.rust,
        borderRadius: 4,
      }],
    },
    options: baseLineOptions(),
  });

  // ---- Day of week profile ----
  new Chart(document.getElementById("chart-dow"), {
    type: "bar",
    data: {
      labels: data.dayofweek_profile.labels,
      datasets: [{
        data: data.dayofweek_profile.values,
        backgroundColor: PALETTE.sage,
        borderRadius: 4,
      }],
    },
    options: baseLineOptions(),
  });

  // ---- Actual vs predicted (test set) ----
  new Chart(document.getElementById("chart-actual-vs-pred"), {
    type: "line",
    data: {
      labels: data.actual_vs_predicted.labels,
      datasets: [
        {
          label: "Actual",
          data: data.actual_vs_predicted.actual,
          borderColor: PALETTE.sage,
          backgroundColor: "transparent",
          pointRadius: 0,
          borderWidth: 2,
        },
        {
          label: "Predicted",
          data: data.actual_vs_predicted.predicted,
          borderColor: PALETTE.amber,
          backgroundColor: "transparent",
          borderDash: [5, 4],
          pointRadius: 0,
          borderWidth: 2,
        },
      ],
    },
    options: baseLineOptions({
      plugins: { legend: { display: true, position: "top", align: "end" } },
    }),
  });

  // ---- Feature importance ----
  const importance = await fetchJSON("/api/feature-importance");
  const entries = Object.entries(importance);
  new Chart(document.getElementById("chart-importance"), {
    type: "bar",
    data: {
      labels: entries.map(([k]) => k),
      datasets: [{
        data: entries.map(([, v]) => v),
        backgroundColor: PALETTE.amber,
        borderRadius: 4,
      }],
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { color: PALETTE.grid } },
        y: { grid: { display: false } },
      },
    },
  });

  // ---- Metrics table ----
  const metrics = await fetchJSON("/api/metrics");
  const rf = metrics.models.random_forest;
  const lr = metrics.models.linear_regression;
  document.getElementById("stat-mape").textContent = `${rf.mape}%`;

  const rows = [
    { name: "Random Forest (chosen)", m: rf, winner: true },
    { name: "Linear Regression (baseline)", m: lr, winner: false },
  ];
  document.getElementById("model-table-body").innerHTML = rows.map(r => `
    <tr class="${r.winner ? "winner" : ""}">
      <td>${r.name}</td>
      <td class="metric">${r.m.mae.toLocaleString()}</td>
      <td class="metric">${r.m.rmse.toLocaleString()}</td>
      <td class="metric">${r.m.mape}%</td>
      <td class="metric">${r.m.r2}</td>
    </tr>
  `).join("");

  document.getElementById("split-note").textContent =
    `Train: ${metrics.train_period}  ·  Test: ${metrics.test_period}`;

  // Default the date picker to a sensible value within the dataset range.
  const dateInput = document.getElementById("input-date");
  dateInput.min = data.summary.dataset_start;
  dateInput.value = data.summary.dataset_end;
}

function wirePredictForm() {
  const hourInput = document.getElementById("input-hour");
  const hourReadout = document.getElementById("hour-readout");
  hourInput.addEventListener("input", () => {
    hourReadout.textContent = `${String(hourInput.value).padStart(2, "0")}:00`;
  });

  document.getElementById("predict-btn").addEventListener("click", async () => {
    const dateVal = document.getElementById("input-date").value;
    const hourVal = hourInput.value;
    const resultValue = document.getElementById("result-value");
    const resultMeta = document.getElementById("result-meta");

    if (!dateVal) {
      resultMeta.textContent = "Pick a date first.";
      return;
    }

    resultValue.textContent = "…";
    resultMeta.textContent = "Running the model…";

    try {
      const res = await fetchJSON("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ date: dateVal, hour: hourVal }),
      });
      resultValue.textContent = fmtMW(res.predicted_mw);
      resultMeta.innerHTML =
        `<strong>${res.day_of_week}</strong>, ${res.date} at ${String(res.hour).padStart(2, "0")}:00` +
        (res.is_weekend ? " &middot; weekend" : " &middot; weekday");
    } catch (err) {
      resultMeta.textContent = "Something went wrong -- check the date and try again.";
    }
  });
}

renderCharts().catch((err) => console.error(err));
wirePredictForm();
