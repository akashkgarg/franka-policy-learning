"use strict";

const BLOCK_RGB = { red: "#d63a3a", green: "#33b34a", blue: "#3a6bdc" };
const ACTION_COLOR = { move_to: "#58a6ff", close_gripper: "#f0883e", open_gripper: "#3fb950", done: "#a371f7" };
const FPS = 15;
const PLOT_BG = { paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)", font: { color: "#c9d1d9", size: 11 } };
const $ = (id) => document.getElementById(id);
const fmt = (v, d = 3) => (v === null || v === undefined || Number.isNaN(v) ? "–" : Number(v).toFixed(d));
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const api = async (path, options) => {
  const response = await fetch(path, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || response.statusText);
  return data;
};

const state = { episodes: [], current: null, playing: false, lastPlotUpdate: 0, activeJob: null, labStatus: null };

// ------------------------------------------------------------------------------------------------- tabs
function showView(name) {
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.dataset.view === name));
  document.querySelectorAll(".view").forEach((v) => v.classList.toggle("active", v.id === `view-${name}`));
  if (name === "overview") renderOverview();
  if (name === "run") refreshLab();
  window.dispatchEvent(new Event("resize"));
}
document.querySelectorAll(".tab").forEach((t) => (t.onclick = () => showView(t.dataset.view)));

// ------------------------------------------------------------------------------------------------- episode list
async function loadEpisodes() {
  state.episodes = await api("/api/episodes");
  const sources = [...new Set(state.episodes.map((e) => e.source))];
  const sourceSel = $("f-source");
  const previous = sourceSel.value;
  sourceSel.innerHTML = `<option value="">all sources</option>` +
    sources.map((s) => `<option value="${s}">${sourceLabel(s)} (${state.episodes.filter((e) => e.source === s).length})</option>`).join("");
  sourceSel.value = previous || (sources.includes("llm") ? "llm" : "");
  const tasks = [...new Set(state.episodes.map((e) => `${e.top} on ${e.bottom}`))].sort();
  const taskSel = $("f-task");
  const prevTask = taskSel.value;
  taskSel.innerHTML = `<option value="">all tasks</option>` + tasks.map((t) => `<option>${t}</option>`).join("");
  taskSel.value = prevTask;
  renderList();
  const total = state.episodes.length;
  const llm = state.episodes.filter((e) => e.source === "llm");
  $("topbar-status").textContent = `${total} episodes · LLM dataset ${llm.filter((e) => e.success).length}/${llm.length} successful`;
}

function sourceLabel(s) {
  if (s === "llm") return "LLM dataset";
  if (s === "lab") return "My runs";
  if (s === "vision_wood") return "LLM vision-only, wood table";
  if (s === "vision_grid") return "LLM vision-only, grid table (partial)";
  if (s === "vision_pilot") return "LLM vision-only pilot (grid)";
  if (s === "vision_wood_pilot") return "LLM vision-only pilot (wood)";
  if (s.startsWith("eval_")) return `eval: ${s.slice(5)}`;
  return s;
}

function filteredEpisodes() {
  const source = $("f-source").value, task = $("f-task").value, outcome = $("f-outcome").value;
  const search = $("f-search").value.trim().toLowerCase();
  let list = state.episodes.filter((e) =>
    (!source || e.source === source) &&
    (!task || `${e.top} on ${e.bottom}` === task) &&
    (!outcome || (outcome === "success") === !!e.success) &&
    (!search || e.name.toLowerCase().includes(search) || String(e.seed).includes(search)));
  const sort = $("f-sort").value;
  const key = { steps: (e) => -(e.num_steps || 0), error: (e) => -(e.xy_error_m || 0), latency: (e) => -(e.llm_latency_s || 0) }[sort];
  if (key) list = [...list].sort((a, b) => key(a) - key(b));
  return list;
}

function renderList() {
  const list = filteredEpisodes();
  $("list-count").textContent = `${list.length} episodes`;
  const container = $("episode-list");
  container.innerHTML = list.slice(0, 1500).map((e) => `
    <div class="ep-item ${state.current && state.current.summary.id === e.id ? "active" : ""}" data-id="${e.id}">
      <span class="dot" style="background:${e.success ? "var(--ok)" : "var(--bad)"}"></span>
      <div>
        <div class="name">${esc(e.name)}</div>
        <div class="sub"><span class="swatch" style="background:${BLOCK_RGB[e.top]}"></span>${e.top} on
          <span class="swatch" style="background:${BLOCK_RGB[e.bottom]}"></span>${e.bottom} · ${sourceLabel(e.source)}</div>
      </div>
      <div class="meta">${e.mode === "llm" ? `${e.num_steps} steps` : "policy"}<br>${fmt(e.xy_error_m * 100, 1)} cm</div>
    </div>`).join("");
  container.querySelectorAll(".ep-item").forEach((el) => (el.onclick = () => openEpisode(el.dataset.id)));
}
["f-source", "f-task", "f-outcome", "f-sort"].forEach((id) => ($(id).onchange = renderList));
$("f-search").oninput = renderList;

// ------------------------------------------------------------------------------------------------- episode detail
async function openEpisode(id) {
  showView("episodes");
  const main = $("episode-main");
  main.innerHTML = `<div class="empty">Loading ${esc(id)} (first open extracts the timeline from the bag)…</div>`;
  let data;
  try {
    data = await api(`/api/episode/${id}`);
  } catch (err) {
    main.innerHTML = `<div class="empty">Failed to load: ${esc(err.message)}</div>`;
    return;
  }
  state.current = data;
  document.querySelectorAll(".ep-item").forEach((el) => el.classList.toggle("active", el.dataset.id === id));
  renderEpisode(data);
}

function renderEpisode(data) {
  const { episode, timeline, summary } = data;
  const isLLM = episode.mode === "llm";
  const main = $("episode-main");
  const outcome = episode.success ? `<span class="badge ok">stacked</span>` : `<span class="badge bad">not stacked</span>`;
  main.innerHTML = `
    <div class="ep-header">
      <h2>${esc(summary.name)}</h2>${outcome}
      <span class="badge">${sourceLabel(summary.source)}</span>
      <span class="badge">${isLLM ? esc(episode.model) : "policy"}</span>
      ${isLLM ? `<span class="badge">${summary.observation === "vision" ? "vision only" : "full state"}</span>` : ""}
      <span class="badge">${summary.table} table</span>
      <div class="ep-actions">
        <button class="secondary" id="btn-rerun">Run this layout…</button>
      </div>
    </div>
    <div style="margin-bottom:6px">${esc(summary.instruction)}</div>
    <div class="ep-stats">
      <span>seed ${episode.seed}</span>
      <span>${isLLM ? `${episode.num_steps} LLM steps` : `${episode.num_steps} policy queries`}</span>
      <span>stack XY error ${fmt(episode.xy_error_m * 100, 1)} cm</span>
      <span>height diff ${fmt(episode.height_diff_m * 100, 1)} cm</span>
      <span>${fmt(episode.sim_seconds, 1)} s sim · ${fmt(episode.wall_seconds, 0)} s wall</span>
      ${summary.llm_latency_s ? `<span>${fmt(summary.llm_latency_s, 1)} s waiting on the LLM</span>` : ""}
      ${episode.error ? `<span class="badge bad">${esc(episode.error)}</span>` : ""}
    </div>
    <div class="media-row" style="margin-top:12px">
      <div class="card">
        ${summary.has_bag ? `<video id="video" controls muted playsinline preload="auto" src="/api/video/${summary.id}.mp4"></video>
        <div class="video-labels"><span>scene camera</span><span>wrist camera</span></div>` :
          `<div class="empty">No bag recorded for this episode (only episode.json was kept).</div>`}
      </div>
      <div class="card"><div id="plot3d" class="plot3d"></div></div>
    </div>
    ${timeline ? `<div class="timeline" id="timeline"></div>
    <div class="legend">
      <span><i style="background:var(--move)"></i>move_to</span><span><i style="background:var(--close)"></i>close_gripper</span>
      <span><i style="background:var(--open)"></i>open_gripper</span><span><i style="background:var(--done)"></i>done</span>
      <span>Click the bar or a step to seek. The LLM's thinking time takes no simulated time, so it does not appear here.</span>
    </div>` : `<div class="empty">${esc(data.timeline_error || "No timeline (no bag).")}</div>`}
    <div class="detail-row">
      <div class="card"><h3>${isLLM ? "LLM decisions" : "Policy"}</h3><div class="steps" id="steps"></div></div>
      <div class="card"><h3>Gripper height &amp; fingers</h3><div id="plot2d" class="plot2d"></div>
        <h3 style="margin-top:10px">Block positions (start → end)</h3><div class="kv" id="block-table"></div></div>
    </div>`;
  $("btn-rerun").onclick = () => prefillRun(episode);
  renderSteps(episode, timeline, summary);
  renderBlockTable(episode, timeline);
  if (timeline) {
    renderTimelineBar(timeline, episode);
    render3D(timeline, episode);
    render2D(timeline);
  }
  const video = $("video");
  if (video) {
    video.addEventListener("timeupdate", () => syncToVideo(false));
    video.addEventListener("seeked", () => syncToVideo(true));
    video.addEventListener("play", () => { state.playing = true; requestAnimationFrame(playLoop); });
    video.addEventListener("pause", () => (state.playing = false));
    video.addEventListener("error", () => {
      video.insertAdjacentHTML("afterend", `<div class="error">Video generation failed; check the server log.</div>`);
    });
  }
}

function renderBlockTable(episode, timeline) {
  const rows = [];
  for (const color of ["red", "green", "blue"]) {
    const start = timeline ? timeline.blocks[color][0] : (episode.initial_block_positions || {})[color];
    const end = (episode.block_positions || {})[color];
    const s = start ? `(${start.map((v) => fmt(v, 3)).join(", ")})` : "–";
    const e = end ? `(${end.map((v) => fmt(v, 3)).join(", ")})` : "–";
    rows.push(`<div><span class="swatch" style="background:${BLOCK_RGB[color]}"></span>${color}</div><div>${s} → ${e}</div>`);
  }
  $("block-table").innerHTML = rows.join("");
}

function llmImages(summary, s) {
  if (!summary.has_llm_images) return "";
  const pad = String(s.index).padStart(2, "0");
  const d = s.decision;
  return `<div class="llm-images">${["scene", "wrist"].map((cam) => {
    const marker = d.action === "locate" && d.camera === cam ?
      `<span class="pointer" style="left:${(100 * d.u) / 640}%;top:${(100 * d.v) / 480}%"></span>` : "";
    return `<a href="/api/llm_image/${summary.id}/step${pad}_${cam}.jpg" target="_blank">
      <img loading="lazy" src="/api/llm_image/${summary.id}/step${pad}_${cam}.jpg">${marker}</a>`;
  }).join("")}</div>`;
}

function renderSteps(episode, timeline, summary) {
  const container = $("steps");
  if (episode.mode !== "llm") {
    const n = episode.steps.length;
    container.innerHTML = `<div class="hint">The policy was queried ${n} times (every ${n ? fmt(episode.sim_seconds / n, 2) : "–"} s of sim
      time). Each query returns a chunk of 16 future actions (grasp-point goal + gripper open/closed); the first 8 are
      executed before the next query. Grey diamonds in the 3D view are the first goal of every chunk.</div>`;
    return;
  }
  const segments = timeline ? timeline.segments : [];
  container.innerHTML = episode.steps.map((s, i) => {
    if (!s.decision) return `<div class="step"><div class="step-head"><span class="idx">${i + 1}</span><span class="act">error</span></div>
      <div class="why">${esc(s.error)}</div></div>`;
    const d = s.decision;
    const args = actionArgs(d);
    const r = s.result || {};
    let result = "";
    if (d.action === "locate") {
      result = r.point ? `depth point (${r.point.map((v) => fmt(v)).join(", ")})` : esc(r.error);
      if (s.debug) result += ` · ground truth: ${fmt(s.debug.xy_error_to_top_center_m * 1000, 1)} mm from the ${s.debug.nearest_block} block's top center (not shown to the LLM)`;
    } else if (d.action === "move_to") result = `${r.reached ? "reached" : "NOT reached"} · error ${fmt((r.position_error_m || 0) * 1000, 1)} mm · ${fmt(r.duration_s, 2)} s`;
    else if (r.finger_positions_m) result = `fingers ${r.finger_positions_m.map((v) => fmt(v, 3)).join(" / ")} m · ${fmt(r.duration_s, 2)} s`;
    if (r.clipped_target) result += ` · clipped to (${r.clipped_target.join(", ")})`;
    const seg = segments.find((g) => g.index === s.index);
    return `<div class="step ${d.action}" data-start="${seg ? seg.start : ""}" data-index="${s.index}">
      <div class="step-head"><span class="idx">${s.index + 1}</span><span class="act">${d.action}${args}</span>
        <span class="lat">${s.llm ? `${fmt(s.llm.latency_s, 1)} s · ${s.llm.prompt_tokens}+${s.llm.completion_tokens} tok` : ""}</span></div>
      <div class="why">${esc(d.reasoning)}</div>
      ${result ? `<div class="res">${result}</div>` : ""}
      ${llmImages(summary, s)}
      <details><summary>prompt sent to the LLM</summary><pre>${esc(s.prompt)}</pre></details>
    </div>`;
  }).join("");
  container.querySelectorAll(".step").forEach((el) => {
    el.onclick = (ev) => {
      if (ev.target.closest("details")) return;
      if (el.dataset.start !== "") seekToSim(parseFloat(el.dataset.start) + 0.01);
    };
  });
}

function actionArgs(d) {
  if (d.action === "move_to") return `(${fmt(d.x)}, ${fmt(d.y)}, ${fmt(d.z)})`;
  if (d.action === "locate") return `(${d.camera}, ${Math.round(d.u)}, ${Math.round(d.v)})`;
  return "()";
}

function renderTimelineBar(timeline, episode) {
  const bar = $("timeline");
  const total = timeline.t[timeline.t.length - 1] || 1;
  let html = "";
  if (episode.mode === "llm") {
    for (const s of timeline.segments) {
      const left = (100 * s.start) / total, width = Math.max(0.3, (100 * (s.end - s.start)) / total);
      const label = s.action === "move_to" ? `${s.index + 1} move` : s.action === "close_gripper" ? `${s.index + 1} close` :
        s.action === "open_gripper" ? `${s.index + 1} open` : s.action === "locate" ? "" : "done";
      html += `<div class="seg ${s.action}" style="left:${left}%;width:${width}%" title="step ${s.index + 1}: ${s.action}">${label}</div>`;
    }
  } else {
    html += `<div class="seg policy" style="left:0;width:100%">policy control</div>`;
  }
  html += `<div class="playhead" id="playhead" style="left:0"></div>`;
  bar.innerHTML = html;
  bar.onclick = (ev) => {
    const rect = bar.getBoundingClientRect();
    seekToSim(((ev.clientX - rect.left) / rect.width) * total);
  };
}

function render3D(timeline, episode) {
  const g = timeline.grasp;
  const closed = timeline.gripper_open_cmd.length ? timeline.gripper_open_cmd.map((v) => 1 - v) : g.map(() => 0);
  const traces = [];
  traces.push({
    type: "scatter3d", mode: "lines", name: "grasp point (green open / orange closed)",
    x: g.map((p) => p[0]), y: g.map((p) => p[1]), z: g.map((p) => p[2]),
    line: { width: 5, color: closed, colorscale: [[0, "#3fb950"], [1, "#f0883e"]], cmin: 0, cmax: 1 },
    hovertemplate: "t=%{text}s<br>(%{x:.3f}, %{y:.3f}, %{z:.3f})<extra></extra>", text: timeline.t.map((t) => t.toFixed(2)),
  });
  for (const color of ["red", "green", "blue"]) {
    const b = timeline.blocks[color];
    traces.push({ type: "scatter3d", mode: "lines", name: `${color} block path`, showlegend: false,
      x: b.map((p) => p[0]), y: b.map((p) => p[1]), z: b.map((p) => p[2]),
      line: { width: 3, color: BLOCK_RGB[color], dash: "dot" }, hoverinfo: "skip" });
    traces.push({ type: "scatter3d", mode: "markers", name: `${color} start`, showlegend: false,
      x: [b[0][0]], y: [b[0][1]], z: [b[0][2]],
      marker: { size: 9, symbol: "square-open", color: BLOCK_RGB[color], line: { width: 2 } },
      hovertemplate: `${color} block start<extra></extra>` });
  }
  const located = episode.steps.filter((s) => s.result && s.result.point);
  if (located.length) {
    traces.push({ type: "scatter3d", mode: "markers", name: "locate() depth readings",
      x: located.map((s) => s.result.point[0]), y: located.map((s) => s.result.point[1]), z: located.map((s) => s.result.point[2]),
      text: located.map((s) => `step ${s.index + 1}: ${s.decision.camera} (${Math.round(s.decision.u)}, ${Math.round(s.decision.v)})`),
      marker: { size: 5, symbol: "cross", color: "#ffd33d" }, hovertemplate: "%{text}<br>(%{x:.3f}, %{y:.3f}, %{z:.3f})<extra></extra>" });
  }
  if (episode.mode === "llm") {
    const moves = timeline.segments.filter((s) => s.target);
    traces.push({ type: "scatter3d", mode: "markers+text", name: "LLM move_to targets",
      x: moves.map((s) => s.target[0]), y: moves.map((s) => s.target[1]), z: moves.map((s) => s.target[2]),
      text: moves.map((s) => String(s.index + 1)), textposition: "top center", textfont: { color: "#9ecbff", size: 11 },
      marker: { size: 5, symbol: "diamond", color: "#58a6ff" },
      hovertemplate: "step %{text}: move_to(%{x:.3f}, %{y:.3f}, %{z:.3f})<extra></extra>" });
  } else if (timeline.policy_targets.length) {
    const p = timeline.policy_targets;
    traces.push({ type: "scatter3d", mode: "markers", name: "policy goals (first of each chunk)",
      x: p.map((v) => v[1]), y: p.map((v) => v[2]), z: p.map((v) => v[3]),
      marker: { size: 3, symbol: "diamond", color: p.map((v) => (v[4] > 0.5 ? "#8b949e" : "#f0883e")) },
      hovertemplate: "goal (%{x:.3f}, %{y:.3f}, %{z:.3f})<extra></extra>" });
  }
  // dynamic traces: current grasp point + current blocks
  state.dynamicTrace = traces.length;
  traces.push({ type: "scatter3d", mode: "markers", name: "now", showlegend: false,
    x: [g[0][0]], y: [g[0][1]], z: [g[0][2]], marker: { size: 7, color: "#ffffff", line: { color: "#000", width: 1 } }, hoverinfo: "skip" });
  traces.push({ type: "scatter3d", mode: "markers", name: "blocks now", showlegend: false,
    x: ["red", "green", "blue"].map((c) => timeline.blocks[c][0][0]), y: ["red", "green", "blue"].map((c) => timeline.blocks[c][0][1]),
    z: ["red", "green", "blue"].map((c) => timeline.blocks[c][0][2]),
    marker: { size: 11, symbol: "square", color: ["red", "green", "blue"].map((c) => BLOCK_RGB[c]), opacity: 0.95 }, hoverinfo: "skip" });
  const layout = {
    ...PLOT_BG, margin: { l: 0, r: 0, t: 0, b: 0 }, showlegend: true,
    legend: { x: 0, y: 1, bgcolor: "rgba(13,17,23,.6)", font: { size: 10 } },
    scene: {
      xaxis: { title: "x", range: [0.15, 0.75], gridcolor: "#30363d", zerolinecolor: "#30363d" },
      yaxis: { title: "y", range: [-0.4, 0.4], gridcolor: "#30363d", zerolinecolor: "#30363d" },
      zaxis: { title: "z", range: [0, 0.4], gridcolor: "#30363d", zerolinecolor: "#30363d" },
      aspectmode: "manual", aspectratio: { x: 1.2, y: 1.6, z: 0.8 },
      camera: { eye: { x: 1.6, y: 0.9, z: 0.9 } },
    },
  };
  Plotly.newPlot("plot3d", traces, layout, { displaylogo: false, responsive: true });
}

function render2D(timeline) {
  const t = timeline.t;
  const traces = [
    { x: t, y: timeline.grasp.map((p) => p[2]), name: "grasp z (m)", line: { color: "#58a6ff" } },
  ];
  if (timeline.fingers.length) traces.push({ x: t, y: timeline.fingers.map((f) => f[0] + f[1]), name: "finger opening (m)", line: { color: "#f0883e" } });
  if (timeline.gripper_open_cmd.length) traces.push({ x: t, y: timeline.gripper_open_cmd.map((v) => v * 0.08), name: "open command", line: { color: "#3fb950", dash: "dot", shape: "hv" } });
  for (const color of ["red", "green", "blue"]) {
    traces.push({ x: t, y: timeline.blocks[color].map((p) => p[2]), name: `${color} block z`, line: { color: BLOCK_RGB[color], width: 1 } });
  }
  const layout = { ...PLOT_BG, margin: { l: 40, r: 10, t: 10, b: 30 }, xaxis: { title: "sim time (s)", gridcolor: "#30363d" },
    yaxis: { gridcolor: "#30363d" }, legend: { orientation: "h", y: -0.25, font: { size: 10 } },
    shapes: [{ type: "line", x0: 0, x1: 0, yref: "paper", y0: 0, y1: 1, line: { color: "#fff", width: 1 } }] };
  Plotly.newPlot("plot2d", traces, layout, { displaylogo: false, responsive: true });
  $("plot2d").on("plotly_click", (ev) => seekToSim(ev.points[0].x));
}

// ------------------------------------------------------------------------------------------------- sync
function videoSimTime() {
  const video = $("video");
  const tl = state.current && state.current.timeline;
  if (!video || !tl) return 0;
  const frames = tl.video_times;
  if (!frames.length) return video.currentTime;
  const i = Math.min(frames.length - 1, Math.max(0, Math.floor(video.currentTime * FPS + 1e-3)));
  return frames[i];
}

function seekToSim(simT) {
  const video = $("video");
  const tl = state.current && state.current.timeline;
  if (!video || !tl) return;
  const frames = tl.video_times;
  let i = frames.findIndex((t) => t >= simT - 1e-3);
  if (i < 0) i = frames.length - 1;
  video.currentTime = (i + 0.5) / FPS;
  syncToVideo(true);
}

function playLoop() {
  syncToVideo(false);
  if (state.playing) requestAnimationFrame(playLoop);
}

function syncToVideo(force) {
  const tl = state.current && state.current.timeline;
  if (!tl) return;
  const now = performance.now();
  if (!force && now - state.lastPlotUpdate < 90) return;
  state.lastPlotUpdate = now;
  const simT = videoSimTime();
  const total = tl.t[tl.t.length - 1] || 1;
  const playhead = $("playhead");
  if (playhead) playhead.style.left = `${(100 * simT) / total}%`;
  const k = Math.min(tl.t.length - 1, Math.max(0, Math.round(simT * 30)));
  const g = tl.grasp[k];
  const colors = ["red", "green", "blue"];
  if ($("plot3d") && $("plot3d").data) {
    Plotly.restyle("plot3d", {
      x: [[g[0]], colors.map((c) => tl.blocks[c][k][0])],
      y: [[g[1]], colors.map((c) => tl.blocks[c][k][1])],
      z: [[g[2]], colors.map((c) => tl.blocks[c][k][2])],
    }, [state.dynamicTrace, state.dynamicTrace + 1]);
  }
  if ($("plot2d") && $("plot2d").layout) Plotly.relayout("plot2d", { "shapes[0].x0": simT, "shapes[0].x1": simT });
  let active = null;
  for (const s of tl.segments) if (simT >= s.start - 1e-3) active = s.index;
  document.querySelectorAll(".step").forEach((el) => {
    const on = el.dataset.index !== undefined && Number(el.dataset.index) === active;
    if (on && !el.classList.contains("active")) {
      const list = el.parentElement;
      const top = el.offsetTop;
      if (top < list.scrollTop || top + el.offsetHeight > list.scrollTop + list.clientHeight) list.scrollTop = top - 8;
    }
    el.classList.toggle("active", on);
  });
}

// ------------------------------------------------------------------------------------------------- overview
async function renderOverview() {
  const root = $("overview");
  root.innerHTML = `<div class="empty wide">Computing statistics…</div>`;
  const s = await api("/api/stats");
  const L = s.llm;
  const sourceNames = Object.keys(s.by_source);
  const rate = (k) => s.by_source[k].successes / Math.max(1, s.by_source[k].episodes);
  const median = (a) => { const b = [...a].sort((x, y) => x - y); return b.length ? b[Math.floor(b.length / 2)] : null; };
  root.innerHTML = `
    <div class="card wide"><h3>LLM dataset</h3><div class="big-numbers">
      <div><b>${s.by_source.llm ? s.by_source.llm.episodes : 0}</b><span>episodes</span></div>
      <div><b>${s.by_source.llm ? (100 * rate("llm")).toFixed(1) : 0}%</b><span>stacked</span></div>
      <div><b>${L.latencies.length}</b><span>LLM calls</span></div>
      <div><b>${fmt(median(L.latencies), 1)} s</b><span>median latency</span></div>
      <div><b>${(L.tokens.prompt / 1e6).toFixed(2)}M / ${(L.tokens.completion / 1e6).toFixed(2)}M</b><span>prompt / completion tokens</span></div>
      <div><b>${fmt(median(L.xy_errors) * 1000, 1)} mm</b><span>median stack error</span></div>
    </div></div>
    <div class="card"><h3>Success rate by source</h3><div class="plot" id="ov-sources"></div></div>
    <div class="card"><h3>LLM steps per episode</h3><div class="plot" id="ov-steps"></div></div>
    <div class="card"><h3>Where the LLM grasped and released (top-down)</h3><div class="plot" id="ov-xy"></div></div>
    <div class="card"><h3>LLM call latency</h3><div class="plot" id="ov-latency"></div></div>
    <div class="card"><h3>Release height chosen by the LLM</h3><div class="plot" id="ov-place"></div></div>
    <div class="card"><h3>Stack XY error</h3><div class="plot" id="ov-error"></div></div>
    <div class="card"><h3>Actions</h3><div class="plot" id="ov-actions"></div></div>
    <div class="card"><h3>Tasks</h3><div class="plot" id="ov-tasks"></div></div>
    ${Object.keys(s.vision.steps_hist).length ? `
    <div class="card wide"><h3>Vision-only LLM (cameras + depth locate, no block positions)</h3><div class="big-numbers">
      <div><b>${s.vision.xy_errors.length}</b><span>episodes</span></div>
      <div><b>${(() => { const v = Object.entries(s.by_source).filter(([k]) => k.startsWith("vision"));
        return `${v.reduce((a, [, x]) => a + x.successes, 0)}/${v.reduce((a, [, x]) => a + x.episodes, 0)}`; })()}</b><span>stacked (all vision sources)</span></div>
      <div><b>${fmt(median(s.vision.latencies), 1)} s</b><span>median latency</span></div>
      <div><b>${fmt(median(s.vision.xy_errors.filter((v) => v !== null)) * 1000, 1)} mm</b><span>median stack error</span></div>
      <div><b>${fmt(median(s.vision.locate_errors.map((v) => v[1])) * 1000, 1)} mm</b><span>median locate error (ground truth)</span></div>
    </div></div>
    <div class="card"><h3>Vision: steps per episode</h3><div class="plot" id="ov-v-steps"></div></div>
    <div class="card"><h3>Vision: locate calls per episode</h3><div class="plot" id="ov-v-locates"></div></div>
    <div class="card"><h3>Vision: locate error vs. block top center</h3><div class="plot" id="ov-v-error"></div></div>` : ""}
    <div class="card wide explain"><h3>How the policy's action space relates to the LLM's</h3>
      <p>The LLM acts in an <b>event-based</b> space: one of 4 discrete primitives per decision (<code>move_to(x,y,z)</code>,
      <code>open_gripper</code>, <code>close_gripper</code>, <code>done</code>), and the next decision only comes after the
      primitive finishes. A typical episode is 8–9 decisions.</p>
      <p>The trained policy acts in a <b>time-based</b> space: every 1/15 s it outputs a continuous 4-number action
      <code>(goal x, goal y, goal z, gripper open?)</code> — exactly the LLM's current <code>move_to</code> target and gripper
      state, resampled at the camera rate. "16" is the <b>chunk length</b>, not a number of classes: each query predicts the
      next 16 of these 4-D actions (≈1 s), and the first 8 are executed before re-querying.</p>
    </div>`;
  const cfg = { displaylogo: false, responsive: true };
  const base = (extra) => ({ ...PLOT_BG, margin: { l: 45, r: 10, t: 10, b: 40 }, xaxis: { gridcolor: "#30363d" }, yaxis: { gridcolor: "#30363d" }, ...extra });
  Plotly.newPlot("ov-sources", [{ type: "bar", x: sourceNames.map(sourceLabel), y: sourceNames.map((k) => 100 * rate(k)),
    text: sourceNames.map((k) => `${s.by_source[k].successes}/${s.by_source[k].episodes}`), textposition: "auto",
    marker: { color: sourceNames.map((k) => (k === "llm" ? "#58a6ff" : k.startsWith("vision") ? "#ffd33d" : k.includes("goal_state") ? "#3fb950" : "#8b949e")) } }],
    base({ yaxis: { title: "% stacked", range: [0, 105], gridcolor: "#30363d" } }), cfg);
  Plotly.newPlot("ov-steps", [{ type: "bar", x: Object.keys(L.steps_hist), y: Object.values(L.steps_hist), marker: { color: "#58a6ff" } }],
    base({ xaxis: { title: "decisions", dtick: 1, gridcolor: "#30363d" } }), cfg);
  const xy = [];
  for (const c of ["red", "green", "blue"]) {
    const p = L.picks.filter((v) => v[2] === c), q = L.places.filter((v) => v[2] === c);
    xy.push({ type: "scattergl", mode: "markers", name: `grasp ${c}`, x: p.map((v) => v[1]), y: p.map((v) => v[0]), marker: { color: BLOCK_RGB[c], size: 5, symbol: "circle" } });
    xy.push({ type: "scattergl", mode: "markers", name: `release on ${c}`, x: q.map((v) => v[1]), y: q.map((v) => v[0]), marker: { color: BLOCK_RGB[c], size: 6, symbol: "x" } });
  }
  Plotly.newPlot("ov-xy", xy, base({ xaxis: { title: "y (m)", autorange: "reversed", gridcolor: "#30363d" }, yaxis: { title: "x (m)", scaleanchor: "x", gridcolor: "#30363d" }, legend: { font: { size: 10 } } }), cfg);
  Plotly.newPlot("ov-latency", [{ type: "histogram", x: L.latencies, xbins: { size: 0.5 }, marker: { color: "#a371f7" } }], base({ xaxis: { title: "seconds", range: [0, 20], gridcolor: "#30363d" } }), cfg);
  Plotly.newPlot("ov-place", [{ type: "histogram", x: L.place_heights, xbins: { size: 0.005 }, marker: { color: "#3fb950" } }],
    base({ xaxis: { title: "grasp-point z at open_gripper (m); 0.075 = block-on-block center", gridcolor: "#30363d" } }), cfg);
  Plotly.newPlot("ov-error", [{ type: "histogram", x: L.xy_errors.map((v) => v * 1000), xbins: { size: 1 }, marker: { color: "#f0883e" } }], base({ xaxis: { title: "mm", gridcolor: "#30363d" } }), cfg);
  Plotly.newPlot("ov-actions", [{ type: "pie", labels: Object.keys(L.actions), values: Object.values(L.actions), hole: 0.45,
    marker: { colors: Object.keys(L.actions).map((a) => ACTION_COLOR[a] || "#8b949e") } }], { ...PLOT_BG, margin: { l: 10, r: 10, t: 10, b: 10 } }, cfg);
  Plotly.newPlot("ov-tasks", [{ type: "bar", x: Object.keys(L.per_task), y: Object.values(L.per_task), marker: { color: "#8b949e" } }], base({}), cfg);
  if ($("ov-v-steps")) {
    const V = s.vision;
    Plotly.newPlot("ov-v-steps", [{ type: "bar", x: Object.keys(V.steps_hist), y: Object.values(V.steps_hist), marker: { color: "#58a6ff" } }],
      base({ xaxis: { title: "decisions", gridcolor: "#30363d" } }), cfg);
    Plotly.newPlot("ov-v-locates", [{ type: "bar", x: Object.keys(V.locates_hist), y: Object.values(V.locates_hist), marker: { color: "#ffd33d" } }],
      base({ xaxis: { title: "locate calls", dtick: 1, gridcolor: "#30363d" } }), cfg);
    Plotly.newPlot("ov-v-error", ["scene", "wrist"].map((c) => ({ type: "histogram", name: `${c} camera`, opacity: 0.75,
      x: V.locate_errors.filter((v) => v[0] === c).map((v) => v[1] * 1000), xbins: { size: 2 } })),
      base({ barmode: "overlay", xaxis: { title: "XY distance from the nearest block's top center (mm)", gridcolor: "#30363d" } }), cfg);
  }
}

// ------------------------------------------------------------------------------------------------- run
function prefillRun(episode) {
  $("r-top").value = episode.task.top;
  $("r-bottom").value = episode.task.bottom;
  $("r-seed").value = episode.seed;
  if (episode.mode === "llm") state.pendingController = episode.observation === "vision" ? "llm-vision" : "llm";
  $("r-table").value = episode.table || "grid";
  showView("run");
}

async function refreshLab() {
  let s;
  try { s = await api("/api/lab/status"); } catch (err) { $("r-worker").textContent = err.message; return; }
  state.labStatus = s;
  const sel = $("r-controller");
  const previous = sel.value;
  const opts = [`<option value="llm">LLM, full state (sees block positions)</option>`,
    `<option value="llm-vision">LLM, vision only (cameras + depth locate)</option>`];
  for (const [name, p] of Object.entries(s.policies)) {
    opts.push(`<option value="policy:${name}">policy: ${name}${p.eval ? ` (eval ${p.eval})` : ""}</option>`);
  }
  sel.innerHTML = opts.join("");
  sel.value = state.pendingController || previous || (s.policies.goal_state ? "policy:goal_state" : "llm");
  state.pendingController = null;
  updateControllerHint();
  const workerText = { ready: "ready", starting: "starting Isaac Sim (about a minute)…", stopped: "not running (starts with the first run)" }[s.worker];
  $("r-worker").innerHTML = `<span class="badge ${s.worker === "ready" ? "ok" : s.worker === "starting" ? "warn" : ""}">${workerText}</span>
    <span class="hint">GPU ${s.gpu}</span> ${s.worker === "stopped" ? `<button class="secondary" id="r-start">Start now</button>` : ""}`;
  if ($("r-start")) $("r-start").onclick = async () => { await api("/api/lab/start", { method: "POST", body: "{}" }); refreshLab(); };
  const t = s.token;
  $("r-token").innerHTML = !t.present ? `<span class="badge bad">no token</span>` :
    `<span class="badge ${t.expired ? "bad" : "ok"}">${t.expired ? "expired" : "valid"}</span>
     <span class="hint">${esc(t.subject || "")} · expires ${new Date(t.expires * 1000).toLocaleString()}</span>`;
  $("job-list").innerHTML = s.jobs.map((j) => `<div class="job">
      <span class="badge ${j.state === "done" ? (j.success ? "ok" : "bad") : j.state === "error" ? "bad" : "warn"}">${j.state === "done" ? (j.success ? "stacked" : "failed") : j.state}</span>
      <span>${esc(j.job.mode === "llm" ? j.job.model : `policy ${j.job.policy}`)}</span>
      <span>${j.job.top} on ${j.job.bottom} · seed ${j.job.seed}</span>
      <a data-watch="${j.id}">watch</a>${j.state === "done" ? `<a data-open="lab/${j.id}">open</a>` : ""}
    </div>`).join("") || `<div class="hint">No runs yet.</div>`;
  document.querySelectorAll("[data-watch]").forEach((a) => (a.onclick = () => watchJob(a.dataset.watch)));
  document.querySelectorAll("[data-open]").forEach((a) => (a.onclick = async () => { await loadEpisodes(); openEpisode(a.dataset.open); }));
}

function updateControllerHint() {
  const value = $("r-controller").value;
  const isLLM = value.startsWith("llm");
  $("r-llm-options").classList.toggle("hidden", !isLLM);
  $("r-policy-options").classList.toggle("hidden", isLLM);
  const s = state.labStatus;
  $("r-max-steps").value = value === "llm-vision" ? 40 : 24;
  if (value === "llm-vision") {
    $("r-controller-hint").textContent = "The LLM only sees the two camera images and the robot's own gripper state. " +
      "It finds blocks with locate(camera, u, v), which reads the depth camera at a pixel it points to.";
  } else if (isLLM) {
    $("r-controller-hint").textContent = "Each step sends the measured state, history and both camera images to the LLM, which picks one primitive.";
  } else if (s) {
    const p = s.policies[value.split(":")[1]] || {};
    $("r-controller-hint").textContent = `Inputs: ${p.inputs || "?"}. Trained on the LLM episodes; validation L1 ${fmt(p.val_l1, 3)}.`;
  }
}
$("r-controller").onchange = updateControllerHint;
$("r-random").onclick = () => ($("r-seed").value = Math.floor(1e6 + Math.random() * 8.9e6));

$("r-submit").onclick = async () => {
  $("r-error").textContent = "";
  const body = {
    controller: $("r-controller").value, top: $("r-top").value, bottom: $("r-bottom").value, seed: $("r-seed").value,
    table: $("r-table").value,
    instruction: $("r-instruction").value, model: $("r-model").value, max_steps: $("r-max-steps").value,
    policy_seconds: $("r-policy-seconds").value,
  };
  if (body.controller.startsWith("llm") && state.labStatus && (!state.labStatus.token.present || state.labStatus.token.expired)) {
    $("r-error").textContent = "The LLM Gateway token is missing or expired; paste a new one below.";
    return;
  }
  $("r-submit").disabled = true;
  try {
    const job = await api("/api/lab/run", { method: "POST", body: JSON.stringify(body) });
    watchJob(job.id);
  } catch (err) {
    $("r-error").textContent = err.message;
  } finally {
    $("r-submit").disabled = false;
    refreshLab();
  }
};

$("r-token-save").onclick = async () => {
  try {
    await api("/api/lab/token", { method: "POST", body: JSON.stringify({ token: $("r-token-input").value }) });
    $("r-token-input").value = "";
    refreshLab();
  } catch (err) { $("r-error").textContent = err.message; }
};

function watchJob(id) {
  state.activeJob = id;
  $("live-events").innerHTML = "";
  pollJob();
}

async function pollJob() {
  const id = state.activeJob;
  if (!id) return;
  let s;
  try { s = await api(`/api/lab/job/${id}`); } catch (err) { setTimeout(pollJob, 1000); return; }
  if (id !== state.activeJob) return;
  const j = s.job;
  $("live-title").textContent = `${j.mode === "llm" ? j.model : `policy ${j.policy}`} · ${j.top} on ${j.bottom} · seed ${j.seed}`;
  const stateText = s.state === "done" ? (s.result.success ? "stacked" : "not stacked") : s.state;
  $("live-state").className = `badge ${s.state === "done" ? (s.result.success ? "ok" : "bad") : s.state === "error" ? "bad" : "warn"}`;
  $("live-state").textContent = stateText;
  if (s.state === "running" || s.state === "done") {
    const bust = Date.now();
    $("live-scene").src = `/api/lab/live/${id}/scene.jpg?${bust}`;
    $("live-wrist").src = `/api/lab/live/${id}/wrist.jpg?${bust}`;
  }
  const live = s.live;
  $("live-telemetry").innerHTML = live ? `t=${fmt(live.sim_time, 2)} s · grasp point (${live.grasp_point.map((v) => fmt(v)).join(", ")}) ·
    target (${live.target.map((v) => fmt(v)).join(", ")}) · gripper ${live.gripper} (${live.fingers.map((v) => fmt(v)).join(" / ")})` :
    (s.state === "queued" ? "waiting for the sim worker…" : "");
  $("live-events").innerHTML = s.events.slice().reverse().map((e) => {
    if (e.type === "decision") return `<div class="event decision"><b>${e.index + 1}. ${e.decision.action}${actionArgs(e.decision)}</b>
      · ${fmt(e.llm.latency_s, 1)} s<br>${esc(e.decision.reasoning)}</div>`;
    if (e.type === "result") return `<div class="event result">${e.index + 1}. ${esc(e.summary)} → ${esc(JSON.stringify(e.result))}</div>`;
    if (e.type === "thinking") return `<div class="event">${e.index + 1}. waiting for the LLM…</div>`;
    if (e.type === "policy") return `<div class="event">policy t=${fmt(e.elapsed_s, 1)} s · goal (${e.first_target.slice(0, 3).map((v) => fmt(v)).join(", ")}) · gripper ${e.first_target[3] > 0.5 ? "open" : "closed"}</div>`;
    if (e.type === "error") return `<div class="event error">${esc(e.error)}</div>`;
    return "";
  }).join("") + (s.state === "done" ? "" : "");
  if (s.state === "done") {
    $("live-events").insertAdjacentHTML("afterbegin", `<div class="event result">Finished: ${s.result.success ? "stacked" : "not stacked"} ·
      XY error ${fmt(s.result.xy_error_m * 100, 1)} cm · <a style="color:var(--accent);cursor:pointer" id="open-finished">open in the episode browser</a></div>`);
    $("open-finished").onclick = async () => { await loadEpisodes(); openEpisode(`lab/${id}`); };
    refreshLab();
    return;
  }
  if (s.state === "error") { $("live-events").insertAdjacentHTML("afterbegin", `<div class="event error">${esc(s.error)}</div>`); refreshLab(); return; }
  setTimeout(pollJob, 400);
}

setInterval(() => { if ($("view-run").classList.contains("active")) refreshLab(); }, 5000);
loadEpisodes().then(() => {
  const first = filteredEpisodes()[0];
  if (first) openEpisode(first.id);
});
