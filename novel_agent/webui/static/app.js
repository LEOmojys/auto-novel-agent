"use strict";

const PHASE_LABELS = {
  brief: "需求分析", planning: "规划 (Plan)", executing: "执行 (Execute)",
  review: "等待审查", finished: "已完成", failed: "已失败", cancelled: "已取消",
};
const NODE_GLYPH = { done: "✓", running: "●", pending: "○", blocked: "✗" };
const NODE_CLASS = { done: "done", running: "running", pending: "pending", blocked: "blocked" };
const NODE_STATUS_LABEL = { done: "完成", running: "进行中", pending: "待处理", blocked: "受阻" };
const NODE_LABEL = { world: "世界观设定", cast: "角色设定", outline: "故事大纲" };
const STATE_CLASS = { succeeded: "ok", failed: "err", waiting_review: "warn", running: "run", paused: "warn", queued: "idle", cancelled: "err" };
const RUN_STATE_LABEL = { succeeded: "已完成", failed: "失败", running: "运行中", queued: "排队中", paused: "已暂停", cancelled: "已取消", waiting_review: "待审查" };
const CHAIN_EVENTS = new Set([
  "stage_entered", "brief_analyzed", "plan_generated", "task_graph_built",
  "chapter_staged", "chapter_accepted", "provider_call_started", "provider_call_finished",
]);
const EVENT_LABEL = {
  stage_entered: "进入阶段", brief_analyzed: "需求分析完成", plan_generated: "规划已生成",
  task_graph_built: "任务图已构建", chapter_staged: "章节待确认", chapter_accepted: "章节已确认",
  provider_call_started: "调用模型", provider_call_finished: "模型调用完成",
};

function nodePhase(nodeId) {
  if (nodeId === "world" || nodeId === "cast") return "brief";
  if (nodeId === "outline") return "planning";
  return "executing";
}

function nodeLabel(nodeId) {
  if (NODE_LABEL[nodeId]) return NODE_LABEL[nodeId];
  if (nodeId.startsWith("chapter_")) return `第 ${nodeId.slice("chapter_".length)} 章`;
  return nodeId;
}

const state = {
  project: "", run_id: "", provider: "offline_fake",
  events_seen: -1, artifacts_loaded: false,
  polling: null, chain_version: 0, chain_event_seen: -1,
  pending_key: "", pending_artifact: null, pending_original: "", pending_loaded: false,
};

const $ = (id) => document.getElementById(id);

async function api(path, options) {
  const resp = await fetch(path, options);
  const data = await resp.json();
  if (!resp.ok || data.error) throw new Error(data.error || `HTTP ${resp.status}`);
  return data;
}

function post(path, body) {
  return api(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

function formatTime(iso) {
  if (!iso) return "";
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return "";
  const pad = (n) => String(n).padStart(2, "0");
  return `${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`;
}

const CALLER_LABEL = {
  brief: "需求分析",
  plan: "任务规划",
  governed_world: "世界观生成",
  governed_cast: "角色生成",
  governed_cast_revise: "角色重做",
  governed_outline: "大纲生成",
  governed_chapter: "章节创作",
  governed_summary: "章节摘要",
  quality: "质量评审",
  schema_repair: "JSON 修复",
};

function summarizePayload(payload) {
  if (!payload || !Object.keys(payload).length) return "";
  if (payload.stage) return `→ ${payload.stage}`;
  if (payload.chapter_number != null) return `章节 ${payload.chapter_number}`;
  if (payload.kind) return payload.kind;
  if (payload.caller) {
    const label = CALLER_LABEL[payload.caller] || payload.caller;
    return payload.model ? `${label}（${payload.model}）` : label;
  }
  if (payload.degraded && payload.degraded_reason) return `降级原因：${payload.degraded_reason}`;
  const text = JSON.stringify(payload);
  return text.length > 90 ? text.slice(0, 90) + "…" : text;
}

// ---------- 初始化 ----------

async function loadProjects() {
  const data = await api("/api/projects");
  const select = $("project-select");
  select.innerHTML = "";
  for (const project of data.projects) {
    const option = document.createElement("option");
    option.value = project;
    option.textContent = project;
    select.appendChild(option);
  }
  state.project = select.value;
  await loadHistory();
}

async function loadHistory() {
  const data = await api("/api/runs");
  const list = $("run-history");
  list.innerHTML = "";
  for (const run of data.runs) {
    const item = document.createElement("li");
    item.className = `run-item ${STATE_CLASS[run.state] || "idle"}`;
    item.textContent = `${run.run_id} · ${RUN_STATE_LABEL[run.state] || run.state}`;
    item.title = `${run.project} / ${run.run_id} (${run.updated_at})`;
    item.addEventListener("click", () => selectRun(run.project, run.run_id));
    list.appendChild(item);
  }
}

function selectRun(project, run_id) {
  state.project = project;
  state.run_id = run_id;
  $("project-select").value = project;
  resetStreamState();
  startPolling();
}

function clearRunView() {
  if (state.polling) clearInterval(state.polling);
  state.polling = null;
  state.run_id = "";
  state.events_seen = -1;
  state.artifacts_loaded = false;
  $("events").innerHTML = "";
  $("chain").innerHTML = "";
  $("graph").textContent = "（任务图尚未构建）";
  $("task-list").innerHTML = "";
  $("artifact-tabs").innerHTML = "";
  $("artifact-meta").textContent = "";
  $("artifact-body").textContent = "（暂无产物）";
  $("plan-panel").style.display = "none";
  $("pending-panel").style.display = "none";
  $("rebase-panel").style.display = "none";
  $("activity").textContent = "—";
  setBadge("idle", "未启动");
  $("start-btn").disabled = false;
  $("pause-btn").disabled = true;
  $("resume-btn").disabled = true;
  $("cancel-btn").disabled = true;
  $("repair-btn").disabled = true;
}

function resetStreamState() {
  state.events_seen = -1;
  state.artifacts_loaded = false;
  state.chain_version = 0;
  state.chain_event_seen = -1;
  state.pending_key = "";
  state.pending_artifact = null;
  state.pending_original = "";
  state.pending_loaded = false;
  $("events").innerHTML = "";
  $("chain").innerHTML = "";
}

// ---------- 命令 ----------

async function startRun() {
  const project = $("project-select").value;
  const requirements = $("requirements").value.trim();
  if (!project) {
    setBadge("err", "请先选择项目");
    return;
  }
  if (!requirements) {
    setBadge("err", "请先输入创作需求");
    $("requirements").focus();
    return;
  }
  try {
    // Multi-turn console mode: no run_id is entered — the service assigns it
    // automatically (retrying a terminal run reuses its id; a new requirement
    // gets the next id).  The start button is only disabled while the project
    // has a run in progress.
    const data = await post("/api/run/start", {
      project,
      requirements,
      provider: $("provider-select").value,
      chapter_count: $("chapter-count").value ? Number($("chapter-count").value) : null,
      auto_approve: $("auto-approve").checked,
    });
    state.run_id = data.run_id;
    state.project = project;
    resetStreamState();
    $("requirements").value = "";
    renderSnapshot(data);
    startPolling();
    await loadHistory();
  } catch (error) {
    setBadge("err", error.message);
    return;
  }
}

async function control(command) {
  try {
    const data = await post("/api/run/control", {
      project: state.project, run_id: state.run_id, command,
    });
    renderSnapshot(data);
  } catch (error) {
    setBadge("err", error.message);
  }
}

async function confirmRebase() {
  try {
    const data = await post("/api/run/rebase", {
      project: state.project, run_id: state.run_id,
    });
    renderSnapshot(data);
    await loadHistory();
  } catch (error) {
    setBadge("err", error.message);
  }
}

async function rewriteChapter() {
  const chapterNumber = $("rewrite-chapter").value;
  if (!chapterNumber) {
    setBadge("err", "请先输入要重写的章号");
    return;
  }
  try {
    const data = await post("/api/run/rewrite", {
      project: state.project, run_id: state.run_id, chapter_number: parseInt(chapterNumber, 10),
    });
    renderSnapshot(data);
    await loadHistory();
  } catch (error) {
    setBadge("err", error.message);
  }
}

async function exportFiles() {
  try {
    const data = await post("/api/export", { project: state.project });
    const activity = $("activity");
    activity.textContent = `已导出 ${data.files.length} 个旧版格式文件`;
    activity.style.color = "var(--ok)";
    setTimeout(() => { activity.style.color = ""; refresh(); }, 2500);
  } catch (error) {
    setBadge("err", error.message);
  }
}

async function approveArtifact() {
  if (!state.pending_artifact) return;
  const edited = $("pending-edit").value;
  const body = {
    project: state.project,
    run_id: state.run_id,
    artifact_id: state.pending_artifact.artifact_id,
    revision: state.pending_artifact.revision,
  };
  if (edited !== state.pending_original) body.content = edited;
  try {
    const data = await post("/api/run/approve", body);
    clearPendingState();
    state.artifacts_loaded = false;
    renderSnapshot(data);
    await loadHistory();
  } catch (error) {
    setBadge("err", error.message);
  }
}

async function regenerateArtifact() {
  if (!state.pending_artifact) return;
  try {
    const data = await post("/api/run/regenerate", {
      project: state.project,
      run_id: state.run_id,
      artifact_id: state.pending_artifact.artifact_id,
      revision: state.pending_artifact.revision,
    });
    clearPendingState();
    state.artifacts_loaded = false;
    renderSnapshot(data);
    await loadHistory();
  } catch (error) {
    setBadge("err", error.message);
  }
}

function clearPendingState() {
  state.pending_key = "";
  state.pending_artifact = null;
  state.pending_original = "";
  state.pending_loaded = false;
}

// ---------- 轮询 ----------

function startPolling() {
  if (state.polling) clearInterval(state.polling);
  state.polling = setInterval(refresh, 2000);
  refresh();
}

function inFlightCall(events) {
  let latestStart = null;
  for (const event of events || []) {
    if (event.event_type === "provider_call_started") {
      latestStart = event;
    } else if (event.event_type === "provider_call_finished") {
      if (latestStart && latestStart.payload && event.payload &&
          latestStart.payload.caller === event.payload.caller) {
        latestStart = null;
      }
    }
  }
  return latestStart;
}

function renderInFlightCall(events) {
  const call = inFlightCall(events);
  if (!call) return;
  const caller = (call.payload && call.payload.caller) || "";
  const model = (call.payload && call.payload.model) || "";
  const label = CALLER_LABEL[caller] || caller || "模型";
  const started = call.created_at ? new Date(call.created_at).getTime() : Date.now();
  const seconds = Math.max(0, Math.floor((Date.now() - started) / 1000));
  const badge = $("status-badge");
  badge.className = "badge run";
  badge.textContent = "调用中";
  const activity = $("activity");
  activity.textContent = `${label}（${model}）已等待 ${seconds}s…`;
}

async function refresh() {
  if (!state.run_id) return;
  try {
    const snapshot = await api(`/api/run/snapshot?project=${encodeURIComponent(state.project)}&run_id=${encodeURIComponent(state.run_id)}`);
    renderSnapshot(snapshot);
    renderEvents(snapshot.events || []);

    const workflow = await api(`/api/run/workflow?project=${encodeURIComponent(state.project)}&run_id=${encodeURIComponent(state.run_id)}`);
    renderWorkflow(workflow);
    // 最后覆盖：若当前有未完成的模型调用，显示"调用中"而非阶段文案
    renderInFlightCall(snapshot.events || []);

    if (!state.artifacts_loaded || snapshot.status === "succeeded" || snapshot.status === "failed" || snapshot.status === "waiting_review") {
      await loadArtifacts();
    }
    await loadAudit();
  } catch (error) {
    // 运行可能已删除或项目不存在；保持安静，由刷新按钮处理
  }
}

// ---------- 渲染 ----------

function setBadge(kind, text) {
  const badge = $("status-badge");
  badge.className = `badge ${kind}`;
  badge.textContent = text;
}

function renderSnapshot(snapshot) {
  const phaseText = PHASE_LABELS[snapshot.status] || snapshot.status;
  setBadge(STATE_CLASS[snapshot.status] || "idle", phaseText);
  const review = snapshot.review_reason ? `｜复核：${snapshot.review_reason}` : "";
  $("activity").textContent = `${snapshot.run_id || state.run_id} · 事件 ${snapshot.event_count}${review}`;
  // Multi-turn console: start is blocked only while the displayed run is in
  // progress (queued/running/cancel_requested) or waiting on an
  // approval/pause flow that has its own dedicated buttons.  After a failure,
  // cancellation or success the user can start again on the same run
  // (retry/continue) — the service reuses the run id when the requirement is
  // unchanged.
  $("start-btn").disabled = ["queued", "running", "waiting_review", "paused", "cancel_requested"].includes(snapshot.status);
  $("pause-btn").disabled = snapshot.status !== "running";
  $("resume-btn").disabled = snapshot.status !== "paused";
  $("cancel-btn").disabled = !["queued", "running", "paused", "waiting_review"].includes(snapshot.status);
  $("repair-btn").disabled = !(snapshot.status === "waiting_review" && snapshot.review_reason && snapshot.review_reason !== "chapter_rebase_confirmation_required");
  const rebaseRequired = snapshot.status === "waiting_review" && snapshot.review_reason === "chapter_rebase_confirmation_required";
  $("rebase-panel").style.display = rebaseRequired ? "block" : "none";
  $("rebase-btn").disabled = !rebaseRequired;

  const pendingRequired = snapshot.status === "waiting_review" && snapshot.review_reason === "artifact_approval_required";
  $("pending-panel").style.display = pendingRequired ? "block" : "none";
  $("approve-btn").disabled = !pendingRequired;
  $("regenerate-btn").disabled = !pendingRequired;
  if (pendingRequired) loadPendingArtifact();

  const planRequired = snapshot.status === "waiting_review" && snapshot.review_reason === "plan_approval_required";
  $("plan-panel").style.display = planRequired ? "block" : "none";
  $("plan-approve-btn").disabled = !planRequired;
  $("plan-regen-btn").disabled = !planRequired;
  if (planRequired) loadPlanSteps(snapshot.events || []);
}

const PLAN_TOOL_LABEL = {
  "world.generate@1.0.0": "世界观生成",
  "character.generate@1.0.0": "角色设定",
  "character.revise@1.0.0": "角色重做",
  "outline.generate@1.0.0": "大纲生成",
  "chapter.draft@1.0.0": "章节创作",
};

function loadPlanSteps(events) {
  const list = $("plan-steps");
  list.innerHTML = "";
  let stepsText = "";
  // Multi-turn streams append events chronologically; scan from the newest
  // event backwards so a re-run's plan panel never shows a previous cycle's
  // steps (a requeued run keeps its full event history).
  for (let i = events.length - 1; i >= 0; i--) {
    const event = events[i];
    if (event.event_type === "plan_approval_required") {
      stepsText = (event.payload && event.payload.steps) || "";
      break;
    }
    if (event.event_type === "plan_generated" && !stepsText) {
      stepsText = (event.payload && event.payload.steps) || "";
      break;
    }
  }
  if (!stepsText) {
    list.innerHTML = "<li>（计划内容不可用，请重新规划）</li>";
    return;
  }
  for (const item of stepsText.split(";")) {
    const parts = item.split(":");
    const id = parts[0];
    const tool = parts[1];
    const description = parts.slice(2).join(":");
    if (!id || !tool) continue;
    const li = document.createElement("li");
    const label = PLAN_TOOL_LABEL[tool] || tool;
    li.textContent = `${id}. ${label} — ${description || ""}`;
    list.appendChild(li);
  }
}

async function approvePlan() {
  try {
    const data = await post("/api/run/plan/approve", {
      project: state.project, run_id: state.run_id,
    });
    renderSnapshot(data);
    await loadHistory();
  } catch (error) {
    setBadge("err", error.message);
  }
}

async function regeneratePlan() {
  try {
    const data = await post("/api/run/plan/regenerate", {
      project: state.project, run_id: state.run_id,
    });
    renderSnapshot(data);
    await loadHistory();
  } catch (error) {
    setBadge("err", error.message);
  }
}

function renderWorkflow(workflow) {
  const phase = PHASE_LABELS[workflow.phase] || workflow.phase;
  $("activity").textContent = `${phase}｜${workflow.current_activity}`;
  const graph = $("graph");
  if (!workflow.nodes.length) {
    graph.textContent = "（任务图尚未构建）";
  } else {
    graph.innerHTML = "";
    workflow.nodes.forEach((node, index) => {
      if (index > 0) {
        const arrow = document.createElement("span");
        arrow.className = "graph-arrow";
        arrow.textContent = "→";
        graph.appendChild(arrow);
      }
      const span = document.createElement("span");
      span.className = `node-badge ${NODE_CLASS[node.status] || "pending"}`;
      span.textContent = `${NODE_GLYPH[node.status] || "○"} ${nodeLabel(node.node_id)}`;
      graph.appendChild(span);
    });
  }
  renderTaskList(workflow.nodes);
  renderChain(workflow.chain);
}

function renderTaskList(nodes) {
  const box = $("task-list");
  if (!nodes.length) {
    box.innerHTML = "";
    const empty = document.createElement("div");
    empty.className = "task-empty";
    empty.textContent = "（任务清单尚未生成）";
    box.appendChild(empty);
    return;
  }
  box.innerHTML = "";
  const phases = [
    { key: "brief", label: "需求分析" },
    { key: "planning", label: "任务规划" },
    { key: "executing", label: "执行" },
  ];
  for (const phase of phases) {
    const phaseNodes = nodes.filter((node) => nodePhase(node.node_id) === phase.key);
    if (!phaseNodes.length) continue;
    const section = document.createElement("div");
    section.className = "task-phase";

    const head = document.createElement("div");
    head.className = "task-phase-head";
    head.textContent = phase.label;
    section.appendChild(head);

    for (const node of phaseNodes) {
      const item = document.createElement("div");
      item.className = `task-item ${NODE_CLASS[node.status] || "pending"}`;

      const glyph = document.createElement("span");
      glyph.className = "task-glyph";
      glyph.textContent = NODE_GLYPH[node.status] || "○";

      const label = document.createElement("span");
      label.className = "task-label";
      label.textContent = nodeLabel(node.node_id);

      const badge = document.createElement("span");
      badge.className = "task-status";
      badge.textContent = NODE_STATUS_LABEL[node.status] || node.status;

      item.append(glyph, label, badge);
      section.appendChild(item);
    }
    box.appendChild(section);
  }
}

function renderChain(chain) {
  if (chain.length <= state.chain_version) return;
  const box = $("chain");
  const fragment = document.createDocumentFragment();
  for (let i = state.chain_version; i < chain.length; i++) {
    const line = document.createElement("div");
    line.className = "chain-line";
    line.textContent = chain[i];
    fragment.appendChild(line);
  }
  box.appendChild(fragment);
  state.chain_version = chain.length;
  box.scrollTop = box.scrollHeight;
}

function renderEvents(events) {
  const box = $("events");
  const fragment = document.createDocumentFragment();
  let appended = false;
  for (const event of events) {
    if (event.sequence <= state.events_seen) continue;
    appended = true;
    const line = document.createElement("div");
    line.className = `event event-${event.event_type}`;
    const time = formatTime(event.created_at);
    const payload = event.payload && Object.keys(event.payload).length
      ? " " + JSON.stringify(event.payload)
      : "";
    line.textContent = `#${event.sequence}  ${time}  ${event.event_type}${payload}`;
    fragment.appendChild(line);
    appendChainEvent(event);
    if (event.sequence > state.events_seen) state.events_seen = event.sequence;
  }
  if (appended) {
    box.appendChild(fragment);
    $("event-count").textContent = String(state.events_seen + 1);
    box.scrollTop = box.scrollHeight;
  }
}

function appendChainEvent(event) {
  if (!CHAIN_EVENTS.has(event.event_type)) return;
  if (event.sequence <= state.chain_event_seen) return;
  state.chain_event_seen = event.sequence;
  const box = $("chain");
  const line = document.createElement("div");
  line.className = "chain-line chain-event";
  const time = formatTime(event.created_at);
  const label = EVENT_LABEL[event.event_type] || event.event_type;
  const detail = summarizePayload(event.payload);
  line.textContent = `${time}  ◆ ${label}${detail ? "  " + detail : ""}`;
  box.appendChild(line);
  box.scrollTop = box.scrollHeight;
}

async function loadArtifacts() {
  const data = await api(`/api/run/artifacts?project=${encodeURIComponent(state.project)}&run_id=${encodeURIComponent(state.run_id)}`);
  state.artifacts_loaded = true;
  const tabs = $("artifact-tabs");
  tabs.innerHTML = "";
  if (!data.artifacts.length) {
    $("artifact-meta").textContent = "";
    $("artifact-body").textContent = "（暂无产物）";
    return;
  }
  for (const artifact of data.artifacts) {
    const tab = document.createElement("button");
    tab.className = "tab";
    const statusMark = artifact.status === "staged" ? "⏳ " : "";
    tab.textContent = `${statusMark}${artifact.kind} · v${artifact.revision}${artifact.status === "staged" ? "（待确认）" : ""}`;
    tab.addEventListener("click", () => viewArtifact(artifact.artifact_id, artifact.revision));
    tabs.appendChild(tab);
  }
  // Prefer showing the pending (staged) artifact so approval mode is obvious.
  const staged = data.artifacts.filter((artifact) => artifact.status === "staged");
  const first = staged.length ? staged[staged.length - 1] : data.artifacts[data.artifacts.length - 1];
  await viewArtifact(first.artifact_id, first.revision);
}

async function viewArtifact(artifact_id, revision) {
  const data = await api(
    `/api/run/artifact?project=${encodeURIComponent(state.project)}&run_id=${encodeURIComponent(state.run_id)}` +
    `&artifact_id=${encodeURIComponent(artifact_id)}&revision=${revision}`
  );
  $("artifact-meta").textContent = `${data.kind} / ${data.status} / v${data.revision} / SHA-256 ${data.sha256.slice(0, 16)}…`;
  $("artifact-body").textContent = data.text;
}

async function loadAudit() {
  const data = await api(`/api/run/audit?project=${encodeURIComponent(state.project)}&run_id=${encodeURIComponent(state.run_id)}`);
  const tools = $("tools");
  tools.innerHTML = "";
  const manifests = data.manifests || [];
  if (!manifests.length) {
    const empty = document.createElement("div");
    empty.className = "tools-empty";
    empty.textContent = "（暂无审计清单）";
    tools.appendChild(empty);
  } else {
    for (const manifest of manifests) {
      const line = document.createElement("div");
      line.className = "tool-line";
      line.textContent = `审计 ${manifest.manifest_id} · ${(manifest.sha256 || "").slice(0, 16)}…`;
      tools.appendChild(line);
    }
  }
  const quality = $("quality");
  quality.innerHTML = "";
  const reports = data.quality || [];
  if (!reports.length) {
    const empty = document.createElement("div");
    empty.className = "quality-empty";
    empty.textContent = "（暂无质量评审）";
    quality.appendChild(empty);
  } else {
    for (const report of reports) {
      const line = document.createElement("div");
      line.className = `quality-line ${report.decision === "ACCEPT" ? "ok" : "warn"}`;
      const scores = report.report && report.report.scores
        ? Object.entries(report.report.scores).map(([k, v]) => `${k}=${v}`).join(" ")
        : JSON.stringify(report.report || {});
      line.textContent = `${report.decision} · ${scores}`;
      quality.appendChild(line);
    }
  }
  const failures = data.provider_failures || [];
  if (failures.length) {
    const failBox = document.createElement("div");
    failBox.className = "provider-failures";
    const heading = document.createElement("div");
    heading.className = "provider-failures-head";
    heading.textContent = `LLM 调用失败（${failures.length}）`;
    failBox.appendChild(heading);
    for (const failure of failures) {
      const line = document.createElement("div");
      line.className = "provider-failure-line";
      line.textContent = `${failure.created_at} ${failure.caller} [${failure.failure_kind}] ${failure.reason}`;
      line.title = `${failure.model_name} · ${failure.prompt_sha256}`;
      failBox.appendChild(line);
    }
    tools.appendChild(failBox);
  }
}

// ---------- 待确认产物 ----------

async function loadPendingArtifact() {
  try {
    const data = await api(`/api/run/artifacts?project=${encodeURIComponent(state.project)}&run_id=${encodeURIComponent(state.run_id)}`);
    const staged = data.artifacts.filter((artifact) => artifact.status === "staged");
    if (!staged.length) return;
    const latest = staged[staged.length - 1];
    const key = `${latest.artifact_id}@${latest.revision}`;
    if (state.pending_loaded && state.pending_key === key) return;
    const detail = await api(
      `/api/run/artifact?project=${encodeURIComponent(state.project)}&run_id=${encodeURIComponent(state.run_id)}` +
      `&artifact_id=${encodeURIComponent(latest.artifact_id)}&revision=${latest.revision}`
    );
    state.pending_key = key;
    state.pending_artifact = detail;
    state.pending_original = detail.text;
    state.pending_loaded = true;
    $("pending-meta").textContent = `${detail.kind} / ${detail.status} / v${detail.revision} / SHA-256 ${detail.sha256.slice(0, 16)}…`;
    $("pending-body").textContent = detail.text;
    $("pending-edit").value = detail.text;
  } catch (error) {
    // 产物可能在确认前被其他路径处理；静默等待下一轮
  }
}

// ---------- LLM 连通性检查 ----------

async function checkProvider() {
  const element = $("llm-status");
  element.textContent = "LLM API: 检查中…";
  element.className = "llm-status";
  try {
    const data = await api("/api/health");
    if (data.ok) {
      element.textContent = `LLM API: ✓ 已连接（${data.latency_ms}ms）`;
      element.className = "llm-status ok";
    } else if (data.reason === "missing_key") {
      element.textContent = `LLM API: ✗ 未配置 Key（${data.hint}）`;
      element.className = "llm-status err";
    } else {
      element.textContent = `LLM API: ✗ ${data.reason}`;
      element.className = "llm-status err";
    }
  } catch (error) {
    element.textContent = "LLM API: ✗ 检查失败";
    element.className = "llm-status err";
  }
}

// ---------- 事件绑定 ----------

$("start-btn").addEventListener("click", startRun);
$("pause-btn").addEventListener("click", () => control("pause"));
$("resume-btn").addEventListener("click", () => control("resume"));
$("cancel-btn").addEventListener("click", () => control("cancel"));
$("repair-btn").addEventListener("click", () => control("repair"));
$("rewrite-btn").addEventListener("click", rewriteChapter);
$("refresh-btn").addEventListener("click", refresh);
$("export-btn").addEventListener("click", exportFiles);
$("rebase-btn").addEventListener("click", confirmRebase);
$("approve-btn").addEventListener("click", approveArtifact);
$("regenerate-btn").addEventListener("click", regenerateArtifact);
$("plan-approve-btn").addEventListener("click", approvePlan);
$("plan-regen-btn").addEventListener("click", regeneratePlan);
$("project-select").addEventListener("change", () => { state.project = $("project-select").value; });

loadProjects().catch((error) => setBadge("err", error.message));
checkProvider();
