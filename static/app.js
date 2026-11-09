// Voice-to-Insight Agent — Enhanced Frontend Logic
// Supports live recording, file upload, audio playback player,
// task filtering, batch clear, undoable delete, and Voice RAG.

const els = {
  recordBtn: document.getElementById("recordBtn"),
  recDot: document.getElementById("recDot"),
  recStatus: document.getElementById("recStatus"),
  recTimer: document.getElementById("recTimer"),
  meter: document.getElementById("meter"),
  uploadAudioBtn: document.getElementById("uploadAudioBtn"),
  audioFileInput: document.getElementById("audioFileInput"),
  audioPlaybackWrap: document.getElementById("audioPlaybackWrap"),
  audioPlayer: document.getElementById("audioPlayer"),
  textFallback: document.getElementById("textFallback"),
  runTextBtn: document.getElementById("runTextBtn"),
  samples: document.getElementById("samples"),
  pipeline: document.getElementById("pipeline"),
  pipelineStatus: document.getElementById("pipelineStatus"),
  tasks: document.getElementById("tasks"),
  historyList: document.getElementById("historyList"),
  tasksView: document.getElementById("tasksView"),
  historyView: document.getElementById("historyView"),
  tabTasksBtn: document.getElementById("tabTasksBtn"),
  tabHistoryBtn: document.getElementById("tabHistoryBtn"),
  taskSearchInput: document.getElementById("taskSearchInput"),
  historySearchInput: document.getElementById("historySearchInput"),
  pillAll: document.getElementById("pillAll"),
  pillPending: document.getElementById("pillPending"),
  pillCompleted: document.getElementById("pillCompleted"),
  countAll: document.getElementById("countAll"),
  countPending: document.getElementById("countPending"),
  countCompleted: document.getElementById("countCompleted"),
  clearCompletedBtn: document.getElementById("clearCompletedBtn"),
  errorBanner: document.getElementById("errorBanner"),
  sttChip: document.getElementById("sttChip"),
  llmChip: document.getElementById("llmChip"),
  dbChip: document.getElementById("dbChip"),
  notionChip: document.getElementById("notionChip"),
  toast: document.getElementById("toast"),
  toastMsg: document.getElementById("toastMsg"),
  toastUndoBtn: document.getElementById("toastUndoBtn"),
};

// Global State
let currentFilter = "all";
let cachedTasks = [];
let cachedHistory = [];
let toastTimeout = null;

// Multilingual Samples
const SAMPLES = [
  { label: "Task (Arabic)", text: "ذكرني بكرا اتصل بالمحاسب، أولوية متوسطة." },
  { label: "Task (English)", text: "Remind Sarah to send the Q3 financial report by next Friday, high priority." },
  { label: "Content (Arabic)", text: "اكتبلي بوست للينكد إن عن إطلاق ميزة الذكاء الاصطناعي الصوتي بطريقة احترافية." },
  { label: "Summary (French)", text: "Résume cette note de réunion: l'équipe marketing a validé le budget, et le lancement est confirmé pour le mois prochain." },
  { label: "Ask RAG (Arabic)", text: "ما هي المهام المطلوبة المسجلة؟" },
  { label: "Ask RAG (English)", text: "What did I record about the contract or report?" },
];

// Helper: Check if string contains Arabic characters
function isArabic(text) {
  return /[\u0600-\u06FF]/.test(text || "");
}

// Helper: Copy to clipboard
window.copyText = function(text) {
  navigator.clipboard.writeText(text).then(() => {
    showToast("Copied to clipboard!");
  }).catch((err) => {
    console.error("Copy failed", err);
  });
};

// ---------- Toast Notifications ----------
function showToast(msg, undoFn = null) {
  if (toastTimeout) clearTimeout(toastTimeout);
  els.toastMsg.textContent = msg;
  if (undoFn) {
    els.toastUndoBtn.style.display = "inline-block";
    els.toastUndoBtn.onclick = () => {
      undoFn();
      hideToast();
    };
  } else {
    els.toastUndoBtn.style.display = "none";
  }
  els.toast.style.display = "flex";
  toastTimeout = setTimeout(hideToast, 5000);
}

function hideToast() {
  els.toast.style.display = "none";
}

// ---------- Error banner ----------
function showError(msg) {
  els.errorBanner.style.display = "block";
  els.errorBanner.textContent = "⚠ " + msg;
}
function clearError() {
  els.errorBanner.style.display = "none";
  els.errorBanner.textContent = "";
}

// ---------- Health & Configuration ----------
async function checkHealth() {
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    if (els.sttChip) els.sttChip.textContent = `STT: ${data.transcribe_backend.toUpperCase()}`;
    if (els.llmChip) els.llmChip.textContent = `LLM: ${data.understand_backend.toUpperCase()}`;
    if (els.notionChip) {
      if (data.notion_configured) {
        els.notionChip.classList.add("active");
        els.notionChip.textContent = "Notion: Connected";
      } else {
        els.notionChip.classList.remove("active");
        els.notionChip.textContent = "Notion: Optional";
      }
    }
  } catch (e) {
    console.error("Health check failed", e);
  }
}

// ---------- Pipeline Rendering ----------
function renderPipeline(state) {
  const s = (cls, label, bodyHtml, copyVal = null) => {
    const copyBtn = copyVal
      ? `<button class="copy-btn" data-copy="${escapeHtml(copyVal)}">📋 Copy</button>`
      : "";
    return `
      <div class="stage ${cls}">
        <div class="stage-header">
          <span class="label">${label}</span>
          ${copyBtn}
        </div>
        <div class="body">${bodyHtml}</div>
      </div>`;
  };

  let html = "";

  // 1. Transcription Stage
  if (state.transcript !== undefined) {
    if (state.transcript.status === "active") {
      html += s("active", "① Speech-to-Text", "<em>Transcribing audio with Whisper…</em>");
      if (els.pipelineStatus) els.pipelineStatus.textContent = "Transcribing…";
    } else if (state.transcript.status === "done") {
      const text = state.transcript.text || "(empty)";
      const rtlCls = isArabic(text) ? "rtl-text" : "";
      html += s(
        "done",
        `① Speech-to-Text <span class="locale">(${escapeHtml(state.transcript.locale || "detected")})</span>`,
        `<div class="${rtlCls}">${escapeHtml(text)}</div>`,
        text
      );
    }
  }

  // 2. Classification Stage
  if (state.classify) {
    if (state.classify.status === "active") {
      html += s("active", "② Understanding & Extraction", "<em>Classifying intent & extracting JSON…</em>");
      if (els.pipelineStatus) els.pipelineStatus.textContent = "Understanding…";
    } else if (state.classify.status === "done") {
      const intent = state.classify.intent;
      html += s(
        "done",
        "② Understanding & Extraction",
        `<span class="badge ${escapeHtml(intent)}">${escapeHtml(intent)}</span>` +
        `<pre>${escapeHtml(JSON.stringify(state.classify.data, null, 2))}</pre>`,
        JSON.stringify(state.classify.data, null, 2)
      );
    } else if (state.classify.status === "error") {
      html += s(
        "err",
        "② Understanding & Extraction",
        `<span class="badge error">error</span><pre>${escapeHtml(state.classify.error)}</pre>`
      );
    }
  }

  // 3. Action Stage
  if (state.action) {
    html += s("done", "③ Action & Workspace Storage", renderAction(state.action));
    if (els.pipelineStatus) els.pipelineStatus.textContent = "Completed";
  }

  els.pipeline.innerHTML = html || '<p class="empty">Record or run text to see the chain execute live.</p>';
}

function renderAction(action) {
  if (action.type === "task_added") {
    const t = action.task;
    let notionHtml = "";
    if (action.notion && action.notion.pushed) {
      const url = action.notion.url || (t && t.notion_url);
      notionHtml = url
        ? ` <a href="${escapeHtml(url)}" target="_blank" class="notion-pill">Open in Notion ↗</a>`
        : ' <span class="notion-pill">✓ Synced to Notion</span>';
    }
    const rtlCls = isArabic(t.title) ? "rtl-text" : "";
    return `
      <div>
        ✅ Task saved to database: <strong class="${rtlCls}">${escapeHtml(t.title)}</strong>${notionHtml}
      </div>`;
  }

  if (action.type === "caption_generated") {
    const rtlCls = isArabic(action.caption) ? "rtl-text" : "";
    return `
      <div>
        📝 Caption for <strong>${escapeHtml(action.platform || "Social")}</strong> (${escapeHtml(action.tone || "General")}):
        <div class="rag-box ${rtlCls}">${escapeHtml(action.caption)}</div>
        <button class="btn-secondary copy-btn" style="margin-top:8px;" data-copy="${escapeHtml(action.caption)}">📋 Copy Caption</button>
      </div>`;
  }

  if (action.type === "summary_shown") {
    const rtlCls = isArabic(action.summary_text) ? "rtl-text" : "";
    const pts = (action.key_points || []).map((p) => `<li>${escapeHtml(p)}</li>`).join("");
    return `
      <div class="${rtlCls}">
        🧾 <strong>Summary:</strong> ${escapeHtml(action.summary_text)}
        <ul style="margin: 6px 0 0 16px;">${pts}</ul>
      </div>`;
  }

  if (action.type === "rag_answer") {
    const rtlCls = isArabic(action.answer) ? "rtl-text" : "";
    let citationsHtml = "";
    if (action.citations && action.citations.length) {
      citationsHtml = '<div class="citations-header">Source Citations:</div>' +
        action.citations.map((c) =>
          `<div class="citation-item"><strong>#${c.id} [${escapeHtml(c.intent)}]:</strong> ${escapeHtml(c.transcript)}</div>`
        ).join("");
    }
    return `
      <div class="rag-box">
        <div class="rag-answer ${rtlCls}">💬 ${escapeHtml(action.answer)}</div>
        ${citationsHtml}
        <button class="btn-secondary copy-btn" style="margin-top:8px;" data-copy="${escapeHtml(action.answer)}">📋 Copy Answer</button>
      </div>`;
  }

  return escapeHtml(JSON.stringify(action));
}

function escapeHtml(str) {
  return String(str || "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// Copy buttons carry their text in an HTML-escaped data-copy attribute and are
// handled by one delegated listener. Never interpolate model output into inline
// onclick handlers: HTML entities in the text are decoded before the JS runs.
document.addEventListener("click", (ev) => {
  const btn = ev.target.closest("[data-copy]");
  if (btn) copyText(btn.dataset.copy);
});

// ---------- Pipeline Driver ----------
async function runUnderstand(transcript, state, locale = "") {
  state.classify = { status: "active" };
  renderPipeline(state);

  let res;
  try {
    const r = await fetch("/api/understand", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ transcript, locale }),
    });
    res = await r.json();
    // Non-2xx (e.g. 429 rate limit) may come without intent: "error"; treat it as one.
    if (!r.ok && res.intent !== "error") {
      res = { intent: "error", error: res.error || `HTTP ${r.status}` };
    }
  } catch (e) {
    state.classify = { status: "error", error: "Network error: " + e.message };
    renderPipeline(state);
    showError(state.classify.error);
    return;
  }

  if (res.intent === "error") {
    state.classify = { status: "error", error: res.error + (res.raw ? "\n" + JSON.stringify(res.raw, null, 2) : "") };
    renderPipeline(state);
    showError("Understanding failed: " + res.error);
    return;
  }

  state.classify = { status: "done", intent: res.intent, data: res.data };
  state.action = res.action;
  renderPipeline(state);

  loadTasks();
  loadHistory();
}

// ---------- Text Fallback ----------
async function runFromText() {
  const text = els.textFallback.value.trim();
  if (!text) { showError("Type or pick a sample prompt first."); return; }
  clearError();
  const state = {};
  await runUnderstand(text, state);
}

// ---------- Mic Recording & Audio Upload ----------
let mediaRecorder = null;
let chunks = [];
let audioCtx = null;
let analyser = null;
let meterRAF = null;
let recStartTime = 0;
let recTimerInterval = null;

function buildMeter() {
  els.meter.innerHTML = "";
  for (let i = 0; i < 16; i++) els.meter.appendChild(document.createElement("span"));
}
buildMeter();

function animateMeter() {
  if (!analyser) return;
  const data = new Uint8Array(analyser.frequencyBinCount);
  analyser.getByteFrequencyData(data);
  const bars = els.meter.children;
  const step = Math.floor(data.length / bars.length);
  for (let i = 0; i < bars.length; i++) {
    const v = data[i * step] / 255;
    bars[i].style.height = 4 + v * 28 + "px";
  }
  meterRAF = requestAnimationFrame(animateMeter);
}

function updateRecTimer() {
  const elapsed = Math.floor((Date.now() - recStartTime) / 1000);
  const m = String(Math.floor(elapsed / 60)).padStart(2, "0");
  const s = String(elapsed % 60).padStart(2, "0");
  els.recTimer.textContent = `${m}:${s}`;
}

async function startRecording() {
  clearError();
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (e) {
    showError("Microphone access denied or unavailable: " + e.message + "\nYou can use 'Upload Audio' or the text input below.");
    return;
  }

  chunks = [];
  mediaRecorder = new MediaRecorder(stream);
  mediaRecorder.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };
  mediaRecorder.onstop = () => onRecordingStopped(stream);
  mediaRecorder.start();

  recStartTime = Date.now();
  els.recTimer.textContent = "00:00";
  els.recTimer.style.display = "inline-block";
  recTimerInterval = setInterval(updateRecTimer, 1000);

  audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  const src = audioCtx.createMediaStreamSource(stream);
  analyser = audioCtx.createAnalyser();
  analyser.fftSize = 64;
  src.connect(analyser);
  animateMeter();

  els.recordBtn.classList.add("recording");
  els.recordBtn.textContent = "■ Stop Recording";
  els.recDot.classList.add("live");
  els.recStatus.textContent = "Recording…";
}

function stopRecording() {
  if (mediaRecorder && mediaRecorder.state !== "inactive") mediaRecorder.stop();
  els.recordBtn.classList.remove("recording");
  els.recordBtn.textContent = "● Record Voice";
  els.recDot.classList.remove("live");
  els.recStatus.textContent = "Transcribing…";
  clearInterval(recTimerInterval);
  els.recTimer.style.display = "none";
  if (meterRAF) cancelAnimationFrame(meterRAF);
}

async function onRecordingStopped(stream) {
  stream.getTracks().forEach((t) => t.stop());
  if (audioCtx) { audioCtx.close(); audioCtx = null; analyser = null; }

  const mime = mediaRecorder.mimeType || "audio/webm";
  const blob = new Blob(chunks, { type: mime });

  // Attach audio to playback player
  attachAudioPlayback(blob);

  // Send for transcription
  await processAudioBlob(blob, mime);
}

function attachAudioPlayback(blob) {
  const url = URL.createObjectURL(blob);
  els.audioPlayer.src = url;
  els.audioPlaybackWrap.style.display = "flex";
}

async function processAudioBlob(blob, mime) {
  const state = { transcript: { status: "active" } };
  renderPipeline(state);

  const ext = mime.includes("mp4") || mime.includes("m4a") ? "mp4"
    : mime.includes("ogg") ? "ogg"
    : mime.includes("wav") ? "wav"
    : "webm";

  const form = new FormData();
  form.append("audio", blob, "recording." + ext);

  let res;
  try {
    const r = await fetch("/api/transcribe", { method: "POST", body: form });
    res = await r.json();
    if (!r.ok && !res.error) res.error = `HTTP ${r.status}`;
  } catch (e) {
    showError("Network error calling /api/transcribe: " + e.message);
    els.recStatus.textContent = "Idle";
    return;
  }

  els.recStatus.textContent = "Idle";

  if (res.error) {
    state.transcript = undefined;
    renderPipeline(state);
    showError("Transcription failed: " + res.error);
    return;
  }

  state.transcript = { status: "done", text: res.transcript, locale: res.locale };
  renderPipeline(state);

  await runUnderstand(res.transcript, state, res.locale);
}

// Audio File Upload
els.uploadAudioBtn.addEventListener("click", () => els.audioFileInput.click());
els.audioFileInput.addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  clearError();
  attachAudioPlayback(file);
  await processAudioBlob(file, file.type || "audio/wav");
  els.audioFileInput.value = "";
});

// ---------- Task List & Management ----------
async function loadTasks() {
  try {
    const r = await fetch("/api/tasks");
    const { tasks } = await r.json();
    cachedTasks = tasks || [];
    renderTaskList();
  } catch (e) {
    console.error("Could not load tasks:", e);
  }
}

// Local calendar date as YYYY-MM-DD (comparable with the server's due_date_iso strings).
function todayIso() {
  const d = new Date();
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
}

function renderTaskList() {
  const query = (els.taskSearchInput.value || "").trim().toLowerCase();

  const pendingCount = cachedTasks.filter((t) => t.status === "pending").length;
  const completedCount = cachedTasks.filter((t) => t.status === "completed").length;
  els.countAll.textContent = cachedTasks.length;
  els.countPending.textContent = pendingCount;
  els.countCompleted.textContent = completedCount;

  let filtered = cachedTasks.filter((t) => {
    if (currentFilter === "pending" && t.status !== "pending") return false;
    if (currentFilter === "completed" && t.status !== "completed") return false;
    if (query) {
      const matchTitle = (t.title || "").toLowerCase().includes(query);
      const matchOwner = (t.owner || "").toLowerCase().includes(query);
      return matchTitle || matchOwner;
    }
    return true;
  });

  if (!filtered.length) {
    els.tasks.innerHTML = '<p class="empty">No tasks found.</p>';
    return;
  }

  els.tasks.innerHTML = filtered.map((t) => {
    const isDone = t.status === "completed";
    const rtlCls = isArabic(t.title) ? "rtl-text" : "";
    const overdue = !isDone && t.due_date_iso && t.due_date_iso < todayIso();
    const dueIso = t.due_date_iso && t.due_date_iso !== t.due_date
      ? ` <span class="due-iso">(${escapeHtml(t.due_date_iso)})</span>` : "";
    const dueHtml = t.due_date
      ? " · 📅 " + escapeHtml(t.due_date) + dueIso
      : (t.due_date_iso ? " · 📅 " + escapeHtml(t.due_date_iso) : "");
    const notionLink = t.notion_url
      ? `<a href="${escapeHtml(t.notion_url)}" target="_blank" class="notion-pill">Notion ↗</a>`
      : "";
    return `
      <div class="task-item ${isDone ? 'completed' : ''}" id="task-${t.id}">
        <div class="task-left">
          <input type="checkbox" class="task-check" ${isDone ? "checked" : ""} onchange="toggleTask(${t.id})">
          <div>
            <div class="t ${rtlCls}">${escapeHtml(t.title)}</div>
            <div class="meta">
              <span class="pri ${escapeHtml(t.priority)}">${escapeHtml(t.priority)}</span>
              ${overdue ? '<span class="overdue">overdue</span>' : ""}
              ${t.owner ? " · 👤 " + escapeHtml(t.owner) : ""}
              ${dueHtml}
              ${notionLink}
            </div>
          </div>
        </div>
        <div class="task-actions">
          <button class="del-btn" title="Delete Task" onclick="deleteTask(${t.id})">🗑️</button>
        </div>
      </div>`;
  }).join("");
}

window.toggleTask = async function(id) {
  // Optimistic UI update
  const task = cachedTasks.find((t) => t.id === id);
  if (task) {
    task.status = task.status === "pending" ? "completed" : "pending";
    renderTaskList();
  }
  try {
    const r = await fetch(`/api/tasks/${id}/toggle`, { method: "POST" });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    loadTasks();
  } catch (e) {
    showError("Failed to update task: " + e.message);
    loadTasks();
  }
};

window.deleteTask = async function(id) {
  const taskIndex = cachedTasks.findIndex((t) => t.id === id);
  if (taskIndex !== -1) {
    cachedTasks.splice(taskIndex, 1);
    renderTaskList();
  }

  try {
    const r = await fetch(`/api/tasks/${id}`, { method: "DELETE" });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    loadHistory();
    showToast("Task deleted", async () => {
      // Undo: the server soft-deletes, so restore the exact same task (no LLM call).
      try {
        const rr = await fetch(`/api/tasks/${id}/restore`, { method: "POST" });
        if (!rr.ok) throw new Error((await rr.json()).error || `HTTP ${rr.status}`);
      } catch (e) {
        showError("Could not undo delete: " + e.message);
      }
      loadTasks();
      loadHistory();
    });
  } catch (e) {
    showError("Failed to delete task: " + e.message);
    loadTasks();
  }
};

// Batch Clear Completed Tasks
els.clearCompletedBtn.addEventListener("click", async () => {
  const completedCount = cachedTasks.filter((t) => t.status === "completed").length;
  if (!completedCount) {
    showToast("No completed tasks to clear.");
    return;
  }
  try {
    const r = await fetch("/api/tasks/clear-completed", { method: "POST" });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    loadHistory();
    showToast(`Cleared ${completedCount} completed task(s).`);
    loadTasks();
  } catch (e) {
    showError("Failed to clear tasks: " + e.message);
  }
});

// Filter Pills
[els.pillAll, els.pillPending, els.pillCompleted].forEach((pill) => {
  pill.addEventListener("click", () => {
    [els.pillAll, els.pillPending, els.pillCompleted].forEach((p) => p.classList.remove("active"));
    pill.classList.add("active");
    currentFilter = pill.dataset.filter;
    renderTaskList();
  });
});

els.taskSearchInput.addEventListener("input", renderTaskList);

// ---------- History View ----------
async function loadHistory() {
  try {
    const r = await fetch("/api/history?limit=40");
    const { history } = await r.json();
    cachedHistory = history || [];
    renderHistoryList();
  } catch (e) {
    console.error("Could not load history:", e);
  }
}

function renderHistoryList() {
  const query = (els.historySearchInput.value || "").trim().toLowerCase();
  const filtered = cachedHistory.filter((h) => {
    if (!query) return true;
    return (h.transcript || "").toLowerCase().includes(query) || (h.intent || "").toLowerCase().includes(query);
  });

  if (!filtered.length) {
    els.historyList.innerHTML = '<p class="empty">No recorded history found.</p>';
    return;
  }

  els.historyList.innerHTML = filtered.map((h) => {
    const rtlCls = isArabic(h.transcript) ? "rtl-text" : "";
    return `
      <div class="history-item">
        <div class="h-meta">
          <span>#${h.id} · <span class="badge ${h.intent}">${h.intent}</span></span>
          <span>${(h.created_at || '').substring(0, 16).replace('T', ' ')}</span>
        </div>
        <div class="${rtlCls}">${escapeHtml(h.transcript)}</div>
      </div>
    `;
  }).join("");
}

els.historySearchInput.addEventListener("input", renderHistoryList);

// ---------- Tabs & Wiring ----------
els.tabTasksBtn.addEventListener("click", () => {
  els.tabTasksBtn.classList.add("active");
  els.tabHistoryBtn.classList.remove("active");
  els.tasksView.style.display = "block";
  els.historyView.style.display = "none";
});

els.tabHistoryBtn.addEventListener("click", () => {
  els.tabHistoryBtn.classList.add("active");
  els.tabTasksBtn.classList.remove("active");
  els.tasksView.style.display = "none";
  els.historyView.style.display = "block";
  loadHistory();
});

els.recordBtn.addEventListener("click", () => {
  if (mediaRecorder && mediaRecorder.state === "recording") stopRecording();
  else startRecording();
});

els.runTextBtn.addEventListener("click", runFromText);

// Build Sample Buttons
SAMPLES.forEach((s) => {
  const b = document.createElement("button");
  b.textContent = s.label;
  b.title = s.text;
  b.addEventListener("click", () => {
    els.textFallback.value = s.text;
    els.textFallback.focus();
  });
  els.samples.appendChild(b);
});

// Initial boot
checkHealth();
loadTasks();
loadHistory();
