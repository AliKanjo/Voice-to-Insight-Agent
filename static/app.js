// Voice-to-Insight Agent — frontend logic.
// Records mic audio via MediaRecorder, converts it to 16 kHz mono WAV *in the
// browser* (so the backend needs no ffmpeg), then walks the pipeline stage by
// stage so a live audience can watch each step complete.

const els = {
  recordBtn: document.getElementById("recordBtn"),
  recDot: document.getElementById("recDot"),
  recStatus: document.getElementById("recStatus"),
  meter: document.getElementById("meter"),
  textFallback: document.getElementById("textFallback"),
  runTextBtn: document.getElementById("runTextBtn"),
  samples: document.getElementById("samples"),
  pipeline: document.getElementById("pipeline"),
  tasks: document.getElementById("tasks"),
  errorBanner: document.getElementById("errorBanner"),
};

// Sample phrases for the fallback path (no mic needed).
const SAMPLES = [
  "Remind Sarah to send the Q3 report by next Friday, it's high priority.",
  "اكتبلي بوست عن الإنجازات تبعنا هالشهر للينكد إن، بطريقة احترافية.",
  "Summarize this: we met the client, they liked the demo, but want pricing options and a follow-up call next week.",
  "ذكرني بكرا اتصل بالمحاسب، أولوية متوسطة.",
];

// ---------- Error banner ----------
function showError(msg) {
  els.errorBanner.style.display = "block";
  els.errorBanner.textContent = "⚠ " + msg;
}
function clearError() {
  els.errorBanner.style.display = "none";
  els.errorBanner.textContent = "";
}

// ---------- Pipeline rendering ----------
// Rebuilds the pipeline panel from a stage state object each time it changes.
function renderPipeline(state) {
  const s = (name, cls, label, bodyHtml) =>
    `<div class="stage ${cls}"><div class="label">${label}</div><div class="body">${bodyHtml}</div></div>`;

  let html = "";
  // Transcript stage (skipped when using text fallback)
  if (state.transcript !== undefined) {
    if (state.transcript.status === "active")
      html += s("transcript", "active", "① Transcript", "<em>Transcribing…</em>");
    else if (state.transcript.status === "done")
      html += s("transcript", "done", `① Transcript <span class="locale">(${state.transcript.locale || "?"})</span>`,
        escapeHtml(state.transcript.text || "(empty)"));
  }

  // Classification stage
  if (state.classify) {
    if (state.classify.status === "active")
      html += s("classify", "active", "② Classification", "<em>Understanding…</em>");
    else if (state.classify.status === "done") {
      const intent = state.classify.intent;
      html += s("classify", "done", "② Classification",
        `<span class="badge ${intent}">${intent}</span>` +
        `<pre>${escapeHtml(JSON.stringify(state.classify.data, null, 2))}</pre>`);
    } else if (state.classify.status === "error") {
      html += s("classify", "err", "② Classification",
        `<span class="badge error">error</span><pre>${escapeHtml(state.classify.error)}</pre>`);
    }
  }

  // Action stage
  if (state.action) {
    html += s("action", "done", "③ Action", renderAction(state.action));
  }

  els.pipeline.innerHTML = html || '<p class="empty">Record or run text to see the chain light up.</p>';
}

function renderAction(action) {
  if (action.type === "task_added") {
    const t = action.task;
    const notion = action.notion && action.notion.pushed
      ? '<div class="meta">✓ Pushed to Notion</div>'
      : `<div class="meta">Notion: ${escapeHtml((action.notion && action.notion.reason) || "n/a")}</div>`;
    return `✅ Task added: <strong>${escapeHtml(t.title)}</strong>${notion}`;
  }
  if (action.type === "caption_generated") {
    return `📝 Caption for <strong>${escapeHtml(action.platform)}</strong> (${escapeHtml(action.tone || "")}):` +
      `<pre>${escapeHtml(action.caption)}</pre>`;
  }
  if (action.type === "summary_shown") {
    const pts = (action.key_points || []).map((p) => `<li>${escapeHtml(p)}</li>`).join("");
    return `🧾 ${escapeHtml(action.summary_text)}<ul>${pts}</ul>`;
  }
  return escapeHtml(JSON.stringify(action));
}

function escapeHtml(str) {
  return String(str).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// ---------- The pipeline driver ----------
// Runs classify + action given a transcript, updating `state` as it goes.
async function runUnderstand(transcript, state) {
  state.classify = { status: "active" };
  renderPipeline(state);

  let res;
  try {
    const r = await fetch("/api/understand", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ transcript }),
    });
    res = await r.json();
  } catch (e) {
    state.classify = { status: "error", error: "Network error calling /api/understand: " + e.message };
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

  if (res.intent === "task") loadTasks();
}

// ---------- Text fallback path ----------
async function runFromText() {
  const text = els.textFallback.value.trim();
  if (!text) { showError("Type or pick a sample first."); return; }
  clearError();
  const state = {};
  // No transcript stage — we already have text.
  await runUnderstand(text, state);
}

// ---------- Mic recording -> WAV -> transcribe -> understand ----------
let mediaRecorder = null;
let chunks = [];
let audioCtx = null;
let analyser = null;
let meterRAF = null;

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
    bars[i].style.height = 4 + v * 36 + "px";
  }
  meterRAF = requestAnimationFrame(animateMeter);
}

async function startRecording() {
  clearError();
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (e) {
    showError("Microphone access denied or unavailable: " + e.message + "\nUse the text fallback below instead.");
    return;
  }

  chunks = [];
  mediaRecorder = new MediaRecorder(stream);
  mediaRecorder.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };
  mediaRecorder.onstop = () => onRecordingStopped(stream);
  mediaRecorder.start();

  // live meter
  audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  const src = audioCtx.createMediaStreamSource(stream);
  analyser = audioCtx.createAnalyser();
  analyser.fftSize = 64;
  src.connect(analyser);
  animateMeter();

  els.recordBtn.classList.add("recording");
  els.recordBtn.textContent = "■ Stop";
  els.recDot.classList.add("live");
  els.recStatus.textContent = "Recording…";
}

function stopRecording() {
  if (mediaRecorder && mediaRecorder.state !== "inactive") mediaRecorder.stop();
  els.recordBtn.classList.remove("recording");
  els.recordBtn.textContent = "● Record";
  els.recDot.classList.remove("live");
  els.recStatus.textContent = "Processing…";
  if (meterRAF) cancelAnimationFrame(meterRAF);
}

async function onRecordingStopped(stream) {
  stream.getTracks().forEach((t) => t.stop());
  if (audioCtx) { audioCtx.close(); audioCtx = null; analyser = null; }

  const mime = mediaRecorder.mimeType || "audio/webm";
  const blob = new Blob(chunks, { type: mime });

  const state = { transcript: { status: "active" } };
  renderPipeline(state);

  // OpenAI's transcription API accepts the recorded audio directly (webm/mp4/…),
  // so we send the raw blob — no in-browser conversion needed. Pick a filename
  // extension that matches the recorder's mime type so the API infers the format.
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

  // Feed the transcript into the understanding + action stages.
  await runUnderstand(res.transcript, state);
}

// ---------- Task list ----------
async function loadTasks() {
  try {
    const r = await fetch("/api/tasks");
    const { tasks } = await r.json();
    if (!tasks.length) { els.tasks.innerHTML = '<p class="empty">No tasks yet.</p>'; return; }
    els.tasks.innerHTML = tasks.map((t) => `
      <div class="task-item">
        <div class="t">${escapeHtml(t.title)}</div>
        <div class="meta">
          <span class="pri ${t.priority}">${t.priority}</span>
          ${t.owner ? " · 👤 " + escapeHtml(t.owner) : ""}
          ${t.due_date ? " · 📅 " + escapeHtml(t.due_date) : ""}
        </div>
      </div>`).join("");
  } catch (e) {
    showError("Could not load tasks: " + e.message);
  }
}

// ---------- Wire up ----------
els.recordBtn.addEventListener("click", () => {
  if (mediaRecorder && mediaRecorder.state === "recording") stopRecording();
  else startRecording();
});
els.runTextBtn.addEventListener("click", runFromText);

SAMPLES.forEach((text, i) => {
  const b = document.createElement("button");
  b.textContent = "Sample " + (i + 1);
  b.title = text;
  b.addEventListener("click", () => { els.textFallback.value = text; });
  els.samples.appendChild(b);
});

loadTasks();
