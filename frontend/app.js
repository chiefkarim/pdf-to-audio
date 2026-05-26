"use strict";

// ── DOM refs ────────────────────────────────────────────────────────────────
const form            = document.getElementById("convert-form");
const fileInput       = document.getElementById("file-input");
const fileDrop        = document.getElementById("file-drop");
const fileLabel       = document.getElementById("file-label");
const convertBtn      = document.getElementById("convert-btn");

const progressSection = document.getElementById("progress-section");
const progressBar     = document.getElementById("progress-bar");
const progressPct     = document.getElementById("progress-pct");
const progressStatus  = document.getElementById("progress-status");
const progressLabel   = document.getElementById("progress-label");

const downloadSection = document.getElementById("download-section");
const downloadLink    = document.getElementById("download-link");

const errorSection    = document.getElementById("error-section");
const errorMessage    = document.getElementById("error-message");
const retryBtn        = document.getElementById("retry-btn");

const jobsList        = document.getElementById("jobs-list");
const jobsEmpty       = document.getElementById("jobs-empty");
const jobsSection     = document.getElementById("jobs-section");
const voiceField      = document.getElementById("voice-field");

// ── State ───────────────────────────────────────────────────────────────────
let pollTimer = null;

// ── Voice field visibility ───────────────────────────────────────────────────
function updateVoiceVisibility() {
  const mode = form.querySelector("input[name='mode']:checked")?.value;
  voiceField.hidden = mode !== "quality";
}
form.querySelectorAll("input[name='mode']").forEach((r) =>
  r.addEventListener("change", updateVoiceVisibility)
);

// ── File input wiring ───────────────────────────────────────────────────────
fileInput.addEventListener("change", () => {
  onFileChosen(fileInput.files[0] || null);
});

// Drag-and-drop support on the drop zone.
fileDrop.addEventListener("dragover", (e) => {
  e.preventDefault();
  fileDrop.classList.add("drag-over");
});

["dragleave", "dragend"].forEach((evt) => {
  fileDrop.addEventListener(evt, () => fileDrop.classList.remove("drag-over"));
});

fileDrop.addEventListener("drop", (e) => {
  e.preventDefault();
  fileDrop.classList.remove("drag-over");

  const file = e.dataTransfer.files[0] || null;
  if (file && file.type === "application/pdf") {
    // Push the dropped file into the real input so FormData picks it up.
    const dt = new DataTransfer();
    dt.items.add(file);
    fileInput.files = dt.files;
    onFileChosen(file);
  } else if (file) {
    showError("Only PDF files are accepted.");
  }
});

function onFileChosen(file) {
  if (!file) {
    fileLabel.textContent = "Choose a PDF or drag it here";
    fileDrop.classList.remove("has-file");
    convertBtn.disabled = true;
    return;
  }
  fileLabel.textContent = file.name;
  fileDrop.classList.add("has-file");
  convertBtn.disabled = false;

  // Clear any previous result / error when a new file is picked.
  hideAll();
}

// ── Form submit ─────────────────────────────────────────────────────────────
form.addEventListener("submit", async (e) => {
  e.preventDefault();
  hideAll();

  const file   = fileInput.files[0];
  const format = form.querySelector("input[name='format']:checked").value;
  const mode   = form.querySelector("input[name='mode']:checked").value;
  const voice  = form.querySelector("input[name='voice']:checked")?.value ?? "male";

  if (!file) {
    showError("Please select a PDF file.");
    return;
  }

  const body = new FormData();
  body.append("file",   file);
  body.append("format", format);
  body.append("mode",   mode);
  body.append("voice",  voice);

  convertBtn.disabled = true;
  setProgress(0, "Uploading…", "queued");
  progressSection.hidden = false;

  try {
    const res = await fetch("/upload", { method: "POST", body });
    if (!res.ok) {
      const text = await res.text().catch(() => "");
      throw new Error("Upload failed (HTTP " + res.status + ")" + (text ? ": " + text : ""));
    }
    const { job_id } = await res.json();
    if (!job_id) { showError("Server error: missing job ID"); return; }

    progressLabel.textContent = "Processing…";
    progressSection.hidden = true;
    convertBtn.disabled = false;

    // Scroll jobs panel into view — the global 2 s poll will pick up the new job.
    jobsSection.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (err) {
    showError(err.message || "Upload failed. Check your connection and try again.");
  }
});

// ── UI helpers (upload flow) ─────────────────────────────────────────────────
function setProgress(value, label, statusText) {
  progressBar.value       = value;
  progressPct.textContent = value + "%";
  if (label)      progressLabel.textContent  = label;
  if (statusText) progressStatus.textContent = statusText;
}

function showError(msg) {
  progressSection.hidden   = true;
  errorMessage.textContent = msg;
  errorSection.hidden = false;
  convertBtn.disabled = false;
}

function hideAll() {
  progressSection.hidden = true;
  downloadSection.hidden = true;
  errorSection.hidden    = true;
}

// ── Retry button ─────────────────────────────────────────────────────────────
retryBtn.addEventListener("click", () => {
  hideAll();
  fileInput.value = "";
  onFileChosen(null);
});

// ── Relative time helper ─────────────────────────────────────────────────────
function relative_time(isoString) {
  const secs = Math.floor((Date.now() - new Date(isoString).getTime()) / 1000);
  if (secs < 60)  return "just now";
  const mins = Math.floor(secs / 60);
  if (mins < 60)  return mins + "m ago";
  const hrs  = Math.floor(mins / 60);
  if (hrs  < 24)  return hrs  + "h ago";
  const days = Math.floor(hrs / 24);
  return days + "d ago";
}

// ── DOM-safe element factory ─────────────────────────────────────────────────
function el(tag, props, ...children) {
  const node = document.createElement(tag);
  if (props) {
    Object.entries(props).forEach(([k, v]) => {
      if (k === "class")      node.className = v;
      else if (k === "title") node.title = v;
      else if (k === "href")  node.href = v;
      else if (k === "download") node.download = v;
      else if (k === "max")   node.max = v;
      else if (k === "value") node.value = v;
      else if (k.startsWith("data-")) node.dataset[k.slice(5)] = v;
      else node.setAttribute(k, v);
    });
  }
  children.forEach((c) => {
    if (c == null) return;
    node.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
  });
  return node;
}

// ── Jobs panel ───────────────────────────────────────────────────────────────
function buildJobRow(job) {
  const isTerminal = job.status === "done" || job.status === "error";
  const pagesTotal = job.pages_total ?? 0;
  const pagesDone  = job.pages_done  ?? 0;

  // meta row
  const meta = el("div", { class: "job-meta" },
    el("span", { class: "job-id", title: job.job_id }, job.job_id.slice(0, 8)),
    el("span", { class: "job-badge job-badge--" + job.status }, job.status),
    el("span", { class: "job-format" }, job.format),
    job.mode ? el("span", { class: "job-mode job-mode--" + job.mode }, job.mode) : null,
    job.ocr_pages > 0 ? el("span", { class: "job-ocr", title: job.ocr_pages + " page(s) needed OCR" }, "OCR ×" + job.ocr_pages) : null,
    job.filename ? el("span", { class: "job-filename", title: job.filename }, job.filename) : null,
    el("span", { class: "job-time" }, relative_time(job.created_at))
  );

  // progress bar + optional page counter (hidden when terminal)
  let progressWrap = null;
  if (!isTerminal) {
    if (pagesTotal > 0) {
      // Show page-level progress alongside the bar
      progressWrap = el("div", { class: "job-progress-wrap" },
        el("progress", { class: "job-progress-bar", max: "100", value: String(job.progress ?? 0) }),
        el("span", { class: "job-pages" }, pagesDone + " / " + pagesTotal + " pages")
      );
    } else {
      // Fallback: percentage only (backward compat)
      progressWrap = el("div", { class: "job-progress-wrap" },
        el("progress", { class: "job-progress-bar", max: "100", value: String(job.progress ?? 0) }),
        el("span", { class: "job-progress-pct" }, (job.progress ?? 0) + "%")
      );
    }
  }

  // "Download so far" partial link — shown when pages have been processed
  const partialLink = (pagesDone > 0 && !isTerminal)
    ? el("a", {
        class: "btn-job btn-job--partial",
        href: "/jobs/" + job.job_id + "/partial",
        download: "audio_partial." + job.format
      }, "Download so far")
    : null;

  // primary action
  let actionNode = null;
  if (job.status === "processing") {
    const btn = el("button", { class: "btn-job btn-job--pause", "data-id": job.job_id, type: "button" }, "Pause");
    btn.addEventListener("click", () => jobAction(job.job_id, "pause"));
    actionNode = btn;
  } else if (job.status === "paused") {
    const btn = el("button", { class: "btn-job btn-job--resume", "data-id": job.job_id, type: "button" }, "Resume");
    btn.addEventListener("click", () => jobAction(job.job_id, "resume"));
    actionNode = btn;
  } else if (job.status === "done") {
    actionNode = el("a",
      { class: "btn-job btn-job--download", href: "/jobs/" + job.job_id + "/download", download: "audio." + job.format },
      "Download"
    );
  } else if (job.status === "error") {
    actionNode = el("span", { class: "job-error-msg" }, job.error || "Failed");
  }

  const actions = el("div", { class: "job-actions" }, actionNode, partialLink);

  const li = el("li", { class: "job-row" }, meta, progressWrap, actions);
  li.dataset.jobId = job.job_id;
  return li;
}

function renderJobs(jobs) {
  // Remove all rows except the empty-state sentinel.
  Array.from(jobsList.children).forEach((child) => {
    if (child !== jobsEmpty) child.remove();
  });

  if (!jobs || jobs.length === 0) {
    jobsEmpty.hidden = false;
    return;
  }

  jobsEmpty.hidden = true;

  // Newest first.
  const sorted = [...jobs].sort(
    (a, b) => new Date(b.created_at) - new Date(a.created_at)
  );

  const frag = document.createDocumentFragment();
  sorted.forEach((job) => frag.appendChild(buildJobRow(job)));
  jobsList.appendChild(frag);
}

async function jobAction(jobId, action) {
  try {
    const res = await fetch("/jobs/" + encodeURIComponent(jobId) + "/" + action, { method: "POST" });
    if (!res.ok && res.status !== 409) {
      console.warn("Job " + action + " failed: HTTP " + res.status);
    }
    // Immediate refresh so the UI feels responsive.
    await fetchAndRenderJobs();
  } catch (err) {
    console.warn("Job action error:", err);
  }
}

async function fetchAndRenderJobs() {
  try {
    const res = await fetch("/jobs");
    if (!res.ok) throw new Error("HTTP " + res.status);
    const jobs = await res.json();
    renderJobs(jobs);
  } catch (err) {
    console.warn("Could not fetch jobs:", err);
  }
}

// ── Global poll (2 s) ────────────────────────────────────────────────────────
fetchAndRenderJobs();
pollTimer = setInterval(fetchAndRenderJobs, 2000);
