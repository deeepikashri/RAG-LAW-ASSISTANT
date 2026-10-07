const API_BASE = ""; // same-origin; set e.g. "https://your-api.onrender.com" if frontend is hosted separately

const chatScroll = document.getElementById("chatScroll");
const composer = document.getElementById("composer");
const questionInput = document.getElementById("questionInput");
const sendBtn = document.getElementById("sendBtn");
const statusDot = document.getElementById("statusDot");
const statusText = document.getElementById("statusText");
const suggestions = document.getElementById("suggestions");

let isReady = false;

// --- Autosize the textarea -------------------------------------------------
questionInput.addEventListener("input", () => {
  questionInput.style.height = "auto";
  questionInput.style.height = Math.min(questionInput.scrollHeight, 140) + "px";
});

questionInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    composer.requestSubmit();
  }
});

suggestions.addEventListener("click", (e) => {
  const btn = e.target.closest(".suggestion-chip");
  if (!btn) return;
  questionInput.value = btn.textContent;
  questionInput.dispatchEvent(new Event("input"));
  questionInput.focus();
});

// --- Health polling ----------------------------------------------------------
async function pollHealth() {
  try {
    const res = await fetch(`${API_BASE}/api/health`);
    const data = await res.json();
    if (data.error) {
      setStatus("error", "Backend error — see server logs");
      return;
    }
    if (data.ready) {
      isReady = true;
      setStatus("ready", "Ready");
      return;
    }
    setStatus("loading", "Loading models…");
  } catch (err) {
    setStatus("error", "Can't reach backend");
  }
  if (!isReady) setTimeout(pollHealth, 3000);
}

function setStatus(kind, text) {
  statusDot.className = "status-dot" + (kind === "ready" ? " ready" : kind === "error" ? " error" : "");
  statusText.textContent = text;
}

pollHealth();

// --- Chat submission ---------------------------------------------------------
composer.addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = questionInput.value.trim();
  if (!question) return;

  addUserMessage(question);
  questionInput.value = "";
  questionInput.style.height = "auto";
  sendBtn.disabled = true;

  const typingEl = addTypingIndicator();

  try {
    const res = await fetch(`${API_BASE}/api/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });

    typingEl.remove();

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      addErrorMessage(err.detail || `Request failed (${res.status})`);
      return;
    }

    const data = await res.json();
    addAssistantMessage(data);
  } catch (err) {
    typingEl.remove();
    addErrorMessage("Network error — is the backend running?");
  } finally {
    sendBtn.disabled = false;
  }
});

// --- Rendering helpers ---------------------------------------------------------
function addUserMessage(text) {
  const wrap = document.createElement("div");
  wrap.className = "chat-msg user";
  wrap.innerHTML = `<div class="msg-bubble">${escapeHtml(text)}</div>`;
  chatScroll.appendChild(wrap);
  scrollToBottom();
}

function addTypingIndicator() {
  const wrap = document.createElement("div");
  wrap.className = "chat-msg assistant";
  wrap.innerHTML = `<div class="msg-bubble"><div class="typing-dots"><span></span><span></span><span></span></div></div>`;
  chatScroll.appendChild(wrap);
  scrollToBottom();
  return wrap;
}

function addErrorMessage(text) {
  const wrap = document.createElement("div");
  wrap.className = "chat-msg assistant";
  wrap.innerHTML = `<div class="msg-bubble error-text">${escapeHtml(text)}</div>`;
  chatScroll.appendChild(wrap);
  scrollToBottom();
}

function addAssistantMessage(data) {
  const { answer, decision, scores, sources } = data;
  const { section, body } = splitSectionAnswer(answer);

  const wrap = document.createElement("div");
  wrap.className = "chat-msg assistant";

  const decisionClass = (decision || "").toLowerCase();
  const scoreId = "scores-" + Math.random().toString(36).slice(2, 9);

  let scoresHtml = "";
  if (scores) {
    scoresHtml = `
      <button class="scores-toggle" data-target="${scoreId}">Show critique scores</button>
      <div class="scores-grid" id="${scoreId}">
        ${["faithfulness", "completeness", "relevance", "safety"]
          .map(
            (key) => `
          <div class="score-cell">
            <span class="score-label">${key}</span>
            <span class="score-value">${scores[key]}/5</span>
          </div>`
          )
          .join("")}
      </div>`;
  }

  const sourcesHtml =
    sources && sources.length
      ? `<div class="citation-meta">${sources.map((s) => `<span class="meta-chip">${escapeHtml(s)}</span>`).join("")}</div>`
      : "";

  wrap.innerHTML = `
    <div class="msg-bubble citation-card">
      <div class="citation-header">
        ${section ? `<span class="section-tag">${escapeHtml(section)}</span>` : "<span></span>"}
        ${decision ? `<span class="decision-badge ${decisionClass}">${escapeHtml(decision)}</span>` : ""}
      </div>
      <div class="citation-answer">${escapeHtml(body || answer)}</div>
      ${sourcesHtml}
      ${scoresHtml}
    </div>
  `;

  chatScroll.appendChild(wrap);

  const toggle = wrap.querySelector(".scores-toggle");
  if (toggle) {
    toggle.addEventListener("click", () => {
      const el = document.getElementById(toggle.dataset.target);
      el.classList.toggle("open");
      toggle.textContent = el.classList.contains("open") ? "Hide critique scores" : "Show critique scores";
    });
  }

  scrollToBottom();
}

function splitSectionAnswer(text) {
  if (!text) return { section: null, body: text };
  const match = text.match(/Section\s*:\s*(.+?)\n\s*Answer\s*:\s*([\s\S]+)/i);
  if (!match) return { section: null, body: text };
  return { section: `Section: ${match[1].trim()}`, body: match[2].trim() };
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

function scrollToBottom() {
  chatScroll.scrollTop = chatScroll.scrollHeight;
}
