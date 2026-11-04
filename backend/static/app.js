// EduHinglish chat page. Talks only to the backend's /api/* routes (same ones the Flutter app will use).
const API = "/api";

const $ = (sel) => document.querySelector(sel);
const els = {
  chatList: $("#chat-list"),
  messages: $("#messages"),
  empty: $("#empty"),
  form: $("#composer"),
  input: $("#input"),
  send: $("#send"),
  subject: $("#subject"),
  classNum: $("#class_num"),
  language: $("#language"),
  health: $("#health"),
};

const state = {
  chatId: null,      // null = new chat not saved yet
  chats: [],
  pending: false,    // waiting for the model
};

const SUBJECTS = { biology: "Biology", chemistry: "Chemistry", physics: "Physics",
                   mathematics: "Maths", social_science: "Social Science" };

// ── API helper ───────────────────────────────────────────────────────────────
async function api(path, options = {}) {
  const res = await fetch(API + path, {
    headers: { "Content-Type": "application/json" },
    ...options,
    body: options.body ? JSON.stringify(options.body) : undefined,
  });
  if (res.status === 204) return null;
  let data = null;
  try { data = await res.json(); } catch { /* non-JSON error page */ }
  if (!res.ok) {
    let detail = data && data.detail;
    if (Array.isArray(detail)) detail = detail.map((d) => d.msg).join("; ");
    throw new Error(detail || `Request failed (${res.status})`);
  }
  return data;
}

// ── Formatting (escape first, then a little markdown) ────────────────────────
function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function inline(s) {
  return s
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/`([^`]+)`/g, "<code>$1</code>");
}

function formatAnswer(text) {
  const lines = escapeHtml(text.trim()).split("\n");
  let html = "", list = null, para = [];
  const flushPara = () => { if (para.length) { html += `<p>${inline(para.join("<br>"))}</p>`; para = []; } };
  const flushList = () => { if (list) { html += `</${list}>`; list = null; } };
  for (const raw of lines) {
    const line = raw.trim();
    const ul = line.match(/^[-*•]\s+(.*)/);
    const ol = line.match(/^\d+[.)]\s+(.*)/);
    if (ul || ol) {
      flushPara();
      const tag = ul ? "ul" : "ol";
      if (list !== tag) { flushList(); html += `<${tag}>`; list = tag; }
      html += `<li>${inline((ul || ol)[1])}</li>`;
    } else if (!line) {
      flushPara(); flushList();
    } else {
      flushList(); para.push(line);
    }
  }
  flushPara(); flushList();
  return html;
}

// ── Sidebar ──────────────────────────────────────────────────────────────────
async function loadChats() {
  try {
    state.chats = await api("/chats");
  } catch (e) {
    state.chats = [];
  }
  renderChatList();
}

function renderChatList() {
  els.chatList.innerHTML = "";
  for (const chat of state.chats) {
    const item = document.createElement("div");
    item.className = "chat-item" + (chat.id === state.chatId ? " active" : "");
    const title = document.createElement("button");
    title.className = "title";
    title.textContent = chat.title;
    title.title = chat.title;
    title.onclick = () => openChat(chat.id);
    const del = document.createElement("button");
    del.className = "delete";
    del.textContent = "✕";
    del.title = "Delete chat";
    del.onclick = () => deleteChat(chat.id);
    item.append(title, del);
    els.chatList.append(item);
  }
}

async function openChat(id) {
  closeSidebar();
  if (state.pending) return;
  try {
    const chat = await api(`/chats/${id}`);
    state.chatId = chat.id;
    els.subject.value = chat.subject || "";
    els.classNum.value = chat.class_num || "";
    els.language.value = chat.language || "hinglish";
    clearMessages();
    chat.messages.forEach(renderMessage);
    renderChatList();
    scrollDown();
  } catch (e) {
    alert(e.message);
    loadChats();
  }
}

function newChat() {
  closeSidebar();
  if (state.pending) return;
  state.chatId = null;
  clearMessages();
  els.empty.hidden = false;
  renderChatList();
  els.input.focus();
}

async function deleteChat(id) {
  if (!confirm("Delete this chat?")) return;
  try {
    await api(`/chats/${id}`, { method: "DELETE" });
  } catch (e) {
    alert(e.message);
  }
  if (id === state.chatId) newChat();
  loadChats();
}

// ── Messages ─────────────────────────────────────────────────────────────────
function clearMessages() {
  els.messages.querySelectorAll(".msg").forEach((m) => m.remove());
  els.empty.hidden = true;
}

function scrollDown() {
  els.messages.scrollTop = els.messages.scrollHeight;
}

function addRow(role) {
  els.empty.hidden = true;
  const row = document.createElement("div");
  row.className = `msg ${role}`;
  els.messages.append(row);
  return row;
}

function renderMessage(msg) {
  const row = addRow(msg.role);
  if (msg.role === "user") {
    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.textContent = msg.content;
    row.append(bubble);
    return row;
  }

  const content = document.createElement("div");
  content.className = "content";
  content.innerHTML = formatAnswer(msg.content || "(empty answer)");
  row.append(content);

  if (msg.sources && msg.sources.length) {
    const box = document.createElement("div");
    box.className = "sources";
    for (const s of msg.sources) {
      const tag = document.createElement("span");
      tag.className = "source";
      const subj = SUBJECTS[s.subject] || s.subject;
      tag.textContent = [s.chapter, subj, s.class && `Class ${s.class}`].filter(Boolean).join(" · ");
      tag.title = `Relevance ${Number(s.score || 0).toFixed(2)}\n${s.text_preview || ""}`;
      box.append(tag);
    }
    row.append(box);
  }

  const actions = document.createElement("div");
  actions.className = "actions";
  const up = feedbackButton("👍", 1, msg);
  const down = feedbackButton("👎", -1, msg);
  const copy = document.createElement("button");
  copy.textContent = "⧉";
  copy.title = "Copy";
  copy.onclick = () => navigator.clipboard && navigator.clipboard.writeText(msg.content);
  actions.append(up, down, copy);
  const meta = [];
  if (msg.intent) meta.push(msg.intent.replace(/_/g, " "));
  if (msg.timings && msg.timings.total) meta.push(`${msg.timings.total.toFixed(1)} s`);
  if (meta.length) {
    const m = document.createElement("span");
    m.className = "meta";
    m.textContent = meta.join(" · ");
    actions.append(m);
  }
  row.append(actions);
  return row;
}

function feedbackButton(label, value, msg) {
  const btn = document.createElement("button");
  btn.textContent = label;
  btn.title = value > 0 ? "Good answer" : "Bad answer";
  btn.dataset.value = value;
  if (msg.feedback === value) btn.classList.add("on");
  btn.onclick = async () => {
    const next = msg.feedback === value ? 0 : value;
    try {
      const saved = await api(`/messages/${msg.id}/feedback`, { method: "POST", body: { value: next } });
      msg.feedback = saved.feedback;
      btn.parentElement.querySelectorAll("button[data-value]").forEach((b) =>
        b.classList.toggle("on", Number(b.dataset.value) === msg.feedback));
    } catch (e) {
      alert(e.message);
    }
  };
  return btn;
}

function showTyping() {
  const row = addRow("assistant");
  row.innerHTML = '<div class="typing"><span></span><span></span><span></span>Soch raha hoon…</div>';
  scrollDown();
  return row;
}

function showError(row, message, retry) {
  row.innerHTML = "";
  const box = document.createElement("div");
  box.className = "error";
  box.textContent = message;
  const btn = document.createElement("button");
  btn.textContent = "Retry";
  btn.onclick = retry;
  box.append(btn);
  row.append(box);
}

// ── Sending ──────────────────────────────────────────────────────────────────
async function send(text) {
  text = text.trim();
  if (text.length < 2 || state.pending) return;
  state.pending = true;
  updateSendButton();

  const chatIdAtSend = state.chatId;
  const userRow = renderMessage({ role: "user", content: text });
  const typingRow = showTyping();

  try {
    const res = await api("/chat", {
      method: "POST",
      body: {
        message: text,
        chat_id: chatIdAtSend,
        subject: els.subject.value || null,
        class_num: els.classNum.value || null,
        language: els.language.value,
      },
    });
    state.chatId = res.chat.id;
    typingRow.remove();
    renderMessage(res.assistant_message);
    scrollDown();
    loadChats();
  } catch (e) {
    // Nothing was saved on the server; retry sends the same question again.
    showError(typingRow, e.message, () => {
      userRow.remove();
      typingRow.remove();
      send(text);
    });
  } finally {
    state.pending = false;
    updateSendButton();
  }
}

function updateSendButton() {
  els.send.disabled = state.pending || els.input.value.trim().length < 2;
}

function autosize() {
  els.input.style.height = "auto";
  els.input.style.height = Math.min(els.input.scrollHeight, 200) + "px";
}

els.form.addEventListener("submit", (e) => {
  e.preventDefault();
  const text = els.input.value;
  if (state.pending || text.trim().length < 2) return;
  els.input.value = "";
  autosize();
  send(text);
});
els.input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    els.form.requestSubmit();
  }
});
els.input.addEventListener("input", () => { autosize(); updateSendButton(); });
document.querySelectorAll(".example").forEach((b) => (b.onclick = () => send(b.textContent)));
$("#new-chat").onclick = newChat;

// ── Phone sidebar ────────────────────────────────────────────────────────────
function closeSidebar() { document.body.classList.remove("sidebar-open"); }
$("#menu").onclick = () => document.body.classList.add("sidebar-open");
$("#backdrop").onclick = closeSidebar;

// ── Model status ─────────────────────────────────────────────────────────────
async function checkHealth() {
  const text = els.health.querySelector(".health-text");
  try {
    const h = await api("/health");
    const m = h.model_server || {};
    els.health.className = "health " + (m.status === "ok" ? "ok" : m.status === "loading" ? "" : "offline");
    if (m.status === "ok") {
      text.textContent = m.busy ? "Model online · busy" : "Model online";
      els.health.title = `${h.model_server_url}\nLoRA: ${m.lora_loaded ? "yes" : "no"} · KB chunks: ${m.kb_chunks}`;
    } else if (m.status === "loading") {
      text.textContent = "Model loading…";
    } else {
      text.textContent = "Model offline";
      els.health.title = m.detail || "";
    }
  } catch {
    els.health.className = "health offline";
    text.textContent = "Backend offline";
  }
}

loadChats();
checkHealth();
setInterval(checkHealth, 30000);
