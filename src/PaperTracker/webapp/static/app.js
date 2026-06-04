const state = {
  briefs: [],
  collections: [],
  activeBriefId: null,
  activeCollectionId: null,
  activeView: "brief",
  renameTarget: null,
  currentConversationId: null,
  pendingPaperForCollection: null,
  llmEnabled: false,
  rag: {},
};

const API_ORIGIN = window.location.origin === "http://127.0.0.1:8000"
  ? ""
  : "http://127.0.0.1:8000";

function byId(id) {
  return document.getElementById(id);
}

async function api(path, options = {}) {
  const url = `${API_ORIGIN}${path}`;
  let response;
  try {
    response = await fetch(url, {
      headers: { "Content-Type": "application/json" },
      ...options,
    });
  } catch (error) {
    throw new Error(`无法连接 ResearchMind 后端：${error.message || "network error"}。当前页面：${window.location.href}；请求地址：${url}。请确认服务仍在运行，并优先从 http://127.0.0.1:8000 打开页面。`);
  }
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.detail || `Request failed: ${response.status}`);
  }
  if (response.status === 204) return null;
  return response.json();
}

async function streamApi(path, options = {}, onEvent) {
  const url = `${API_ORIGIN}${path}`;
  let response;
  try {
    response = await fetch(url, {
      headers: { "Content-Type": "application/json" },
      ...options,
    });
  } catch (error) {
    throw new Error(`无法连接 ResearchMind 后端：${error.message || "network error"}。当前页面：${window.location.href}；请求地址：${url}。请确认服务仍在运行，并优先从 http://127.0.0.1:8000 打开页面。`);
  }
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.detail || `Request failed: ${response.status}`);
  }
  if (!response.body) throw new Error("当前浏览器不支持流式响应");

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const events = buffer.split("\n\n");
    buffer = events.pop() || "";
    for (const eventText of events) {
      const dataLines = eventText
        .split("\n")
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.slice(5).trim());
      if (!dataLines.length) continue;
      onEvent(JSON.parse(dataLines.join("\n")));
    }
  }
  buffer += decoder.decode();
  if (buffer.trim()) {
    const data = buffer
      .split("\n")
      .filter((line) => line.startsWith("data:"))
      .map((line) => line.slice(5).trim())
      .join("\n");
    if (data) onEvent(JSON.parse(data));
  }
}

function formatTime(timestamp) {
  if (!timestamp) return "-";
  return new Date(timestamp * 1000).toLocaleString();
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function renderAssistantContent(markdown) {
  const raw = String(markdown || "");
  if (!window.marked || !window.DOMPurify) {
    return `<div class="message-content plain-text">${escapeHtml(raw)}</div>`;
  }

  marked.setOptions({
    gfm: true,
    breaks: true,
  });

  const rendered = marked.parse(raw);
  const sanitized = DOMPurify.sanitize(rendered, {
    USE_PROFILES: { html: true },
  });

  return `<div class="message-content rich-text">${sanitized}</div>`;
}

function renderMath(container) {
  if (!container || !window.renderMathInElement) return;
  renderMathInElement(container, {
    delimiters: [
      { left: "$$", right: "$$", display: true },
      { left: "\\[", right: "\\]", display: true },
      { left: "$", right: "$", display: false },
      { left: "\\(", right: "\\)", display: false },
    ],
    throwOnError: false,
    strict: "ignore",
  });
}

function setView(view) {
  state.activeView = view;
  byId("brief-view")?.classList.toggle("hidden", view !== "brief");
  byId("collection-view")?.classList.toggle("hidden", view !== "collection");
}

function openCollectionModal() {
  byId("collection-modal")?.showModal();
}

function closeCollectionModal() {
  byId("collection-modal")?.close();
}

function openRenameModal(target) {
  state.renameTarget = target;
  const titleNode = byId("rename-modal-title");
  if (titleNode) titleNode.textContent = target.type === "brief" ? "重命名 Brief" : "编辑 Knowledge Base";
  byId("rename-name").value = target.name || "";
  const descriptionInput = byId("rename-description");
  descriptionInput.value = target.description || "";
  byId("rename-description-wrap").style.display = target.type === "collection" ? "block" : "none";
  byId("rename-modal")?.showModal();
}

function closeRenameModal() {
  byId("rename-modal")?.close();
  state.renameTarget = null;
}

function openAddPaperModal(source, sourceId) {
  state.pendingPaperForCollection = { source, source_id: sourceId };
  renderCollectionSelect(byId("paper-target-collection"));
  byId("add-paper-modal")?.showModal();
}

function closeAddPaperModal() {
  byId("add-paper-modal")?.close();
  state.pendingPaperForCollection = null;
}

function focusSearchLanding() {
  state.activeBriefId = null;
  setView("brief");
  renderLandingState();
  renderNavLists();
  byId("landing-search-query")?.focus();
}

function renderNavLists() {
  const briefList = byId("brief-list");
  const collectionList = byId("collection-list");

  if (!briefList || !collectionList) return;

  briefList.innerHTML = "";
  collectionList.innerHTML = "";

  state.briefs.forEach((brief) => {
    const shell = document.createElement("div");
    shell.className = `nav-item-shell ${brief.id === state.activeBriefId ? "is-active" : ""}`;
    shell.innerHTML = `
      <button type="button" class="nav-item ${brief.id === state.activeBriefId ? "is-active" : ""}">
        <strong>${escapeHtml(brief.title)}</strong>
        <small>${brief.paper_count} papers | ${formatTime(brief.last_run_at)}</small>
      </button>
      <div class="nav-actions">
        <button type="button" class="nav-action" data-rename-brief="${brief.id}" aria-label="重命名 Brief">⋯</button>
        <button type="button" class="nav-action" data-delete-brief="${brief.id}" aria-label="删除 Brief">×</button>
      </div>
    `;
    shell.querySelector(".nav-item").addEventListener("click", () => runAction(() => loadBrief(brief.id)));
    shell.querySelector("[data-rename-brief]").addEventListener("click", (event) => {
      event.stopPropagation();
      openRenameModal({ type: "brief", id: brief.id, name: brief.title });
    });
    shell.querySelector("[data-delete-brief]").addEventListener("click", (event) => {
      event.stopPropagation();
      runAction(() => onDeleteBrief(brief.id));
    });
    briefList.appendChild(shell);
  });

  state.collections.forEach((collection) => {
    const shell = document.createElement("div");
    shell.className = `nav-item-shell ${collection.id === state.activeCollectionId ? "is-active" : ""}`;
    shell.innerHTML = `
      <button type="button" class="nav-item ${collection.id === state.activeCollectionId ? "is-active" : ""}">
        <strong>${escapeHtml(collection.name)}</strong>
        <small>${escapeHtml(collection.description || "Knowledge Base")}</small>
      </button>
      <div class="nav-actions">
        <button type="button" class="nav-action" data-rename-collection="${collection.id}" aria-label="编辑 Knowledge Base">⋯</button>
        <button type="button" class="nav-action" data-delete-collection="${collection.id}" aria-label="删除 Knowledge Base">×</button>
      </div>
    `;
    shell.querySelector(".nav-item").addEventListener("click", () => runAction(() => loadCollection(collection.id)));
    shell.querySelector("[data-rename-collection]").addEventListener("click", (event) => {
      event.stopPropagation();
      openRenameModal({
        type: "collection",
        id: collection.id,
        name: collection.name,
        description: collection.description || "",
      });
    });
    shell.querySelector("[data-delete-collection]").addEventListener("click", (event) => {
      event.stopPropagation();
      runAction(() => onDeleteCollection(collection.id));
    });
    collectionList.appendChild(shell);

  });

  if (!state.activeCollectionId && state.collections.length) {
    state.activeCollectionId = state.collections[0].id;
  }
  renderCollectionSelect(byId("paper-target-collection"));
}

function renderCollectionSelect(selectNode) {
  if (!selectNode) return;
  selectNode.innerHTML = "";
  state.collections.forEach((collection) => {
    const option = document.createElement("option");
    option.value = String(collection.id);
    option.textContent = collection.name;
    selectNode.appendChild(option);
  });
  if (state.activeCollectionId) {
    selectNode.value = String(state.activeCollectionId);
  }
}

async function loadBootstrap() {
  const payload = await api("/api/bootstrap");
  state.briefs = payload.briefs;
  state.collections = payload.collections;
  state.llmEnabled = Boolean(payload.llm_enabled);
  state.rag = payload.rag || {};
  byId("landing-search-max-results").value = payload.search_defaults.max_results;
  if (byId("rag-mode")) byId("rag-mode").value = state.rag.default_mode || "standard";
  const ragMode = byId("rag-mode");
  if (ragMode) {
    const decomposeOption = ragMode.querySelector('option[value="decompose"]');
    const agentOption = ragMode.querySelector('option[value="agent"]');
    decomposeOption?.toggleAttribute("disabled", !state.rag.decompose_enabled);
    agentOption?.toggleAttribute("disabled", !state.rag.agent_enabled);
  }
  renderNavLists();
  focusSearchLanding();
}

async function refreshBriefs() {
  const payload = await api("/api/briefs");
  state.briefs = payload.briefs;
  renderNavLists();
}

async function refreshCollections() {
  const payload = await api("/api/collections");
  state.collections = payload.collections;
  renderNavLists();
}

async function createBriefSearch(query, maxResults) {
  const payload = await api("/api/briefs/search", {
    method: "POST",
    body: JSON.stringify({ query, max_results: maxResults }),
  });
  await refreshBriefs();
  await loadBrief(payload.brief.id);
  if (payload.existing_brief) {
    alert(`已存在同名 Brief，本次搜索结果已追加到「${payload.brief.title}」。新增 ${payload.inserted_count} 篇。`);
  }
  if (payload.warning) {
    alert(`Brief 已创建，但本次搜索没有成功完成：${payload.warning}`);
  }
}

async function onCreateBriefSearchFromLanding() {
  const query = byId("landing-search-query").value.trim();
  const maxResults = Number(byId("landing-search-max-results").value || 5);
  if (!query) throw new Error("请输入搜索关键词");
  await createBriefSearch(query, maxResults);
  byId("landing-search-query").value = "";
}

async function loadBrief(briefId) {
  const payload = await api(`/api/briefs/${briefId}`);
  state.activeBriefId = briefId;
  setView("brief");
  renderNavLists();
  renderBrief(payload);
}

async function onRenameTarget(event) {
  event.preventDefault();
  if (!state.renameTarget) return;
  const name = byId("rename-name").value.trim();
  const description = byId("rename-description").value.trim();
  if (!name) throw new Error("请输入名称");

  if (state.renameTarget.type === "brief") {
    await api(`/api/briefs/${state.renameTarget.id}`, {
      method: "PATCH",
      body: JSON.stringify({ title: name }),
    });
    await refreshBriefs();
    if (state.activeBriefId === state.renameTarget.id) {
      await loadBrief(state.activeBriefId);
    }
  } else {
    await api(`/api/collections/${state.renameTarget.id}`, {
      method: "PATCH",
      body: JSON.stringify({ name, description }),
    });
    await refreshCollections();
    if (state.activeCollectionId === state.renameTarget.id) {
      await loadCollection(state.activeCollectionId);
    }
  }
  closeRenameModal();
}

async function onDeleteBrief(briefId) {
  if (!confirm("确定删除这个 Brief 吗？")) return;
  await api(`/api/briefs/${briefId}`, { method: "DELETE" });
  if (state.activeBriefId === briefId) {
    state.activeBriefId = null;
    focusSearchLanding();
  }
  await refreshBriefs();
}

async function onDeleteCollection(collectionId) {
  if (!confirm("确定删除这个 Knowledge Base 吗？")) return;
  await api(`/api/collections/${collectionId}`, { method: "DELETE" });
  if (state.activeCollectionId === collectionId) {
    state.activeCollectionId = null;
    renderEmptyCollection();
    if (state.activeView === "collection") {
      focusSearchLanding();
    }
  }
  await refreshCollections();
}

async function onRerunBrief() {
  if (!state.activeBriefId) throw new Error("请先选择一个 Brief");
  const maxResults = Number(byId("brief-rerun-max-results").value || 5);
  const payload = await api(`/api/briefs/${state.activeBriefId}/rerun`, {
    method: "POST",
    body: JSON.stringify({ max_results: maxResults }),
  });
  await refreshBriefs();
  renderBrief(payload);
  if (payload.inserted_count > 0) {
    alert(`已追加 ${payload.inserted_count} 篇新论文。`);
  } else {
    alert("本次 rerun 没有发现新的论文，可以稍后再试或调大返回数量。");
  }
}

async function onDeleteBriefPaper(source, sourceId) {
  if (!state.activeBriefId) return;
  const payload = await api(`/api/briefs/${state.activeBriefId}/papers/${encodeURIComponent(source)}/${encodeURIComponent(sourceId)}`, {
    method: "DELETE",
  });
  await refreshBriefs();
  renderBrief(payload);
}

async function onAddSelectedToCollection() {
  if (!state.activeBriefId) throw new Error("请先选择一个 Brief");
  const targetCollection = byId("paper-target-collection");
  if (!targetCollection) throw new Error("请选择目标 Knowledge Base");
  const collectionId = Number(targetCollection.value);
  if (!collectionId) throw new Error("请先创建并选择一个 Knowledge Base");
  const selected = state.pendingPaperForCollection ? [state.pendingPaperForCollection] : [];
  if (!selected.length) throw new Error("请先选择一篇论文");
  await api(`/api/briefs/${state.activeBriefId}/add-to-collection`, {
    method: "POST",
    body: JSON.stringify({
      collection_id: collectionId,
      selected_papers: selected,
      ingest_immediately: true,
    }),
  });
  state.activeCollectionId = collectionId;
  closeAddPaperModal();
  await refreshCollections();
  await loadCollection(collectionId);
}

async function onIngestCollection() {
  if (!state.activeCollectionId) throw new Error("请先选择一个 Knowledge Base");
  const button = byId("ingest-collection");
  const previousLabel = button?.textContent || "更新索引";
  if (button) {
    button.disabled = true;
    button.textContent = "已加入队列";
  }
  try {
    const payload = await api(`/api/collections/${state.activeCollectionId}/ingest`, {
      method: "POST",
    });
    alert(`已开始更新 ${payload.scheduled} 篇论文的索引。完成后会显示「索引」。`);
    await loadCollection(state.activeCollectionId);
  } finally {
    if (button) {
      button.disabled = false;
      button.textContent = previousLabel;
    }
  }
}

function renderLandingState() {
  byId("brief-landing").classList.remove("hidden");
  byId("brief-detail-hero").classList.add("hidden");
  byId("brief-detail-body").classList.add("hidden");
}

function hideLandingState() {
  byId("brief-landing").classList.add("hidden");
  byId("brief-detail-hero").classList.remove("hidden");
  byId("brief-detail-body").classList.remove("hidden");
}

function renderBrief(payload) {
  hideLandingState();
  const brief = payload.brief;
  const papers = payload.papers || [];

  byId("brief-title").textContent = brief.title;
  byId("brief-meta").textContent = `Sources: ${brief.sources.join(", ")} | last run ${formatTime(brief.last_run_at)}`;
  byId("brief-paper-count").textContent = `${papers.length} papers`;
  byId("brief-rerun-max-results").value = brief.max_results || 5;

  const container = byId("brief-papers");
  container.innerHTML = "";
  if (!papers.length) {
    container.innerHTML = `<div class="empty-state">这个 Brief 目前没有论文。你可以 rerun search，或修改搜索词后创建新的 Brief。</div>`;
    return;
  }

  papers.forEach((paper) => {
    const card = document.createElement("div");
    card.className = "paper-card";
    card.innerHTML = `
      <div class="checkbox-row row">
        <div class="paper-title-block">
          <h4>${escapeHtml(paper.title)}</h4>
          <div class="paper-meta">${escapeHtml((paper.authors || []).join(", ") || "Unknown authors")}</div>
        </div>
        <div class="paper-actions">
          <button class="secondary-action add-paper-button" type="button">加入知识库</button>
          <button class="danger-link" type="button">删除</button>
        </div>
      </div>
      <div class="badge-row">
        <span class="badge">${escapeHtml(paper.source)}</span>
      </div>
      <p class="paper-meta">${paper.pdf_url ? `<a href="${paper.pdf_url}" target="_blank" rel="noopener noreferrer">PDF</a>` : "No PDF link"}</p>
      <details class="abstract-details">
        <summary>摘要</summary>
        <p>${escapeHtml(paper.abstract || "No abstract available.")}</p>
      </details>
    `;
    card.querySelector(".add-paper-button").addEventListener("click", () => openAddPaperModal(paper.source, paper.source_id));
    card.querySelector(".danger-link").addEventListener("click", () => runAction(() => onDeleteBriefPaper(paper.source, paper.source_id)));
    container.appendChild(card);
  });
}

async function onCreateCollection(event) {
  event.preventDefault();
  const name = byId("collection-name").value.trim();
  const description = byId("collection-description").value.trim();
  if (!name) throw new Error("请输入 Knowledge Base 名称");
  const payload = await api("/api/collections", {
    method: "POST",
    body: JSON.stringify({ name, description }),
  });
  byId("collection-form").reset();
  closeCollectionModal();
  state.activeCollectionId = payload.collection.id;
  await refreshCollections();
  await loadCollection(payload.collection.id);
}

async function loadCollection(collectionId) {
  const payload = await api(`/api/collections/${collectionId}`);
  state.activeCollectionId = collectionId;
  state.currentConversationId = payload.conversations?.[0]?.id || null;
  setView("collection");
  renderNavLists();
  await renderCollection(payload);
}

async function renderCollection(payload) {
  const collection = payload.collection;
  const papers = payload.papers || [];
  if (byId("collection-title")) byId("collection-title").textContent = collection.name;
  if (byId("chat-meta")) byId("chat-meta").textContent = state.currentConversationId
    ? ""
    : state.llmEnabled
      ? "RAG + LLM 已启用"
      : "RAG 检索已启用，LLM 未配置";

  const container = byId("collection-papers");
  container.innerHTML = "";
  if (!papers.length) {
    container.innerHTML = `<div class="empty-state">这个 Knowledge Base 还没有论文。请从 Brief 中勾选论文并加入这里。</div>`;
    return;
  }
  papers.forEach((paper) => {
    const card = document.createElement("div");
    card.className = "paper-card";
    card.innerHTML = `
      <div class="checkbox-row row">
        <div class="paper-title-block">
          <h4>${escapeHtml(paper.title)}</h4>
        </div>
        ${renderIndexBadge(paper)}
      </div>
      <p class="paper-meta">${escapeHtml((paper.authors || []).join(", ") || "Unknown authors")}</p>
      <p class="paper-meta">${paper.pdf_url ? `<a href="${paper.pdf_url}" target="_blank" rel="noopener noreferrer">PDF</a>` : "No PDF link"}</p>
      <details class="abstract-details">
        <summary>摘要</summary>
        <p>${escapeHtml(paper.abstract || "No abstract available.")}</p>
      </details>
    `;
    container.appendChild(card);
  });

  await renderConversation();
}

function renderEmptyCollection() {
  if (byId("collection-title")) byId("collection-title").textContent = "还没有 Knowledge Base";
  if (byId("collection-papers")) byId("collection-papers").innerHTML = `<div class="empty-state">已入库论文会显示在这里。</div>`;
  if (byId("chat-history")) byId("chat-history").innerHTML = `<div class="empty-state">选择一个 Knowledge Base 后，这里会显示问答历史。</div>`;
  if (byId("chat-meta")) byId("chat-meta").textContent = "";
}

async function onAsk() {
  if (!state.activeCollectionId) throw new Error("请先选择一个 Knowledge Base");
  const question = byId("ask-question").value.trim();
  if (!question) throw new Error("请输入问题");
  const askButton = byId("ask-button");
  const previousLabel = askButton?.textContent || "发送";
  if (askButton) {
    askButton.disabled = true;
    askButton.textContent = "检索与生成中...";
  }
  const selectedMode = byId("rag-mode")?.value || "standard";
  if (byId("chat-meta")) byId("chat-meta").textContent = "正在检索证据并生成回答...";
  const history = byId("chat-history");
  const existingEmpty = history?.querySelector(".empty-state");
  existingEmpty?.remove();
  const userItem = document.createElement("div");
  userItem.className = "chat-message user";
  userItem.textContent = question;
  const assistantItem = document.createElement("div");
  assistantItem.className = "chat-message assistant";
  let streamedAnswer = "";
  let donePayload = null;
  let agentTrace = [];
  assistantItem.innerHTML = `${renderAgentTrace(agentTrace, { pending: selectedMode === "agent" })}${renderAssistantContent("")}`;
  history?.appendChild(userItem);
  history?.appendChild(assistantItem);
  if (history) history.scrollTop = history.scrollHeight;

  try {
    await streamApi("/api/ask/stream", {
      method: "POST",
      body: JSON.stringify({
        collection_id: state.activeCollectionId,
        question,
        conversation_id: state.currentConversationId,
        mode: selectedMode,
      }),
    }, (event) => {
      if (event.type === "meta") {
        state.currentConversationId = event.conversation_id;
      } else if (event.type === "trace_delta") {
        if (event.item) agentTrace.push(event.item);
        assistantItem.innerHTML = `${renderAgentTrace(agentTrace, { enabled: selectedMode === "agent" })}${renderAssistantContent(streamedAnswer)}`;
        renderMath(assistantItem.querySelector(".message-content"));
        scrollAgentTrace(assistantItem);
        if (history) history.scrollTop = history.scrollHeight;
      } else if (event.type === "trace") {
        agentTrace = event.items || [];
        assistantItem.innerHTML = `${renderAgentTrace(agentTrace, { enabled: selectedMode === "agent" })}${renderAssistantContent(streamedAnswer)}`;
        renderMath(assistantItem.querySelector(".message-content"));
        scrollAgentTrace(assistantItem);
        if (history) history.scrollTop = history.scrollHeight;
      } else if (event.type === "token") {
        streamedAnswer += event.content || "";
        assistantItem.innerHTML = `${renderAgentTrace(agentTrace, { enabled: selectedMode === "agent" })}${renderAssistantContent(streamedAnswer)}`;
        renderMath(assistantItem.querySelector(".message-content"));
        if (history) history.scrollTop = history.scrollHeight;
      } else if (event.type === "done") {
        donePayload = event;
      } else if (event.type === "error") {
        throw new Error(event.detail || "生成回答失败");
      }
    });
  } finally {
    if (askButton) {
      askButton.disabled = false;
      askButton.textContent = previousLabel;
    }
  }
  if (donePayload) {
    state.currentConversationId = donePayload.conversation_id || state.currentConversationId;
    agentTrace = donePayload.retrieval_debug?.trace || agentTrace;
    assistantItem.innerHTML = `
      ${renderAgentTrace(agentTrace, { enabled: selectedMode === "agent" })}
      ${renderAssistantContent(donePayload.answer || streamedAnswer)}
      ${renderMessageCitations(donePayload.citations || [])}
    `;
    renderMath(assistantItem.querySelector(".message-content"));
  }
  byId("ask-question").value = "";
  if (byId("chat-meta")) byId("chat-meta").textContent = "";
}

function renderAgentTrace(trace, options = {}) {
  if (!options.enabled && !options.pending) return "";
  const items = Array.isArray(trace) ? trace : [];
  if (!items.length && !options.pending) return "";
  const rows = items.length
    ? items.map((item, index) => {
        const queries = Array.isArray(item.queries) && item.queries.length
          ? `<div class="agent-query-row">${item.queries.map((query) => `
              <span class="agent-query-chip" title="${escapeHtml(query.query || "")}">
                <span>${escapeHtml(query.query || "")}</span>
                ${query.hit_count !== undefined ? `<em>${escapeHtml(String(query.hit_count))} 条</em>` : ""}
              </span>
            `).join("")}</div>`
          : "";
        return `
          <li class="agent-trace-step">
            <span class="agent-step-index">${index + 1}</span>
            <div>
              <p>${escapeHtml(item.message || item.stage || "执行一步检索")}</p>
              ${item.observation ? `<p class="agent-note"><strong>观察</strong>${escapeHtml(item.observation)}</p>` : ""}
              ${item.rationale ? `<p class="agent-note"><strong>判断</strong>${escapeHtml(item.rationale)}</p>` : ""}
              ${item.action ? `<span class="agent-action">${escapeHtml(item.action)}</span>` : ""}
              ${queries}
            </div>
          </li>
        `;
      }).join("")
    : `<li><span class="agent-step-index">…</span><div><p>正在启动检索流程...</p></div></li>`;
  return `
    <details class="agent-trace" open>
      <summary>Agent 检索过程</summary>
      <div class="agent-trace-body">
        <ol>${rows}</ol>
      </div>
    </details>
  `;
}

function scrollAgentTrace(container) {
  const traceBody = container?.querySelector(".agent-trace-body");
  if (traceBody) traceBody.scrollTop = traceBody.scrollHeight;
}

function formatScore(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return Number(value).toFixed(4);
}

function renderIndexBadge(paper) {
  const status = paper.index_status || "queued";
  if (status === "completed") return `<span class="badge status-badge indexed">索引</span>`;
  if (status === "failed") return `<span class="badge status-badge failed">失败</span>`;
  return `<span class="badge status-badge pending">入库</span>`;
}

async function renderConversation() {
  const history = byId("chat-history");
  history.innerHTML = "";
  if (!state.currentConversationId) {
    history.innerHTML = `<div class="empty-state">这里会显示围绕当前 Knowledge Base 的历史问答。</div>`;
    return;
  }
  const payload = await api(`/api/conversations/${state.currentConversationId}`);
  const messages = payload.messages || [];
  if (!messages.length) {
    history.innerHTML = `<div class="empty-state">这里会显示围绕当前 Knowledge Base 的历史问答。</div>`;
    return;
  }
  messages.forEach((message) => {
    const item = document.createElement("div");
    item.className = `chat-message ${message.role === "user" ? "user" : "assistant"}`;
    if (message.role === "assistant") {
      item.innerHTML = `
        ${renderAssistantContent(message.content)}
        ${renderMessageCitations(message.citations || [])}
      `;
      renderMath(item.querySelector(".message-content"));
    } else {
      item.textContent = message.content;
    }
    history.appendChild(item);
  });
  history.scrollTop = history.scrollHeight;
}

function renderMessageCitations(citations) {
  if (!citations.length) return "";
  const cards = citations.map((citation, index) => `
    <article class="message-citation">
      <strong>[${index + 1}] ${escapeHtml(citation.paper_title || citation.source_id || "Untitled")}</strong>
      <p class="paper-meta">
        ${escapeHtml(citation.source || "-")} ${escapeHtml(citation.source_id || "")}
        · pages ${escapeHtml(String(citation.page_start || "?"))}-${escapeHtml(String(citation.page_end || "?"))}
        · score ${escapeHtml(formatScore(citation.score))}
      </p>
      ${citation.section_title ? `<p class="paper-meta">${escapeHtml(citation.section_title)}</p>` : ""}
      ${citation.image_url ? `<img class="citation-image" src="${escapeHtml(citation.image_url)}" alt="${escapeHtml(citation.paper_title || "Retrieved figure")}">` : ""}
      <p>${escapeHtml(citation.quote_text || "")}</p>
    </article>
  `).join("");
  return `
    <details class="message-citations">
      <summary>查看检索来源（${citations.length}）</summary>
      <div class="message-citation-list">${cards}</div>
    </details>
  `;
}

function bindModalDismiss(dialogId, closeFn) {
  byId(dialogId)?.addEventListener("click", (event) => {
    if (event.target.id === dialogId) {
      closeFn();
    }
  });
}

function bindEvents() {
  byId("landing-search-button").addEventListener("click", () => runAction(onCreateBriefSearchFromLanding));
  byId("brief-rerun").addEventListener("click", () => runAction(onRerunBrief));
  byId("collection-form").addEventListener("submit", (event) => runAction(() => onCreateCollection(event)));
  byId("rename-form").addEventListener("submit", (event) => runAction(() => onRenameTarget(event)));
  byId("add-paper-form").addEventListener("submit", (event) => {
    event.preventDefault();
    runAction(onAddSelectedToCollection);
  });
  byId("refresh-collection").addEventListener("click", () => runAction(() => loadCollection(state.activeCollectionId)));
  byId("ingest-collection").addEventListener("click", () => runAction(onIngestCollection));
  byId("ask-button").addEventListener("click", () => runAction(onAsk));
  byId("sidebar-new-search").addEventListener("click", focusSearchLanding);
  byId("sidebar-new-kb").addEventListener("click", openCollectionModal);
  byId("close-collection-modal").addEventListener("click", closeCollectionModal);
  byId("close-rename-modal").addEventListener("click", closeRenameModal);
  byId("close-add-paper-modal").addEventListener("click", closeAddPaperModal);
  bindModalDismiss("collection-modal", closeCollectionModal);
  bindModalDismiss("rename-modal", closeRenameModal);
  bindModalDismiss("add-paper-modal", closeAddPaperModal);
}

async function runAction(action) {
  try {
    await action();
  } catch (error) {
    alert(error.message);
  }
}

bindEvents();
runAction(loadBootstrap);
