// Must match the gateway's MAX_HISTORY_TURNS setting (chatbot/config.py);
// the gateway 422s if more turns are sent.
const MAX_HISTORY_TURNS = 12;
// Cap stored conversations to keep localStorage bounded (drop oldest).
const MAX_CONVERSATIONS = 20;
const ACTIVE_RESEARCH_JOB_KEY = 'active_research_job_id';
const ACTIVE_RESEARCH_CONVERSATION_KEY = 'active_research_conversation_id';
const RESEARCH_TOKEN_KEY = 'combined-chat-research-token';

// State management
let conversations = [];
let currentConversationId = null;
let isStreaming = false;
let abortController = null;

// Elements
const btnNewChat = document.getElementById('btn-new-chat');
const historyList = document.getElementById('history-list');
const queryMode = document.getElementById('query-mode');
const messagesWindow = document.getElementById('messages-window');
const welcomeScreen = document.getElementById('welcome-screen');
const inputForm = document.getElementById('input-form');
const textInput = document.getElementById('text-input');
const btnSend = document.getElementById('btn-send');
const btnStop = document.getElementById('btn-stop');
const charCounter = document.getElementById('char-counter');
const healthDot = document.getElementById('health-dot');
const researchProgress = document.getElementById('research-progress');
const researchStatus = document.getElementById('research-status');
const researchEvidence = document.getElementById('research-evidence');
const researchEstimatedCost = document.getElementById('research-estimated-cost');
const researchActualCost = document.getElementById('research-actual-cost');
const researchCancel = document.getElementById('research-cancel');
let activeResearchJobId = null;

// Init App
document.addEventListener('DOMContentLoaded', () => {
  loadHistoryFromStorage();
  createNewConversation();
  checkHealth();
  reattachActiveResearch();

  // Char count listener + auto-grow
  textInput.addEventListener('input', () => {
    const count = textInput.value.length;
    charCounter.textContent = `${count}/4000`;
    autoResizeInput();
  });

  // Enter sends; Shift+Enter inserts a newline (default textarea behavior).
  textInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      if (typeof inputForm.requestSubmit === 'function') {
        inputForm.requestSubmit();
      } else {
        inputForm.dispatchEvent(new Event('submit', { cancelable: true }));
      }
    }
  });

  // Handle suggestion cards clicks
  document.addEventListener('click', (e) => {
    const card = e.target.closest('.suggestion-card');
    if (card) {
      const query = card.dataset.query;
      if (query && !isStreaming) {
        textInput.value = query;
        // Update character count display
        charCounter.textContent = `${query.length}/4000`;
        handleSendMessage(query);
      }
    }
  });
});

// Health indicator: fetch /api/health -> {ok, upstream}
async function checkHealth() {
  if (!healthDot) return;
  try {
    const r = await fetch('/api/health');
    const data = await r.json();
    if (data.ok) {
      healthDot.className = 'health-dot ok';
      healthDot.title = `Knowledge graph service online (${data.upstream})`;
    } else {
      healthDot.className = 'health-dot down';
      healthDot.title = `Knowledge graph service unreachable (${data.upstream})`;
    }
  } catch (e) {
    healthDot.className = 'health-dot down';
    healthDot.title = 'Health check failed';
  }
}

// Storage Management
function loadHistoryFromStorage() {
  try {
    const data = localStorage.getItem('rag_chatbot_history');
    conversations = data ? JSON.parse(data) : [];
    renderHistoryList();
  } catch (e) {
    console.error('Failed to load local history:', e);
    conversations = [];
  }
}

function saveHistoryToStorage() {
  try {
    // Cap stored conversations at MAX_CONVERSATIONS (newest are unshifted to
    // the front, so slicing keeps the most recent and drops the oldest).
    if (conversations.length > MAX_CONVERSATIONS) {
      conversations = conversations.slice(0, MAX_CONVERSATIONS);
    }
    localStorage.setItem('rag_chatbot_history', JSON.stringify(conversations));
  } catch (e) {
    console.error('Failed to save local history:', e);
  }
}

// Render History list
function renderHistoryList() {
  historyList.innerHTML = '';
  if (conversations.length === 0) {
    const empty = document.createElement('div');
    empty.className = 'history-empty';
    empty.textContent = 'No past conversations';
    historyList.appendChild(empty);
    return;
  }

  conversations.forEach(c => {
    const item = document.createElement('div');
    item.className = `history-item ${c.id === currentConversationId ? 'active' : ''}`;
    item.dataset.id = c.id;

    const title = document.createElement('span');
    title.className = 'history-item-title';
    title.textContent = c.title || 'Untitled Chat';

    const btnDelete = document.createElement('button');
    btnDelete.className = 'btn-delete-history';
    btnDelete.textContent = '\u{1F5D1}';
    btnDelete.title = 'Delete chat';

    item.appendChild(title);
    item.appendChild(btnDelete);

    // Listeners
    item.addEventListener('click', (e) => {
      if (e.target !== btnDelete) {
        selectConversation(c.id);
      }
    });

    btnDelete.addEventListener('click', (e) => {
      e.stopPropagation();
      deleteConversation(c.id);
    });

    historyList.appendChild(item);
  });
}

// Conversation Management
function createNewConversation() {
  if (isStreaming) return;

  currentConversationId = 'conv_' + Date.now();
  const newConv = {
    id: currentConversationId,
    title: '',
    messages: [],
    mode: queryMode.value
  };

  conversations.unshift(newConv);
  selectConversation(currentConversationId);
}

function selectConversation(id) {
  if (isStreaming) return;

  currentConversationId = id;
  const conv = conversations.find(c => c.id === id);

  // Update UI selection
  renderHistoryList();

  // Sync query mode dropdown
  if (conv.mode) {
    queryMode.value = conv.mode;
  }

  // Render Messages
  messagesWindow.innerHTML = '';

  if (!conv.messages || conv.messages.length === 0) {
    messagesWindow.appendChild(welcomeScreen);
    welcomeScreen.classList.remove('hidden');
  } else {
    welcomeScreen.classList.add('hidden');
    conv.messages.forEach(msg => {
      appendMessageUI(msg.role, msg.content, msg.sources, false);
    });
    highlightAndAddCopyButtons(messagesWindow);
    scrollToBottom(true);
  }
  textInput.focus();
}

function deleteConversation(id) {
  if (isStreaming && currentConversationId === id) return;

  conversations = conversations.filter(c => c.id !== id);
  saveHistoryToStorage();

  if (currentConversationId === id) {
    if (conversations.length > 0) {
      selectConversation(conversations[0].id);
    } else {
      createNewConversation();
    }
  } else {
    renderHistoryList();
  }
}

// New Chat handler
btnNewChat.addEventListener('click', () => {
  createNewConversation();
});

queryMode.addEventListener('change', () => {
  const conv = conversations.find(c => c.id === currentConversationId);
  if (conv) {
    conv.mode = queryMode.value;
    saveHistoryToStorage();
  }
});

// Helper functions for DOM creation
function createBadge(text, domain) {
  const span = document.createElement('span');
  span.className = `source-badge badge-${domain || 'unknown'}`;
  span.textContent = text;

  let tooltip = 'Unknown provenance';
  if (domain === 'memory') tooltip = 'Retrieved from local memory working database (:9621)';
  else if (domain === 'optical') tooltip = 'Retrieved from local optical communication working database (:9622)';
  else if (domain === 'storage') tooltip = 'Retrieved from local storage systems working database (:9624)';
  else if (domain === 'both') tooltip = 'Bridge entity present in both database schemas';

  span.title = tooltip;
  return span;
}

// Each `+`-joined provenance value becomes its own labelled domain chip.
// `both` remains a single legacy bridge chip for provenance files from two-way merges.
function createProvenanceBadges(domain) {
  const badges = document.createElement('div');
  badges.className = 'source-badges-flex';
  const domains = String(domain || 'unknown').split('+')
    .map(value => value.trim())
    .filter(Boolean);
  (domains.length ? domains : ['unknown']).forEach(domain => {
    badges.appendChild(createBadge(domain, domain));
  });
  return badges;
}

// First <SEP>-segment of a description, truncated to ~160 chars.
function descriptionSnippet(desc) {
  if (!desc) return '';
  const first = String(desc).split('<SEP>')[0].trim();
  if (first.length <= 160) return first;
  return first.slice(0, 160).trimEnd() + '…';
}

// Count legacy both-domain and multi-domain entities + relationships.
function countCrossDomain(sources) {
  let n = 0;
  const isCrossDomain = domain => domain === 'both' || String(domain || '').includes('+');
  (sources.entities || []).forEach(e => { if (isCrossDomain(e.domain)) n++; });
  (sources.relationships || []).forEach(r => { if (isCrossDomain(r.domain)) n++; });
  return n;
}

// Scroll chat window to bottom only if near bottom
function scrollToBottom(force = false) {
  const threshold = 150;
  const isNearBottom = messagesWindow.scrollHeight - messagesWindow.scrollTop - messagesWindow.clientHeight < threshold;
  if (force || isNearBottom) {
    messagesWindow.scrollTop = messagesWindow.scrollHeight;
  }
}

function hasSourceContent(sources) {
  if (!sources) return false;
  return Boolean(
    sources.entities?.length ||
    sources.relationships?.length ||
    sources.references?.length ||
    sources.error
  );
}

// HTML Rendering of Message
function appendMessageUI(role, text, sources = null, active = false) {
  const row = document.createElement('div');
  row.className = `message-row ${role}`;

  const avatar = document.createElement('div');
  avatar.className = `message-avatar ${role}`;
  if (role === 'assistant') {
    avatar.innerHTML = `<svg class="svg-avatar" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 8V4H8"></path><rect width="16" height="12" x="4" y="8" rx="2"></rect><path d="M2 14h2"></path><path d="M20 14h2"></path><path d="M15 13v2"></path><path d="M9 13v2"></path></svg>`;
  } else {
    avatar.innerHTML = `<svg class="svg-avatar" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path><circle cx="12" cy="7" r="4"></circle></svg>`;
  }

  const bubble = document.createElement('div');
  bubble.className = 'message-bubble';

  const textContainer = document.createElement('div');
  textContainer.className = 'message-text';

  // Parse markdown
  if (role === 'assistant') {
    textContainer.innerHTML = DOMPurify.sanitize(marked.parse(text));
  } else {
    textContainer.textContent = text;
  }

  bubble.appendChild(textContainer);

  if (role === 'assistant') {
    row.appendChild(avatar);
    row.appendChild(bubble);
  } else {
    row.appendChild(bubble);
    row.appendChild(avatar);
  }

  // If sources block is passed, append it
  if (hasSourceContent(sources)) {
    appendSourcesBlock(bubble, sources);
  }

  messagesWindow.appendChild(row);
  return { row, bubble, textContainer };
}

function appendSourcesBlock(bubble, sources) {
  const sourcesContainer = document.createElement('div');
  sourcesContainer.className = 'sources-container';

  const toggle = document.createElement('button');
  toggle.className = 'sources-toggle';
  toggle.textContent = 'View Sources ';

  // Cross-domain ("both") count in the summary line -- the product's core value.
  const crossDomain = countCrossDomain(sources);
  if (crossDomain > 0) {
    const count = document.createElement('span');
    count.className = 'cross-domain-count';
    count.textContent = `(${crossDomain} cross-domain)`;
    toggle.appendChild(count);
  }

  const content = document.createElement('div');
  content.className = 'sources-content';

  const grid = document.createElement('div');
  grid.className = 'sources-grid';

  // 0. Sources-unavailable error (surface instead of an empty toggle)
  if (sources.error) {
    const errBox = document.createElement('div');
    errBox.className = 'sources-error';
    errBox.textContent = sources.error;
    grid.appendChild(errBox);
  }

  // 1. Entities (badge + short description snippet)
  if (sources.entities && sources.entities.length > 0) {
    const sec = document.createElement('div');
    sec.className = 'source-section-title';
    sec.textContent = 'Entities';
    grid.appendChild(sec);

    sources.entities.forEach(ent => {
      const item = document.createElement('div');
      item.className = 'source-entity';
      const name = document.createElement('div');
      name.className = 'source-item-name';
      name.textContent = ent.name;
      item.appendChild(name);
      item.appendChild(createProvenanceBadges(ent.domain));
      const snippet = descriptionSnippet(ent.description);
      if (snippet) {
        const desc = document.createElement('div');
        desc.className = 'source-entity-desc';
        desc.textContent = snippet;
        item.appendChild(desc);
      }
      grid.appendChild(item);
    });
  }

  // 2. Relationships
  if (sources.relationships && sources.relationships.length > 0) {
    const sec = document.createElement('div');
    sec.className = 'source-section-title';
    sec.textContent = 'Relationships';
    grid.appendChild(sec);

    sources.relationships.forEach(rel => {
      const item = document.createElement('div');
      item.className = 'source-relation';
      const name = document.createElement('div');
      name.className = 'source-item-name';
      name.textContent = `${rel.src} ↔ ${rel.tgt}`;
      item.appendChild(name);
      item.appendChild(createProvenanceBadges(rel.domain));
      grid.appendChild(item);
    });
  }

  // 3. References (Papers)
  if (sources.references && sources.references.length > 0) {
    const sec = document.createElement('div');
    sec.className = 'source-section-title';
    sec.textContent = 'References';
    grid.appendChild(sec);

    const flex = document.createElement('div');
    flex.className = 'source-badges-flex';
    sources.references.forEach(ref => {
      const item = document.createElement('div');
      item.className = 'source-reference';
      item.appendChild(createBadge(ref.file_path, ref.domain || 'unknown'));
      item.appendChild(createProvenanceBadges(ref.domain));
      flex.appendChild(item);
    });
    grid.appendChild(flex);
  }

  content.appendChild(grid);
  sourcesContainer.appendChild(toggle);
  sourcesContainer.appendChild(content);
  bubble.appendChild(sourcesContainer);

  // Click toggle logic
  toggle.addEventListener('click', () => {
    toggle.classList.toggle('open');
    content.classList.toggle('open');
    setTimeout(scrollToBottom, 50); // slight delay to allow transition expand
  });
}

// Toggle Stop/Send button visibility for the streaming state.
function setStreamingUI(streaming) {
  isStreaming = streaming;
  btnSend.disabled = streaming;
  textInput.disabled = streaming;
  if (btnStop) btnStop.classList.toggle('hidden', !streaming);
  if (streaming) {
    btnSend.classList.add('hidden');
  } else {
    btnSend.classList.remove('hidden');
  }
}

async function researchFetch(url, options = {}, mayPrompt = true) {
  const token = localStorage.getItem(RESEARCH_TOKEN_KEY) || '';
  const headers = { ...(options.headers || {}) };
  if (token) headers['X-Research-Token'] = token;
  const response = await fetch(url, { ...options, headers });
  if (response.status !== 403) return response;
  localStorage.removeItem(RESEARCH_TOKEN_KEY);
  if (!mayPrompt) throw new Error('The research token was rejected.');
  const entered = (prompt(
    'This gateway is remote and requires a research token:'
  ) || '').trim();
  if (!entered) throw new Error('Research authorization is required.');
  localStorage.setItem(RESEARCH_TOKEN_KEY, entered);
  return researchFetch(url, options, false);
}

function formatUsd(value) {
  return Number.isFinite(Number(value))
    ? `$${Number(value).toFixed(4)}`
    : 'pending';
}

function showResearchProgress() {
  researchProgress?.classList.remove('hidden');
  welcomeScreen.classList.add('hidden');
}

function clearActiveResearch() {
  activeResearchJobId = null;
  localStorage.removeItem(ACTIVE_RESEARCH_JOB_KEY);
  localStorage.removeItem(ACTIVE_RESEARCH_CONVERSATION_KEY);
}

function renderResearchEvent(eventData) {
  const event = eventData || {};
  const name = event.name || '';
  const data = event.data && typeof event.data === 'object'
    ? event.data : event;
  if (name.startsWith('domain_scanned:')) {
    const domain = name.split(':')[1];
    const target = document.getElementById(`research-${domain}-coverage`);
    const value = target?.querySelector('strong');
    if (value) {
      value.textContent = `${Number(data.nodes).toLocaleString()} nodes · ` +
        `${Number(data.edges).toLocaleString()} edges`;
    }
    researchStatus.textContent = `Completed ${domain} graph scan`;
  } else if (name === 'evidence_collected') {
    researchEvidence.textContent =
      `Evidence: ${Number(data.papers).toLocaleString()} papers · ` +
      `${Number(data.chunks).toLocaleString()} chunks`;
  } else if (name === 'cost_estimated') {
    researchEstimatedCost.textContent =
      `Estimated cost: ${formatUsd(data.total_with_reserve_usd)}`;
    researchStatus.textContent = 'Evidence selected; model compilation starting';
  } else if (name === 'evidence_batch_started') {
    researchStatus.textContent =
      `Processing evidence batch ${data.batch} / ${data.total}`;
  } else if (name === 'evidence_reduce_started') {
    researchStatus.textContent =
      `Reducing evidence batch ${data.batch} / ${data.total}`;
  } else if (name === 'synthesis_started') {
    researchStatus.textContent = 'Sonnet is synthesizing the final answer';
  } else if (name === 'cost_actual') {
    researchActualCost.textContent =
      `Actual cost: ${formatUsd(data.actual_cost_usd)} (${data.cost_status})`;
  }
}

function researchSources(citations) {
  return {
    entities: [],
    relationships: [],
    references: (citations || []).map(citation => ({
      id: citation.chunk_id,
      file_path: citation.title || citation.file_path || citation.chunk_id,
      domain: (citation.domains || []).length
        ? citation.domains.join('+') : 'unknown'
    }))
  };
}

function finishResearch(record) {
  if (record?.kind === 'hypotheses') {
    const markdown = record?.result?.result?.markdown || '';
    if (!markdown) {
      throw new Error('Research completed without a hypothesis dossier.');
    }
    hypRenderStages(5, true);
    hypRenderDossier(markdown);
    researchActualCost.textContent =
      `Actual cost: ${formatUsd(record.actual_cost_usd)} (${record.cost_status})`;
    researchStatus.textContent = 'Complete';
    clearActiveResearch();
    setStreamingUI(false);
    hypSetRunning(false);
    hypLoadRecent();
    return;
  }
  const result = record?.result || {};
  const markdown = result.markdown || '';
  if (!markdown) throw new Error('Research completed without an answer.');
  const sources = researchSources(result.citations);
  const convId = localStorage.getItem(ACTIVE_RESEARCH_CONVERSATION_KEY);
  const conv = conversations.find(item => item.id === convId) ||
    conversations.find(item => item.id === currentConversationId);
  if (conv) {
    if (conv.messages.some(message => message.research_job_id === record.id)) {
      clearActiveResearch();
      setStreamingUI(false);
      return;
    }
    if (currentConversationId !== conv.id) selectConversation(conv.id);
    const { bubble, textContainer } = appendMessageUI(
      'assistant', '', null, false
    );
    textContainer.innerHTML = DOMPurify.sanitize(marked.parse(markdown));
    appendSourcesBlock(bubble, sources);
    highlightAndAddCopyButtons(bubble);
    conv.messages.push({
      role: 'assistant',
      content: markdown,
      sources,
      coverage: result.coverage,
      estimated_cost_usd: record.estimated_cost_usd,
      actual_cost_usd: record.actual_cost_usd,
      research_job_id: record.id
    });
    saveHistoryToStorage();
  }
  researchActualCost.textContent =
    `Actual cost: ${formatUsd(record.actual_cost_usd)} (${record.cost_status})`;
  researchStatus.textContent = 'Complete';
  clearActiveResearch();
  setStreamingUI(false);
  scrollToBottom(true);
}

async function attachResearchJob(jobId) {
  activeResearchJobId = jobId;
  localStorage.setItem(ACTIVE_RESEARCH_JOB_KEY, jobId);
  showResearchProgress();
  setStreamingUI(true);
  const response = await researchFetch(`/api/research/${jobId}/events`);
  if (!response.ok) {
    if (response.status === 404) clearActiveResearch();
    const detail = await response.json().catch(() => ({}));
    throw new Error(detail.detail || `Could not attach to job ${jobId}`);
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const blocks = buffer.split('\n\n');
    buffer = blocks.pop();
    for (const block of blocks) {
      let event = '';
      let data = null;
      for (const line of block.split('\n')) {
        if (line.startsWith('event: ')) event = line.slice(7).trim();
        if (line.startsWith('data: ')) {
          try { data = JSON.parse(line.slice(6)); } catch (_) { data = null; }
        }
      }
      if (event === 'progress') renderResearchEvent(data);
      if (event === 'snapshot' && data?.last_progress) {
        renderResearchEvent(data.last_progress);
      }
      if (
        (event === 'done') ||
        (event === 'snapshot' && data?.status === 'done')
      ) {
        finishResearch(data);
        return;
      }
      if (event === 'error') {
        clearActiveResearch();
        throw new Error(data?.error || data?.message || 'Research failed');
      }
      if (event === 'cancelled') {
        researchStatus.textContent = 'Cancelled';
        clearActiveResearch();
        setStreamingUI(false);
        return;
      }
    }
  }
  setStreamingUI(false);
  researchStatus.textContent = 'Connection lost; reload to reattach';
  throw new Error(`Research event stream ended before job ${jobId} finished.`);
}

async function reattachActiveResearch() {
  try {
    const response = await researchFetch('/api/research/active');
    if (!response.ok) return;
    const data = await response.json();
    const jobId = data.active_job_id ||
      localStorage.getItem(ACTIVE_RESEARCH_JOB_KEY);
    if (!jobId) return;
    const convId = localStorage.getItem(ACTIVE_RESEARCH_CONVERSATION_KEY);
    if (convId && conversations.some(item => item.id === convId)) {
      selectConversation(convId);
    }
    await attachResearchJob(jobId);
  } catch (error) {
    console.error('Research reattachment failed:', error);
    setStreamingUI(false);
  }
}

async function handleSendMessage(queryText) {
  if (isStreaming || !queryText.trim()) return;
  const conv = conversations.find(c => c.id === currentConversationId);
  if (!conv) return;
  const payloadHistory = conv.messages
    .slice(-MAX_HISTORY_TURNS)
    .map(message => ({ role: message.role, content: message.content }));
  setStreamingUI(true);
  try {
    const response = await researchFetch('/api/research/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: queryText, history: payloadHistory })
    });
    const data = await response.json().catch(() => ({}));
    if (
      response.status === 503 &&
      data.detail === 'exhaustive research is disabled'
    ) {
      setStreamingUI(false);
      return handleLegacySendMessage(queryText);
    }
    if (response.status === 409 && data.active_job_id) {
      await attachResearchJob(data.active_job_id);
      return;
    }
    if (!response.ok) {
      throw new Error(data.detail || `Server error: ${response.status}`);
    }
    if (conv.messages.length === 0) {
      conv.title = queryText.length > 25
        ? queryText.slice(0, 22) + '...' : queryText;
    }
    conv.messages.push({ role: 'user', content: queryText });
    appendMessageUI('user', queryText, null, false);
    saveHistoryToStorage();
    renderHistoryList();
    textInput.value = '';
    charCounter.textContent = '0/4000';
    autoResizeInput();
    localStorage.setItem(
      ACTIVE_RESEARCH_CONVERSATION_KEY, conv.id
    );
    await attachResearchJob(data.job_id);
  } catch (error) {
    console.error('Exhaustive research failed:', error);
    researchStatus.textContent = `Failed: ${error.message}`;
    showResearchProgress();
    setStreamingUI(false);
  }
}

// Legacy bounded stream remains available while exhaustive mode is disabled.
async function handleLegacySendMessage(queryText) {
  if (isStreaming || !queryText.trim()) return;

  setStreamingUI(true);
  welcomeScreen.classList.add('hidden');

  const conv = conversations.find(c => c.id === currentConversationId);
  if (!conv) {
    setStreamingUI(false);
    return;
  }

  // Update title if first message
  if (conv.messages.length === 0) {
    conv.title = queryText.length > 25 ? queryText.slice(0, 22) + '...' : queryText;
    renderHistoryList();
  }

  // Append user message
  const userMsg = { role: 'user', content: queryText };
  conv.messages.push(userMsg);
  appendMessageUI('user', queryText, null, false);
  scrollToBottom(true);

  // Clear input
  textInput.value = '';
  charCounter.textContent = '0/4000';
  autoResizeInput();

  // Append streaming bot shell
  const { bubble, textContainer } = appendMessageUI('assistant', '', null, true);
  scrollToBottom(true);

  let botContent = '';
  let botSources = null;

  // Prepare history payload: exclude the just-typed message, then cap to the
  // last MAX_HISTORY_TURNS turns (gateway 422s beyond that).
  const payloadHistory = conv.messages
    .slice(0, -1)
    .slice(-MAX_HISTORY_TURNS)
    .map(m => ({ role: m.role, content: m.content }));

  abortController = new AbortController();

  try {
    const response = await fetch('/api/chat', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        query: queryText,
        mode: queryMode.value,
        history: payloadHistory
      }),
      signal: abortController.signal
    });

    if (!response.ok) {
      if (response.status === 429) {
        throw new Error('Rate limit reached — wait a minute and try again');
      }
      const errData = await response.json().catch(() => ({}));
      throw new Error(errData.detail || `Server error: ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split('\n\n');
      buffer = parts.pop(); // keep partial block in buffer

      for (const block of parts) {
        if (!block.trim()) continue;

        let event = '';
        let data = null;

        for (const line of block.split('\n')) {
          if (line.startsWith('event: ')) {
            event = line.substring(7).trim();
          } else if (line.startsWith('data: ')) {
            try {
              data = JSON.parse(line.substring(6).trim());
            } catch (e) {
              console.error('Failed to parse SSE data block:', e);
            }
          }
        }

        if (event === 'token' && data) {
          botContent += data.text;
          textContainer.innerHTML = DOMPurify.sanitize(marked.parse(botContent));
          scrollToBottom();
        } else if (event === 'sources' && data) {
          botSources = data;
          if (hasSourceContent(botSources)) {
            appendSourcesBlock(bubble, botSources);
          }
          scrollToBottom();
        } else if (event === 'error' && data) {
          throw new Error(data.message || 'Stream error occurred');
        }
      }
    }

    // Save assistant message to state
    conv.messages.push({
      role: 'assistant',
      content: botContent,
      sources: botSources
    });
    saveHistoryToStorage();

  } catch (err) {
    // A user-initiated Stop aborts the fetch: keep the partial text, no banner.
    if (err.name === 'AbortError') {
      conv.messages.push({
        role: 'assistant',
        content: botContent,
        sources: botSources
      });
      saveHistoryToStorage();
    } else {
      console.error('Chat error:', err);

      // Error display (built with textContent, never innerHTML interpolation)
      const banner = document.createElement('div');
      banner.className = 'error-banner';

      const msgSpan = document.createElement('span');
      msgSpan.textContent = `⚠️ Failed to get answer: ${err.message}`;

      const btnRetry = document.createElement('button');
      btnRetry.className = 'btn-retry';
      btnRetry.textContent = 'Retry';

      banner.appendChild(msgSpan);
      banner.appendChild(btnRetry);
      bubble.appendChild(banner);
      scrollToBottom();

      // Retry listener
      btnRetry.addEventListener('click', (e) => {
        e.currentTarget.disabled = true;
        banner.remove();

        // Remove the failed user entry from message list before sending
        conv.messages.pop();

        // Re-trigger send
        isStreaming = false;
        handleSendMessage(queryText);
      });
    }
  } finally {
    highlightAndAddCopyButtons(bubble);
    setStreamingUI(false);
    textInput.focus();
    abortController = null;
  }
}

// Stop button: abort the in-flight stream (treated as a normal stop).
if (btnStop) {
  btnStop.addEventListener('click', async () => {
    if (activeResearchJobId) {
      await researchFetch(`/api/research/${activeResearchJobId}`, {
        method: 'DELETE'
      }).catch(error => console.error('Research cancel failed:', error));
    } else if (abortController) {
      abortController.abort();
    }
  });
}

if (researchCancel) {
  researchCancel.addEventListener('click', () => btnStop?.click());
}

// Grow the composer to fit its content, capped by the CSS max-height.
function autoResizeInput() {
  textInput.style.height = 'auto';
  textInput.style.height = Math.min(textInput.scrollHeight, 160) + 'px';
}

// Form Listener
inputForm.addEventListener('submit', (e) => {
  e.preventDefault();
  const val = textInput.value;
  if (val.trim()) {
    handleSendMessage(val);
  }
});

// Helper for Prism Syntax Highlighting and Copy Button injection
function highlightAndAddCopyButtons(container) {
  if (typeof Prism !== 'undefined') {
    Prism.highlightAllUnder(container);
  }

  // Add copy button to each code block
  const blocks = container.querySelectorAll('pre');
  blocks.forEach(block => {
    // Check if copy button already exists to avoid duplicates
    if (block.querySelector('.btn-copy-code')) return;

    const button = document.createElement('button');
    button.className = 'btn-copy-code';
    button.textContent = 'Copy';
    button.type = 'button';

    // Position button parent
    block.style.position = 'relative';

    button.addEventListener('click', async () => {
      const code = block.querySelector('code')?.innerText || block.innerText;
      try {
        await navigator.clipboard.writeText(code);
        button.textContent = 'Copied!';
        button.classList.add('copied');
        setTimeout(() => {
          button.textContent = 'Copy';
          button.classList.remove('copied');
        }, 2000);
      } catch (err) {
        console.error('Failed to copy code:', err);
        button.textContent = 'Error';
      }
    });

    block.appendChild(button);
  });
}

// Export conversation to Markdown
function exportToMarkdown() {
  const conv = conversations.find(c => c.id === currentConversationId);
  if (!conv || !conv.messages || conv.messages.length === 0) return;

  let md = `# MIRA RAG Chat Session - ${conv.title || 'Untitled'}\n\n`;
  md += `*Mode: ${conv.mode || 'mix'}*\n\n---\n\n`;

  conv.messages.forEach(msg => {
    const roleName = msg.role === 'user' ? 'User' : 'MIRA Assistant';
    md += `### 👤 ${roleName}\n\n${msg.content}\n\n`;

    // Add sources if present
    if (msg.sources && (msg.sources.entities?.length || msg.sources.relationships?.length || msg.sources.references?.length)) {
      md += `#### 🔍 Sources:\n`;
      if (msg.sources.entities?.length) {
        md += `* **Entities:** ${msg.sources.entities.map(e => `${e.name} (${e.domain})`).join(', ')}\n`;
      }
      if (msg.sources.relationships?.length) {
        md += `* **Relations:** ${msg.sources.relationships.map(r => `${r.src} ↔ ${r.tgt} (${r.domain})`).join(', ')}\n`;
      }
      if (msg.sources.references?.length) {
        md += `* **References:** ${msg.sources.references.map(r => r.file_path).join(', ')}\n`;
      }
      md += `\n`;
    }
    md += `---\n\n`;
  });

  const blob = new Blob([md], { type: 'text/markdown;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.setAttribute('download', `${conv.title ? conv.title.replace(/[^a-z0-9]/gi, '_').toLowerCase() : 'chat'}_export.md`);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

// Export conversation to JSON
function exportToJSON() {
  const conv = conversations.find(c => c.id === currentConversationId);
  if (!conv || !conv.messages || conv.messages.length === 0) return;

  const blob = new Blob([JSON.stringify(conv, null, 2)], { type: 'application/json;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.setAttribute('download', `${conv.title ? conv.title.replace(/[^a-z0-9]/gi, '_').toLowerCase() : 'chat'}_export.json`);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

// Bind Export Click Handlers
const btnExportMd = document.getElementById('btn-export-md');
const btnExportJson = document.getElementById('btn-export-json');
if (btnExportMd) btnExportMd.addEventListener('click', exportToMarkdown);
if (btnExportJson) btnExportJson.addEventListener('click', exportToJSON);

// Active PDF/Text Ingestion Handler
const fileUpload = document.getElementById('file-upload');
if (fileUpload) {
  fileUpload.addEventListener('change', async () => {
    const file = fileUpload.files[0];
    if (!file) return;

    // Uploads write to the graph, so the gateway requires a shared token on top
    // of the tailnet gate. Ask once, cache in localStorage, re-prompt on 403.
    let uploadToken = localStorage.getItem('combined-chat-upload-token') || '';
    if (!uploadToken) {
      uploadToken = (prompt('Enter the upload token to index documents:') || '').trim();
      if (!uploadToken) { fileUpload.value = ''; return; }
      localStorage.setItem('combined-chat-upload-token', uploadToken);
    }

    welcomeScreen.classList.add('hidden');

    // Append uploading status message
    const statusMsgText = `🔄 Uploading and indexing **${file.name}** into the knowledge graph...`;
    const { bubble, textContainer } = appendMessageUI('assistant', statusMsgText, null, true);
    scrollToBottom(true);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('/api/upload', {
        method: 'POST',
        headers: { 'X-Upload-Token': uploadToken },
        body: formData
      });

      if (response.status === 403) {
        // Stale or wrong token — drop it so the next attempt re-prompts.
        localStorage.removeItem('combined-chat-upload-token');
        throw new Error('Upload not authorized — check the upload token and retry.');
      }
      if (!response.ok) {
        const err = await response.json().catch(() => ({}));
        throw new Error(err.detail || `Upload failed: ${response.status}`);
      }

      const result = await response.json();
      textContainer.innerHTML = DOMPurify.sanitize(marked.parse(
        `✅ Ingested **${file.name}** (${result.chars.toLocaleString()} chars) successfully!\n\n` +
        `The document has been processed and integrated. Try asking a question using **mix mode** to retrieve its facts.`
      ));

      // Save success message to state
      const conv = conversations.find(c => c.id === currentConversationId);
      if (conv) {
        conv.messages.push({
          role: 'assistant',
          content: `Ingested ${file.name} successfully.`,
          sources: null
        });
        saveHistoryToStorage();
      }
    } catch (err) {
      console.error('Upload error:', err);
      textContainer.innerHTML = DOMPurify.sanitize(marked.parse(
        `⚠️ **Failed to index ${file.name}**\n\nError: ${err.message}`
      ));
    } finally {
      // Reset input element
      fileUpload.value = '';
      scrollToBottom(true);
      checkHealth(); // Trigger a quick status health indicator update
    }
  });
}

/* ------------------------------------------------------------------
   Hypothesis dossiers
   An async job surface, not a chat turn: a run is minutes of subprocess
   work, so the panel streams stage progress and then renders the finished
   dossier. SSE is consumed with fetch + manual parsing, exactly like
   /api/chat -- EventSource cannot set the X-Hypothesis-Token header.
   ------------------------------------------------------------------ */

const HYP_TOKEN_KEY = 'combined-chat-hypothesis-token';
const HYP_STAGES = [
  'Loading graph',
  'Mining gap candidates',
  'Novelty check',
  'Synthesizing hypotheses',
  'Writing dossier'
];

const hypPanel = document.getElementById('hyp-panel');
const hypForm = document.getElementById('hyp-form');
const hypChips = document.getElementById('hyp-chips');
const hypTopicFilter = document.getElementById('hyp-topic-filter');
const hypTopicList = document.getElementById('hyp-topic-list');
const hypProfiles = document.getElementById('hyp-profiles');
const hypRun = document.getElementById('hyp-run');
const hypProgress = document.getElementById('hyp-progress');
const hypStages = document.getElementById('hyp-stages');
const hypChecks = document.getElementById('hyp-checks');
const hypWarnings = document.getElementById('hyp-warnings');
const hypWarningList = document.getElementById('hyp-warning-list');
const hypError = document.getElementById('hyp-error');
const hypErrorBody = document.getElementById('hyp-error-body');
const hypResult = document.getElementById('hyp-result');
const hypDossier = document.getElementById('hyp-dossier');
const hypRecent = document.getElementById('hyp-recent');

let hypRunning = false;
// Tracks the in-flight profile fetch so a fast submit waits for it instead of
// reading an empty checkbox list and blaming the user for selecting nothing.
let hypProfilesPromise = null;
// Whether the list actually loaded. An empty checkbox list has two very
// different causes -- the fetch failed, or the user unchecked everything --
// and reporting the wrong one sends people hunting for a button that isn't
// there.
let hypProfilesLoaded = false;
let hypProfilesError = '';
// Full topic list (name + paper count), fetched once and filtered client-side.
let hypAllTopics = [];
let hypSelectedTopics = [];
let hypTopicsPromise = null;
let hypTopicsLoaded = false;
let hypTopicsError = '';
// Cap rendered rows: 733 topics render fine, but there is no value in painting
// hundreds of off-screen rows on every keystroke.
const HYP_TOPIC_ROWS = 120;

// Prompt once, cache, and drop on 403 so the next attempt re-prompts. Same
// pattern as the upload token, but a SEPARATE key: generating dossiers and
// writing to the graph are different capabilities and revoking one must not
// revoke the other.
function hypClearToken() {
  localStorage.removeItem(HYP_TOKEN_KEY);
}

// A loopback gateway needs no token at all, so NEVER prompt up front -- send
// whatever is cached (usually nothing) and only ask if the server actually
// answers 403, which means this gateway is remote and does require one.
async function hypFetch(url, options = {}, mayPrompt = true) {
  const token = localStorage.getItem(HYP_TOKEN_KEY) || '';
  const headers = { ...(options.headers || {}) };
  if (token) headers['X-Hypothesis-Token'] = token;

  const response = await fetch(url, { ...options, headers });
  if (response.status !== 403) return response;

  hypClearToken();
  if (!mayPrompt) {
    throw new Error('Not authorized — the hypothesis token was rejected.');
  }
  const entered = (prompt(
    'This gateway is remote and requires a hypothesis token:') || '').trim();
  if (!entered) {
    throw new Error('Not authorized — this gateway requires a hypothesis token.');
  }
  localStorage.setItem(HYP_TOKEN_KEY, entered);
  return hypFetch(url, options, false);
}

function hypShow(section, visible) {
  if (section) section.classList.toggle('hidden', !visible);
}

// `complete` marks every stage done. Stage 5 is still "active" when its own
// marker arrives, so without this a finished run would sit showing the last
// stage as in-progress forever.
function hypRenderStages(currentStage, complete = false) {
  hypStages.replaceChildren();
  HYP_STAGES.forEach((name, index) => {
    const stageNumber = index + 1;
    const isDone = complete || (currentStage && stageNumber < currentStage);
    const item = document.createElement('li');
    item.className = 'hyp-stage';
    if (isDone) item.classList.add('done');
    if (!complete && currentStage === stageNumber) item.classList.add('active');
    const mark = document.createElement('span');
    mark.className = 'hyp-stage-mark';
    mark.textContent = isDone ? '✓' : String(stageNumber);
    const label = document.createElement('span');
    label.textContent = name;
    item.append(mark, label);
    hypStages.appendChild(item);
  });
}

function hypRenderWarnings(warnings) {
  const items = Array.isArray(warnings) ? warnings : [];
  hypWarningList.replaceChildren();
  items.forEach(warning => {
    const li = document.createElement('li');
    li.textContent = warning;   // textContent: warnings are untrusted output
    hypWarningList.appendChild(li);
  });
  hypShow(hypWarnings, items.length > 0);
}

function hypAddCheck(text) {
  const li = document.createElement('li');
  li.textContent = text;
  hypChecks.appendChild(li);
}

function hypFail(message) {
  hypErrorBody.textContent = message;
  hypShow(hypError, true);
}

function hypSetRunning(running) {
  hypRunning = running;
  hypRun.disabled = running;
  hypRun.textContent = running ? 'Running…' : 'Run hypothesis generation';
}


async function hypLoadTopics() {
  hypTopicsLoaded = false;
  hypTopicsError = '';
  try {
    const response = await hypFetch('/api/hypothesis/topics');
    if (!response.ok) throw new Error(`topics unavailable (HTTP ${response.status})`);
    const data = await response.json();
    hypAllTopics = data.topics || [];
    hypTopicsLoaded = hypAllTopics.length > 0;
    if (!hypTopicsLoaded) hypTopicsError = 'the combined graph reported no topics';
    hypTopicFilter.placeholder = `Type to filter ${hypAllTopics.length} topics…`;
  } catch (err) {
    hypAllTopics = [];
    hypTopicsError = err.message;
  }
  hypRenderTopicList();
}

function hypRenderTopicList() {
  const needle = hypTopicFilter.value.trim().toLowerCase();
  hypTopicList.replaceChildren();

  if (!hypTopicsLoaded) {
    const hint = document.createElement('span');
    hint.className = 'hyp-hint';
    hint.textContent = hypTopicsError
      ? `Could not load topics — ${hypTopicsError}`
      : 'Loading topics…';
    hypTopicList.appendChild(hint);
    return;
  }

  const matches = hypAllTopics.filter(t => t.name.toLowerCase().includes(needle));
  if (!matches.length) {
    const hint = document.createElement('span');
    hint.className = 'hyp-hint';
    hint.textContent = `No topic matches "${hypTopicFilter.value.trim()}".`;
    hypTopicList.appendChild(hint);
    return;
  }

  matches.slice(0, HYP_TOPIC_ROWS).forEach(topic => {
    const row = document.createElement('button');
    row.type = 'button';
    row.className = 'hyp-topic-row';
    if (hypSelectedTopics.includes(topic.name)) row.classList.add('selected');
    const name = document.createElement('span');
    name.className = 'hyp-topic-name';
    name.textContent = topic.name;
    const count = document.createElement('span');
    count.className = 'hyp-topic-count';
    // Graph-wide, not scoped to the ticked profiles -- labelled so the number
    // is not mistaken for the count this run will actually use.
    count.textContent = `${topic.papers} papers in graph`;
    row.append(name, count);
    row.addEventListener('click', () => hypToggleTopic(topic.name));
    hypTopicList.appendChild(row);
  });

  if (matches.length > HYP_TOPIC_ROWS) {
    const more = document.createElement('span');
    more.className = 'hyp-hint';
    more.textContent = `${matches.length - HYP_TOPIC_ROWS} more — keep typing to narrow.`;
    hypTopicList.appendChild(more);
  }
}

function hypToggleTopic(name) {
  const at = hypSelectedTopics.indexOf(name);
  if (at >= 0) hypSelectedTopics.splice(at, 1);
  else hypSelectedTopics.push(name);
  hypRenderChips();
  hypRenderTopicList();
}

function hypRenderChips() {
  hypChips.replaceChildren();
  if (!hypSelectedTopics.length) {
    const hint = document.createElement('span');
    hint.className = 'hyp-hint';
    hint.textContent = 'No topics selected yet.';
    hypChips.appendChild(hint);
    return;
  }
  hypSelectedTopics.forEach(name => {
    const chip = document.createElement('button');
    chip.type = 'button';
    chip.className = 'hyp-chip';
    chip.title = 'Remove';
    chip.textContent = name;
    const x = document.createElement('span');
    x.className = 'hyp-chip-x';
    x.textContent = '×';
    chip.appendChild(x);
    chip.addEventListener('click', () => hypToggleTopic(name));
    hypChips.appendChild(chip);
  });
}

async function hypLoadProfiles() {
  hypProfilesLoaded = false;
  hypProfilesError = '';
  try {
    const response = await hypFetch('/api/hypothesis/profiles');
    if (!response.ok) throw new Error(`profiles unavailable (HTTP ${response.status})`);
    const data = await response.json();
    const profiles = data.profiles || [];
    hypProfiles.replaceChildren();
    if (!profiles.length) {
      hypProfilesError = 'the combined graph reported no profiles';
      const hint = document.createElement('span');
      hint.className = 'hyp-hint';
      hint.textContent = 'No profiles found in the combined graph.';
      hypProfiles.appendChild(hint);
      return;
    }
    profiles.forEach(profile => {
      const label = document.createElement('label');
      label.className = 'hyp-check';
      const box = document.createElement('input');
      box.type = 'checkbox';
      box.value = profile;
      box.checked = true;          // all-on: cross-domain is the default intent
      box.dataset.hypProfile = '1';
      const span = document.createElement('span');
      span.textContent = profile;
      label.append(box, span);
      hypProfiles.appendChild(label);
    });
    hypProfilesLoaded = true;
  } catch (err) {
    hypProfilesError = err.message;
    hypProfiles.replaceChildren();
    const hint = document.createElement('span');
    hint.className = 'hyp-hint';
    hint.textContent = `Could not load profiles — ${err.message}`;
    hypProfiles.appendChild(hint);
    // Give the user a way out without reloading the page: this clears the
    // cached token so the next attempt re-prompts.
    const retry = document.createElement('button');
    retry.type = 'button';
    retry.className = 'hyp-retry';
    retry.textContent = 'Re-enter token and retry';
    retry.addEventListener('click', () => {
      hypClearToken();
      hypProfilesPromise = hypLoadProfiles();
    });
    hypProfiles.appendChild(retry);
  }
}

async function hypLoadRecent() {
  try {
    const response = await hypFetch('/api/hypothesis/dossiers');
    if (!response.ok) throw new Error(`listing unavailable (${response.status})`);
    const entries = await response.json();
    hypRecent.replaceChildren();
    if (!entries.length) {
      const hint = document.createElement('span');
      hint.className = 'hyp-hint';
      hint.textContent = 'No dossiers yet.';
      hypRecent.appendChild(hint);
      return;
    }
    entries
      .slice()
      .sort((a, b) => (b.mtime || 0) - (a.mtime || 0))
      .slice(0, 20)
      .forEach(entry => {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'hyp-recent-item';
        const name = document.createElement('span');
        name.className = 'hyp-recent-name';
        name.textContent = entry.filename;
        const meta = document.createElement('span');
        meta.className = 'hyp-recent-meta';
        const when = entry.mtime ? new Date(entry.mtime * 1000).toLocaleString() : '';
        meta.textContent = `${entry.profile}${when ? ' · ' + when : ''}`;
        button.append(name, meta);
        button.addEventListener('click', () => hypOpenDossier(entry));
        hypRecent.appendChild(button);
      });
  } catch (err) {
    hypRecent.replaceChildren();
    const hint = document.createElement('span');
    hint.className = 'hyp-hint';
    hint.textContent = err.message;
    hypRecent.appendChild(hint);
  }
}

async function hypOpenDossier(entry) {
  try {
    const url = `/api/hypothesis/dossiers/${encodeURIComponent(entry.profile)}/${encodeURIComponent(entry.filename)}`;
    const response = await hypFetch(url);
    if (!response.ok) throw new Error(`dossier unavailable (${response.status})`);
    hypRenderDossier(await response.text());
  } catch (err) {
    hypFail(err.message);
  }
}

function hypRenderDossier(markdown) {
  // Same sanitize pipeline the chat transcript uses.
  hypDossier.innerHTML = DOMPurify.sanitize(marked.parse(markdown || ''));
  highlightAndAddCopyButtons(hypDossier);
  hypShow(hypResult, true);
  hypResult.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

async function hypStream(jobId) {
  const response = await hypFetch(`/api/hypothesis/${jobId}/events`);
  if (!response.ok) throw new Error(`event stream unavailable (${response.status})`);

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split('\n\n');
    buffer = parts.pop();

    for (const block of parts) {
      if (!block.trim()) continue;
      let event = '';
      let data = null;
      for (const line of block.split('\n')) {
        if (line.startsWith('event: ')) {
          event = line.substring(7).trim();
        } else if (line.startsWith('data: ')) {
          try {
            data = JSON.parse(line.substring(6).trim());
          } catch (e) {
            console.error('Failed to parse hypothesis SSE block:', e);
          }
        }
      }
      if (!data) continue;

      if (event === 'snapshot') {
        hypRenderStages(data.stage);
        hypRenderWarnings(data.warnings);
      } else if (event === 'progress') {
        hypRenderStages(data.stage);
        // Per-hypothesis ticks during the long synthesis stage.
        if (data.hypothesis) hypAddCheck(data.hypothesis);
      } else if (event === 'warnings') {
        hypRenderWarnings(data.warnings);
      } else if (event === 'error') {
        return { ok: false, message: data.message || 'Run failed.' };
      } else if (event === 'done') {
        return { ok: true };
      }
    }
  }
  // The stream ended without a terminal event (gateway restart, dropped
  // connection). The run may still have finished, so fall back to the record.
  return { ok: null };
}

async function hypResearchStream(jobId) {
  const response = await researchFetch(`/api/research/${jobId}/events`);
  if (!response.ok) {
    throw new Error(`research event stream unavailable (${response.status})`);
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  while (true) {
    const { value, done } = await reader.read();
    if (done) return { ok: null };
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split('\n\n');
    buffer = parts.pop();
    for (const block of parts) {
      let event = '';
      let data = null;
      for (const line of block.split('\n')) {
        if (line.startsWith('event: ')) event = line.slice(7).trim();
        if (line.startsWith('data: ')) {
          try { data = JSON.parse(line.slice(6)); } catch (_) { data = null; }
        }
      }
      if (event === 'progress') {
        renderResearchEvent(data);
        const name = data?.name || '';
        if (name === 'hypothesis_candidates_started') hypRenderStages(2);
        if (name === 'hypothesis_synthesis_started') hypRenderStages(4);
        if (name === 'hypothesis_critic_started') hypRenderStages(4);
      } else if (event === 'error') {
        return {
          ok: false,
          message: data?.error || data?.message || 'Research failed.'
        };
      } else if (event === 'done') {
        return { ok: true, record: data };
      } else if (event === 'cancelled') {
        return { ok: false, message: 'Research cancelled.' };
      }
    }
  }
}

async function hypSubmit(event) {
  event.preventDefault();
  if (hypRunning) return;

  if (hypTopicsPromise) {
    try { await hypTopicsPromise; } catch (e) { /* handled in the loader */ }
  }
  if (!hypSelectedTopics.length) {
    hypFail(hypTopicsLoaded
      ? 'Pick at least one topic from the list above.'
      : `The topic list could not be loaded — ${hypTopicsError || 'unknown error'}.`);
    return;
  }

  // Wait for the profile list rather than racing it; otherwise a fast submit
  // reads zero checkboxes and reports a selection error that isn't the user's.
  if (hypProfilesPromise) {
    try { await hypProfilesPromise; } catch (e) { /* handled in the loader */ }
  }
  // If the list never loaded, retry once here -- that re-prompts for the token
  // when it is missing or was dismissed.
  if (!hypProfilesLoaded) {
    hypProfilesPromise = hypLoadProfiles();
    try { await hypProfilesPromise; } catch (e) { /* handled in the loader */ }
  }
  if (!hypProfilesLoaded) {
    hypFail(
      `The profile list could not be loaded — ${hypProfilesError || 'unknown error'}.\n\n`
      + 'There are no profiles to select yet, so this is not a selection problem. '
      + 'A 403 means the hypothesis token is missing or wrong: use '
      + '"Re-enter token and retry" above, or reload the page to be prompted again.'
    );
    return;
  }

  const profiles = Array.from(
    hypProfiles.querySelectorAll('input[data-hyp-profile]:checked')
  ).map(box => box.value);
  if (!profiles.length) {
    hypFail('Select at least one profile. All three are checked by default — '
      + 're-check one to continue.');
    return;
  }

  hypShow(hypError, false);
  hypShow(hypResult, false);
  hypShow(hypWarnings, false);
  hypChecks.replaceChildren();
  hypRenderStages(0);
  hypShow(hypProgress, true);
  hypSetRunning(true);

  try {
    let sharedResearch = true;
    let response = await researchFetch('/api/research/hypotheses', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        topics: hypSelectedTopics.slice(),
        profiles,
        max_hypotheses: Number(document.getElementById('hyp-max-hypotheses').value) || 5,
        max_candidates: Number(document.getElementById('hyp-max-candidates').value) || 12,
        critic: document.getElementById('hyp-critic').checked,
        no_external: document.getElementById('hyp-no-external').checked
      })
    });
    if (response.status === 503) {
      const disabled = await response.clone().json().catch(() => ({}));
      if (disabled.detail === 'exhaustive research is disabled') {
        sharedResearch = false;
        response = await hypFetch('/api/hypothesis', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            topics: hypSelectedTopics.slice(),
            profiles,
            max_hypotheses: Number(
              document.getElementById('hyp-max-hypotheses').value
            ) || 5,
            max_candidates: Number(
              document.getElementById('hyp-max-candidates').value
            ) || 12,
            critic: document.getElementById('hyp-critic').checked,
            no_external: document.getElementById('hyp-no-external').checked
          })
        });
      }
    }

    let jobId = null;
    if (response.status === 409 && sharedResearch) {
      const body = await response.json().catch(() => ({}));
      if (!body.active_job_id) {
        throw new Error('A research run is already in progress.');
      }
      jobId = body.active_job_id;
    } else if (response.status === 409) {
      const body = await response.json().catch(() => ({}));
      throw new Error(
        `A hypothesis run is already in progress${body.active_job_id ? ` (job ${body.active_job_id})` : ''}. ` +
        'Only one runs at a time.'
      );
    }
    if (response.status === 503) {
      throw new Error('The gateway is shutting down — try again once it is back.');
    }
    if (!response.ok && !jobId) {
      const body = await response.json().catch(() => ({}));
      // Keep the server's hints, not just its headline. A rejected profile
      // reply also carries the profiles that DO exist, and that list is the
      // part the user can act on.
      const lines = [typeof body.detail === 'string'
        ? body.detail
        : JSON.stringify(body.detail || `Request rejected (${response.status})`, null, 2)];
      if (Array.isArray(body.invalid_profiles) && body.invalid_profiles.length) {
        lines.push(`Rejected: ${body.invalid_profiles.join(', ')}`);
      }
      if (Array.isArray(body.available_profiles) && body.available_profiles.length) {
        lines.push(`Available: ${body.available_profiles.join(', ')}`);
      }
      throw new Error(lines.join('\n'));
    }

    if (!jobId) {
      ({ job_id: jobId } = await response.json());
    }
    if (sharedResearch) {
      activeResearchJobId = jobId;
      localStorage.setItem(ACTIVE_RESEARCH_JOB_KEY, jobId);
      showResearchProgress();
      setStreamingUI(true);
      await attachResearchJob(jobId);
      return;
    }
    const outcome = await hypStream(jobId);

    const record = await hypFetch(`/api/hypothesis/${jobId}`)
      .then(r => (r.ok ? r.json() : null))
      .catch(() => null);

    if (record) hypRenderWarnings(record.warnings);

    if (outcome.ok === false) {
      // The CLI's stderr tail carries nearest-topic and available-profile
      // hints; show it whole rather than the summary line alone.
      hypFail((record && record.error) || outcome.message);
    } else if (record && record.status === 'error') {
      hypFail(record.error || 'Run failed.');
    } else if (record && record.dossier_markdown) {
      hypRenderStages(5, true);
      hypRenderDossier(record.dossier_markdown);
    } else if (outcome.ok === null) {
      hypFail('The event stream ended before the run reported a result. '
        + 'Check "Recent dossiers" — the run may have completed anyway.');
    } else {
      hypFail('The run finished but its dossier could not be read.');
    }
  } catch (err) {
    hypFail(err.message);
  } finally {
    hypSetRunning(false);
    hypLoadRecent();
  }
}

if (hypPanel) {
  const btnHypotheses = document.getElementById('btn-hypotheses');
  const hypClose = document.getElementById('hyp-close');

  if (btnHypotheses) {
    btnHypotheses.addEventListener('click', () => {
      const opening = hypPanel.classList.contains('hidden');
      hypPanel.classList.toggle('hidden', !opening);
      if (opening) {
        hypRenderStages(0);
        hypProfilesPromise = hypLoadProfiles();
        hypTopicsPromise = hypLoadTopics();
        hypRenderChips();
        hypLoadRecent();
        hypTopicFilter.focus();
      }
    });
  }
  if (hypClose) {
    // Closing detaches the view only; the run continues server-side.
    hypClose.addEventListener('click', () => hypPanel.classList.add('hidden'));
  }
  if (hypTopicFilter) {
    hypTopicFilter.addEventListener('input', hypRenderTopicList);
    // Enter filters rather than submitting a half-configured run.
    hypTopicFilter.addEventListener('keydown', e => {
      if (e.key === 'Enter') e.preventDefault();
    });
  }
  if (hypForm) hypForm.addEventListener('submit', hypSubmit);
}
