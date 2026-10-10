'use strict';

/* Nova — chat, browser voice input, and phone speech output. */

const $ = (id) => document.getElementById(id);
const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
const DEFAULT_MODELS = {
  anthropic: 'claude-opus-5',
  google: 'gemini-3.5-flash-lite',
  openai: 'gpt-4o',
  openrouter: 'openrouter/auto',
  ollama: 'phi3:mini',
  lmstudio: '',
  custom: '',
};
const KEY_PROVIDERS = new Set(['anthropic', 'google', 'openai', 'openrouter', 'custom']);
const URL_PROVIDERS = new Set(['ollama', 'lmstudio', 'custom']);

const state = {
  name: 'Nova', speed: 'normal', provider: 'google', model: 'gemini-3.5-flash-lite',
  models: {}, providers: [], keysSaved: {}, baseUrls: {}, apiReady: false,
};

let busy = false;
let speaking = false;
let listening = false;
let wakeWanted = false;
let wakeRunning = false;
let wakeRec = null;
let listenRec = null;
let speakToken = 0;
let typingEl = null;
let clearKeyRequested = false;

/* ---------- status line, dot, and avatar animation ---------- */

const face = {
  bob: $('faceBob'), eyeL: $('eyeL'), eyeR: $('eyeR'),
  blushL: $('blushL'), blushR: $('blushR'),
};
let faceEmotion = 'idle';
let faceT = 0;
let blinkIn = 3.0;
let blinkLeft = 0;
let lastTs = 0;

function inferEmotion(text) {
  const s = text.toLowerCase();
  if (s.includes('listening')) return 'listening';
  if (s.includes('thinking')) return 'thinking';
  if (s.includes('speaking')) return 'talking';
  if (['mic failed', 'could not', 'couldn\'t', 'sorry', 'failed'].some((part) => s.includes(part))) return 'sad';
  if (['saved', 'cleared'].some((part) => s.includes(part))) return 'happy';
  return 'idle';
}

function readyStatus() {
  return `Ready — say "Hey ${state.name}" or tap Mic`;
}

function setStatus(text) {
  $('status').textContent = text;
  faceEmotion = inferEmotion(text);
  const dot = $('statusDot');
  dot.className = 'dot';
  if (faceEmotion === 'listening') dot.classList.add('listening');
  else if (faceEmotion === 'talking') dot.classList.add('speaking');
  else if (faceEmotion === 'thinking') dot.classList.add('thinking');
}

function faceFrame(ts) {
  const dt = lastTs ? Math.min((ts - lastTs) / 1000, 0.1) : 0.08;
  lastTs = ts;
  faceT += dt;
  blinkIn -= dt;
  if (blinkLeft > 0) blinkLeft -= dt;
  else if (blinkIn <= 0) {
    blinkLeft = 0.12;
    blinkIn = 2.6 + Math.random() * 3.0;
  }

  const emotion = faceEmotion;
  const breath = emotion === 'talking' ? 0 : Math.sin(faceT * 0.8);
  const dy = breath * 0.6;
  const scale = breath * 0.004;
  face.bob.setAttribute('transform',
    `translate(50 50) scale(${(1 + scale).toFixed(4)} ${(1 - scale).toFixed(4)}) translate(-50 -50) translate(0 ${dy.toFixed(2)})`);

  const blinking = blinkLeft > 0;
  let width = 8.6, height = 8.6, shear = 0, dx = 0, eyeY = 52;
  if (emotion === 'listening') { width *= 1.18; height *= 1.24; eyeY -= 1.8; }
  else if (emotion === 'thinking') { dx = Math.sin(faceT * 0.9) * 1.8 - 1.2; eyeY -= 1.6; }
  else if (emotion === 'sad') { width *= 0.86; height *= 0.9; eyeY += 1.8; shear = -1.4; }
  else if (emotion === 'happy') { width *= 1.32; height *= 0.78; eyeY -= 0.8; shear = 2.0; }
  else if (emotion === 'talking') height *= 0.85 + 0.4 * Math.abs(Math.sin(faceT * 9));
  else dx = Math.sin(faceT * 0.5) * 0.6;
  if (blinking) height = width * 0.12;

  for (const [eye, side] of [[face.eyeL, -1], [face.eyeR, 1]]) {
    const x = 50 + side * 15.5 + dx;
    const tip = -side * shear;
    eye.setAttribute('x1', (x - tip).toFixed(2));
    eye.setAttribute('y1', (eyeY - height).toFixed(2));
    eye.setAttribute('x2', (x + tip).toFixed(2));
    eye.setAttribute('y2', (eyeY + height).toFixed(2));
    eye.setAttribute('stroke-width', width.toFixed(2));
  }
  const blushBoost = emotion === 'happy' ? 1.15 : 1;
  for (const blush of [face.blushL, face.blushR]) {
    blush.setAttribute('rx', (5.2 * blushBoost).toFixed(2));
    blush.setAttribute('ry', (3.6 * blushBoost).toFixed(2));
  }
  requestAnimationFrame(faceFrame);
}
requestAnimationFrame(faceFrame);

/* ---------- safe chat rendering ---------- */

function appendInline(parent, text) {
  const token = /(\*\*[^*]+\*\*|`[^`]+`|\*[^*]+\*)/g;
  let offset = 0;
  for (const match of text.matchAll(token)) {
    if (match.index > offset) parent.append(document.createTextNode(text.slice(offset, match.index)));
    const value = match[0];
    const element = value.startsWith('**') ? document.createElement('strong')
      : value.startsWith('`') ? document.createElement('code')
        : document.createElement('em');
    element.textContent = value.startsWith('**') ? value.slice(2, -2)
      : value.startsWith('`') ? value.slice(1, -1) : value.slice(1, -1);
    parent.append(element);
    offset = match.index + value.length;
  }
  if (offset < text.length) parent.append(document.createTextNode(text.slice(offset)));
}

function renderMarkdown(target, text) {
  const fragment = document.createDocumentFragment();
  const lines = String(text).split(/\r?\n/);
  let list = null;
  let codeLines = [];
  let inCode = false;
  const closeList = () => { list = null; };
  const appendCode = () => {
    const pre = document.createElement('pre');
    const code = document.createElement('code');
    code.textContent = codeLines.join('\n');
    pre.append(code);
    fragment.append(pre);
    codeLines = [];
  };

  for (const line of lines) {
    if (line.trim().startsWith('```')) {
      closeList();
      if (inCode) appendCode();
      inCode = !inCode;
      continue;
    }
    if (inCode) { codeLines.push(line); continue; }
    const bullet = line.match(/^\s*[-*+]\s+(.+)$/);
    const ordered = line.match(/^\s*\d+[.)]\s+(.+)$/);
    if (bullet || ordered) {
      if (!list || list.tagName !== (bullet ? 'UL' : 'OL')) {
        list = document.createElement(bullet ? 'ul' : 'ol');
        fragment.append(list);
      }
      const item = document.createElement('li');
      appendInline(item, (bullet || ordered)[1]);
      list.append(item);
      continue;
    }
    closeList();
    if (!line.trim()) continue;
    const heading = line.match(/^\s*#{1,4}\s+(.+)$/);
    const quote = line.match(/^\s*>\s?(.*)$/);
    const paragraph = document.createElement(heading ? 'h3' : quote ? 'blockquote' : 'p');
    appendInline(paragraph, (heading || quote || [null, line])[1]);
    fragment.append(paragraph);
  }
  if (inCode) appendCode();
  target.replaceChildren(fragment);
}

function addMessage(role, text) {
  const message = document.createElement('article');
  message.className = `message message-${role}`;
  const header = document.createElement('div');
  header.className = 'message-head';
  const name = document.createElement('span');
  name.className = 'message-name';
  name.textContent = role === 'user' ? 'You' : role === 'assistant' ? state.name : 'Notice';
  header.append(name);
  if (role !== 'system') {
    const copy = document.createElement('button');
    copy.className = 'copy-btn';
    copy.type = 'button';
    copy.textContent = 'Copy';
    copy.addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText(String(text));
        copy.textContent = 'Copied';
        setTimeout(() => { copy.textContent = 'Copy'; }, 1200);
      } catch (_) {
        copy.textContent = 'Unavailable';
      }
    });
    header.append(copy);
  }
  const body = document.createElement('div');
  body.className = role === 'assistant' ? 'message-body reply' : 'message-body';
  if (role === 'assistant') renderMarkdown(body, text);
  else body.textContent = String(text);
  message.append(header, body);
  $('chat').append(message);
  $('chat').scrollTop = $('chat').scrollHeight;
}

function showTyping() {
  hideTyping();
  typingEl = document.createElement('div');
  typingEl.className = 'typing';
  typingEl.innerHTML = `<span>${escapeHTML(state.name)} is thinking</span><i></i><i></i><i></i>`;
  $('chat').append(typingEl);
  $('chat').scrollTop = $('chat').scrollHeight;
}

function hideTyping() {
  if (typingEl) {
    typingEl.remove();
    typingEl = null;
  }
}

function escapeHTML(text) {
  return String(text).replace(/[&<>"']/g, (char) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  })[char]);
}

/* ---------- phone speech output ---------- */

function setMicSpeaking(on) {
  const button = $('micBtn');
  button.classList.toggle('speaking', on);
  button.textContent = on ? 'Stop' : listening ? 'Stop' : 'Mic';
  button.setAttribute('aria-label', on ? 'Stop speech' : listening ? 'Stop listening' : 'Start voice input');
  button.disabled = busy && !on;
}

function setMicListening(on) {
  const button = $('micBtn');
  button.classList.toggle('recording', on);
  button.textContent = on || speaking ? 'Stop' : 'Mic';
  button.setAttribute('aria-label', on ? 'Stop listening' : speaking ? 'Stop speech' : 'Start voice input');
  button.disabled = busy && !speaking;
}

function stopListening(restartWake = true) {
  if (listenRec) {
    try { listenRec.stop(); } catch (_) { /* recognition already ended */ }
  }
  listening = false;
  setMicListening(false);
  setStatus(readyStatus());
  if (restartWake) ensureWake();
}

function speak(text) {
  if (!window.speechSynthesis) {
    addMessage('system', 'Speech output is not supported in this browser.');
    return;
  }
  speechSynthesis.cancel();
  const token = ++speakToken;
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.rate = state.speed === 'fast' ? 1.2 : state.speed === 'slow' ? 0.85 : 1;
  const voices = speechSynthesis.getVoices();
  const preferred = voices.find((voice) => /^en/i.test(voice.lang)) || voices[0];
  if (preferred) utterance.voice = preferred;
  speaking = true;
  setMicSpeaking(true);
  setStatus('Speaking...');
  const done = () => {
    if (token !== speakToken) return;
    speaking = false;
    setMicSpeaking(false);
    setStatus(readyStatus());
    ensureWake();
  };
  utterance.onend = done;
  utterance.onerror = done;
  try {
    speechSynthesis.speak(utterance);
  } catch (error) {
    speaking = false;
    setMicSpeaking(false);
    setStatus(readyStatus());
    addMessage('system', `Speech output failed: ${error.message || 'unsupported browser'}.`);
  }
}

function stopSpeech(restartWake = true) {
  speakToken += 1;
  if (window.speechSynthesis) speechSynthesis.cancel();
  speaking = false;
  setMicSpeaking(false);
  setStatus(readyStatus());
  if (restartWake) ensureWake();
}

function beep() {
  try {
    const context = new (window.AudioContext || window.webkitAudioContext)();
    const oscillator = context.createOscillator();
    const gain = context.createGain();
    oscillator.frequency.value = 880;
    gain.gain.value = 0.18;
    oscillator.connect(gain);
    gain.connect(context.destination);
    oscillator.start();
    oscillator.stop(context.currentTime + 0.12);
  } catch (_) { /* optional audio feedback */ }
}

/* ---------- send and clear ---------- */

async function send(text) {
  text = String(text || '').trim();
  if (!text || busy) return;
  if (listening) stopListening(false);
  if (wakeRunning) stopWake();
  stopSpeech(false);
  addMessage('user', text);
  busy = true;
  setBusy(true);
  setStatus('Thinking...');
  showTyping();
  try {
    const response = await fetch('/api/message', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });
    const data = await response.json();
    hideTyping();
    if (!response.ok) throw new Error(data.error || 'The chat request failed.');
    const reply = data.reply || '(no response)';
    addMessage('assistant', reply);
    speak(reply);
  } catch (error) {
    hideTyping();
    const reply = `I couldn't reach the chat server: ${error.message}`;
    addMessage('system', reply);
    speak(reply);
  } finally {
    busy = false;
    setBusy(false);
    if (!speaking && !listening) setStatus(readyStatus());
    ensureWake();
  }
}

function setBusy(on) {
  $('sendBtn').disabled = on;
  $('entry').disabled = on;
  $('modelBtn').disabled = on;
  $('clearBtn').disabled = on;
  $('micBtn').disabled = on && !speaking;
}

async function clearChat() {
  if (busy) return;
  stopSpeech();
  try {
    const response = await fetch('/api/clear', { method: 'POST' });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Unable to start a new chat.');
    $('chat').replaceChildren();
    addMessage('system', 'New chat started.');
  } catch (error) {
    addMessage('system', `Couldn't clear the chat: ${error.message}`);
  }
  setStatus(readyStatus());
}

/* ---------- browser speech recognition ---------- */

function startListening() {
  if (!SR) {
    addMessage('system', 'Voice input is not supported in this browser — type instead.');
    return;
  }
  if (listening || busy || speaking) return;
  stopWake();
  listening = true;
  setMicListening(true);
  setStatus('Listening...');
  try {
    listenRec = new SR();
    listenRec.lang = navigator.language || 'en-US';
    listenRec.interimResults = true;
    listenRec.continuous = false;
    listenRec.onresult = (event) => {
      let transcript = '';
      for (let index = event.resultIndex; index < event.results.length; index++) {
        transcript += event.results[index][0].transcript;
      }
      if (event.results.length && event.results[event.results.length - 1].isFinal) {
        listening = false;
        setMicListening(false);
        send(transcript.trim());
      }
    };
    listenRec.onerror = (event) => {
      listening = false;
      setMicListening(false);
      const message = event.error === 'not-allowed' || event.error === 'service-not-allowed'
        ? 'Microphone permission denied — type instead.'
        : 'Mic failed — type instead.';
      addMessage('system', message);
      setStatus(readyStatus());
      ensureWake();
    };
    listenRec.onend = () => {
      if (listening) {
        listening = false;
        setMicListening(false);
        setStatus(readyStatus());
        ensureWake();
      }
    };
    listenRec.start();
  } catch (_) {
    listening = false;
    setMicListening(false);
    addMessage('system', 'Mic failed — type instead.');
    setStatus(readyStatus());
  }
}

/* ---------- optional wake word ---------- */

const WAKE_ONLY = ['', 'i have a question', "i've got a question", 'ive got a question',
  'question', 'yes', 'yeah', 'hello', 'hi', 'hey', 'are you there',
  'can you hear me', 'wake up', 'you there', "it's me", 'its me'];

function wakeQuestion(text, name) {
  if (!name) return null;
  const escaped = name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const strict = text.match(new RegExp(
    `^\\s*(?:please\\s+)?(?:(?:hey(?:\\s+there)?|ok(?:ay)?|hi|hello|yo)\\s+)?${escaped}\\b(?<tail>.*)$`, 'i'));
  if (strict) {
    const tail = (strict.groups.tail || '').trim().replace(/[ .,:!?]+$/, '');
    return WAKE_ONLY.includes(tail.toLowerCase()) ? '' : tail;
  }
  if (new RegExp(`\\b${escaped}\\b`, 'i').test(text)) {
    return text.replace(new RegExp(`\\b(?:hey\\s+|ok(?:ay)?\\s+)?${escaped}\\b`, 'i'), '').trim();
  }
  return null;
}

function startWake() {
  if (!SR || wakeRunning || listening || busy || speaking) return;
  try {
    wakeRec = new SR();
    wakeRec.lang = navigator.language || 'en-US';
    wakeRec.continuous = true;
    wakeRec.interimResults = true;
    wakeRec.onresult = (event) => {
      for (let index = event.resultIndex; index < event.results.length; index++) {
        if (!event.results[index].isFinal) continue;
        const transcript = event.results[index][0].transcript.trim();
        const question = wakeQuestion(transcript, state.name);
        if (question === null) continue;
        stopWake();
        beep();
        if (question) send(question);
        else startListening();
        return;
      }
    };
    wakeRec.onend = () => {
      wakeRunning = false;
      if (wakeWanted && !listening && !busy && !speaking) {
        setTimeout(() => {
          if (wakeWanted && !listening && !busy && !speaking) startWake();
        }, 250);
      }
    };
    wakeRec.onerror = () => { wakeRunning = false; };
    wakeRec.start();
    wakeRunning = true;
    $('wakeBtn').classList.add('active');
    setStatus(`Listening...`);
  } catch (_) {
    wakeRunning = false;
    addMessage('system', 'Wake listening could not start. Use Mic or type instead.');
  }
}

function stopWake() {
  if (wakeRec) {
    try { wakeRec.stop(); } catch (_) { /* already stopped */ }
  }
  wakeRunning = false;
}

function ensureWake() {
  if (wakeWanted && !wakeRunning && !listening && !busy && !speaking) startWake();
}

function updateWakeButton() {
  $('wakeBtn').classList.toggle('active', wakeWanted);
  try { localStorage.setItem('novaWakeWord', wakeWanted ? 'on' : 'off'); } catch (_) {}
}

/* ---------- provider/model settings ---------- */

function defaultBaseUrl(provider) {
  if (provider === 'ollama') return 'http://localhost:11434';
  if (provider === 'lmstudio') return 'http://localhost:1234/v1';
  return '';
}

function selectedProvider() {
  return $('providerSelect').value || state.provider;
}

function updateProviderFields() {
  const provider = selectedProvider();
  const spec = state.providers.find((item) => item.id === provider) || { label: provider };
  const saved = Boolean(state.keysSaved[provider]);
  $('keySection').hidden = !KEY_PROVIDERS.has(provider);
  $('keyLabel').textContent = `${String(spec.label || provider).toUpperCase()} API KEY`;
  $('setKey').value = '';
  $('setKey').placeholder = saved ? 'Saved key — leave blank to keep it' : 'Enter API key';
  $('keyHint').textContent = saved ? 'A key is already saved for this provider.'
    : provider === 'custom' ? 'Optional for local OpenAI-compatible servers.' : 'Stored in config.py on the server PC.';
  $('clearKey').hidden = !saved;
  clearKeyRequested = false;

  const hasUrl = URL_PROVIDERS.has(provider);
  $('urlSection').hidden = !hasUrl;
  $('urlLabel').textContent = provider === 'ollama' ? 'OLLAMA SERVER URL' : 'SERVER BASE URL';
  $('baseUrl').value = state.baseUrls[provider] || defaultBaseUrl(provider);
  $('baseUrl').placeholder = defaultBaseUrl(provider) || 'https://your-server.example/v1';
  $('modelField').value = state.models[provider] || DEFAULT_MODELS[provider] || '';
  $('modelHint').textContent = 'Load models from this provider or enter a model ID directly.';
  $('modelOptions').replaceChildren();
}

function openSheet() {
  $('setName').value = state.name;
  $('providerSelect').value = state.provider;
  const speedRadio = document.querySelector(`input[name="speed"][value="${state.speed}"]`);
  if (speedRadio) speedRadio.checked = true;
  $('settingsStatus').textContent = '';
  updateProviderFields();
  $('sheetBackdrop').classList.remove('hidden');
  $('providerSelect').focus();
}

function closeSheet() {
  $('sheetBackdrop').classList.add('hidden');
}

async function loadModels() {
  const provider = selectedProvider();
  $('loadModels').disabled = true;
  $('modelHint').textContent = `Loading ${provider} models…`;
  const payload = { provider };
  const key = $('setKey').value.trim();
  if (key) payload.api_key = key;
  if (URL_PROVIDERS.has(provider)) payload.base_url = $('baseUrl').value.trim();
  try {
    const response = await fetch('/api/models', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Could not load models.');
    const models = Array.isArray(data.models) ? data.models : [];
    const options = document.createDocumentFragment();
    for (const model of models) {
      const option = document.createElement('option');
      option.value = model;
      options.append(option);
    }
    $('modelOptions').replaceChildren(options);
    if (models.length && !models.includes($('modelField').value.trim())) {
      const preferred = models.find((model) => model.toLowerCase().includes('flash')) || models[0];
      $('modelField').value = preferred;
    }
    $('modelHint').textContent = models.length
      ? `Loaded ${models.length} model${models.length === 1 ? '' : 's'}.`
      : 'No chat models were returned.';
  } catch (error) {
    $('modelHint').textContent = error.message;
  } finally {
    $('loadModels').disabled = false;
  }
}

async function saveSettings() {
  const provider = selectedProvider();
  const selectedSpeed = document.querySelector('input[name="speed"]:checked');
  const payload = {
    name: $('setName').value.trim() || 'Nova',
    provider,
    model: $('modelField').value.trim(),
    speed: selectedSpeed ? selectedSpeed.value : 'normal',
    clear_key: clearKeyRequested,
  };
  const key = $('setKey').value.trim();
  if (key) payload.api_key = key;
  if (URL_PROVIDERS.has(provider)) payload.base_url = $('baseUrl').value.trim();
  $('saveBtn').disabled = true;
  $('settingsStatus').textContent = 'Saving…';
  try {
    const response = await fetch('/api/settings', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Could not save settings.');
    applyState(data);
    closeSheet();
    setStatus('Saved');
    setTimeout(() => { if (!busy && !speaking && !listening) setStatus(readyStatus()); }, 900);
  } catch (error) {
    $('settingsStatus').textContent = error.message;
  } finally {
    $('saveBtn').disabled = false;
  }
}

function applyState(data) {
  state.name = data.name || state.name;
  state.provider = data.provider || state.provider;
  if (typeof data.model === 'string') state.model = data.model;
  state.models = data.models || state.models;
  state.providers = data.providers || state.providers;
  state.keysSaved = data.keys_saved || state.keysSaved;
  state.baseUrls = data.base_urls || state.baseUrls;
  state.apiReady = Boolean(data.api_ready);
  state.speed = data.speed || state.speed;
  $('titleName').textContent = state.name;
  document.title = `${state.name} — Voice Chat`;
  const activeModel = state.model || DEFAULT_MODELS[state.provider] || 'Choose a model';
  $('modelBtn').textContent = `${data.provider_label || state.provider} · ${activeModel} ▴`;
}

function clearSavedKey() {
  clearKeyRequested = true;
  $('setKey').value = '';
  $('keyHint').textContent = 'Saved key will be removed when you save.';
}

/* ---------- wiring and startup ---------- */

async function loadState() {
  let savedHistory = [];
  try {
    const response = await fetch('/api/state');
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Could not load settings.');
    applyState(data);
    savedHistory = Array.isArray(data.history) ? data.history : [];
    const options = document.createDocumentFragment();
    for (const provider of state.providers) {
      const option = document.createElement('option');
      option.value = provider.id;
      option.textContent = provider.label;
      options.append(option);
    }
    $('providerSelect').replaceChildren(options);
  } catch (error) {
    addMessage('system', `Settings unavailable: ${error.message}`);
  }
  if (savedHistory.length) {
    for (const message of savedHistory) {
      if (message && ['user', 'assistant'].includes(message.role)) {
        addMessage(message.role, message.content || '');
      }
    }
  } else {
    addMessage('assistant', `Hi, I'm ${state.name}. Ask me anything, or tap Mic to speak.`);
  }
  setStatus(readyStatus());
  try { wakeWanted = localStorage.getItem('novaWakeWord') === 'on'; } catch (_) {}
  updateWakeButton();
  ensureWake();
}

$('sendBtn').addEventListener('click', () => {
  const input = $('entry');
  const text = input.value;
  input.value = '';
  input.style.height = 'auto';
  send(text);
});
$('entry').addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    $('sendBtn').click();
  }
});
$('entry').addEventListener('input', () => {
  const input = $('entry');
  input.style.height = 'auto';
  input.style.height = `${Math.min(input.scrollHeight, 120)}px`;
});
$('micBtn').addEventListener('click', () => {
  if (speaking) stopSpeech();
  else if (listening) stopListening();
  else if (!busy) startListening();
});
$('wakeBtn').addEventListener('click', () => {
  if (!SR) {
    addMessage('system', 'Wake listening is not supported in this browser. Use Mic or type instead.');
    return;
  }
  wakeWanted = !wakeWanted;
  updateWakeButton();
  if (wakeWanted) startWake();
  else { stopWake(); setStatus(readyStatus()); }
});
$('gearBtn').addEventListener('click', openSheet);
$('modelBtn').addEventListener('click', openSheet);
$('closeSheet').addEventListener('click', closeSheet);
$('cancelBtn').addEventListener('click', closeSheet);
$('sheetBackdrop').addEventListener('click', (event) => {
  if (event.target === $('sheetBackdrop')) closeSheet();
});
$('providerSelect').addEventListener('change', updateProviderFields);
$('loadModels').addEventListener('click', () => void loadModels());
$('clearKey').addEventListener('click', clearSavedKey);
$('saveBtn').addEventListener('click', () => void saveSettings());
$('clearBtn').addEventListener('click', () => void clearChat());

if (!SR) {
  $('micBtn').disabled = true;
  $('wakeBtn').disabled = true;
}
if ('serviceWorker' in navigator) navigator.serviceWorker.register('/sw.js').catch(() => {});
if (window.speechSynthesis) speechSynthesis.getVoices();

loadState();
