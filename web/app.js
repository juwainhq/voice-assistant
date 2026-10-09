'use strict';

/* Nova — mobile web client: chat, browser speech, wake word, phone TTS. */

const $ = (id) => document.getElementById(id);
const SR = window.SpeechRecognition || window.webkitSpeechRecognition;

const state = {
  name: 'Nova', apiReady: false, speed: 'normal', apps: {},
  user_name: '',
};

let busy = false;        // waiting for the server
let speaking = false;    // speechSynthesis is playing
let listening = false;   // question recognition is running
let wakeWanted = false;  // user toggled the wake listener on
let wakeRunning = false;
let wakeRec = null, listenRec = null;
let speakToken = 0;
let selectedApp = null;

/* ---------- status line, dot, and face ---------- */

const face = {
  bob: $('faceBob'),
  eyeL: $('eyeL'), eyeR: $('eyeR'),
  blushL: $('blushL'), blushR: $('blushR'),
};
let faceEmotion = 'idle', faceT = 0, blinkIn = 3.0, blinkLeft = 0, lastTs = 0;

function inferEmotion(text) {
  const s = text.toLowerCase();
  if (s.includes('listening')) return 'listening';
  if (s.includes('thinking') || s.includes('searching')) return 'thinking';
  if (s.includes('speaking')) return 'talking';
  if (['mic failed', "can't understand", 'sorry', 'failed', 'denied'].some(b => s.includes(b))) return 'sad';
  if (['copied', 'saved', 'cleared', 'added', 'removed', 'got it', "i'll remember"].some(g => s.includes(g))) return 'happy';
  return 'idle';
}

function readyStatus() {
  return `Ready — say "Hey ${state.name}" to talk`;
}

function setStatus(text) {
  $('status').textContent = text;
  const emo = inferEmotion(text);
  faceEmotion = emo;
  const dot = $('statusDot');
  dot.className = 'dot';
  if (emo === 'listening') dot.classList.add('listening');
  else if (emo === 'talking') dot.classList.add('speaking');
  else if (emo === 'thinking') dot.classList.add('thinking');
}

function faceFrame(ts) {
  const dt = lastTs ? Math.min((ts - lastTs) / 1000, 0.1) : 0.08;
  lastTs = ts; faceT += dt;
  blinkIn -= dt;
  if (blinkLeft > 0) blinkLeft -= dt;
  else if (blinkIn <= 0) { blinkLeft = 0.12; blinkIn = 2.6 + Math.random() * 3.0; }

  const emo = faceEmotion;
  const breath = emo === 'talking' ? 0 : Math.sin(faceT * 0.8);
  const dy = breath * 0.6, sq = breath * 0.004;
  face.bob.setAttribute('transform',
    `translate(50 50) scale(${(1 + sq).toFixed(4)} ${(1 - sq).toFixed(4)}) translate(-50 -50) translate(0 ${dy.toFixed(2)})`);

  const blinking = blinkLeft > 0;
  let w = 8.6, h = 8.6, shear = 0, dx = 0, ey = 52;
  if (emo === 'listening') { w *= 1.18; h *= 1.24; ey -= 1.8; }
  else if (emo === 'thinking') { dx = Math.sin(faceT * 0.9) * 1.8 - 1.2; ey -= 1.6; }
  else if (emo === 'sad') { w *= 0.86; h *= 0.9; ey += 1.8; shear = -1.4; }
  else if (emo === 'happy') { w *= 1.32; h *= 0.78; ey -= 0.8; shear = 2.0; }
  else if (emo === 'talking') { h *= 0.85 + 0.4 * Math.abs(Math.sin(faceT * 9)); }
  else { dx = Math.sin(faceT * 0.5) * 0.6; }
  if (blinking) h = w * 0.12;

  for (const [eye, side] of [[face.eyeL, -1], [face.eyeR, 1]]) {
    const ex = 50 + side * 15.5 + dx;
    const tip = -side * shear;
    eye.setAttribute('x1', (ex - tip).toFixed(2)); eye.setAttribute('y1', (ey - h).toFixed(2));
    eye.setAttribute('x2', (ex + tip).toFixed(2)); eye.setAttribute('y2', (ey + h).toFixed(2));
    eye.setAttribute('stroke-width', w.toFixed(2));
  }
  const boost = emo === 'happy' ? 1.15 : 1.0;
  for (const blush of [face.blushL, face.blushR]) {
    blush.setAttribute('rx', (5.2 * boost).toFixed(2));
    blush.setAttribute('ry', (3.6 * boost).toFixed(2));
  }
  requestAnimationFrame(faceFrame);
}
requestAnimationFrame(faceFrame);

/* ---------- chat window ---------- */

function addMsg(role, text) {
  const el = document.createElement('div');
  el.className = 'msg msg-' + role;
  const name = role === 'user' ? 'You' : role === 'system' ? 'Notice' : state.name;
  const meta = document.createElement('div');
  meta.className = 'meta';
  meta.textContent = name;
  if (role !== 'system') {
    const copy = document.createElement('span');
    copy.className = 'copy';
    copy.textContent = '[Copy]';
    copy.addEventListener('click', () => navigator.clipboard.writeText(text));
    meta.appendChild(copy);
  }
  const body = document.createElement('div');
  body.textContent = text;
  el.appendChild(meta);
  el.appendChild(body);
  $('chat').appendChild(el);
  $('chat').scrollTop = $('chat').scrollHeight;
}

let typingEl = null;
function showTyping() {
  hideTyping();
  typingEl = document.createElement('div');
  typingEl.className = 'typing';
  let n = 0;
  typingEl.textContent = `${state.name} is typing`;
  $('chat').appendChild(typingEl);
  const timer = setInterval(() => {
    if (!typingEl) { clearInterval(timer); return; }
    n = (n + 1) % 4;
    typingEl.textContent = `${state.name} is typing${'.'.repeat(n)}`;
    $('chat').scrollTop = $('chat').scrollHeight;
  }, 350);
}
function hideTyping() {
  if (typingEl) { typingEl.remove(); typingEl = null; }
}

/* ---------- speech output (phone voice) ---------- */

function speak(text) {
  if (!window.speechSynthesis) {
    addMsg('system', 'Speech output is not supported in this browser.');
    return;
  }
  speechSynthesis.cancel();
  const token = ++speakToken;
  const u = new SpeechSynthesisUtterance(text);
  u.rate = state.speed === 'fast' ? 1.2 : state.speed === 'slow' ? 0.85 : 1.0;
  const voices = speechSynthesis.getVoices();
  const preferred =
    voices.find(v => /^en/i.test(v.lang) && /female|google|natural|samantha|zira|aria/i.test(v.name)) ||
    voices.find(v => /^en/i.test(v.lang));
  if (preferred) u.voice = preferred;
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
  u.onend = done;
  u.onerror = done;
  speechSynthesis.speak(u);
}

function stopSpeech() {
  speakToken++;
  if (window.speechSynthesis) speechSynthesis.cancel();
  speaking = false;
  setMicSpeaking(false);
  setStatus(readyStatus());
  ensureWake();
}

function setMicSpeaking(on) {
  const btn = $('micBtn');
  btn.classList.toggle('speaking', on);
  btn.textContent = on ? 'Stop' : 'Mic';
}

function beep() {
  try {
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const osc = ctx.createOscillator(), gain = ctx.createGain();
    osc.frequency.value = 880; gain.gain.value = 0.2;
    osc.connect(gain); gain.connect(ctx.destination);
    osc.start(); osc.stop(ctx.currentTime + 0.12);
  } catch (err) { /* audio feedback is optional */ }
}

/* ---------- sending messages ---------- */

async function send(text) {
  text = (text || '').trim();
  if (!text) return;
  stopSpeech();

  if (/^(stop|stop speaking|be quiet|quiet)$/i.test(text)) {
    addMsg('user', text);
    addMsg('assistant', 'Okay, stopping.');
    return;
  }

  addMsg('user', text);
  busy = true;
  setStatus('Thinking...');
  showTyping();
  try {
    const res = await fetch('/api/message', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });
    const data = await res.json();
    hideTyping();
    addMsg('assistant', data.reply || '(no response)');
    speak(data.reply || '');
  } catch (err) {
    hideTyping();
    addMsg('system', 'Cannot reach the server: ' + err.message);
    setStatus(readyStatus());
  }
  busy = false;
}

/* ---------- speech input (browser mic) ---------- */

function startListening() {
  if (!SR) {
    addMsg('system', 'Voice input is not supported in this browser — type instead.');
    return;
  }
  if (listening || busy) return;
  stopWake();
  listening = true;
  setStatus('Listening...');
  try {
    listenRec = new SR();
    listenRec.lang = 'en-US';
    listenRec.interimResults = true;
    listenRec.continuous = false;
    listenRec.onresult = (e) => {
      let transcript = '';
      for (let i = e.resultIndex; i < e.results.length; i++) {
        transcript += e.results[i][0].transcript;
      }
      if (e.results.length && e.results[e.results.length - 1].isFinal) {
        listening = false;
        send(transcript.trim());
      }
    };
    listenRec.onerror = (e) => {
      listening = false;
      if (e.error === 'not-allowed' || e.error === 'service-not-allowed') {
        addMsg('system', 'Microphone permission denied — type instead.');
      } else {
        addMsg('system', 'Mic failed — type instead.');
      }
      setStatus(readyStatus());
      ensureWake();
    };
    listenRec.onend = () => {
      if (listening) {
        listening = false;
        setStatus(readyStatus());
        ensureWake();
      }
    };
    listenRec.start();
  } catch (err) {
    listening = false;
    setStatus(readyStatus());
    addMsg('system', 'Mic failed — type instead.');
  }
}

/* ---------- wake word ("Hey Nova") ---------- */

const WAKE_ONLY = ['', 'i have a question', "i've got a question", 'ive got a question',
  'question', 'yes', 'yeah', 'hello', 'hi', 'hey', 'are you there',
  'can you hear me', 'wake up', 'you there', "it's me", 'its me'];

function wakeQuestion(text, name) {
  if (!name) return null;
  const esc = name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const strict = text.match(new RegExp(
    `^\\s*(?:please\\s+)?(?:(?:hey(?:\\s+there)?|ok(?:ay)?|hi|hello|yo)\\s+)?${esc}\\b(?<tail>.*)$`,
    'i'));
  if (strict) {
    const tail = (strict.groups.tail || '').trim().replace(/[ .,:!?]+$/, '');
    return WAKE_ONLY.includes(tail.toLowerCase()) ? '' : tail;
  }
  if (new RegExp(`\\b${esc}\\b`, 'i').test(text)) {
    return text.replace(new RegExp(`\\b(?:hey\\s+|ok(?:ay)?\\s+)?${esc}\\b`, 'i'), '').trim();
  }
  return null;
}

function startWake() {
  if (!SR || wakeRunning || listening || busy || speaking) return;
  try {
    wakeRec = new SR();
    wakeRec.lang = 'en-US';
    wakeRec.continuous = true;
    wakeRec.interimResults = true;
    wakeRec.onresult = (e) => {
      for (let i = e.resultIndex; i < e.results.length; i++) {
        if (!e.results[i].isFinal) continue;
        const t = e.results[i][0].transcript.trim();
        const q = wakeQuestion(t, state.name);
        if (q === null) continue;
        stopWake();
        beep();
        if (q) send(q);
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
    setStatus(`Wake word on — say "Hey ${state.name}"`);
    setTimeout(() => { if (!listening && !busy && !speaking) setStatus(readyStatus()); }, 1600);
  } catch (err) { wakeRunning = false; }
}

function stopWake() {
  if (wakeRec) { try { wakeRec.stop(); } catch (err) { /* already stopped */ } }
  wakeRunning = false;
}

function ensureWake() {
  if (wakeWanted && !wakeRunning && !listening && !busy && !speaking) startWake();
}

function updateWakeButton() {
  $('wakeBtn').classList.toggle('active', wakeWanted);
}

/* ---------- settings sheet ---------- */

function renderApps() {
  const list = $('appsList');
  list.innerHTML = '';
  const names = Object.keys(state.apps).sort();
  if (!names.length) {
    const row = document.createElement('div');
    row.className = 'app-row';
    row.textContent = 'No apps yet';
    list.appendChild(row);
    return;
  }
  for (const name of names) {
    const row = document.createElement('div');
    row.className = 'app-row' + (selectedApp === name ? ' selected' : '');
    row.textContent = name;
    row.addEventListener('click', () => {
      selectedApp = name;
      $('appName').value = name;
      $('appPath').value = state.apps[name];
      renderApps();
    });
    list.appendChild(row);
  }
}

function openSheet() {
  $('setKey').value = '';
  $('setKey').placeholder = state.apiReady ? '•••••••• (saved)' : 'YOUR_GEMINI_API_KEY_HERE';
  $('setName').value = state.name;
  const radio = document.querySelector(`input[name="speed"][value="${state.speed}"]`);
  if (radio) radio.checked = true;
  selectedApp = null;
  $('appName').value = '';
  $('appPath').value = '';
  renderApps();
  $('sheetBackdrop').classList.remove('hidden');
}

function closeSheet() {
  $('sheetBackdrop').classList.add('hidden');
}

async function saveSettings() {
  const payload = {
    name: $('setName').value.trim() || 'Nova',
    apps: state.apps,
    speed: document.querySelector('input[name="speed"]:checked').value,
  };
  const key = $('setKey').value.trim();
  if (key) payload.api_key = key;
  try {
    const res = await fetch('/api/settings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    state.name = data.name || payload.name;
    state.apiReady = !!data.api_ready;
    state.apps = data.apps || state.apps;
    state.speed = payload.speed;
    $('titleName').textContent = state.name;
    document.title = state.name;
    closeSheet();
    addMsg('system', 'Settings saved.');
  } catch (err) {
    addMsg('system', 'Cannot reach the server: ' + err.message);
  }
}

async function clearChat() {
  try {
    const res = await fetch('/api/clear', { method: 'POST' });
    const data = await res.json();
    $('chat').innerHTML = '';
    addMsg('assistant', data.reply || 'Chat cleared.');
    closeSheet();
  } catch (err) {
    addMsg('system', 'Cannot reach the server: ' + err.message);
  }
}

/* ---------- wiring ---------- */

async function loadState() {
  try {
    const res = await fetch('/api/state');
    const data = await res.json();
    state.name = data.name || 'Nova';
    state.apiReady = !!data.api_ready;
    state.apps = data.apps || {};
    state.user_name = data.user_name || '';
    $('titleName').textContent = state.name;
    document.title = state.name;
  } catch (err) { /* keep defaults */ }
  const hello = state.user_name ? `Hello, ${state.user_name}!` : 'Hello!';
  addMsg('assistant',
    `${hello} I'm ${state.name}, your voice assistant. Tap the Mic or type below, ` +
    `turn on the wake word (moon button) and just say 'Hey ${state.name}', ` +
    `or say 'search for' to look something up. Say 'help' to hear what I can do.`);
  setStatus(readyStatus());
}

$('sendBtn').addEventListener('click', () => {
  const text = $('entry').value;
  $('entry').value = '';
  send(text);
});
$('entry').addEventListener('keydown', (e) => {
  if (e.key === 'Enter') {
    const text = $('entry').value;
    $('entry').value = '';
    send(text);
  }
});
$('micBtn').addEventListener('click', () => {
  if (speaking) { stopSpeech(); return; }
  if (busy) return;
  startListening();
});
$('wakeBtn').addEventListener('click', () => {
  wakeWanted = !wakeWanted;
  updateWakeButton();
  if (wakeWanted) startWake();
  else { stopWake(); setStatus(readyStatus()); }
});
$('gearBtn').addEventListener('click', openSheet);
$('closeSheet').addEventListener('click', closeSheet);
$('sheetBackdrop').addEventListener('click', (e) => {
  if (e.target === $('sheetBackdrop')) closeSheet();
});
$('saveBtn').addEventListener('click', saveSettings);
$('clearBtn').addEventListener('click', clearChat);
$('addApp').addEventListener('click', () => {
  const name = $('appName').value.trim();
  const path = $('appPath').value.trim();
  if (!name || !path) { addMsg('system', 'Enter both an app name and a file path.'); return; }
  state.apps[name] = path;
  selectedApp = name;
  renderApps();
});
$('removeApp').addEventListener('click', () => {
  if (!selectedApp) { addMsg('system', 'Select an app to remove.'); return; }
  delete state.apps[selectedApp];
  selectedApp = null;
  $('appName').value = '';
  $('appPath').value = '';
  renderApps();
});

if (!SR) {
  $('micBtn').style.opacity = '0.5';
  $('wakeBtn').style.opacity = '0.5';
}

if ('serviceWorker' in navigator) {
  navigator.serviceWorker.register('/sw.js').catch(() => {});
}
window.speechSynthesis && window.speechSynthesis.getVoices();

loadState();
