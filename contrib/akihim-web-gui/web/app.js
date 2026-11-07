'use strict';
/* Anime Denoise — front end.
 *
 * Mental model shown to the user (and mirrored here):
 *   result = original  →  denoise(settings)  →  local protection layer on top
 * - Denoise settings belong to each image. The button only asks for a run when the settings differ from
 *   the result on screen; every run starts from the original image.
 * - Local protection is a soft-edged mask layer. Where painted, the original is blended back over the
 *   denoised result. It is previewed live, auto-saved, and survives re-runs (the server re-applies it).
 * - While a new denoise runs, the previous result stays visible.
 */

const $ = s => document.querySelector(s);
const $$ = s => [...document.querySelectorAll(s)];
const TOKEN = $('meta[name=token]').content;
const API_VERSION = 2;

const SMOOTH_STEPS = [0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001, 0.0012, 0.0015, 0.0018, 0.002, 0.0025, 0.003, 0.004, 0.005];
const PRESETS = {
  gentle: { ratio: 80, smooth: 0.0006 },
  recommended: { ratio: 60, smooth: 0.001 },
  strong: { ratio: 45, smooth: 0.0018 },
};
const MAX_BYTES = 32 * 1024 * 1024;
const DEFAULT_PRESERVE = 0.8;   // protection brush: how much of the original comes back where painted

const S = {
  jobs: [], current: null, activeId: null,
  view: 'compare',            // preferred view; falls back to 'original' while there is no result
  peek: false,                // holding Z: temporarily show the original
  tool: 'pan',                // pan | paint | erase
  scale: 1, tx: 0, ty: 0, autoFit: true,
  split: 0.5,                 // divider position as a fraction of the IMAGE width, so it moves with the image
  spaceHeld: false, finishedUnseen: false, maskVisible: true,
};
const drafts = new Map();     // job id → settings the user changed but has not run yet

const viewport = $('#viewport');
const canvasEl = $('#canvas');
const originalImg = $('#original');
const resultImg = $('#result');
const overlay = $('#mask-overlay');
const preview = $('#preview');
const oCtx = overlay.getContext('2d');
const pCtx = preview.getContext('2d');

/* ================================================================ utilities */

async function api(path, options = {}) {
  let res;
  try {
    res = await fetch(path, { ...options, headers: { 'X-Token': TOKEN, ...(options.headers || {}) } });
  } catch {
    throw new Error('連不上本機服務。工具可能已結束，請重新啟動工具。');
  }
  let data = {};
  try { data = await res.json(); } catch { /* non-JSON */ }
  if (!res.ok) throw new Error(data.error || `操作失敗（${res.status}）`);
  return data;
}
const post = (path, body) => api(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });

function toast(message, { type = 'info', action, duration } = {}) {
  const box = $('#toasts');
  while (box.children.length >= 3) box.firstChild.remove();
  const el = document.createElement('div');
  el.className = 'toast' + (type === 'error' ? ' error' : '');
  el.setAttribute('role', type === 'error' ? 'alert' : 'status');
  const text = document.createElement('span');
  text.textContent = message;
  el.append(text);
  if (action) {
    const b = document.createElement('button');
    b.className = 'act';
    b.textContent = action.label;
    b.onclick = () => { el.remove(); action.fn(); };
    el.append(b);
  }
  const close = document.createElement('button');
  close.className = 'x';
  close.textContent = '×';
  close.setAttribute('aria-label', '關閉');
  close.onclick = () => el.remove();
  el.append(close);
  box.append(el);
  setTimeout(() => el.remove(), duration ?? (type === 'error' ? 9000 : action ? 6000 : 3500));
}

const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const fmtSmooth = v => String(+v.toFixed(4));
const fmtParams = p => `${p.ratio}% · ${fmtSmooth(p.smooth)}`;
const sameParams = (a, b) => a.ratio === b.ratio && Math.abs(a.smooth - b.smooth) < 1e-9;
const hasResult = j => !!(j && j.output);
const isRunning = j => !!(j && j.status === 'running');
const resultParams = j => ({ ratio: Math.round((j.result_downscale ?? j.downscale ?? 0.6) * 100), smooth: j.result_strength ?? j.strength ?? 0.001 });
const requestedParams = j => ({ ratio: Math.round((j.downscale ?? 0.6) * 100), smooth: j.strength ?? 0.001 });
const effectiveView = () => (!hasResult(S.current) || S.peek ? 'original' : S.view);
const canPaint = () => !!S.current && !isRunning(S.current) && !M.loading;
function store(key, value) { try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* storage unavailable */ } }
function recall(key) { try { return JSON.parse(localStorage.getItem(key)); } catch { return null; } }
function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error('圖片載入失敗'));
    img.src = src;
  });
}

/* ================================================================ theme */

const THEMES = {
  system: { label: '跟隨系統', icon: '<rect x="3" y="4" width="18" height="12" rx="2"/><path d="M8 20h8M12 16v4"/>' },
  light: { label: '淺色', icon: '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>' },
  dark: { label: '深色', icon: '<path d="M20 14.5A8 8 0 0 1 9.5 4a8 8 0 1 0 10.5 10.5z"/>' },
};
function applyTheme(mode) {
  if (!THEMES[mode]) mode = 'system';
  if (mode === 'system') delete document.documentElement.dataset.theme;
  else document.documentElement.dataset.theme = mode;
  $('#theme-icon').innerHTML = THEMES[mode].icon;
  $('#theme-label').textContent = THEMES[mode].label;
  $('#theme-btn').title = `外觀：${THEMES[mode].label}（點擊切換）`;
  store('ad.theme', mode);
}
$('#theme-btn').onclick = () => {
  const order = Object.keys(THEMES);
  applyTheme(order[(order.indexOf(recall('ad.theme') || 'system') + 1) % order.length]);
};
applyTheme(recall('ad.theme') || 'system');

/* ================================================================ denoise settings (per image) */

function getParams() { return { ratio: +$('#ratio').value, smooth: SMOOTH_STEPS[+$('#smooth').value] }; }
function nearestSmooth(v) {
  let best = 0;
  SMOOTH_STEPS.forEach((s, i) => { if (Math.abs(s - v) < Math.abs(SMOOTH_STEPS[best] - v)) best = i; });
  return best;
}
function paramsFor(job) {
  if (drafts.has(job.id)) return drafts.get(job.id);
  if (isRunning(job)) return requestedParams(job);
  if (hasResult(job)) return resultParams(job);
  return recall('ad.params') || PRESETS.recommended;
}
// Show settings in the UI. fromUser = true means the user changed them (becomes this image's draft).
function setParams(p, fromUser = false) {
  $('#ratio').value = clamp(Math.round(p.ratio), 10, 100);
  $('#smooth').value = nearestSmooth(p.smooth);
  const q = getParams();
  $('#ratio-out').textContent = q.ratio + '%';
  $('#smooth-out').textContent = fmtSmooth(q.smooth);
  $('#tune-summary').textContent = fmtParams(q);
  let matched = false;
  for (const [name, preset] of Object.entries(PRESETS)) {
    const on = sameParams(preset, q);
    matched ||= on;
    $(`[data-preset=${name}]`).setAttribute('aria-checked', on);
  }
  if (!matched) $('#tune').open = true;
  if (fromUser) {
    store('ad.params', q);  // new images start from the last settings used
    const j = S.current;
    if (j) {
      if (hasResult(j) && sameParams(q, resultParams(j))) drafts.delete(j.id);
      else drafts.set(j.id, q);
    }
  }
  renderDenoise();
}
$$('[data-preset]').forEach(b => b.onclick = () => setParams(PRESETS[b.dataset.preset], true));
$('#ratio').oninput = () => setParams(getParams(), true);
$('#smooth').oninput = () => setParams(getParams(), true);

function denoiseState(j) {
  if (!j) return 'none';
  if (isRunning(j)) return 'running';
  if (!hasResult(j)) return j.status === 'error' ? 'error' : 'todo';
  return sameParams(getParams(), resultParams(j)) ? 'fresh' : 'modified';
}

/* ================================================================ protection mask */

const M = {
  jobId: null, canvas: document.createElement('canvas'), base: null,
  actions: [], redo: [], stroke: null, loading: false, dirty: false, version: 0,
};
M.ctx = M.canvas.getContext('2d');

function brushSize() { return Math.max(1, Math.round(Math.pow(600, +$('#brush-size').value / 100))); } // 1 … 600 px
function updateBrushLabel() { $('#brush-size-out').textContent = brushSize(); }

function stamp(ctx, x, y, size, erase) {
  if (size <= 1) {   // same as the original brush: only a 1 px brush is hard-edged; everything else is soft
    ctx.globalCompositeOperation = erase ? 'destination-out' : 'source-over';
    ctx.fillStyle = '#fff';
    ctx.fillRect(Math.floor(x), Math.floor(y), 1, 1);
  } else {
    const r = size / 2;
    const g = ctx.createRadialGradient(x, y, r * 0.55, x, y, r);
    g.addColorStop(0, 'rgba(255,255,255,1)');
    g.addColorStop(1, 'rgba(255,255,255,0)');
    ctx.globalCompositeOperation = erase ? 'destination-out' : 'source-over';
    ctx.fillStyle = g;
    ctx.fillRect(x - r, y - r, size, size);
  }
  ctx.globalCompositeOperation = 'source-over';
}
function segment(ctx, a, b, size, erase) {
  const steps = Math.max(1, Math.ceil(Math.hypot(b.x - a.x, b.y - a.y) / Math.max(0.5, size / 8)));
  for (let i = 1; i <= steps; i++) stamp(ctx, a.x + (b.x - a.x) * i / steps, a.y + (b.y - a.y) * i / steps, size, erase);
}
function applyAction(ctx, action) {
  if (action.clear) { ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height); return; }
  const p = action.points;
  stamp(ctx, p[0].x, p[0].y, action.size, action.erase);
  for (let i = 1; i < p.length; i++) segment(ctx, p[i - 1], p[i], action.size, action.erase);
}
function replay() {
  if (M.actions.length > 80) {   // keep replays cheap: bake the oldest strokes into the base
    const baked = document.createElement('canvas');
    baked.width = M.canvas.width; baked.height = M.canvas.height;
    const c = baked.getContext('2d');
    if (M.base) c.drawImage(M.base, 0, 0);
    M.actions.splice(0, 40).forEach(a => applyAction(c, a));
    M.base = baked;
  }
  M.ctx.clearRect(0, 0, M.canvas.width, M.canvas.height);
  if (M.base) M.ctx.drawImage(M.base, 0, 0);
  M.actions.forEach(a => applyAction(M.ctx, a));
  invalidate();
}

// Repaint the green guide overlay and the live "original restored" preview, limited to the changed area.
let dirtyRect = null, frame = 0;
function invalidate(x, y, w, h) {
  const W = M.canvas.width, H = M.canvas.height;
  const r = x === undefined ? { x0: 0, y0: 0, x1: W, y1: H } : { x0: x, y0: y, x1: x + w, y1: y + h };
  dirtyRect = dirtyRect ? { x0: Math.min(dirtyRect.x0, r.x0), y0: Math.min(dirtyRect.y0, r.y0), x1: Math.max(dirtyRect.x1, r.x1), y1: Math.max(dirtyRect.y1, r.y1) } : r;
  if (!frame) frame = requestAnimationFrame(paintMask);
}
function paintMask() {
  frame = 0;
  if (!dirtyRect) return;
  const W = M.canvas.width, H = M.canvas.height;
  const x = clamp(Math.floor(dirtyRect.x0), 0, W), y = clamp(Math.floor(dirtyRect.y0), 0, H);
  const w = clamp(Math.ceil(dirtyRect.x1), 0, W) - x, h = clamp(Math.ceil(dirtyRect.y1), 0, H) - y;
  dirtyRect = null;
  if (w <= 0 || h <= 0) return;
  const layers = [[oCtx, c => { c.fillStyle = 'rgba(32,185,137,.45)'; c.fillRect(x, y, w, h); }]];
  if (originalImg.complete && originalImg.naturalWidth) layers.push([pCtx, c => c.drawImage(originalImg, 0, 0, W, H)]);
  for (const [c, fill] of layers) {
    c.save();
    c.beginPath(); c.rect(x, y, w, h); c.clip();
    c.clearRect(x, y, w, h);
    c.drawImage(M.canvas, 0, 0);
    c.globalCompositeOperation = 'source-in';
    fill(c);
    c.restore();
  }
}

async function loadMask(job) {
  const version = ++M.version;
  Object.assign(M, { jobId: job.id, actions: [], redo: [], base: null, stroke: null, dirty: false, loading: true });
  for (const c of [M.canvas, overlay, preview]) { c.width = job.width; c.height = job.height; }
  $('#preserve').value = Math.round((job.preserve ?? DEFAULT_PRESERVE) * 100);
  updatePreserve(false);
  try {
    if (job.mask_version) {
      const img = await loadImage(`/image/${job.id}/mask?v=${job.mask_version}`);
      if (version !== M.version) return;
      const c = document.createElement('canvas');
      c.width = job.width; c.height = job.height;
      const ctx = c.getContext('2d');
      ctx.drawImage(img, 0, 0);
      const px = ctx.getImageData(0, 0, c.width, c.height);
      const d = px.data;
      for (let i = 0; i < d.length; i += 4) { d[i + 3] = d[i]; d[i] = d[i + 1] = d[i + 2] = 255; }
      ctx.putImageData(px, 0, 0);
      M.base = c;
    }
  } catch (e) {
    if (version === M.version) toast('保護區載入失敗：' + e.message, { type: 'error' });
  } finally {
    if (version === M.version) { M.loading = false; replay(); renderAll(); }
  }
}

function exportMask() {
  const c = document.createElement('canvas');
  c.width = M.canvas.width; c.height = M.canvas.height;
  const ctx = c.getContext('2d');
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, c.width, c.height);
  ctx.drawImage(M.canvas, 0, 0);
  return { protection_mask: c.toDataURL('image/png'), preserve: +$('#preserve').value / 100 };
}

// Auto-save: changes are queued per image and sent in order; there is no "apply" step.
let saveTimer = 0, saveLoop = null, saveFailed = false;
const pendingSaves = new Map();
function maskChanged(delay = 150) {
  M.dirty = true;
  setSaveState('未儲存');
  clearTimeout(saveTimer);
  saveTimer = setTimeout(flushMask, delay);
}
function flushMask() {
  clearTimeout(saveTimer);
  if (M.dirty && M.jobId && !M.loading) {
    pendingSaves.set(M.jobId, exportMask());
    M.dirty = false;
  }
  if (!saveLoop && pendingSaves.size) saveLoop = runSaves();
  return saveLoop || Promise.resolve();
}
async function runSaves() {
  setSaveState('儲存中…');
  while (pendingSaves.size) {
    const [id, body] = pendingSaves.entries().next().value;
    pendingSaves.delete(id);
    try {
      upsert(await post('/api/protection', { id, ...body }));
      saveFailed = false;
    } catch (e) {
      saveFailed = true;
      toast('保護區儲存失敗：' + e.message, { type: 'error' });
    }
  }
  saveLoop = null;
  setSaveState(saveFailed ? '儲存失敗' : '已儲存');
  renderAll();
}
function setSaveState(text) {
  const el = $('#save-state');
  el.textContent = text;
  el.classList.toggle('error', text === '儲存失敗');
}
const savePending = () => M.dirty || !!saveLoop || pendingSaves.size > 0;

function undo() { if (!canPaint() || !M.actions.length) return; M.redo.push(M.actions.pop()); replay(); maskChanged(); renderViewer(); }
function redo() { if (!canPaint() || !M.redo.length) return; M.actions.push(M.redo.pop()); replay(); maskChanged(); renderViewer(); }
function clearMask() {
  if (!canPaint()) return;
  M.actions.push({ clear: true }); M.redo = []; replay(); maskChanged(); renderViewer();
  toast('已清除保護區', { action: { label: '撤銷', fn: undo } });
}
function updatePreserve(save = true) {
  const v = +$('#preserve').value;
  $('#preserve-out').textContent = v + '%';
  preview.style.opacity = v / 100;
  if (save) maskChanged(400);
}
$('#brush-size').oninput = updateBrushLabel;
$('#preserve').oninput = () => updatePreserve(true);
$('#brush-undo').onclick = undo;
$('#brush-redo').onclick = redo;
$('#brush-clear').onclick = clearMask;
$('#protect-clear').onclick = clearMask;
function toggleMaskVisible() { S.maskVisible = !S.maskVisible; $('#mask-eye').classList.toggle('on', S.maskVisible); renderViewer(); }
$('#mask-eye').onclick = toggleMaskVisible;
$('#brush-done').onclick = () => setTool('pan');
$('#protect-toggle').onclick = () => setTool(S.tool === 'pan' ? 'paint' : 'pan');

/* ================================================================ jobs & selection */

function upsert(job) {
  const i = S.jobs.findIndex(j => j.id === job.id);
  if (i < 0) S.jobs.unshift(job); else S.jobs[i] = job;
  if (S.current && S.current.id === job.id) S.current = job;
}

let loadedResult = '';
function syncResultImage() {
  const j = S.current;
  const key = hasResult(j) ? `${j.id}/${j.base_output || j.output}` : '';
  if (key === loadedResult) return;
  loadedResult = key;
  if (key) resultImg.src = `/image/${j.id}/base?v=${encodeURIComponent(j.base_output || j.output)}`;
  else resultImg.removeAttribute('src');
}

function select(job) {
  if (!job) {
    flushMask();
    S.current = null; M.jobId = null; loadedResult = '';
    originalImg.removeAttribute('src'); resultImg.removeAttribute('src');
    S.tool = 'pan';
    renderAll();
    return;
  }
  if (S.current && S.current.id === job.id) return;
  flushMask();
  S.current = job;
  S.split = 0.5;
  if (S.tool !== 'pan' && isRunning(job)) S.tool = 'pan';
  store('ad.selected', job.id);
  setParams(paramsFor(job));
  originalImg.onload = () => invalidate();
  originalImg.src = `/image/${job.id}/original`;
  canvasEl.style.width = job.width + 'px';
  canvasEl.style.height = job.height + 'px';
  syncResultImage();
  loadMask(job);
  renderAll();
  fit();
  const thumb = thumbs.get(job.id);
  if (thumb) thumb.scrollIntoView({ block: 'nearest', inline: 'nearest' });
}
function step(delta) {
  if (!S.jobs.length) return;
  const i = S.current ? S.jobs.findIndex(j => j.id === S.current.id) : -1;
  select(S.jobs[clamp(i + delta, 0, S.jobs.length - 1)]);
}

/* ================================================================ view transform */

function fit() {
  const j = S.current;
  if (!j) return;
  const pad = 20, top = 54, vw = viewport.clientWidth, vh = viewport.clientHeight;  // top room for the brush bar
  S.scale = Math.max(0.02, Math.min((vw - pad * 2) / j.width, (vh - pad - top) / j.height));
  S.tx = (vw - j.width * S.scale) / 2;
  S.ty = top + (vh - pad - top - j.height * S.scale) / 2;
  S.autoFit = true;
  renderViewer();
}
function zoomTo(next, cx = viewport.clientWidth / 2, cy = viewport.clientHeight / 2) {
  if (!S.current) return;
  next = clamp(next, 0.02, 32);
  S.tx = cx - (cx - S.tx) * next / S.scale;
  S.ty = cy - (cy - S.ty) * next / S.scale;
  S.scale = next;
  S.autoFit = false;
  renderViewer();
}
function localPoint(e) {
  const r = viewport.getBoundingClientRect();
  return { x: e.clientX - r.left - viewport.clientLeft, y: e.clientY - r.top - viewport.clientTop };
}
function imagePoint(e) {
  const p = localPoint(e);
  return { x: (p.x - S.tx) / S.scale, y: (p.y - S.ty) / S.scale };
}

/* ================================================================ rendering */

function renderAll() {
  renderImageInfo();
  renderDenoise();
  renderProtect();
  renderOutput();
  renderToolbar();
  syncResultImage();
  renderViewer();
  renderStrip();
  renderTitle();
}

function renderImageInfo() {
  const j = S.current;
  $('#image-empty').hidden = !!j;
  $('#image-info').hidden = !j;
  if (!j) return;
  $('#filename').textContent = j.name;
  $('#filename').title = j.name;
  $('#dimensions').textContent = `${j.width} × ${j.height}${j.alpha ? ' · 透明背景' : ''}`;
  const thumb = `/image/${j.id}/thumb`;
  if ($('#file-thumb').getAttribute('src') !== thumb) $('#file-thumb').src = thumb;
}

function renderDenoise() {
  const j = S.current, state = denoiseState(j);
  const active = S.activeId && S.activeId !== j?.id && S.jobs.find(x => x.id === S.activeId);
  const btn = $('#run'), cap = $('#run-caption'), chip = $('#denoise-chip');
  const locked = state === 'running';
  $$('[data-preset], #ratio, #smooth').forEach(el => { el.disabled = locked; });

  let label = '開始去噪', disabled = !j, tone = '', progress = 0;
  btn.className = 'primary';
  cap.className = 'caption';
  cap.replaceChildren();
  const chipSet = (text, cls = '') => { chip.textContent = text; chip.className = 'chip ' + cls; };
  chipSet('');

  if (state === 'running') {
    label = `去噪中 ${j.percent || 0}%`; disabled = true; progress = j.percent || 3;
    btn.classList.add('running');
    chipSet('處理中', 'busy');
    cap.textContent = j.stage || '';
  } else if (active) {
    label = '等待中…'; disabled = true;
    cap.textContent = `「${active.name}」處理中，完成後才能開始這張。`;
  } else if (state === 'todo') {
    chipSet('未處理');
  } else if (state === 'error') {
    label = '再試一次';
    chipSet('失敗', 'err');
    cap.textContent = j.error || '處理失敗';
    tone = 'error';
  } else if (state === 'modified') {
    label = '用新設定重新去噪';
    chipSet('設定已改', 'warn');
    tone = 'warn';
    cap.append(`畫面上是 ${fmtParams(resultParams(j))} 的結果。`);
    const back = document.createElement('button');
    back.className = 'link';
    back.textContent = '改回';
    back.onclick = () => { drafts.delete(j.id); setParams(resultParams(j)); };
    cap.append(back);
  } else if (state === 'fresh') {
    label = '重新生成';
    btn.classList.add('fresh');   // still available, but visually secondary: settings already match the result
    chipSet('已完成', 'ok');
  }
  if (tone) cap.classList.add(tone);
  btn.disabled = disabled;
  btn.title = state === 'fresh' ? '用相同設定從原圖再算一次（Ctrl+Enter）' : 'Ctrl+Enter · 每次都從原圖計算';
  $('#run-label').textContent = label;
  $('#run-progress').style.width = progress + '%';
}

function renderProtect() {
  const j = S.current, result = hasResult(j);
  const toggle = $('#protect-toggle');
  toggle.disabled = !canPaint();
  toggle.classList.toggle('on', S.tool !== 'pan');
  $('#protect-toggle-label').textContent = S.tool !== 'pan' ? '完成塗抹' : (j && j.protected ? '修改保護區' : '保護筆刷');
  toggle.title = j && isRunning(j) ? '去噪處理中，完成後才能塗抹' : 'B';
  const on = !!(j && j.protected);
  const chip = $('#protect-chip');
  chip.textContent = j ? (on ? '使用中' : '未使用') : '';
  chip.className = 'chip' + (on ? ' ok' : '');
  $('#protect-summary').hidden = !on;
  if (on) {
    $('#protect-summary-text').textContent = result
      ? `● 塗抹處恢復原圖 ${Math.round((j.preserve ?? DEFAULT_PRESERVE) * 100)}%`
      : '● 已塗好，去噪完成後自動套用';
    $('#protect-clear').disabled = !canPaint();
  }
}

function renderOutput() {
  const j = S.current, info = $('#result-info');
  const result = hasResult(j);
  $('#download').disabled = !result;
  $('#reveal').disabled = !result;
  info.replaceChildren();
  info.title = '';
  if (!j) return;
  if (!result) { info.textContent = '去噪後即可下載'; return; }
  if (isRunning(j)) { info.textContent = '新結果處理中，現在下載的是上一版'; return; }
  const parts = [fmtParams(resultParams(j))];
  if (j.protected && (j.preserve ?? DEFAULT_PRESERVE) > 0) parts.push('含局部保護');
  if (j.elapsed) parts.push(`${j.elapsed} 秒`);
  info.textContent = '已自動存到 outputs · ' + parts.join(' · ');
  info.title = `${j.device || '本機'}\n${j.output}`;
}

function renderToolbar() {
  const j = S.current, view = effectiveView(), result = hasResult(j);
  $$('[data-view]').forEach(b => {
    b.setAttribute('aria-pressed', b.dataset.view === view);
    b.disabled = !j || (b.dataset.view !== 'original' && !result);
  });
  $$('[data-tool]').forEach(b => {
    b.setAttribute('aria-pressed', b.dataset.tool === S.tool);
    b.disabled = !canPaint();
  });
  const name = $('#view-name');
  name.classList.toggle('peek', S.peek && result);
  name.textContent = !j ? ''
    : S.peek && result ? '暫時顯示原圖（放開 Z 返回）'
    : !result ? '尚未去噪，顯示的是原圖'
    : { compare: '左：原圖　右：結果', original: '原圖', result: j.protected ? '結果（去噪＋局部保護）' : '結果' }[view];
  for (const id of ['zoom-in', 'zoom-out', 'fit', 'actual']) $('#' + id).disabled = !j;
  $('#zoom-label').textContent = j ? Math.round(S.scale * 100) + '%' : '—';
}

function renderViewer() {
  const j = S.current, view = effectiveView(), result = hasResult(j);
  const brushing = !!j && S.tool !== 'pan';
  $('#empty').hidden = !!j;
  canvasEl.hidden = !j;
  viewport.classList.toggle('drop-target', !j);
  viewport.classList.toggle('has-image', !!j);
  viewport.classList.toggle('brushing', brushing && !S.spaceHeld);
  $('#zoom-label').textContent = j ? Math.round(S.scale * 100) + '%' : '—';
  if (!j) {
    for (const id of ['#divider', '#labels', '#brushbar', '#brush-tip', '#busy', '#brush-cursor']) $(id).hidden = true;
    return;
  }
  canvasEl.style.transform = `translate(${S.tx}px,${S.ty}px) scale(${S.scale})`;
  canvasEl.classList.toggle('pixelated', S.scale >= 2);

  const layer = $('#result-layer');
  layer.hidden = !result || view === 'original';
  layer.style.clipPath = view === 'compare' ? `inset(0 0 0 ${S.split * 100}%)` : 'none';
  originalImg.style.visibility = view === 'result' ? 'hidden' : 'visible';
  // The green mask is only an editing guide, shown while the brush is active.
  overlay.hidden = !(brushing && S.maskVisible);

  const divider = $('#divider');
  divider.hidden = !(result && view === 'compare');
  const x = S.tx + S.split * j.width * S.scale;   // pinned to the image: follows zoom and pan
  const shown = clamp(x, 0, viewport.clientWidth);
  divider.style.left = shown + 'px';
  divider.classList.toggle('edge', shown !== x);
  divider.setAttribute('aria-valuenow', Math.round(S.split * 100));
  $('#labels').hidden = divider.hidden;
  $('#label-right').textContent = j.protected ? '結果（含保護）' : '結果';

  $('#brushbar').hidden = !brushing;
  $('#brush-undo').disabled = !M.actions.length;
  $('#brush-redo').disabled = !M.redo.length;
  const tip = $('#brush-tip');
  tip.hidden = !brushing;
  if (brushing) {
    tip.textContent = !result
      ? '塗在要保留的地方（綠色）· 去噪完成後自動恢復原圖'
      : view === 'original'
        ? '現在看的是原圖 · 按 1 或 3 切到對比／結果才看得到保護效果'
        : '塗到處即時恢復原圖 · 空白鍵拖動畫面 · [ ] 筆刷大小 · 右鍵擦除';
  }

  const running = isRunning(j);
  const busy = $('#busy');
  busy.hidden = !running;
  if (running) {
    $('#busy-stage').textContent = `${j.stage || '處理中'} · ${j.percent || 0}%`;
    $('#busy-bar').style.width = (j.percent || 3) + '%';
    const secs = Math.max(0, Math.floor(Date.now() / 1000 - (j.started || Date.now() / 1000)));
    $('#busy-time').textContent = result ? `${secs} 秒 · 目前顯示上一版` : `${secs} 秒`;
  }
}

const thumbs = new Map();
function renderStrip() {
  const box = $('#strip-items');
  const seen = new Set();
  S.jobs.forEach((j, index) => {
    seen.add(j.id);
    let el = thumbs.get(j.id);
    if (!el) {
      el = document.createElement('div');
      el.className = 'thumb';
      const pick = document.createElement('button');
      pick.className = 'pick';
      const img = document.createElement('img');
      img.alt = '';
      img.src = `/image/${j.id}/thumb`;
      pick.append(img);
      pick.onclick = () => select(S.jobs.find(x => x.id === j.id));
      const badge = document.createElement('span');
      const shield = document.createElement('span');
      shield.className = 'badge shield';
      shield.title = '有局部保護';
      const remove = document.createElement('button');
      remove.className = 'remove';
      remove.textContent = '×';
      remove.title = '從列表移除（outputs 裡的結果檔會保留）';
      remove.onclick = e => { e.stopPropagation(); removeJob(j.id, remove); };
      el.append(pick, badge, shield, remove);
      thumbs.set(j.id, el);
    }
    const [pick, badge, shield, remove] = el.children;
    el.classList.toggle('selected', !!S.current && S.current.id === j.id);
    const status = j.status === 'running' ? 'running' : j.status === 'error' && !j.output ? 'error' : j.output ? 'done' : '';
    badge.className = 'badge ' + status;
    badge.hidden = !status;
    badge.textContent = { running: (j.percent || 0) + '%', error: '!', done: '✓' }[status] || '';
    shield.hidden = !j.protected;
    pick.title = `${j.name}\n${j.width} × ${j.height}` + (j.output ? `\n${fmtParams(resultParams(j))}` : '\n尚未去噪');
    remove.hidden = j.status === 'running';
    if (box.children[index] !== el) box.insertBefore(el, box.children[index] || null);
  });
  for (const [id, el] of thumbs) if (!seen.has(id)) { el.remove(); thumbs.delete(id); }
}

function renderTitle() {
  const active = S.activeId && S.jobs.find(j => j.id === S.activeId);
  document.title = active ? `${active.percent || 0}% 去噪中 · Anime Denoise`
    : S.finishedUnseen ? '✓ 去噪完成 · Anime Denoise' : 'Anime Denoise · 動漫去噪';
}
document.addEventListener('visibilitychange', () => { if (!document.hidden && S.finishedUnseen) { S.finishedUnseen = false; renderTitle(); } });

/* ================================================================ actions */

function setView(view) {
  if (!S.current || (view !== 'original' && !hasResult(S.current))) return;
  S.view = view;
  renderToolbar();
  renderViewer();
}
$$('[data-view]').forEach(b => b.onclick = () => setView(b.dataset.view));

function setTool(tool) {
  if (tool !== 'pan' && !canPaint()) {
    if (S.current && isRunning(S.current)) toast('這張圖片正在去噪，完成後才能塗抹保護區。');
    return;
  }
  S.tool = tool;
  $('#brush-cursor').hidden = true;
  if (tool === 'pan') flushMask();
  renderProtect();
  renderToolbar();
  renderViewer();
}

async function run() {
  const j = S.current;
  const state = denoiseState(j);
  if (!j || S.activeId || state === 'running') return;
  try {
    await flushMask();
    const p = getParams();
    const job = await post('/api/process', { id: j.id, strength: p.smooth, downscale: p.ratio / 100 });
    drafts.delete(j.id);
    upsert(job);
    S.activeId = job.id;
    S.tool = 'pan';
    renderAll();
    poll();
  } catch (e) {
    toast(e.message, { type: 'error' });
  }
}
$('#run').onclick = run;

let pollTimer = 0, pollFailures = 0;
function poll() {
  clearTimeout(pollTimer);
  pollTimer = setTimeout(async () => {
    if (!S.activeId) return;
    try {
      const job = await api('/api/job/' + S.activeId);
      pollFailures = 0;
      upsert(job);
      if (job.status !== 'running') { S.activeId = null; finished(job); }
      renderAll();
    } catch {
      if (++pollFailures === 3) toast('與本機服務的連線中斷，正在重試…', { type: 'error' });
    }
    if (S.activeId) poll();
  }, 700);
}
function finished(job) {
  if (document.hidden) S.finishedUnseen = true;
  const viewing = S.current && S.current.id === job.id;
  if (viewing && !drafts.has(job.id)) setParams(paramsFor(job));
  if (job.status === 'error') {
    toast(`「${job.name}」${job.error || '處理失敗'}`, { type: 'error' });
  } else if (viewing) {
    if (S.view === 'original') S.view = 'compare';
    toast(job.protected ? '去噪完成，局部保護已自動套用。' : '去噪完成。若有細節被抹掉，可用「保護筆刷」塗回來。', { duration: 5000 });
  } else {
    toast(`「${job.name}」去噪完成`, { action: { label: '查看', fn: () => select(S.jobs.find(x => x.id === job.id)) } });
  }
}
setInterval(() => { if (S.current && isRunning(S.current)) renderViewer(); }, 1000);

async function addFiles(list) {
  const files = [...list].filter(f => /^image\//.test(f.type) || /\.(png|jpe?g|webp|bmp)$/i.test(f.name));
  if (!files.length) { toast('請加入圖片檔（PNG、JPEG、WebP、BMP）。', { type: 'error' }); return; }
  let first = null, added = 0;
  for (const file of files) {
    if (file.size > MAX_BYTES) { toast(`「${file.name}」超過 32 MB 上限。`, { type: 'error' }); continue; }
    const name = file.name && file.name !== 'image.png' ? file.name : `貼上的圖片 ${new Date().toLocaleTimeString('zh-TW', { hour12: false })}.png`;
    try {
      const job = await api('/api/upload', { method: 'POST', body: file, headers: { 'X-Filename': encodeURIComponent(name) } });
      S.jobs.unshift(job);
      added++;
      if (!first) { first = job; select(job); } else renderStrip();
    } catch (e) {
      toast(`「${name}」${e.message}`, { type: 'error' });
    }
  }
  if (added > 1) toast(`已加入 ${added} 張圖片，可在下方縮圖切換（← →）。`);
}
function pickFiles() { $('#file').click(); }
$('#file').onchange = e => { if (e.target.files.length) addFiles(e.target.files); e.target.value = ''; };
$('#add-image').onclick = $('#strip-add').onclick = $('#empty-pick').onclick = pickFiles;
$('#empty').onclick = e => { if (!e.target.closest('button')) pickFiles(); };

async function removeJob(id, button) {
  if (!button.classList.contains('confirm')) {
    button.classList.add('confirm');
    button.textContent = '移除？';
    setTimeout(() => { button.classList.remove('confirm'); button.textContent = '×'; }, 3000);
    return;
  }
  try {
    if (M.jobId === id) { M.dirty = false; clearTimeout(saveTimer); }
    pendingSaves.delete(id);
    await post('/api/delete', { id });
    drafts.delete(id);
    const index = S.jobs.findIndex(j => j.id === id);
    S.jobs.splice(index, 1);
    if (S.current && S.current.id === id) select(S.jobs[Math.min(index, S.jobs.length - 1)] || null);
    renderAll();
  } catch (e) {
    toast(e.message, { type: 'error' });
  }
}

async function download() {
  if (!hasResult(S.current)) return;
  await flushMask();
  const j = S.current;
  const a = document.createElement('a');
  a.href = `/image/${j.id}/result?download=1&v=${encodeURIComponent(j.output + (j.mask_version || ''))}`;
  a.download = j.name.replace(/\.[^.]+$/, '') + '_denoise.png';
  document.body.append(a); a.click(); a.remove();
}
$('#download').onclick = download;
$('#reveal').onclick = async () => {
  await flushMask();
  try { await post('/api/reveal', { id: S.current.id }); } catch (e) { toast(e.message, { type: 'error' }); }
};
$('#open-folder').onclick = async () => {
  try { await post('/api/reveal', {}); } catch (e) { toast(e.message, { type: 'error' }); }
};
$('#help-btn').onclick = () => $('#help').showModal();
$('#quit').onclick = async () => {
  if (S.activeId) { toast('圖片還在處理，完成後才能結束工具。', { type: 'error' }); return; }
  if (!confirm('結束工具？之後要重新啟動才能使用。')) return;
  try {
    await flushMask();
    await post('/api/shutdown', {});
    document.body.innerHTML = '<p style="margin:30vh auto;text-align:center;font-size:17px;color:var(--muted)">工具已結束，可以關閉這個分頁。</p>';
  } catch (e) {
    toast(e.message, { type: 'error' });
  }
};
$('#fit').onclick = fit;
$('#actual').onclick = () => zoomTo(1);
$('#zoom-in').onclick = () => zoomTo(S.scale * 1.25);
$('#zoom-out').onclick = () => zoomTo(S.scale / 1.25);

/* ================================================================ pointer input on the viewer */

let gesture = null, lastPointer = null;
const cursor = $('#brush-cursor');
function updateCursor(e) {
  const show = S.current && S.tool !== 'pan' && !S.spaceHeld && canPaint() && !(gesture && gesture.type === 'pan');
  if (!show || !e) { cursor.hidden = true; return; }
  const p = localPoint(e), d = Math.max(4, brushSize() * S.scale);
  cursor.hidden = false;
  cursor.style.width = cursor.style.height = d + 'px';
  cursor.style.left = p.x + 'px';
  cursor.style.top = p.y + 'px';
  cursor.classList.toggle('erase', S.tool === 'erase' || !!(gesture && gesture.erase));
}

viewport.addEventListener('pointerdown', e => {
  if (!S.current || e.target.closest('#brushbar, #busy')) return;
  const onDivider = !!e.target.closest('#divider');
  let type;
  if (onDivider && e.button === 0) type = 'divider';
  else if (e.button === 1 || S.spaceHeld || (S.tool === 'pan' && e.button === 0)) type = 'pan';
  else if (S.tool !== 'pan' && (e.button === 0 || e.button === 2) && canPaint()) type = 'stroke';
  else return;
  e.preventDefault();
  viewport.setPointerCapture(e.pointerId);
  gesture = { type, id: e.pointerId, x: e.clientX, y: e.clientY, tx: S.tx, ty: S.ty };
  if (type === 'pan') viewport.classList.add('panning');
  if (type === 'divider') moveDivider(e);
  if (type === 'stroke') {
    const erase = S.tool === 'erase' || e.button === 2;
    const p = imagePoint(e);
    M.stroke = { size: brushSize(), erase, points: [p] };
    M.actions.push(M.stroke);
    M.redo = [];
    gesture.erase = erase;
    stamp(M.ctx, p.x, p.y, M.stroke.size, erase);
    const r = M.stroke.size / 2 + 2;
    invalidate(p.x - r, p.y - r, r * 2, r * 2);
  }
  updateCursor(e);
});
viewport.addEventListener('pointermove', e => {
  lastPointer = e;
  updateCursor(e);
  if (!gesture || gesture.id !== e.pointerId) return;
  if (gesture.type === 'pan') {
    S.tx = gesture.tx + e.clientX - gesture.x;
    S.ty = gesture.ty + e.clientY - gesture.y;
    S.autoFit = false;
    renderViewer();
  } else if (gesture.type === 'divider') {
    moveDivider(e);
  } else if (gesture.type === 'stroke' && M.stroke) {
    const events = e.getCoalescedEvents ? e.getCoalescedEvents() : [];
    for (const ev of events.length ? events : [e]) {
      const a = M.stroke.points.at(-1), b = imagePoint(ev);
      segment(M.ctx, a, b, M.stroke.size, M.stroke.erase);
      M.stroke.points.push(b);
      const r = M.stroke.size / 2 + 2;
      invalidate(Math.min(a.x, b.x) - r, Math.min(a.y, b.y) - r, Math.abs(a.x - b.x) + r * 2, Math.abs(a.y - b.y) + r * 2);
    }
  }
});
function endGesture(e) {
  if (!gesture || (e && gesture.id !== e.pointerId)) return;
  if (gesture.type === 'stroke' && M.stroke) { M.stroke = null; maskChanged(); renderViewer(); }
  viewport.classList.remove('panning');
  gesture = null;
}
viewport.addEventListener('pointerup', endGesture);
viewport.addEventListener('pointercancel', endGesture);
viewport.addEventListener('lostpointercapture', endGesture);
viewport.addEventListener('pointerleave', () => { if (!gesture) cursor.hidden = true; });
viewport.addEventListener('contextmenu', e => { if (S.current) e.preventDefault(); });
viewport.addEventListener('dblclick', e => {
  if (S.current && S.tool === 'pan' && !e.target.closest('#divider, #brushbar, #busy')) fit();
});
viewport.addEventListener('wheel', e => {
  if (!S.current) return;
  e.preventDefault();
  const p = localPoint(e);
  const dy = e.deltaMode === 1 ? e.deltaY * 30 : e.deltaY;
  zoomTo(S.scale * Math.exp(-dy * 0.0015), p.x, p.y);
  updateCursor(e);
}, { passive: false });

function moveDivider(e) {
  const j = S.current, p = localPoint(e);
  S.split = clamp((p.x - S.tx) / (j.width * S.scale), 0, 1);
  renderViewer();
}
$('#divider').addEventListener('keydown', e => {
  const stepSize = e.shiftKey ? 0.1 : 0.02;
  if (e.key === 'ArrowLeft') S.split = clamp(S.split - stepSize, 0, 1);
  else if (e.key === 'ArrowRight') S.split = clamp(S.split + stepSize, 0, 1);
  else if (e.key === 'Home') S.split = 0;
  else if (e.key === 'End') S.split = 1;
  else return;
  e.preventDefault();
  renderViewer();
});

new ResizeObserver(() => { if (S.autoFit) fit(); else renderViewer(); }).observe(viewport);

/* ================================================================ keyboard, paste, drop */

function setPeek(on) {
  if (S.peek === on) return;
  S.peek = on;
  renderToolbar();
  renderViewer();
}
document.addEventListener('keydown', e => {
  if ($('#help').open || e.defaultPrevented) return;
  const el = e.target instanceof Element ? e.target : document.body;
  const typing = el.matches('input[type=text], input[type=number], textarea');
  const ctrl = e.ctrlKey || e.metaKey;
  const key = e.key.toLowerCase();
  if (ctrl) {
    if (key === 'o') { e.preventDefault(); pickFiles(); }
    else if (key === 'enter') { e.preventDefault(); run(); }
    else if (key === 's') { e.preventDefault(); download(); }
    else if (key === 'z' && !typing && S.current) { e.preventDefault(); e.shiftKey ? redo() : undo(); }
    else if (key === 'y' && !typing && S.current) { e.preventDefault(); redo(); }
    return;
  }
  if (e.altKey || typing) return;
  if (e.key === ' ') {
    if (S.current && S.tool !== 'pan') {
      e.preventDefault();
      if (!S.spaceHeld) { S.spaceHeld = true; cursor.hidden = true; renderViewer(); }
    }
    return;
  }
  if (el.matches('input[type=range]') && /^(arrow|home|end|page)/.test(key)) return;
  const actions = {
    '1': () => setView('compare'), '2': () => setView('original'), '3': () => setView('result'),
    z: () => { if (!e.repeat) setPeek(true); },
    arrowleft: () => step(-1), arrowright: () => step(1),
    v: () => setTool('pan'), escape: () => setTool('pan'), b: () => setTool('paint'), e: () => setTool('erase'),
    '[': () => { $('#brush-size').value = +$('#brush-size').value - 5; updateBrushLabel(); updateCursor(lastPointer); },
    ']': () => { $('#brush-size').value = +$('#brush-size').value + 5; updateBrushLabel(); updateCursor(lastPointer); },
    m: toggleMaskVisible,
    f: fit, '0': () => zoomTo(1),
    '+': () => zoomTo(S.scale * 1.25), '=': () => zoomTo(S.scale * 1.25), '-': () => zoomTo(S.scale / 1.25),
    '?': () => $('#help').showModal(),
  };
  const fn = actions[key];
  if (fn && (S.current || key === '?')) { e.preventDefault(); fn(); }
});
document.addEventListener('keyup', e => {
  if (e.key === ' ' && S.spaceHeld) { S.spaceHeld = false; renderViewer(); updateCursor(lastPointer); }
  if (e.key.toLowerCase() === 'z') setPeek(false);
});
window.addEventListener('blur', () => { S.spaceHeld = false; setPeek(false); endGesture(); cursor.hidden = true; hideDrop(); });

document.addEventListener('paste', e => {
  const files = [...(e.clipboardData?.files || [])];
  if (files.length) { e.preventDefault(); addFiles(files); }
});

let dragDepth = 0;
const hasFiles = e => [...(e.dataTransfer?.types || [])].includes('Files');
function hideDrop() { dragDepth = 0; $('#drop-overlay').hidden = true; }
document.addEventListener('dragenter', e => { if (!hasFiles(e)) return; e.preventDefault(); dragDepth++; $('#drop-overlay').hidden = false; });
document.addEventListener('dragover', e => { if (hasFiles(e)) e.preventDefault(); });
document.addEventListener('dragleave', () => { if (--dragDepth <= 0) hideDrop(); });
document.addEventListener('drop', e => { e.preventDefault(); hideDrop(); if (e.dataTransfer.files.length) addFiles(e.dataTransfer.files); });

window.addEventListener('beforeunload', e => {
  if (savePending()) { flushMask(); e.preventDefault(); e.returnValue = ''; }
});

/* ================================================================ start */

function staleServerBanner() {
  const b = document.createElement('div');
  b.className = 'banner';
  b.textContent = '本機服務還是舊版本，部分功能無法使用。請按右上「結束」，再重新啟動工具。';
  document.body.insertBefore(b, $('.layout'));
}

updateBrushLabel();
setParams(recall('ad.params') || PRESETS.recommended);
api('/api/state').then(data => {
  if (data.api !== API_VERSION) staleServerBanner();
  S.jobs = data.jobs;
  S.activeId = data.active;
  const last = recall('ad.selected');
  const job = S.jobs.find(j => j.id === data.active) || S.jobs.find(j => j.id === last) || S.jobs[0];
  if (job) select(job); else renderAll();
  if (S.activeId) poll();
}).catch(e => { toast(e.message, { type: 'error' }); renderAll(); });
