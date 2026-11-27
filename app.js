import { PointViewer } from './viewer.js';

const el = (id) => document.getElementById(id);
const repository = 'https://github.com/temesotejam/mast3r-photo-to-3d';
const isPages = location.hostname.endsWith('.github.io') || location.protocol === 'file:';
const isCodespace = location.hostname.endsWith('.app.github.dev');
const apiBase = isPages ? 'http://127.0.0.1:7861' : location.origin;
const storageKey = 'mast3r:job:' + apiBase;
const MiB = 1024 * 1024;
const files = [];
let busy = false;
let viewing = false;
let jobId = null;
let downloadUrl = '';
let remoteLinkNeeded = false;
const viewer = new PointViewer(el('cloud'));

function message(id, text, isError = false) {
  const target = el(id);
  target.textContent = text;
  target.hidden = !text;
  target.classList.toggle('error', isError);
}

function showControls() {
  el('input-title').textContent = '入力写真';
  el('launch').hidden = true;
  el('controls').hidden = false;
  el('counter').hidden = false;
}

if (isPages) {
  el('input-title').textContent = '復元を開始';
  el('controls').hidden = true;
  el('counter').hidden = true;
  el('launch').hidden = false;
  el('use-loopback').addEventListener('click', showControls);
} else if (isCodespace) {
  el('engine-name').textContent = 'このCodespaceのMASt3R';
  el('engine-note').textContent = '選択した写真をこのCodespaceへ送信して復元します。';
} else {
  el('engine-note').textContent = 'この画面を配信しているサーバーで復元します。';
}
el('repo-link').href = repository;
el('setup').href = repository + '#readme';

function syncButtons() {
  el('run').disabled = busy || files.length < 2;
  el('run').classList.toggle('busy', busy);
  el('run').querySelector('span').textContent = busy ? '処理中' : '3Dを復元';
  el('pick').disabled = busy;
  el('photos').disabled = busy;
  el('open').disabled = busy || viewing;
  el('sample').disabled = busy || viewing;
  for (const button of el('thumbs').querySelectorAll('button')) button.disabled = busy;
}

function renderFiles() {
  const nodes = files.map(({ file, preview }, idx) => {
    const item = document.createElement('li');
    const image = document.createElement('img');
    image.src = preview;
    image.alt = file.name;
    item.title = file.name;
    const remove = document.createElement('button');
    remove.type = 'button';
    remove.className = 'remove';
    remove.textContent = '×';
    remove.setAttribute('aria-label', file.name + ' を外す');
    remove.addEventListener('click', () => {
      URL.revokeObjectURL(files[idx].preview);
      files.splice(idx, 1);
      message('notice', '');
      renderFiles();
    });
    item.append(image, remove);
    return item;
  });
  el('thumbs').replaceChildren(...nodes);
  el('counter').textContent = files.length + ' / 10';
  syncButtons();
}

function addFiles(list) {
  if (busy) return;
  const errors = new Set();
  let total = files.reduce((sum, entry) => sum + entry.file.size, 0);
  for (const file of list) {
    const duplicate = files.some(entry => entry.file.name === file.name && entry.file.size === file.size && entry.file.lastModified === file.lastModified);
    if (duplicate) continue;
    if (!/\.(jpe?g|png)$/i.test(file.name)) { errors.add('JPG・PNGの写真を選んでください。'); continue; }
    if (!file.size || file.size > 12 * MiB) { errors.add('各写真は12 MiB以下にしてください。'); continue; }
    if (files.length >= 10) { errors.add('選択できる写真は10枚までです。'); break; }
    if (total + file.size >= 60 * MiB) { errors.add('写真の合計を60 MiB未満にしてください。'); continue; }
    total += file.size;
    files.push({ file, preview: URL.createObjectURL(file) });
  }
  message('notice', [...errors].join(' '));
  renderFiles();
}

el('pick').addEventListener('click', () => el('photos').click());
el('photos').addEventListener('change', (event) => {
  addFiles(event.target.files);
  event.target.value = '';
});
for (const type of ['dragenter', 'dragover', 'dragleave', 'drop']) {
  el('pick').addEventListener(type, (e) => {
    e.preventDefault();
    el('pick').classList.toggle('drag', type === 'dragenter' || type === 'dragover');
    if (type === 'drop') addFiles(e.dataTransfer.files);
  });
}

function saveJob(id) {
  jobId = id;
  try {
    if (id) sessionStorage.setItem(storageKey, id);
    else sessionStorage.removeItem(storageKey);
  } catch { /* Private windows may disable session storage. */ }
}

class ApiError extends Error {
  constructor(text, retryable = false, status = 0) {
    super(text);
    this.retryable = retryable;
    this.status = status;
  }
}

async function request(path, init = {}, timeout = 45000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeout);
  let response;
  try {
    response = await fetch(apiBase + path, { ...init, signal: controller.signal, cache: 'no-store' });
  } catch {
    remoteLinkNeeded = isPages;
    throw new ApiError(isPages ? 'このPCのサーバーに接続できません。サーバーを起動し、ローカルの操作画面を開いてください。' :
      '復元サーバーに接続できません。接続状態とサーバーの起動状況を確認してください。', true);
  } finally { clearTimeout(timer); }
  if (!response.ok) {
    const detail = await response.json().catch(() => ({}));
    throw new ApiError(detail.message || 'サーバーからエラーが返りました (HTTP ' + response.status + ')。', response.status >= 500, response.status);
  }
  return response;
}

async function json(response) {
  try { return await response.json(); }
  catch { throw new ApiError('サーバーの応答を読み取れません。復元サーバーから操作画面を開いてください。'); }
}

function setDownload(blob, filename) {
  if (downloadUrl) URL.revokeObjectURL(downloadUrl);
  downloadUrl = URL.createObjectURL(blob);
  el('save').href = downloadUrl;
  el('save').download = filename;
  el('save').hidden = false;
}

async function displayCloud(blob, filename, fromBackend = false) {
  viewing = true;
  syncButtons();
  message('view-message', '');
  try {
    // The user can still save a finished reconstruction if GPU rendering is unavailable.
    if (fromBackend) setDownload(blob, filename);
    const { count } = await viewer.show(blob);
    if (!fromBackend) setDownload(blob, filename);
    el('point-count').textContent = count.toLocaleString('ja-JP') + ' 点';
    el('cloud-name').textContent = filename;
    el('cloud-name').hidden = false;
    el('empty').hidden = true;
    el('fit').hidden = false;
    el('view').classList.add('ready');
  } catch (error) {
    message('view-message', '表示できません。' + error.message);
  } finally {
    viewing = false;
    syncButtons();
  }
}

el('fit').addEventListener('click', () => viewer.resetView());
el('open').addEventListener('click', () => el('ply').click());
el('ply').addEventListener('change', async (e) => {
  const file = e.target.files[0];
  e.target.value = '';
  if (file) await displayCloud(file, file.name);
});
if (el('sample').dataset.path) {
  el('sample').hidden = false;
  el('sample').addEventListener('click', async () => {
    el('sample').disabled = true;
    try {
      const response = await fetch(el('sample').dataset.path);
      if (!response.ok) throw new Error();
      await displayCloud(await response.blob(), 'sample.ply');
    } catch { message('view-message', 'サンプル点群を読み込めませんでした。'); }
    finally { syncButtons(); }
  });
}

const pause = ms => new Promise(resolve => setTimeout(resolve, ms));
async function pollJob(id) {
  const path = '/api/jobs/' + encodeURIComponent(id);
  let failures = 0;
  while (true) {
    let result;
    try {
      result = await json(await request(path));
      failures = 0;
    } catch (error) {
      if (error.status === 404) saveJob(null);
      if (!error.retryable || ++failures > 3) throw error;
      message('job-state', '接続を再確認しています…');
      await pause(2500 * failures);
      continue;
    }
    if (result.status === 'error') {
      saveJob(null);
      throw new ApiError(result.message || '復元に失敗しました。');
    }
    if (result.status === 'done') {
      message('job-state', '復元完了。点群を読み込んでいます…');
      let response;
      try {
        response = await request(path + '/cloud.ply', {}, 120000);
      } catch (error) {
        if (error.status === 404) saveJob(null);
        throw error;
      }
      await displayCloud(await response.blob(), 'reconstruction.ply', true);
      saveJob(null);
      message('job-state', '復元が完了しました。');
      return;
    }
    if (result.status !== 'running' && result.status !== 'queued') throw new ApiError('ジョブの状態を読み取れませんでした。');
    message('job-state', result.message || (result.status === 'queued' ? '計算の開始を待っています…' : '写真から点群を復元しています…'));
    await pause(2000);
  }
}

async function run(resume = false) {
  if (busy || (!resume && files.length < 2)) return;
  busy = true;
  remoteLinkNeeded = false;
  el('resume').hidden = true;
  el('local-link').hidden = true;
  syncButtons();
  try {
    if (!resume) {
      message('job-state', '復元サーバーに接続しています…');
      const health = await json(await request('/api/health'));
      if (health.engine !== 'mast3r') throw new ApiError('MASt3Rの復元サーバーに接続してください。');
      if (health.ready === false) throw new ApiError(health.message || 'サーバーがモデルを準備中です。');
      message('job-state', files.length + '枚の写真を送信しています…');
      const data = new FormData();
      for (const entry of files) data.append('photos', entry.file, entry.file.name);
      const newJob = await json(await request('/api/jobs', { method: 'POST', body: data }, 180000));
      if (typeof newJob.id !== 'string' || !newJob.id) throw new ApiError('ジョブIDを取得できませんでした。');
      saveJob(newJob.id);
    }
    await pollJob(jobId);
  } catch (error) {
    message('job-state', error.message || '復元に失敗しました。', true);
    el('resume').hidden = !jobId;
    el('local-link').hidden = !remoteLinkNeeded;
  } finally {
    busy = false;
    syncButtons();
  }
}

el('run').addEventListener('click', () => run());
el('resume').addEventListener('click', () => run(true));
try { jobId = sessionStorage.getItem(storageKey); } catch { /* Optional persistence. */ }
if (jobId && /^[A-Za-z0-9_-]{1,128}$/.test(jobId)) {
  showControls();
  message('job-state', '前回の復元ジョブの状態を確認できます。');
  el('resume').hidden = false;
}
