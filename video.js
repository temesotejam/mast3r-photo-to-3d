import { MotionViewer } from './motion-viewer.js';

const get = id => document.getElementById(id);
const source = get('source');
const pages = location.protocol === 'file:' || location.hostname.endsWith('.github.io');
const viewer = new MotionViewer(get('motion'));
const statusLabels = { observed: '観測', interpolated: '補間', unavailable: '追跡不可' };
let activeScene = null;
let times = [];
let currentFrame = -1;
let playing = false;
let startTime = 0;
let playbackTimer = 0;
let resultUrl = '';
let referenceUrl = '';
let sceneGeneration = 0;
let mediaAvailable = false;

function tell(id, text, error = false) {
  const label = get(id);
  label.textContent = text;
  label.classList.toggle('error', error);
}

function timestamps(scene) {
  if (Array.isArray(scene.times) && scene.times.length === scene.frame_count &&
      scene.times.every((t, i, list) => Number.isFinite(t) && t >= 0 && (!i || t > list[i - 1]))) {
    return scene.times;
  }
  if (!Number.isFinite(scene.fps) || scene.fps <= 0 || !Number.isSafeInteger(scene.frame_count) ||
      scene.frame_count <= 0 || scene.frame_count > 100000) throw new Error('フレーム時刻を読み取れません。');
  return Array.from({length:scene.frame_count}, (_, i) => i / scene.fps);
}

function tick(frame) {
  if (!activeScene || currentFrame === frame) return;
  currentFrame = frame;
  viewer.setFrame(frame);
  get('timeline').value = frame;
  get('frame').textContent = (frame + 1) + ' / ' + times.length;
  get('time').textContent = times[frame].toFixed(2) + ' 秒';
  const labels = activeScene.parts.map(part => {
    const state = part.frames[frame]?.status || 'unavailable';
    const text = document.createElement('span');
    text.className = 'part-status ' + state;
    text.append(document.createElement('i'), part.name + ' · ' + (statusLabels[state] || statusLabels.unavailable));
    return text;
  });
  get('parts-status').replaceChildren(...labels);
}

function nearestFrame(seconds) {
  let low = 0, high = times.length - 1;
  while (low < high) {
    const middle = (low + high) >> 1;
    const cutoff = (times[middle] + times[middle + 1]) / 2;
    if (seconds < cutoff) high = middle;
    else low = middle + 1;
  }
  return low;
}

function stop() {
  playing = false;
  cancelAnimationFrame(playbackTimer);
  source.pause();
  get('play').textContent = '再生';
}

function playClock() {
  if (!activeScene || !playing) return;
  const seconds = mediaAvailable ? source.currentTime : (performance.now() - startTime) / 1000;
  tick(nearestFrame(seconds));
  const step = times.length > 1 ? times.at(-1) - times.at(-2) : .25;
  if (!mediaAvailable && seconds >= times.at(-1) + step) {
    stop();
    return;
  }
  playbackTimer = requestAnimationFrame(playClock);
}

function startClock() {
  if (playing || !activeScene) return;
  playing = true;
  startTime = performance.now() - 1000 * times[Math.max(0, currentFrame)];
  get('play').textContent = '一時停止';
  playClock();
}

function seek(frame) {
  tick(frame);
  if (mediaAvailable) source.currentTime = times[frame];
  else if (playing) startTime = performance.now() - times[frame] * 1000;
}

get('timeline').addEventListener('input', () => seek(Number(get('timeline').value)));
get('play').addEventListener('click', async () => {
  if (playing) { stop(); return; }
  if (currentFrame === times.length - 1) seek(0);
  if (mediaAvailable) {
    try { await source.play(); } catch { tell('scene-msg', '元動画の再生を開始できません。動画の再生ボタン、または下のスライダーを使ってください。', true); }
  } else startClock();
});
source.addEventListener('play', startClock);
source.addEventListener('pause', () => { if (mediaAvailable && playing) stop(); });
source.addEventListener('seeked', () => { if (mediaAvailable && activeScene) tick(nearestFrame(source.currentTime)); });
source.addEventListener('ended', () => { if (activeScene) tick(times.length - 1); stop(); });
source.addEventListener('error', () => {
  if (!referenceUrl || !source.getAttribute('src')) return;
  mediaAvailable = false;
  stop();
  get('clip-empty').hidden = false;
  get('clip-empty').textContent = 'この動画形式をブラウザで再生できません';
  source.hidden = true;
});
get('fit-view').addEventListener('click', () => viewer.resetView());

async function loadScene(blob, clip, title, generation) {
  if (blob.size > 128 * 1024 * 1024) throw new Error('JSONは128 MiB以下にしてください。');
  const scene = JSON.parse(await blob.text());
  if (generation !== sceneGeneration) return;
  const stamps = timestamps(scene);
  viewer.load(scene);
  stop();
  activeScene = scene;
  times = stamps;
  currentFrame = -1;
  if (referenceUrl?.startsWith('blob:')) URL.revokeObjectURL(referenceUrl);
  source.removeAttribute('src');
  referenceUrl = clip instanceof Blob ? URL.createObjectURL(clip) : (clip || '');
  mediaAvailable = !!referenceUrl;
  source.hidden = !mediaAvailable;
  get('clip-empty').textContent = '対応する動画はありません';
  get('clip-empty').hidden = mediaAvailable;
  if (mediaAvailable) source.src = referenceUrl;
  source.load();
  if (resultUrl) URL.revokeObjectURL(resultUrl);
  resultUrl = URL.createObjectURL(blob);
  get('save').href = resultUrl;
  get('save').hidden = false;
  get('fit-view').disabled = false;
  get('play').disabled = times.length < 2;
  get('timeline').disabled = times.length < 2;
  get('timeline').max = times.length - 1;
  tick(0);
  const averageRate = times.length > 1 ? (times.length - 1) / (times.at(-1) - times[0]) : null;
  const details = [title];
  if (averageRate) details.push('平均 ' + averageRate.toFixed(1) + ' コマ/秒');
  const focalLabels = {horizontal_fov: '指定値', rigid_plane_tracks: '部品の動きから推定', mast3r: 'MASt3R推定'};
  if (Number.isFinite(scene.horizontal_fov_deg) && Object.hasOwn(focalLabels, scene.intrinsics_source)) {
    details.push('水平画角 ' + scene.horizontal_fov_deg.toFixed(1) + '°（' + focalLabels[scene.intrinsics_source] + '）');
  }
  tell('scene-msg', details.join(' · '));
}

async function sample() {
  const generation = ++sceneGeneration;
  tell('scene-msg', 'サンプルを読み込んでいます…');
  try {
    const response = await fetch('./video-example/scene.json');
    if (!response.ok) throw new Error();
    await loadScene(await response.blob(), './video-example/video.mp4', 'サンプルの復元結果', generation);
    if (generation === sceneGeneration && !window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      source.play().catch(() => {});
    }
  } catch {
    if (generation === sceneGeneration) tell('scene-msg', 'サンプルを読み込めませんでした。「サンプルを開く」で再試行できます。', true);
  }
}

get('sample').addEventListener('click', sample);
get('open-scene').addEventListener('click', () => get('scene-file').click());
get('scene-file').addEventListener('change', async e => {
  const file = e.target.files[0]; e.target.value = '';
  if (!file) return;
  const generation = ++sceneGeneration;
  try { await loadScene(file, null, file.name, generation); }
  catch (error) { if (generation === sceneGeneration) tell('scene-msg', 'JSONを開けません。' + error.message, true); }
});

const capture = get('first-frame');
const canvas = get('boxes');
const brush = canvas.getContext('2d');
const colors = ['#86e0a1', '#ffbf6f', '#75d7f2', '#f5a8dc', '#e5de7b', '#c3b4fa'];
let videoFile = null;
let captureUrl = '';
let boxes = [];
let drag = null;
let busy = false;
let jobId = null;
let submittedFile = null;
const persistKey = 'mast3r-video-job';
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));

function sync() {
  const useWhole = get('whole').checked;
  get('run-video').disabled = busy || !videoFile || get('editor').hidden || (!boxes.length && !useWhole) ||
    (get('camera').value === 'moving' && useWhole);
  get('pick-video').disabled = get('sampling').disabled = get('camera').disabled = busy;
  get('hfov').disabled = busy;
  get('whole').disabled = busy || get('camera').value === 'moving';
  get('pick-video').textContent = videoFile ? '動画を変更' : '動画を選ぶ';
  get('run-video').textContent = busy ? '処理中…' : '3D動作を復元';
  get('part-list').querySelectorAll('input, button').forEach(x => { x.disabled = busy || useWhole; });
}

function area(rect) {
  return [Math.min(rect[0], rect[2]), Math.min(rect[1], rect[3]), Math.max(rect[0], rect[2]), Math.max(rect[1], rect[3])];
}

function drawReference() {
  if (!videoFile || capture.readyState < 2) return;
  brush.drawImage(capture, 0, 0, canvas.width, canvas.height);
  const rects = get('whole').checked ? [[0, 0, 1, 1]] : boxes.map(p => p.box);
  if (drag) rects.push(area(drag.rect));
  const width = Math.max(2, canvas.width / 290);
  for (const [i, box] of rects.entries()) {
    const [x0,y0,x1,y1] = box;
    brush.lineWidth = width;
    brush.strokeStyle = colors[i % colors.length];
    brush.strokeRect(x0 * canvas.width, y0 * canvas.height, (x1 - x0) * canvas.width, (y1 - y0) * canvas.height);
    brush.font = 'bold ' + (width * 4.8) + 'px sans-serif';
    brush.fillStyle = colors[i % colors.length];
    brush.fillText(String(i + 1), x0 * canvas.width + width, Math.min(canvas.height - width, y0 * canvas.height + width * 5.5));
  }
}

function renderBoxes() {
  const rows = boxes.map((part, i) => {
    const row = document.createElement('div'); row.className = 'part-row';
    const num = document.createElement('span'); num.textContent = i + 1; num.className = 'part-tag';
    const name = document.createElement('input'); name.value = part.name; name.maxLength = 48;
    name.setAttribute('aria-label', '部品' + (i + 1) + 'の名前');
    name.addEventListener('input', () => { part.name = name.value; });
    const remove = document.createElement('button'); remove.textContent = '削除'; remove.className = 'subbutton';
    remove.type = 'button';
    remove.addEventListener('click', () => { boxes.splice(i, 1); renderBoxes(); });
    row.append(num,name,remove); return row;
  });
  get('part-list').replaceChildren(...rows);
  drawReference();
  sync();
}

function pointer(e) {
  const r = canvas.getBoundingClientRect();
  return [Math.max(0, Math.min(1,(e.clientX - r.left) / r.width)), Math.max(0, Math.min(1,(e.clientY - r.top) / r.height))];
}

canvas.addEventListener('pointerdown', e => {
  if (busy || drag || get('whole').checked || e.button !== 0) return;
  if (boxes.length >= 6) { tell('job-msg', '指定できる部品は6個までです。', true); return; }
  const pt = pointer(e);
  drag = { id:e.pointerId, rect:[...pt, ...pt] };
  canvas.setPointerCapture(e.pointerId);
});
canvas.addEventListener('pointermove', e => {
  if (drag?.id !== e.pointerId) return;
  drag.rect.splice(2,2,...pointer(e));
  drawReference();
});
canvas.addEventListener('pointerup', e => {
  if (drag?.id !== e.pointerId) return;
  const rect = area([...drag.rect.slice(0,2), ...pointer(e)]);
  drag = null;
  if (Math.round((rect[2]-rect[0])*capture.videoWidth) >= 8 &&
      Math.round((rect[3]-rect[1])*capture.videoHeight) >= 8) {
    let n = boxes.length + 1; while (boxes.some(p => p.name === '部品 ' + n)) n++;
    boxes.push({name: '部品 ' + n, box: rect});
    tell('job-msg', '');
  } else tell('job-msg', '部品の全体を囲むように、もう少し大きくドラッグしてください。', true);
  renderBoxes();
});
canvas.addEventListener('pointercancel', () => { drag = null; drawReference(); });
get('whole').addEventListener('change', renderBoxes);
get('camera').addEventListener('change', () => {
  const moving = get('camera').value === 'moving';
  get('camera-note').hidden = !moving;
  if (moving) get('whole').checked = false;
  renderBoxes();
});

function clearClip() {
  capture.removeAttribute('src');
  capture.load();
  if (captureUrl) URL.revokeObjectURL(captureUrl);
  captureUrl = ''; videoFile = null; drag = null; boxes = []; get('whole').checked = false;
  get('editor').hidden = true;
  get('filename').textContent = '未選択';
  renderBoxes();
}

get('pick-video').addEventListener('click', () => get('video-file').click());
get('video-file').addEventListener('change', e => {
  const file = e.target.files[0]; e.target.value = '';
  if (!file || busy) return;
  clearClip();
  if (!/\.(mp4|mov|webm)$/i.test(file.name) || !file.size || file.size + 16384 >= 60 * 1024 * 1024) {
    tell('job-msg','60 MiB未満のMP4・MOV・WebMを選んでください。',true); return;
  }
  videoFile = file;
  captureUrl = URL.createObjectURL(file);
  capture.src = captureUrl;
  get('filename').textContent = file.name;
  tell('job-msg','最初のフレームを読み込んでいます…');
});
capture.addEventListener('loadeddata', () => {
  if (!videoFile || capture.currentSrc !== captureUrl) return;
  if (!Number.isFinite(capture.duration) || capture.duration <= 0 || capture.duration > 30.05) {
    clearClip(); tell('job-msg','長さを読み取れる30秒以内の動画を選んでください。', true); return;
  }
  const factor = Math.min(1, 1000 / Math.max(capture.videoWidth, capture.videoHeight));
  canvas.width = Math.max(1, Math.round(capture.videoWidth * factor));
  canvas.height = Math.max(1, Math.round(capture.videoHeight * factor));
  get('editor').hidden = false;
  tell('job-msg','');
  renderBoxes();
});
capture.addEventListener('error', () => {
  if (!captureUrl || !capture.getAttribute('src')) return;
  clearClip();
  tell('job-msg','この動画をブラウザで読み取れません。H.264形式のMP4で試してください。', true);
});

class JobError extends Error {
  constructor(msg, status = 0) { super(msg); this.status = status; }
  get temporary() { return this.status === 0 || this.status >= 500; }
}

async function api(path, init = {}, timeoutMs = 45000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const r = await fetch(path, {...init, signal:controller.signal, cache:'no-store'});
    if (!r.ok) {
      const body = await r.json().catch(() => ({}));
      throw new JobError(body.message || '復元サーバーのエラー (HTTP ' + r.status + ')。', r.status);
    }
    return r;
  } catch (error) {
    if (error instanceof JobError) throw error;
    throw new JobError('復元サーバーに接続できません。ネットワークとサーバーを確認してください。');
  } finally { clearTimeout(timer); }
}

function remember(id) {
  jobId = id;
  try {
    if (id) sessionStorage.setItem(persistKey, id);
    else sessionStorage.removeItem(persistKey);
  } catch { /* Browser storage is optional. */ }
}

async function poll() {
  const path = '/api/video-jobs/' + encodeURIComponent(jobId);
  let failures = 0;
  for (;;) {
    let status;
    try {
      status = await (await api(path)).json();
      failures = 0;
    } catch (error) {
      if (!error.temporary || ++failures > 3) throw error;
      tell('job-msg', '接続を再確認しています…');
      await sleep(1500 * failures);
      continue;
    }
    if (status.status === 'error') {
      remember(null);
      throw new JobError(status.message || '復元に失敗しました。', 400);
    }
    if (status.status === 'done') {
      const data = await (await api(path + '/scene.json', {}, 120000)).blob();
      await loadScene(data, submittedFile, '動画の復元結果', ++sceneGeneration);
      remember(null);
      tell('job-msg','復元が完了しました。');
      get('scene-title').scrollIntoView({block:'start'});
      return;
    }
    if (!['queued', 'running'].includes(status.status)) throw new JobError('ジョブ状態を読み取れません。', 400);
    tell('job-msg', status.message || '動画を処理しています…');
    await sleep(2000);
  }
}

async function run(resume = false) {
  if (pages || busy || (!resume && get('run-video').disabled)) return;
  if (!resume && !get('hfov').reportValidity()) return;
  busy = true; sync(); get('retry-job').hidden = true;
  try {
    if (!resume) {
      const parts = get('whole').checked ? [{name:'画面全体', box:[0,0,1,1]}] :
        boxes.map((p, i) => ({name: p.name.trim() || '部品 ' + (i+1), box:p.box}));
      if (new Set(parts.map(p => p.name)).size !== parts.length) throw new JobError('部品ごとに異なる名前を付けてください。', 400);
      tell('job-msg', '復元環境を確認しています…');
      const health = await (await api('/api/health')).json();
      if (health.video_ready === false) throw new JobError(health.video_message || '動画の復元環境が準備できていません。', 503);
      const data = new FormData();
      data.append('video', videoFile, videoFile.name);
      data.append('parts', JSON.stringify(parts));
      data.append('camera', get('camera').value);
      data.append('sampling', get('sampling').value);
      if (get('hfov').value) data.append('horizontal_fov_deg', get('hfov').value);
      tell('job-msg', '動画を送信しています…');
      const job = await (await api('/api/video-jobs', {method:'POST',body:data}, 180000)).json();
      if (typeof job.id !== 'string' || !job.id) throw new JobError('ジョブIDを読み取れませんでした。');
      submittedFile = videoFile;
      remember(job.id);
    }
    await poll();
  } catch (error) {
    if (error.status === 404) remember(null);
    tell('job-msg', error.message, true);
    get('retry-job').hidden = !jobId;
  } finally {
    busy = false; sync();
  }
}

get('run-video').addEventListener('click', () => run());
get('retry-job').addEventListener('click', () => run(true));
get('launch').hidden = !pages;
get('process').hidden = pages;
if (!pages) {
  try { jobId = sessionStorage.getItem(persistKey); } catch { /* Optional persistence. */ }
  if (jobId && /^[\w-]{1,128}$/.test(jobId)) {
    get('retry-job').hidden = false;
    tell('job-msg','前回の復元ジョブの状態を確認できます。');
  }
}
sync();
sample();
