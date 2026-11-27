// Run the generated public sample through the same HTTP API as the video picker.
import assert from 'node:assert/strict';
import { readFile, writeFile, appendFile } from 'node:fs/promises';
import path from 'node:path';
import { setTimeout as delay } from 'node:timers/promises';

const [example, destination, savedScene] = process.argv.slice(2);
if (!example || !destination) throw new Error('Usage: node scripts/verify_video.mjs EXAMPLE_DIR SCENE.json');
const api = 'http://127.0.0.1:7861';
async function request(route, options) {
  const response = await fetch(api + route, {signal: AbortSignal.timeout(90_000), ...options});
  if (!response.ok) throw new Error(`HTTP ${response.status} ${route}: ${await response.text()}`);
  return response;
}

function checkPose(pose) {
  assert.ok(Array.isArray(pose) && pose.length === 12 && pose.every(Number.isFinite), 'Invalid 3×4 pose');
  const rows = [pose.slice(0, 3), pose.slice(4, 7), pose.slice(8, 11)];
  for (let a = 0; a < 3; a++) {
    for (let b = a; b < 3; b++) {
      const dot = rows[a].reduce((sum, v, i) => sum + v * rows[b][i], 0);
      assert.ok(Math.abs(dot - (a === b ? 1 : 0)) < 0.02, 'The part transform must be rigid');
    }
  }
  const [a, b, c] = rows;
  const det = a[0]*(b[1]*c[2] - b[2]*c[1]) + a[1]*(b[2]*c[0] - b[0]*c[2]) + a[2]*(b[0]*c[1] - b[1]*c[0]);
  assert.ok(Math.abs(det - 1) < 0.02, 'Improper rotation');
}

function cameraProjection(local, pose, intrinsics) {
  const xyz = [0, 1, 2].map(row => {
    const k = row * 4;
    return pose[k + 3] + local.reduce((sum, value, i) => sum + value * pose[k + i], 0);
  });
  const projected = [0, 1, 2].map(row => xyz.reduce((sum, v, i) => sum + intrinsics[row * 3 + i]*v, 0));
  assert.ok(projected[2] > 0, 'Part centroid must be in front of the camera');
  return [projected[0]/projected[2], projected[1]/projected[2]];
}

const annotations = await readFile(path.join(example, 'parts.json'), 'utf8');
const truth = JSON.parse(await readFile(path.join(example, 'truth.json'), 'utf8'));
const names = JSON.parse(annotations).map(part => part.name);
async function reconstructExample() {
  const form = new FormData();
  form.append('video', new Blob([await readFile(path.join(example, 'video.mp4'))], {type: 'video/mp4'}), 'video.mp4');
  form.append('parts', annotations);
  form.append('camera', 'fixed');
  const job = await (await request('/api/video-jobs', {method: 'POST', body: form})).json();
  assert.match(job.id, /^[a-fA-F0-9-]+$/);
  const route = '/api/video-jobs/' + job.id;
  const deadline = Date.now() + 60 * 60 * 1000;
  while (true) {
    const status = await (await request(route)).json();
    if (status.status === 'done') break;
    if (status.status === 'error') throw new Error(status.message);
    assert.ok(['running', 'queued'].includes(status.status), 'Unexpected job status');
    if (Date.now() >= deadline) throw new Error('The sample exceeded its one-hour processing window');
    await delay(2000);
  }
  return new Uint8Array(await (await request(route + '/scene.json')).arrayBuffer());
}
const bytes = savedScene ? await readFile(savedScene) : await reconstructExample();
const scene = JSON.parse(new TextDecoder().decode(bytes));
assert.equal(scene.version, 1);
assert.equal(scene.model, 'naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric');
assert.equal(scene.depth_method, 'mast3r-pairwise');
assert.equal(scene.camera_mode, 'fixed');
assert.equal(scene.coordinate_system, 'opencv');
assert.ok(scene.frame_count >= 8 && scene.frame_count <= 24);
assert.ok(scene.fps > 0 && scene.width > 0 && scene.height > 0);
assert.equal(scene.times.length, scene.frame_count);
assert.ok(scene.times.every((t, i) => Number.isFinite(t) && (i === 0 ? t >= 0 : t > scene.times[i-1])));
assert.ok(Array.isArray(scene.intrinsics) && scene.intrinsics.length === 9 && scene.intrinsics.every(Number.isFinite));
assert.deepEqual(scene.parts.map(p => p.name).sort(), [...names].sort());
const report = [];
for (const part of scene.parts) {
  const {points, colors, frames, new_points_per_frame: increments} = part;
  assert.ok(points.length >= 8, `${part.name}: not enough reconstructed points`);
  assert.ok(points.every(point => point.length === 3 && point.every(Number.isFinite)));
  assert.equal(colors.length, points.length);
  assert.ok(colors.every(color => color.length === 3 && color.every(v => Number.isInteger(v) && v >= 0 && v <= 255)));
  assert.equal(frames.length, scene.frame_count);
  assert.equal(increments.length, scene.frame_count);
  assert.ok(increments.every(n => Number.isInteger(n) && n >= 0));
  assert.ok(increments[0] > 0 && increments.slice(1).some(n => n > 0),
            `${part.name}: missing initial shape or newly visible surface additions`);
  const center = [0, 1, 2].map(i => points.reduce((sum, point) => sum + point[i], 0) / points.length);
  const observed = [];
  frames.forEach((frame, index) => {
    assert.ok(['observed', 'interpolated', 'unavailable'].includes(frame.status));
    if (frame.status !== 'observed') assert.equal(increments[index], 0, 'Only observed frames may add geometry');
    if (frame.status === 'unavailable') { assert.equal(frame.pose, null); return; }
    checkPose(frame.pose);
    if (frame.status === 'observed') observed.push({index, xy: cameraProjection(center, frame.pose, scene.intrinsics)});
  });
  assert.ok(observed.length >= 3, `${part.name}: fewer than 3 observed poses`);
  assert.ok(observed.some(({xy}) => Math.hypot(xy[0] - observed[0].xy[0], xy[1] - observed[0].xy[1]) > 2),
            `${part.name}: the projected trajectory is stationary`);
  report.push(`${part.name}: ${points.length} canonical points, ${observed.length}/${frames.length} observed poses, appended ${increments.slice(1).reduce((a,b)=>a+b,0)} after frame 0.`);

  // Canonical cloud origins need not coincide with the CAD centers. Compare
  // image-plane travel from the first visible observation, in original pixels.
  const pairs = [];
  const referenceFrames = scene.times.map(time => {
    const source = truth.frames.reduce((a,b) =>
      Math.abs(a.time - time) < Math.abs(b.time - time) ? a : b);
    const reference = source.parts.find(p => p.name === part.name);
    assert.ok(reference && Number.isFinite(reference.visible_fraction));
    return reference;
  });
  const occluded = referenceFrames.flatMap((reference,i) => reference.visible_fraction < 0.1 ? [i] : []);
  if (occluded.length) {
    const gap = occluded.map(i => `${i}=${frames[i].status}`).join(', ');
    console.log(`${part.name}: near-total occlusion / estimated state: ${gap}`);
    assert.ok(occluded.every(i => frames[i].status !== 'observed'), 'Nearly hidden object marked observed');
    const last = occluded.at(-1);
    const reacquired = observed.find(({index}) => index > last && referenceFrames[index].visible_fraction > 0.5);
    assert.ok(reacquired, `${part.name}: not reacquired when it became visible again`);
    report.push(`${part.name}: occlusion ${gap}; observed again at frame ${reacquired.index}.`);
  }
  for (const {index, xy} of observed) {
    const reference = referenceFrames[index];
    if (reference.visible_fraction < 0.5) continue;
    const scaled = [xy[0]*truth.image_size[0]/scene.width, xy[1]*truth.image_size[1]/scene.height];
    pairs.push({predicted: scaled, expected: reference.center_px});
  }
  assert.ok(pairs.length >= 3, `${part.name}: too few visible observations to validate`);
  const first = pairs[0];
  let squaredError = 0, squaredBaseline = 0;
  for (const {predicted, expected} of pairs) {
    for (const axis of [0,1]) {
      const delta = expected[axis] - first.expected[axis];
      squaredError += ((predicted[axis] - first.predicted[axis]) - delta) ** 2;
      squaredBaseline += delta ** 2;
    }
  }
  const rms = Math.sqrt(squaredError / pairs.length);
  const frozen = Math.sqrt(squaredBaseline / pairs.length);
  report.push(`${part.name}: projected centroid travel RMSE ${rms.toFixed(1)} px (frozen-position baseline ${frozen.toFixed(1)} px, ${pairs.length} visible observations).`);
  console.log(report.at(-1));
  // The generated scene deliberately has independent motion. The offset
  // between the visible surface centroid and the box center can rotate.
  const tolerance = Math.max(0.03 * truth.image_size[0], frozen);
  assert.ok(rms < tolerance, `${part.name}: centroid motion RMSE ${rms.toFixed(1)} exceeds ${tolerance.toFixed(1)} px`);
}
await writeFile(destination, bytes);
console.log(report.join('\n'));
if (process.env.GITHUB_STEP_SUMMARY) {
  await appendFile(process.env.GITHUB_STEP_SUMMARY, '### Generated video validation\n\n'+report.map(x=>'- '+x).join('\n')+'\n');
}
