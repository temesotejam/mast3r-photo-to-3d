// Exercise the same multipart and polling API used by the photo picker, and
// check the resulting PLY with the viewer's own parser before publishing it.
import { readFile, writeFile } from 'node:fs/promises';
import { setTimeout as delay } from 'node:timers/promises';
import { readPoints } from '../docs/viewer.js';

const base = 'http://127.0.0.1:7861';
async function checked(path, options) {
  const res = await fetch(base + path, options);
  if (!res.ok) throw new Error(`${res.status}: ${await res.text()}`);
  return res;
}

const form = new FormData();
for (const id of [286, 287, 288, 289]) {
  const name = `DSC_0${id}.jpg`;
  const bytes = await readFile(new URL(`../docs/examples/courtyard/${name}`, import.meta.url));
  form.append('photos', new Blob([bytes], { type: 'image/jpeg' }), name);
}
const { id } = await (await checked('/api/jobs', { method: 'POST', body: form })).json();
const endpoint = '/api/jobs/' + id;
let job;
do {
  await delay(1500);
  job = await (await checked(endpoint)).json();
  if (job.status === 'error') throw new Error(job.message);
} while (job.status === 'queued' || job.status === 'running');
if (job.status !== 'done') throw new Error('Unexpected terminal job state: ' + job.status);
const result = await (await checked(endpoint + '/cloud.ply')).arrayBuffer();
const cloud = readPoints(result);
if (!cloud.count || cloud.count !== job.point_count) throw new Error('Viewer and server point counts differ');
await writeFile(new URL('../docs/sample.ply', import.meta.url), new Uint8Array(result));
console.log(`4 ETH3D courtyard photos → ${cloud.count.toLocaleString('en-US')} colored points (${result.byteLength} bytes).`);
