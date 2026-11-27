// PLY point positions are expressed in the first camera's coordinate system:
// X to the right, Y down, Z forward.
const scalarTypes = {};
for (const [names, size, method] of [
  ['char int8', 1, 'getInt8'], ['uchar uint8', 1, 'getUint8'],
  ['short int16', 2, 'getInt16'], ['ushort uint16', 2, 'getUint16'],
  ['int int32', 4, 'getInt32'], ['uint uint32', 4, 'getUint32'],
  ['float float32', 4, 'getFloat32'], ['double float64', 8, 'getFloat64'],
]) {
  for (const name of names.split(' ')) scalarTypes[name] = {size, method};
}

function fail(reason) {
  throw new Error(reason);
}

export function readPoints(buffer) {
  const decoder = new TextDecoder();
  const bytes = new Uint8Array(buffer);
  const prefix = decoder.decode(bytes.subarray(0, 65536));
  const marker = /^end_header[ \t]*\r?\n/m.exec(prefix);
  if (!prefix.startsWith('ply\n') && !prefix.startsWith('ply\r\n')) fail('PLY形式のファイルを選択してください。');
  if (!marker) fail('PLYのヘッダーを読み取れません。');
  // Header comments may contain UTF-8, so count bytes rather than characters.
  const start = new TextEncoder().encode(prefix.slice(0, marker.index + marker[0].length)).length;
  let format, element, count, stride = 0;
  const props = new Map();
  for (const line of prefix.slice(0, marker.index).split(/\r?\n/)) {
    const [keyword, a, b] = line.trim().split(/\s+/);
    if (keyword === 'format') format = a;
    if (keyword === 'element') {
      element = a;
      if (element === 'vertex') {
        if (count !== undefined) fail('PLYの頂点定義が重複しています。');
        count = Number(b);
      } else if (count === undefined && Number(b) > 0) {
        fail('頂点が先頭にあるPLYファイルを選択してください。');
      }
    }
    if (keyword === 'property' && element === 'vertex') {
      const type = Object.hasOwn(scalarTypes, a) && scalarTypes[a];
      if (!type || props.has(b)) fail('未対応のPLY頂点プロパティです。');
      props.set(b, {offset: stride, index: props.size, method: type.method});
      stride += type.size;
    }
  }
  const binary = format === 'binary_little_endian' || format === 'binary_big_endian';
  if (!binary && format !== 'ascii') fail('未対応のPLY形式です。');
  if (!Number.isSafeInteger(count) || count <= 0) fail('PLYに点がありません。');
  if (count > 10000000) fail('このビューアの上限は1,000万点です。');
  const coords = ['x', 'y', 'z'].map(name => props.get(name));
  if (coords.some(p => !p)) fail('PLYにx・y・z座標がありません。');
  const rgb = ['red', 'green', 'blue'].map(name => props.get(name));
  const hasColor = rgb.every(Boolean);
  if (bytes.length - start < count * (binary ? stride : 5)) fail('PLYファイルの頂点データが不足しています。');
  const pos = new Float32Array(count * 3);
  const colors = new Uint8Array(count * 3);
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  const view = new DataView(buffer);
  const little = format === 'binary_little_endian';
  let offset = start, accepted = 0;
  for (let n = 0; n < count; n++) {
    let ascii;
    if (!binary) {
      const eol = bytes.indexOf(10, offset);
      const end = eol === -1 ? bytes.length : eol;
      ascii = decoder.decode(bytes.subarray(offset, end)).trim().split(/\s+/);
      if (ascii.length < props.size) fail('PLYの頂点データを読み取れません。');
      offset = end + 1;
    }
    const value = p => binary ? view[p.method](offset + p.offset, little) : Number(ascii[p.index]);
    const at = accepted * 3;
    for (let i = 0; i < 3; i++) pos[at + i] = value(coords[i]);
    if (Number.isFinite(pos[at]) && Number.isFinite(pos[at + 1]) && Number.isFinite(pos[at + 2])) {
      for (let i = 0; i < 3; i++) {
        const x = pos[at + i];
        lo[i] = Math.min(lo[i], x);
        hi[i] = Math.max(hi[i], x);
        colors[at + i] = hasColor ? Math.max(0, Math.min(255, Math.round(value(rgb[i])))) : 202;
      }
      accepted++;
    }
    if (binary) offset += stride;
  }
  if (!accepted) fail('表示できる有限座標の点がありません。');
  const mid = lo.map((v, i) => (v + hi[i]) / 2);
  const radius = Math.hypot(...lo.map((v, i) => (hi[i] - v) / 2)) || 1;
  for (let i = 0; i < accepted * 3; i++) pos[i] = (pos[i] - mid[i % 3]) / radius;
  return {count: accepted, pos: pos.subarray(0, accepted * 3), colors: colors.subarray(0, accepted * 3)};
}

const vertexSource = `
attribute vec3 position;
attribute vec3 color;
varying vec3 tint;
uniform vec4 turn;
uniform vec2 scale;
uniform vec2 pan;
uniform float distanceToCenter;
uniform float pointSize;
void main() {
  float x = turn.x * position.x + turn.y * position.z;
  float z = -turn.y * position.x + turn.x * position.z;
  float y = -turn.z * position.y - turn.w * z;
  float depth = -turn.w * position.y + turn.z * z + distanceToCenter;
  gl_Position = vec4((vec2(x, y) + pan) * scale, 1.0002 * depth - 0.020002, depth);
  gl_PointSize = pointSize;
  tint = color;
}`;
const fragmentSource = `
precision mediump float;
varying vec3 tint;
void main() { gl_FragColor = vec4(tint, 1.0); }`;

function shaderProgram(gl) {
  const program = gl.createProgram();
  for (const [type, source] of [[gl.VERTEX_SHADER, vertexSource], [gl.FRAGMENT_SHADER, fragmentSource]]) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) fail('点群表示用のシェーダーを初期化できません。');
    gl.attachShader(program, shader);
    gl.deleteShader(shader);
  }
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) fail('点群ビューアを初期化できません。');
  return program;
}

export class PointViewer {
  constructor(canvas) {
    this.canvas = canvas;
    this.count = 0;
    this.pointers = new Map();
    this.frame = 0;
    canvas.style.touchAction = 'none';
    new ResizeObserver(() => this.redraw()).observe(canvas);
    canvas.addEventListener('wheel', e => {
      if (!this.count) return;
      e.preventDefault();
      const unit = e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? canvas.clientHeight : 1;
      this.zoom(Math.exp(Math.max(-1, Math.min(1, e.deltaY * unit * 0.0015))));
    }, {passive: false});
    canvas.addEventListener('contextmenu', e => e.preventDefault());
    canvas.addEventListener('pointerdown', e => {
      if (!this.count) return;
      canvas.setPointerCapture(e.pointerId);
      this.pointers.set(e.pointerId, {x: e.clientX, y: e.clientY});
    });
    canvas.addEventListener('pointermove', e => {
      const prev = this.pointers.get(e.pointerId);
      if (!prev) return;
      const next = {x: e.clientX, y: e.clientY};
      const dx = next.x - prev.x, dy = next.y - prev.y;
      this.pointers.set(e.pointerId, next);
      if (this.pointers.size === 2) {
        const other = [...this.pointers].find(([id]) => id !== e.pointerId)[1];
        const before = Math.hypot(other.x - prev.x, other.y - prev.y);
        const after = Math.hypot(other.x - next.x, other.y - next.y);
        if (before > 1 && after > 1) this.zoom(before / after);
        this.pan(dx / 2, dy / 2);
      } else if ((e.buttons & 6) || e.shiftKey) {
        this.pan(dx, dy);
      } else {
        this.yaw += dx * 0.009;
        this.pitch = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, this.pitch - dy * 0.009));
      }
      this.redraw();
    });
    canvas.addEventListener('lostpointercapture', e => this.pointers.delete(e.pointerId));
    canvas.addEventListener('webglcontextlost', e => { e.preventDefault(); this.lost = true; });
    canvas.addEventListener('webglcontextrestored', () => {
      this.lost = false;
      this.gl = null;
      if (this.points) { this.upload(); this.redraw(); }
    });
    this.resetView();
  }

  resetView() {
    this.yaw = this.pitch = this.panX = this.panY = 0;
    this.distance = 3.2;
    this.redraw();
  }

  pan(dx, dy) {
    const unit = this.distance / (1.2 * Math.max(1, Math.min(this.canvas.clientWidth, this.canvas.clientHeight)));
    this.panX += dx * unit;
    this.panY -= dy * unit;
  }

  zoom(factor) { this.distance = Math.max(0.08, Math.min(40, this.distance * factor)); this.redraw(); }

  async show(blob) {
    if (blob.size > 256 * 1024 * 1024) fail('256 MiB以下のPLYファイルを選択してください。');
    const points = readPoints(await blob.arrayBuffer());
    this.points = points;
    this.count = points.count;
    this.upload();
    this.resetView();
    return {count: points.count};
  }

  upload() {
    if (this.lost) return;
    if (!this.gl) {
      const gl = this.canvas.getContext('webgl', {alpha: true, antialias: false});
      if (!gl) fail('このブラウザではWebGLを利用できません。ブラウザのハードウェアアクセラレーションを確認してください。');
      const program = shaderProgram(gl);
      gl.useProgram(program);
      this.uniforms = Object.fromEntries(['turn', 'scale', 'pan', 'distanceToCenter', 'pointSize']
        .map(name => [name, gl.getUniformLocation(program, name)]));
      this.buffers = [];
      for (const [name, type, normalized] of [['position', gl.FLOAT, false], ['color', gl.UNSIGNED_BYTE, true]]) {
        const location = gl.getAttribLocation(program, name);
        const buffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
        gl.enableVertexAttribArray(location);
        gl.vertexAttribPointer(location, 3, type, normalized, 0, 0);
        this.buffers.push(buffer);
      }
      gl.enable(gl.DEPTH_TEST);
      gl.clearColor(0, 0, 0, 0);
      this.gl = gl;
    }
    const gl = this.gl;
    [this.points.pos, this.points.colors].forEach((arr, i) => {
      gl.bindBuffer(gl.ARRAY_BUFFER, this.buffers[i]);
      gl.bufferData(gl.ARRAY_BUFFER, arr, gl.STATIC_DRAW);
    });
  }

  redraw() {
    if (this.frame || !this.gl || this.lost) return;
    this.frame = requestAnimationFrame(() => {
      this.frame = 0;
      if (this.lost) return;
      const gl = this.gl, c = this.canvas, u = this.uniforms;
      const ratio = Math.min(window.devicePixelRatio || 1, 2);
      const w = Math.max(1, Math.round(c.clientWidth * ratio));
      const h = Math.max(1, Math.round(c.clientHeight * ratio));
      if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
      gl.viewport(0, 0, w, h);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      gl.uniform4f(u.turn, Math.cos(this.yaw), Math.sin(this.yaw), Math.cos(this.pitch), Math.sin(this.pitch));
      gl.uniform2f(u.scale, 2.4 * Math.min(w, h) / w, 2.4 * Math.min(w, h) / h);
      gl.uniform2f(u.pan, this.panX, this.panY);
      gl.uniform1f(u.distanceToCenter, this.distance);
      gl.uniform1f(u.pointSize, (this.count < 40000 ? 2.6 : 1.6) * ratio);
      gl.drawArrays(gl.POINTS, 0, this.count);
    });
  }
}
