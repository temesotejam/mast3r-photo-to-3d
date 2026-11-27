import { PointViewer } from './viewer.js';

const newBounds = () => ({ min: [Infinity, Infinity, Infinity], max: [-Infinity, -Infinity, -Infinity] });

function include(bounds, axis, value) {
  if (value < bounds.min[axis]) bounds.min[axis] = value;
  if (value > bounds.max[axis]) bounds.max[axis] = value;
}

function triple(values) {
  return Array.isArray(values) && values.length === 3 && values.every(Number.isFinite);
}

function readPart(part, frameCount, sceneBounds) {
  if (!Array.isArray(part.points) || !Array.isArray(part.colors) ||
      part.colors.length !== part.points.length) {
    throw new Error('部位の点群または色データが正しくありません。');
  }
  if (!Array.isArray(part.frames) || part.frames.length !== frameCount) {
    throw new Error('部位のフレーム数が動画と一致しません。');
  }

  const bounds = newBounds();
  const xyz = new Float32Array(part.points.length * 3);
  const rgb = new Uint8ClampedArray(xyz.length);
  part.points.forEach((point, i) => {
    if (!triple(point) || !triple(part.colors[i])) throw new Error('点群に無効な座標または色があります。');
    xyz.set(point, i * 3);
    rgb.set(part.colors[i], i * 3);
    for (let axis = 0; axis < 3; axis++) include(bounds, axis, point[axis]);
  });

  const poses = part.frames.map(frame => {
    if (frame === null || frame?.status === 'unavailable') return null;
    if (!frame || !['observed', 'interpolated'].includes(frame.status) ||
        !Array.isArray(frame.pose) || frame.pose.length !== 12 || !frame.pose.every(Number.isFinite)) {
      throw new Error('部位の姿勢行列または追跡状態が正しくありません。');
    }
    const pose = frame.pose;
    if (!xyz.length) return null;
    // Transform the reference bounding box by each pose, to keep framing fixed during playback.
    for (let row = 0; row < 3; row++) {
      let low = pose[row * 4 + 3];
      let high = low;
      for (let col = 0; col < 3; col++) {
        const a = pose[row * 4 + col] * bounds.min[col];
        const b = pose[row * 4 + col] * bounds.max[col];
        low += Math.min(a, b);
        high += Math.max(a, b);
      }
      include(sceneBounds, row, low);
      include(sceneBounds, row, high);
    }
    return pose;
  });
  return {xyz, rgb, poses};
}

/** All frames use the same center and scale. Motion is expressed in reference-camera coordinates. */
export class MotionViewer extends PointViewer {
  load(scene) {
    if (!Number.isSafeInteger(scene.frame_count) || scene.frame_count < 1 ||
        !Array.isArray(scene.parts) || !scene.parts.length) {
      throw new Error('部位またはフレームを読み取れません。');
    }
    const bounds = newBounds();
    const parts = scene.parts.map(p => readPart(p, scene.frame_count, bounds));
    if (!bounds.min.every(Number.isFinite) || !bounds.max.every(Number.isFinite)) {
      throw new Error('表示できる姿勢がありません。');
    }
    const elements = parts.reduce((sum, p) => sum + p.xyz.length, 0);
    this.positions = new Float32Array(elements);
    this.tints = new Uint8Array(elements);
    this.parts = parts;
    this.frameCount = scene.frame_count;
    this.center = bounds.min.map((lo, i) => (lo + bounds.max[i]) / 2);
    this.radius = Math.hypot(...bounds.max.map((hi, i) => hi - this.center[i])) || 1;
    this.resetView();
    return this.setFrame(0);
  }

  setFrame(index) {
    if (!this.parts || !Number.isInteger(index) || index < 0 || index >= this.frameCount) {
      return {count: 0, partCount: 0};
    }
    const [cx, cy, cz] = this.center;
    const scale = 1 / this.radius;
    let written = 0, partCount = 0;
    for (const {xyz, rgb, poses} of this.parts) {
      const p = poses[index];
      if (!p) continue;
      this.tints.set(rgb, written);
      for (let i = 0; i < xyz.length; i += 3) {
        const x = xyz[i], y = xyz[i + 1], z = xyz[i + 2];
        this.positions[written++] = (p[0] * x + p[1] * y + p[2] * z + p[3] - cx) * scale;
        // Convert the camera's Y-down, Z-forward coordinates to the orbit renderer's axes.
        this.positions[written++] = -(p[4] * x + p[5] * y + p[6] * z + p[7] - cy) * scale;
        this.positions[written++] = -(p[8] * x + p[9] * y + p[10] * z + p[11] - cz) * scale;
      }
      partCount++;
    }
    this.count = written / 3;
    this.points = {count: this.count, pos: this.positions.subarray(0, written), colors: this.tints.subarray(0, written)};
    this.upload();
    this.redraw();
    return {count: this.count, partCount};
  }

  upload() {
    if (this.gl || this.lost) { super.upload(); return; }
    if (this.context2d) return;
    if (this.canvas.getContext('webgl', {alpha: true, antialias: false})) {
      super.upload();
      return;
    }
    this.context2d = this.canvas.getContext('2d');
    if (!this.context2d) throw new Error('この環境では点群を描画できません。');
  }

  redraw() {
    if (!this.context2d) { super.redraw(); return; }
    if (this.frame) return;
    this.frame = requestAnimationFrame(() => {
      this.frame = 0;
      this.paint2d();
    });
  }

  paint2d() {
    const canvas = this.canvas;
    const width = Math.max(1, Math.round(canvas.clientWidth));
    const height = Math.max(1, Math.round(canvas.clientHeight));
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    if (!this.image || this.image.width !== width || this.image.height !== height) {
      this.image = this.context2d.createImageData(width, height);
      this.zBuffer = new Float32Array(width * height);
    }
    this.image.data.fill(0);
    this.zBuffer.fill(Infinity);
    const cs = Math.cos(this.yaw), sn = Math.sin(this.yaw);
    const cp = Math.cos(this.pitch), sp = Math.sin(this.pitch);
    const lens = 1.2 * Math.min(width, height);
    const pointSize = this.count < 40000 ? 3 : 2;
    const {pos, colors} = this.points;
    for (let i = 0; i < pos.length; i += 3) {
      const x = cs * pos[i] + sn * pos[i + 2];
      const z = -sn * pos[i] + cs * pos[i + 2];
      const y = cp * pos[i + 1] - sp * z;
      const depth = this.distance - sp * pos[i + 1] - cp * z;
      if (depth < 0.01 || depth > 100) continue;
      const left = Math.round(width / 2 + lens * (x + this.panX) / depth - pointSize / 2);
      const top = Math.round(height / 2 - lens * (y + this.panY) / depth - pointSize / 2);
      for (let py = Math.max(0, top); py < Math.min(height, top + pointSize); py++) {
        for (let px = Math.max(0, left); px < Math.min(width, left + pointSize); px++) {
          const pixel = py * width + px;
          if (depth >= this.zBuffer[pixel]) continue;
          this.zBuffer[pixel] = depth;
          this.image.data[pixel * 4] = colors[i];
          this.image.data[pixel * 4 + 1] = colors[i + 1];
          this.image.data[pixel * 4 + 2] = colors[i + 2];
          this.image.data[pixel * 4 + 3] = 255;
        }
      }
    }
    this.context2d.putImageData(this.image, 0, 0);
  }
}
