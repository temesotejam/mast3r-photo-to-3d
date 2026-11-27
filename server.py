"""Serve the reconstruction workspace, processing one photo or video job at a time."""

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import uuid
import warnings
from email.parser import BytesParser
from email.policy import default
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from PIL import Image


ROOT = Path(__file__).resolve().parent
PAGES_ORIGIN = "https://temesotejam.github.io"
MAX_BODY = 60 * 1024**2
MAX_PHOTO = 12 * 1024**2
MAX_PIXELS = 60_000_000
MAX_RESULTS = 3
JOB_ROUTE = re.compile(r"/api/(jobs|video-jobs)/([a-f0-9]{32})(?:/(cloud\.ply|scene\.json))?")
RESULT_FILES = {"jobs": "cloud.ply", "video-jobs": "scene.json"}


def runtime_status():
    vendor = ROOT / (os.environ.get("MAST3R_DIR") or "vendor/mast3r")
    if not all((vendor / path).is_file() for path in (
        "mast3r/model.py", "dust3r/dust3r/model.py", "dust3r/croco/models/croco.py"
    )):
        return False, "MASt3R本体とサブモジュールをセットアップしてください（README参照）。"
    missing = [name for name in ("torch", "torchvision", "huggingface_hub", "numpy")
               if importlib.util.find_spec(name) is None]
    if missing:
        return False, "復元に必要なライブラリをインストールしてください: " + ", ".join(missing)
    return True, "MASt3Rを利用できます。初回実行時はモデルを取得します。"


def video_runtime_status():
    ready, message = runtime_status()
    if not ready:
        return False, message
    missing = [name for name in ("cv2", "numpy") if importlib.util.find_spec(name) is None]
    if missing or not (ROOT / "reconstruct_video.py").is_file():
        return False, "動画復元の実行環境をセットアップしてください（README参照）。"
    return True, "動画復元を利用できます。"


def multipart_fields(stream, length, content_type):
    if not content_type.isascii() or "\r" in content_type or "\n" in content_type:
        raise ValueError("送信形式が不正です。")
    body = stream.read(length)
    if len(body) != length:
        raise ValueError("ファイルの受信が中断されました。")
    message = BytesParser(policy=default).parsebytes(
        b"MIME-Version: 1.0\r\nContent-Type: " + content_type.encode("ascii") + b"\r\n\r\n" + body
    )
    del body
    if message.get_content_type() != "multipart/form-data" or not message.is_multipart() or message.defects:
        raise ValueError("multipart/form-dataでファイルを送信してください。")
    fields = []
    for part in message.iter_parts():
        if part.get_content_disposition() != "form-data" or part.is_multipart():
            raise ValueError("ファイルの送信形式が不正です。")
        fields.append((part.get_param("name", header="content-disposition"), part))
    return fields


def read_photos(photos, destination):
    """Normalize names, verify actual image contents, and ignore client-side paths."""
    if not 2 <= len(photos) <= 10:
        raise ValueError("写真は2〜10枚選んでください。")
    for index, (field, part) in enumerate(photos):
        if field != "photos":
            raise ValueError("写真の入力欄は photos を使用してください。")
        payload = part.get_payload(decode=True)
        if not payload or len(payload) > MAX_PHOTO:
            raise ValueError("写真は1枚12 MiB以内にしてください。")
        candidate = destination / f"photo-{index:02d}"
        candidate.write_bytes(payload)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(candidate) as image:
                    if image.format not in ("JPEG", "PNG") or image.width * image.height > MAX_PIXELS:
                        raise ValueError("JPEGまたはPNG（各60メガピクセル以下）を選んでください。")
                    extension = ".jpg" if image.format == "JPEG" else ".png"
                    image.verify()
        except (OSError, SyntaxError, Image.DecompressionBombWarning,
                Image.DecompressionBombError) as error:
            raise ValueError("読み込みに失敗した画像があります。JPEGまたはPNGを選んでください。") from error
        candidate.rename(candidate.with_suffix(extension))


def read_video(fields, destination):
    parts = dict(fields)
    if (len(parts) != len(fields) or not {"parts", "video"} <= parts.keys() or
            parts.keys() - {"parts", "video", "camera", "sampling", "horizontal_fov_deg"}):
        raise ValueError("動画と部品の選択範囲を送信してください。")
    camera = parts["camera"].get_payload(decode=True) if "camera" in parts else b"fixed"
    if camera not in (b"fixed", b"moving"):
        raise ValueError("カメラ設定は fixed または moving を選んでください。")
    sampling = parts["sampling"].get_payload(decode=True) if "sampling" in parts else b"standard"
    if sampling not in (b"standard", b"dense"):
        raise ValueError("フレームの抽出設定は standard または dense を選んでください。")
    fov = None
    if "horizontal_fov_deg" in parts:
        try:
            fov = float(parts["horizontal_fov_deg"].get_payload(decode=True))
        except (TypeError, ValueError):
            raise ValueError("水平画角を度数で入力してください。") from None
        if not 10 <= fov <= 160:
            raise ValueError("水平画角は10〜160度の範囲で指定してください。")
    data = parts["parts"].get_payload(decode=True)
    if not data or len(data) > 8192:
        raise ValueError("部品情報のサイズが不正です。")
    try:
        selections = json.loads(data.decode("utf-8"))
    except (ValueError, RecursionError):
        raise ValueError("部品情報はJSON配列で送信してください。") from None
    if not isinstance(selections, list) or not 1 <= len(selections) <= 6:
        raise ValueError("動画の部品は1〜6個指定してください。")
    names = set()
    boxes = []
    for part in selections:
        if not isinstance(part, dict):
            raise ValueError("部品の名前と選択範囲を指定してください。")
        name, box = part.get("name"), part.get("box")
        if (not isinstance(name, str) or not 1 <= len(name.strip()) <= 64 or
                not name.strip().isprintable() or name.strip() in names):
            raise ValueError("各部品に異なる1〜64文字の名前を付けてください。")
        if (not isinstance(box, list) or len(box) != 4 or
                not all(type(v) in (float, int) and 0 <= v <= 1 for v in box) or
                box[0] >= box[2] or box[1] >= box[3]):
            raise ValueError("選択範囲は0〜1の正規化座標 [左, 上, 右, 下] で指定してください。")
        names.add(name.strip())
        boxes.append({"name": name.strip(), "box": box})
    upload = parts["video"]
    extension = Path(upload.get_filename() or "").suffix.lower()
    video = upload.get_payload(decode=True)
    if extension not in {".mp4", ".mov", ".webm"} or not video:
        raise ValueError("MP4、MOVまたはWebM動画を選んでください。")
    if extension == ".webm":
        container_ok = video.startswith(b"\x1a\x45\xdf\xa3")
    else:
        container_ok = len(video) >= 8 and video[4:8] in {b"ftyp", b"moov", b"wide", b"free", b"mdat"}
    if not container_ok:
        raise ValueError("動画の内容と拡張子が一致しません。")
    path = destination / ("input" + extension)
    path.write_bytes(video)
    (destination / "parts.json").write_text(json.dumps(boxes, ensure_ascii=False), encoding="utf-8")
    return path, camera.decode("ascii"), sampling.decode("ascii"), fov


def vertex_count(path):
    with path.open("rb") as stream:
        header = stream.read(4096).split(b"end_header\n", 1)[0]
    match = re.search(rb"^element vertex (\d+)\s*$", header, flags=re.MULTILINE)
    if not match or int(match[1]) == 0:
        raise ValueError("出力ファイルに点が含まれていません。")
    return int(match[1])


class JobStore:
    def __init__(self, directory):
        self.directory = directory
        self.lock = threading.Lock()
        self.slot = threading.Lock()
        self.records = {}
        self.process = None
        self.worker = None
        self.closing = False

    def start(self, job_id, kind, video=None):
        with self.lock:
            if self.closing:
                raise RuntimeError("サーバーを終了しています。")
            while len(self.records) >= MAX_RESULTS:
                oldest = next(iter(self.records))
                del self.records[oldest]
                shutil.rmtree(self.directory / oldest, ignore_errors=True)
            self.records[job_id] = (kind, {"status": "queued", "message": "復元を開始します。"})
            self.worker = threading.Thread(target=self.run, args=(job_id, kind, video), daemon=True)
            self.worker.start()

    def get(self, job_id, kind):
        with self.lock:
            job = self.records.get(job_id)
            return job[1] if job and job[0] == kind else None

    def run(self, job_id, kind, video):
        job_dir = self.directory / job_id
        inputs = job_dir / "inputs"
        output = job_dir / RESULT_FILES[kind]
        log_file = job_dir / "reconstruct.log"
        result = {"status": "error", "message": "復元を中断しました。"}
        try:
            with log_file.open("wb") as log:
                if kind == "jobs":
                    command = [sys.executable, str(ROOT / "reconstruct.py"), "--input", str(inputs),
                               "--output", str(output), "--device", "auto"]
                    progress = "MASt3Rで写真を復元しています。"
                else:
                    video_path, camera, sampling, fov = video
                    command = [sys.executable, str(ROOT / "reconstruct_video.py"), "--video", str(video_path),
                               "--parts", str(inputs / "parts.json"), "--output", str(output), "--camera", camera,
                               "--sampling", sampling]
                    if fov is not None:
                        command.extend(["--horizontal-fov-deg", str(fov)])
                    progress = "動画のフレームを復元し、部品ごとの動きを計算しています。"
                with self.lock:
                    if self.closing:
                        raise RuntimeError("サーバーを終了しています。")
                    process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                    self.process = process
                    self.records[job_id] = (kind, {"status": "running", "message": progress})
                try:
                    code = process.wait(timeout=3600)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                    raise RuntimeError("復元が1時間以内に完了しませんでした。入力を減らしてお試しください。") from None
            if code != 0:
                with log_file.open("rb") as log:
                    log.seek(max(0, log_file.stat().st_size - 4096))
                    tail = log.read().decode("utf-8", "replace").strip().splitlines()
                detail = tail[-1][-500:] if tail else f"exit {code}"
                raise RuntimeError(f"MASt3Rの実行に失敗しました。{detail}")
            if kind == "jobs":
                result = {"status": "done", "message": "点群が完成しました。", "point_count": vertex_count(output)}
            else:
                if not output.is_file() or not output.stat().st_size:
                    raise ValueError("動画の3Dシーンを出力できませんでした。")
                result = {"status": "done", "message": "動画の3Dシーンが完成しました。"}
        except (OSError, RuntimeError, ValueError) as error:
            result = {"status": "error", "message": str(error)}
        finally:
            shutil.rmtree(inputs, ignore_errors=True)
            with self.lock:
                self.process = None
                if job_id in self.records:
                    self.records[job_id] = (kind, result)
            self.slot.release()

    def close(self):
        with self.lock:
            self.closing = True
            if self.process and self.process.poll() is None:
                self.process.kill()
            worker = self.worker
        if worker:
            worker.join()


class Handler(SimpleHTTPRequestHandler):
    extensions_map = {**SimpleHTTPRequestHandler.extensions_map, ".js": "text/javascript",
                      ".css": "text/css"}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT / "docs"), **kwargs)

    def setup(self):
        super().setup()
        self.connection.settimeout(90)

    def origin_allowed(self):
        port = self.server.server_port
        hosts = {f"localhost:{port}", f"127.0.0.1:{port}"}
        origins = {PAGES_ORIGIN, *(f"http://{host}" for host in hosts)}
        codespace = os.environ.get("CODESPACE_NAME")
        domain = os.environ.get("GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN")
        if codespace and domain:
            forwarded_host = f"{codespace}-{port}.{domain}"
            hosts.add(forwarded_host)
            origins.add(f"https://{forwarded_host}")
        return self.headers.get("Host") in hosts and self.headers.get("Origin") in {None, *origins}

    def end_headers(self):
        if self.headers.get("Origin") and self.origin_allowed():
            self.send_header("Access-Control-Allow-Origin", self.headers["Origin"])
            self.send_header("Vary", "Origin")
        self.send_header("X-Content-Type-Options", "nosniff")
        super().end_headers()

    def send_json(self, status, value):
        body = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        if not self.origin_allowed():
            self.send_json(403, {"message": "この接続元は利用できません。"})
            return
        self.send_response(204)
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Private-Network", "true")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        if not self.origin_allowed():
            self.send_json(403, {"message": "この接続元は利用できません。"})
            return
        path = urlsplit(self.path).path
        if not path.startswith("/api/"):
            super().do_GET()
            return
        if path == "/api/health":
            ready, message = runtime_status()
            video_ready, video_message = video_runtime_status()
            self.send_json(200, {"engine": "mast3r", "ready": ready, "message": message,
                                 "video_ready": video_ready, "video_message": video_message})
            return
        match = JOB_ROUTE.fullmatch(path)
        store = self.server.jobs
        record = store.get(match[2], match[1]) if match else None
        if not record or (match[3] and match[3] != RESULT_FILES[match[1]]):
            self.send_json(404, {"message": "このジョブは存在しないか、保存期間を過ぎました。"})
            return
        if not match[3]:
            self.send_json(200, record)
            return
        if record["status"] != "done":
            self.send_json(409, {"message": "結果はまだ作成されていません。"})
            return
        try:
            artifact = (store.directory / match[2] / match[3]).open("rb")
        except FileNotFoundError:
            self.send_json(404, {"message": "結果の保存期間を過ぎました。"})
            return
        with artifact:
            self.send_response(200)
            mime = "application/json; charset=utf-8" if match[3] == "scene.json" else "application/octet-stream"
            self.send_header("Content-Type", mime)
            self.send_header("Content-Disposition", f'attachment; filename="{match[3]}"')
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(os.fstat(artifact.fileno()).st_size))
            self.end_headers()
            shutil.copyfileobj(artifact, self.wfile)

    def do_POST(self):
        if not self.origin_allowed():
            self.send_json(403, {"message": "この接続元は利用できません。"})
            return
        path = urlsplit(self.path).path
        if path not in ("/api/jobs", "/api/video-jobs"):
            self.send_json(404, {"message": "URLを確認してください。"})
            return
        kind = path.rsplit("/", 1)[1]
        ready, message = video_runtime_status() if kind == "video-jobs" else runtime_status()
        if not ready:
            self.send_json(503, {"message": message})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length <= 0 or length > MAX_BODY or self.headers.get("Transfer-Encoding"):
            self.send_json(413, {"message": "ファイルの合計は送信データを含め60 MiB以内にしてください。"})
            return
        store = self.server.jobs
        if not store.slot.acquire(blocking=False):
            self.send_json(409, {"message": "別の復元を実行中です。完了してから再度お試しください。"})
            return
        job_id = uuid.uuid4().hex
        job_dir = store.directory / job_id
        try:
            inputs = job_dir / "inputs"
            inputs.mkdir(parents=True)
            fields = multipart_fields(self.rfile, length, self.headers.get("Content-Type", ""))
            video = None
            if kind == "jobs":
                read_photos(fields, inputs)
            else:
                video = read_video(fields, inputs)
            store.start(job_id, kind, video)
        except (OSError, ValueError, RuntimeError) as error:
            shutil.rmtree(job_dir, ignore_errors=True)
            store.slot.release()
            self.send_json(400, {"message": str(error)})
            return
        self.send_json(202, {"id": job_id})


def main():
    with tempfile.TemporaryDirectory(prefix="mast3r-photo-") as temp:
        jobs = JobStore(Path(temp))
        with ThreadingHTTPServer(("127.0.0.1", 7861), Handler) as server:
            server.jobs = jobs
            print("MASt3R Photo to 3D: http://127.0.0.1:7861", flush=True)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
            finally:
                jobs.close()


if __name__ == "__main__":
    main()
