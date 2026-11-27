"""Serve the photo workspace and run one local MASt3R reconstruction at a time."""

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
JOB_ROUTE = re.compile(r"/api/jobs/([a-f0-9]{32})(/cloud\.ply)?")


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


def read_photos(stream, length, content_type, destination):
    """Normalize names, verify actual image contents, and ignore client-side paths."""
    if not content_type.isascii() or "\r" in content_type or "\n" in content_type:
        raise ValueError("写真の送信形式が不正です。")
    body = stream.read(length)
    if len(body) != length:
        raise ValueError("ファイルの受信が中断されました。")
    message = BytesParser(policy=default).parsebytes(
        b"MIME-Version: 1.0\r\nContent-Type: " + content_type.encode("ascii") + b"\r\n\r\n" + body
    )
    del body
    if message.get_content_type() != "multipart/form-data" or not message.is_multipart() or message.defects:
        raise ValueError("multipart/form-dataで写真を送信してください。")
    photos = list(message.iter_parts())
    if not 2 <= len(photos) <= 10:
        raise ValueError("写真は2〜10枚選んでください。")
    for index, part in enumerate(photos):
        if (part.get_content_disposition() != "form-data" or
                part.get_param("name", header="content-disposition") != "photos" or part.is_multipart()):
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

    def start(self, job_id):
        with self.lock:
            if self.closing:
                raise RuntimeError("サーバーを終了しています。")
            while len(self.records) >= MAX_RESULTS:
                oldest = next(iter(self.records))
                del self.records[oldest]
                shutil.rmtree(self.directory / oldest, ignore_errors=True)
            self.records[job_id] = {"status": "queued", "message": "復元を開始します。"}
            self.worker = threading.Thread(target=self.run, args=(job_id,), daemon=True)
            self.worker.start()

    def get(self, job_id):
        with self.lock:
            return self.records.get(job_id)

    def run(self, job_id):
        job_dir = self.directory / job_id
        output = job_dir / "cloud.ply"
        log_file = job_dir / "reconstruct.log"
        result = {"status": "error", "message": "復元を中断しました。"}
        try:
            with log_file.open("wb") as log:
                command = [sys.executable, str(ROOT / "reconstruct.py"), "--input", str(job_dir / "photos"),
                           "--output", str(output), "--device", "auto"]
                with self.lock:
                    if self.closing:
                        raise RuntimeError("サーバーを終了しています。")
                    process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                    self.process = process
                    self.records[job_id] = {"status": "running", "message": "MASt3Rで写真を復元しています。"}
                try:
                    code = process.wait(timeout=3600)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                    raise RuntimeError("復元が1時間以内に完了しませんでした。写真を減らしてお試しください。") from None
            if code != 0:
                with log_file.open("rb") as log:
                    log.seek(max(0, log_file.stat().st_size - 4096))
                    tail = log.read().decode("utf-8", "replace").strip().splitlines()
                detail = tail[-1][-500:] if tail else f"exit {code}"
                raise RuntimeError(f"MASt3Rの実行に失敗しました。{detail}")
            result = {"status": "done", "message": "点群が完成しました。", "point_count": vertex_count(output)}
        except (OSError, RuntimeError, ValueError) as error:
            result = {"status": "error", "message": str(error)}
        finally:
            shutil.rmtree(job_dir / "photos", ignore_errors=True)
            with self.lock:
                self.process = None
                if job_id in self.records:
                    self.records[job_id] = result
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
            self.send_json(200, {"engine": "mast3r", "ready": ready, "message": message})
            return
        match = JOB_ROUTE.fullmatch(path)
        store = self.server.jobs
        record = store.get(match[1]) if match else None
        if not record:
            self.send_json(404, {"message": "このジョブは存在しないか、保存期間を過ぎました。"})
            return
        if not match[2]:
            self.send_json(200, record)
            return
        if record["status"] != "done":
            self.send_json(409, {"message": "点群はまだ作成されていません。"})
            return
        try:
            cloud = (store.directory / match[1] / "cloud.ply").open("rb")
        except FileNotFoundError:
            self.send_json(404, {"message": "点群の保存期間を過ぎました。"})
            return
        with cloud:
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Disposition", 'attachment; filename="cloud.ply"')
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(os.fstat(cloud.fileno()).st_size))
            self.end_headers()
            shutil.copyfileobj(cloud, self.wfile)

    def do_POST(self):
        if not self.origin_allowed():
            self.send_json(403, {"message": "この接続元は利用できません。"})
            return
        if urlsplit(self.path).path != "/api/jobs":
            self.send_json(404, {"message": "URLを確認してください。"})
            return
        ready, message = runtime_status()
        if not ready:
            self.send_json(503, {"message": message})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length <= 0 or length > MAX_BODY or self.headers.get("Transfer-Encoding"):
            self.send_json(413, {"message": "写真の合計は送信データを含め60 MiB以内にしてください。"})
            return
        store = self.server.jobs
        if not store.slot.acquire(blocking=False):
            self.send_json(409, {"message": "別の復元を実行中です。完了してから再度お試しください。"})
            return
        job_id = uuid.uuid4().hex
        job_dir = store.directory / job_id
        try:
            inputs = job_dir / "photos"
            inputs.mkdir(parents=True)
            read_photos(self.rfile, length, self.headers.get("Content-Type", ""), inputs)
            store.start(job_id)
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
