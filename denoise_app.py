"""Local web app for gpt-anime-denoise.

    python denoise_app.py            # opens http://127.0.0.1:7865/ in the browser

Everything runs on this machine; the page only talks to this local server.
Launching it again while it runs just reopens the page.
"""
import hashlib
import io
import json
import sys
import threading
import time
import urllib.request
import webbrowser
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

PORT = 7865  # fixed, so a second launch just reopens the running app


def _already_running():
    try:
        urllib.request.build_opener(urllib.request.ProxyHandler({})).open(
            f"http://127.0.0.1:{PORT}/api/info", timeout=1.5)
        return True
    except Exception:
        return False


if __name__ == "__main__" and _already_running():  # check before the slow torch import
    if "--no-browser" not in sys.argv:
        webbrowser.open(f"http://127.0.0.1:{PORT}/")
    sys.exit(0)

from PIL import Image  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import denoise as dn  # noqa: E402

PAGE = HERE / "app" / "index.html"
MODELS_DIR = Path.home() / ".cache" / "gpt-anime-denoise"
DEVICE = "cuda" if dn.torch.cuda.is_available() else "cpu"
DEVICE_NAME = dn.torch.cuda.get_device_name(0) if DEVICE == "cuda" else "CPU"
MAX_BYTES = 80 * 1024 * 1024

lock = threading.Lock()
models = {}


def get_model(key):
    if key not in models:
        models[key] = dn.load_model(key, MODELS_DIR, DEVICE)
    return models[key]


sr_cache = OrderedDict()  # (image hash, downscale) -> super-resolved image; changing only L0 skips the model
SR_CACHE_SIZE = 6


def process(data, down, lam):
    src = Image.open(io.BytesIO(data)).convert("RGB")
    key = (hashlib.sha1(data).hexdigest(), round(down, 4))
    with lock:  # one image at a time on the GPU
        t0 = time.time()
        sr = sr_cache.get(key)
        cached = sr is not None
        if cached:
            sr_cache.move_to_end(key)
        else:
            dn.DOWN = down
            sr = dn.sr_at(get_model("esrgan"), src, DEVICE)
            sr_cache[key] = sr
            while len(sr_cache) > SR_CACHE_SIZE:
                sr_cache.popitem(last=False)
        t_sr = time.time() - t0
        out = dn.to_img(dn.l0_smooth(dn.to_arr(sr), lam, device=DEVICE)) if lam > 0 else sr
        if DEVICE == "cuda":
            dn.torch.cuda.empty_cache()
        elapsed = time.time() - t0
    buf = io.BytesIO()
    out.save(buf, "PNG", compress_level=1)  # local transfer only: fast encode matters more than size
    return buf.getvalue(), elapsed, src.size, cached, t_sr


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype, headers=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode(), "application/json; charset=utf-8")

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
        elif path == "/api/info":
            self._json({"device": DEVICE_NAME, "down": 0.6, "l0": dn.L0})
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        u = urlparse(self.path)
        if u.path == "/api/quit":
            self._json({"ok": True})
            threading.Thread(target=self.server.shutdown, daemon=True).start()
            return
        if u.path != "/api/denoise":
            return self._send(404, b"not found", "text/plain")
        n = int(self.headers.get("Content-Length", 0))
        if not 0 < n <= MAX_BYTES:
            return self._json({"error": "图片为空或太大"}, 400)
        q = parse_qs(u.query)
        try:
            down = min(max(float(q.get("down", ["0.6"])[0]), 0.1), 1.0)
            lam = min(max(float(q.get("l0", [str(dn.L0)])[0]), 0.0), 0.05)
            png, elapsed, (w, h), cached, t_sr = process(self.rfile.read(n), down, lam)
        except Exception as e:
            return self._json({"error": str(e)}, 500)
        self._send(200, png, "image/png", {"X-Elapsed": f"{elapsed:.1f}", "X-Device": DEVICE_NAME,
                                           "X-Size": f"{w}x{h}", "X-SR-Cached": "1" if cached else "0",
                                           "X-SR-Time": f"{t_sr:.2f}"})


class Server(ThreadingHTTPServer):
    allow_reuse_address = False  # on Windows reuse would let two copies share the port

    def server_bind(self):
        import socket
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


def main():
    try:
        srv = Server(("127.0.0.1", PORT), Handler)
    except OSError:  # started at the same moment as another copy, or the port is taken by something else
        if _already_running():
            if "--no-browser" not in sys.argv:
                webbrowser.open(f"http://127.0.0.1:{PORT}/")
            return
        srv = Server(("127.0.0.1", 0), Handler)
    url = f"http://127.0.0.1:{srv.server_port}/"
    print(f"Anime Denoise running at {url}  ({DEVICE_NAME})", flush=True)
    if "--no-browser" not in sys.argv:
        webbrowser.open(url)
    srv.serve_forever()


if __name__ == "__main__":
    main()
