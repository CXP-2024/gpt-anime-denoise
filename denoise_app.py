"""Local web app for gpt-anime-denoise.

    python denoise_app.py            # opens http://127.0.0.1:7865/ in the browser

Everything runs on this machine; the page only talks to this local server.
Launching it again while it runs just reopens the page (an older copy left running after an update is replaced).
Finished images are also saved to ./outputs (override with the ANIME_DENOISE_OUTPUT environment variable).
"""
import hashlib
import io
import json
import os
import re
import secrets
import signal
import subprocess
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
HERE = Path(__file__).resolve().parent
# Fingerprint of the backend code: a launch replaces a running copy whose code is older.
VERSION = hashlib.sha1(b"".join((HERE / n).read_bytes() for n in ("denoise_app.py", "denoise.py"))).hexdigest()[:12]
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _running_info():
    try:
        return json.loads(OPENER.open(f"http://127.0.0.1:{PORT}/api/info", timeout=1.5).read())
    except Exception:
        return None


if __name__ == "__main__":  # check before the slow torch import
    info = _running_info()
    if info and info.get("app") == "anime-denoise":
        if info.get("version") == VERSION or info.get("busy"):
            if "--no-browser" not in sys.argv:
                webbrowser.open(f"http://127.0.0.1:{PORT}/")
            sys.exit(0)
        try:  # outdated copy from before an update: replace it
            os.kill(int(info["pid"]), signal.SIGTERM)
            time.sleep(0.8)
        except Exception:
            pass
    elif info and "device" in info and "app" not in info:  # first app version (no pid, quit had no token)
        try:
            OPENER.open(urllib.request.Request(f"http://127.0.0.1:{PORT}/api/quit", data=b"", method="POST"), timeout=2)
            time.sleep(0.8)
        except Exception:
            pass

from PIL import Image, UnidentifiedImageError  # noqa: E402

sys.path.insert(0, str(HERE))
import denoise as dn  # noqa: E402

PAGE = HERE / "app" / "index.html"
ICON = HERE / "app" / "icon.ico"
OUTPUT = Path(os.environ.get("ANIME_DENOISE_OUTPUT", HERE / "outputs"))
MODELS_DIR = Path.home() / ".cache" / "gpt-anime-denoise"
DEVICE = dn.default_device()
DEVICE_NAME = (dn.torch.cuda.get_device_name(0) if DEVICE == "cuda"
               else "Apple GPU (MPS)" if DEVICE == "mps" else "CPU")  # ASCII: it also goes into a response header
MAX_BYTES = 1024 * 1024 * 1024  # a safety net only; the real limit is MAX_PIXELS
DRAIN_MAX = 2 * 1024 * 1024 * 1024  # an oversized upload up to this size is read and dropped so the error reaches the page
# Size guidance, in pixels (file size says little: the same image is a few MB as JPEG and tens of MB as PNG).
# Up to RECOMMEND_PIXELS: GPT output sizes, a few seconds. Above it the page notes it; above WARN_PIXELS it warns
# (measured on an RTX 5080 16 GB: 24 MP 25 s, 47 MP 6 min because L0 no longer fits on the GPU), but still processes.
# MAX_PIXELS is a hard limit only because the page keeps several full-size canvases and browsers give up beyond it.
RECOMMEND_PIXELS = 10_000_000
WARN_PIXELS = 25_000_000
MAX_PIXELS = 100_000_000
Image.MAX_IMAGE_PIXELS = MAX_PIXELS * 2
TOKEN = secrets.token_urlsafe(24)  # injected into the page; every POST must carry it

lock = threading.Lock()
models = {}
progress = {"busy": False, "stage": "", "started": 0.0, "job": ""}
cancelled = set()  # job ids the page gave up on; checked between steps and inside the L0 loop


def stage(text):
    progress["stage"] = text


def get_model(key):
    if key not in models:
        stage("首次使用：加载模型（第一次会自动下载约 18 MB）")
        try:
            models[key] = dn.load_model(key, MODELS_DIR, DEVICE)
        except SystemExit as e:  # denoise.py exits when a downloaded file fails its SHA-256 check
            raise RuntimeError("模型下载不完整（校验不符），请检查网络后重试") from e
        except OSError as e:
            raise RuntimeError(f"模型下载失败，请检查网络后重试（{e}）") from None
    return models[key]


def get_segmenter():
    """The anime character segmenter, downloaded (~195 MB, SHA-256 checked) the first time it is asked for."""
    if "animeseg" not in models:
        import animeseg
        stage("首次使用：下载人物识别模型（约 195 MB，只下载一次）")
        try:
            models["animeseg"] = animeseg.load(dn.fetch("animeseg", MODELS_DIR), DEVICE)
        except SystemExit as e:
            raise RuntimeError("人物识别模型下载不完整（校验不符），请检查网络后重试") from e
        except OSError as e:
            raise RuntimeError(f"人物识别模型下载失败，请检查网络后重试（{e}）") from None
    return models["animeseg"]


def segment(data):
    """PNG (8-bit grey) of how likely each pixel belongs to a character; transparent pixels count as background."""
    import animeseg
    src, alpha, _ = decode(data)
    with lock:  # one model on the GPU at a time
        progress.update(busy=True, started=time.time(), job="")
        try:
            net = get_segmenter()
            stage("识别人物")
            m = animeseg.character_mask(net, dn.to_arr(src).astype("float32"), DEVICE)
        finally:
            progress.update(busy=False, stage="", job="")
            dn._clear(DEVICE)
    if alpha is not None:
        m = m * (dn.np.asarray(alpha, dn.np.float32) / 255)
    buf = io.BytesIO()
    Image.fromarray((m * 255 + 0.5).astype("uint8"), "L").save(buf, "PNG", compress_level=1)
    return buf.getvalue(), float(m.mean())


def decode(data):
    """Size / animation checks, then denoise.open_image: (rgb, alpha or None, icc or None)."""
    try:
        with Image.open(io.BytesIO(data)) as im:
            if getattr(im, "is_animated", False):
                raise ValueError("请使用静态图片；动图请先导出要处理的那一帧")
            if im.width * im.height > MAX_PIXELS:
                raise ValueError(f"图片太大（{im.width}×{im.height}，{im.width * im.height / 1e4:.0f} 万像素），"
                                 "超过 1 亿像素，浏览器无法显示和编辑，请先缩小")
            if min(im.size) < 16:
                raise ValueError("图片至少要 16×16")
        return dn.open_image(io.BytesIO(data))
    except UnidentifiedImageError:
        raise ValueError("读不出这张图，支持 PNG、JPEG、WebP、BMP、GIF、TIFF、AVIF") from None
    except Image.DecompressionBombError:
        raise ValueError("图片太大，超过 1 亿像素，请先缩小") from None


def normalize(data):
    """The source as the server sees it, as PNG: the page shows and blends exactly these pixels."""
    rgb, alpha, icc = decode(data)
    buf = io.BytesIO()
    dn.save_png(rgb, buf, alpha, icc, compress_level=1)
    return buf.getvalue(), rgb.size, alpha is not None


sr_cache = OrderedDict()  # (image hash, downscale) -> super-resolved image; changing only L0 skips the model
SR_CACHE_SIZE = 8


def process(data, down, lam, job=""):
    src, alpha, icc = decode(data)
    key = (hashlib.sha1(data).hexdigest(), round(down, 4))
    stop = (lambda: job in cancelled) if job else None
    with lock:  # one image at a time on the GPU
        dn._check(stop)  # cancelled while waiting for the previous image
        progress.update(busy=True, started=time.time(), job=job)
        try:
            t0 = time.time()
            sr = sr_cache.get(key)
            cached = sr is not None
            if cached:
                sr_cache.move_to_end(key)
                stage("复用超分结果")
            else:
                model = get_model("esrgan")
                stage(f"预缩 {round(down * 100)}% · 动漫超分重建")
                dn.DOWN = down
                sr = dn.sr_at(model, src, DEVICE, stop)
                sr_cache[key] = sr  # kept even if L0 is cancelled next: the re-run will reuse it
                while len(sr_cache) > SR_CACHE_SIZE:
                    sr_cache.popitem(last=False)
            dn._check(stop)
            stage("L0 平滑 · 压平细碎纹理")
            out = dn.to_img(dn.l0_smooth(dn.to_arr(sr), lam, device=DEVICE, stop=stop)) if lam > 0 else sr.copy()
            dn._clear(DEVICE)
            stage("整理结果")
            elapsed = time.time() - t0
            buf = io.BytesIO()
            dn.save_png(out, buf, alpha, icc, compress_level=1)  # local transfer only: fast encode matters more than size
        finally:
            progress.update(busy=False, stage="", job="")
            cancelled.discard(job)
            dn._clear(DEVICE)
    return buf.getvalue(), elapsed, src.size, cached, alpha is not None


def safe_name(name):
    stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", Path(name).stem).strip(" .")[:90] or "image"
    return stem + ".png"


def reveal(path):
    if os.name == "nt":
        if path.is_file():
            # Explorer wants /select,"<path>". A list argument quotes the whole switch once the path has a space,
            # and Explorer then opens Documents instead. Windows paths cannot contain a double quote.
            subprocess.Popen(f'explorer /select,"{path}"')
        else:
            os.startfile(path.parent if not path.is_dir() else path)
    else:
        subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(path if path.is_dir() else path.parent)])


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype, headers=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        if self.close_connection:
            self.send_header("Connection", "close")
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode(), "application/json; charset=utf-8")

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            page = PAGE.read_bytes().replace(b"__TOKEN__", TOKEN.encode())
            self._send(200, page, "text/html; charset=utf-8",
                       {"Content-Security-Policy": "default-src 'self'; img-src 'self' blob: data:; "
                        "style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; "
                        "connect-src 'self'; frame-ancestors 'none'"})
        elif path == "/favicon.ico" and ICON.is_file():
            self._send(200, ICON.read_bytes(), "image/x-icon")
        elif path == "/api/info":
            self._json({"app": "anime-denoise", "version": VERSION, "pid": os.getpid(), "busy": progress["busy"],
                        "device": DEVICE_NAME, "gpu_note": dn.GPU_NOTE, "down": 0.6, "l0": dn.L0, "output": str(OUTPUT),
                        "max_bytes": MAX_BYTES, "pixels": {"recommend": RECOMMEND_PIXELS, "warn": WARN_PIXELS, "max": MAX_PIXELS}, "seg_ready": (MODELS_DIR / dn.MODELS["animeseg"][0]).is_file()})
        elif path == "/api/progress":
            p = dict(progress)
            p["elapsed"] = round(time.time() - p["started"], 1) if p["busy"] else 0
            self._json(p)
        else:
            self._send(404, b"not found", "text/plain")

    def _allowed(self):
        origin = self.headers.get("Origin")
        good_origin = origin is None or re.fullmatch(r"http://(127\.0\.0\.1|localhost):%d" % self.server.server_port, origin)
        return bool(good_origin) and secrets.compare_digest(self.headers.get("X-Token", ""), TOKEN)

    def _discard(self):
        # Read an unwanted request body before answering: a reply sent while the body is still arriving makes
        # Windows reset the connection, and the browser then shows only "Failed to fetch" instead of our message.
        try:
            n = int(self.headers.get("Content-Length", 0))
        except ValueError:
            n = 0
        if n > DRAIN_MAX:  # not worth reading; answer and close, the page checks the size before sending anyway
            self.close_connection = True
            return
        while n > 0:
            chunk = self.rfile.read(min(n, 1 << 20))
            if not chunk:
                break
            n -= len(chunk)

    def _body(self, limit=MAX_BYTES):
        try:
            n = int(self.headers.get("Content-Length", 0))
        except ValueError:
            n = 0
        if not 0 < n <= limit:
            self._discard()
            raise ValueError("图片为空" if n <= 0 else f"图片 {n / 2**20:.0f} MB，超过工具 {limit // 2**20} MB 的上限，请先压缩或裁小再试")
        return self.rfile.read(n)

    def do_POST(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if not self._allowed():
            self._discard()
            return self._json({"error": "页面已过期，请刷新或重新打开工具"}, 403)
        try:
            if u.path == "/api/denoise":
                down = min(max(float(q.get("down", ["0.6"])[0]), 0.1), 1.0)
                lam = min(max(float(q.get("l0", [str(dn.L0)])[0]), 0.0), 0.05)
                job = q.get("job", [""])[0][:64]
                try:
                    png, elapsed, (w, h), cached, has_alpha = process(self._body(), down, lam, job)
                except dn.Cancelled:
                    return self._json({"cancelled": True}, 409)
                return self._send(200, png, "image/png", {
                    "X-Elapsed": f"{elapsed:.1f}", "X-Device": DEVICE_NAME, "X-Size": f"{w}x{h}",
                    "X-SR-Cached": "1" if cached else "0", "X-Alpha": "1" if has_alpha else "0"})
            if u.path == "/api/segment":
                png, share = segment(self._body())
                return self._send(200, png, "image/png", {"X-Share": f"{share:.4f}"})
            if u.path == "/api/convert":
                png, (w, h), has_alpha = normalize(self._body())
                return self._send(200, png, "image/png", {"X-Size": f"{w}x{h}", "X-Alpha": "1" if has_alpha else "0"})
            if u.path == "/api/cancel":
                job = q.get("job", [""])[0][:64]
                if job:
                    cancelled.add(job)
                    if len(cancelled) > 200:  # stale ids of jobs that never arrived
                        cancelled.clear()
                return self._json({"ok": True, "running": progress["job"] == job})
            if u.path == "/api/save":
                data = self._body()
                with Image.open(io.BytesIO(data)) as im:
                    if im.format != "PNG":
                        raise ValueError("只能保存 PNG")
                name = safe_name(q.get("name", ["image.png"])[0])
                OUTPUT.mkdir(parents=True, exist_ok=True)
                tmp = OUTPUT / (name + ".tmp")
                tmp.write_bytes(data)
                tmp.replace(OUTPUT / name)
                return self._json({"file": name, "dir": str(OUTPUT)})
            if u.path == "/api/reveal":
                OUTPUT.mkdir(parents=True, exist_ok=True)
                name = q.get("name", [""])[0]
                reveal(OUTPUT / safe_name(name) if name else OUTPUT)
                return self._json({"ok": True})
            if u.path == "/api/quit":
                self._json({"ok": True})
                threading.Thread(target=self.server.shutdown, daemon=True).start()
                return
            return self._json({"error": "没有这个操作"}, 404)
        except ValueError as e:
            return self._json({"error": str(e)}, 400)
        except Exception as e:
            return self._json({"error": str(e)}, 500)


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
        if _running_info():
            if "--no-browser" not in sys.argv:
                webbrowser.open(f"http://127.0.0.1:{PORT}/")
            return
        srv = Server(("127.0.0.1", 0), Handler)
    url = f"http://127.0.0.1:{srv.server_port}/"
    print(f"Anime Denoise running at {url}  ({DEVICE_NAME}, outputs -> {OUTPUT})", flush=True)
    if "--no-browser" not in sys.argv:
        webbrowser.open(url)
    srv.serve_forever()


if __name__ == "__main__":
    main()
