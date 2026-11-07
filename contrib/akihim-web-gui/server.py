from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from PIL import Image, ImageOps, UnidentifiedImageError
from protection import read_protection, blend
import argparse, hashlib, io, json, logging, math, os, re, secrets, shutil, signal, subprocess, threading, time, urllib.parse, urllib.request, webbrowser

ROOT = Path(__file__).resolve().parent
API_VERSION = 2  # bump when the page needs new server endpoints; checked by web/app.js
# Fingerprint of the backend code: the launcher replaces an already-running server whose code is outdated.
VERSION = hashlib.sha1(b''.join((ROOT / n).read_bytes() for n in ('server.py', 'engine.py', 'protection.py'))).hexdigest()[:12]
DATA = Path(os.environ.get('ANIME_DENOISE_DATA', ROOT / '.data'))
OUTPUT = Path(os.environ.get('ANIME_DENOISE_OUTPUT', ROOT / 'outputs'))
DATA.mkdir(parents=True, exist_ok=True)
OUTPUT.mkdir(parents=True, exist_ok=True)
Image.MAX_IMAGE_PIXELS = 16_777_216
TOKEN = secrets.token_urlsafe(32)
LOCK = threading.RLock()
POOL = ThreadPoolExecutor(max_workers=1)
ACTIVE = None
JOBS = {}
STATIC = {'/': 'index.html', '/app.js': 'app.js', '/style.css': 'style.css'}
for p in DATA.glob('*/record.json'):
    try:
        job = json.loads(p.read_text(encoding='utf-8'))
        if job['status'] == 'running':
            job.update(status='error', error='上次處理已中斷，請重新執行。')
        JOBS[job['id']] = job
    except (ValueError, KeyError):
        pass

def save(job):
    path = DATA / job['id'] / 'record.json'
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(job, ensure_ascii=False), encoding='utf-8')
    temp.replace(path)

def run_job(id, strength, downscale):
    global ACTIVE
    job = JOBS[id]
    def progress(stage, percent):
        with LOCK:
            job.update(stage=stage, percent=percent)
    try:
        from engine import process
        name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', Path(job['name']).stem).rstrip('. ')[:65] or 'image'
        filename = f'{name}_denoise_s{downscale*100:g}_l0-{strength:g}_{id[:8]}_{time.time_ns()}.png'
        device = process(DATA / id / 'original.png', OUTPUT / filename, strength, progress, downscale)
        with LOCK:
            job['base_output'] = filename
            output = apply_protection(job)
            job.update(status='done', percent=100, stage='處理完成', output=output, device=device,
                       elapsed=round(time.time()-job['started'], 1), result_strength=strength, result_downscale=downscale)
    except (Exception, SystemExit) as exc:  # SystemExit: denoise.py exits on a bad model download
        logging.exception('Denoise failed')
        with LOCK:
            job.update(status='error', error='處理失敗：' + str(exc), stage='處理失敗')
    finally:
        with LOCK:
            save(job)
            ACTIVE = None

def apply_protection(job):
    """Return the output filename: the plain result, or one blended with the saved protection mask.
    The protected file name is stable per result, so repeated edits overwrite instead of piling up."""
    base = job.get('base_output') or job['output']
    if not job.get('protected') or job.get('preserve', .8) == 0:
        return base
    filename = Path(base).stem + '_protected.png'
    with Image.open(DATA / job['id'] / 'protection.png') as image:
        mask = image.convert('L')
    blend(DATA / job['id'] / 'original.png', OUTPUT / base, mask, job['preserve'], OUTPUT / filename)
    return filename

def download_name(job):
    stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', Path(job['name']).stem).rstrip('. ') or 'image'
    return stem + '_denoise.png'

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def reply(self, value, status=200):
        self.send_bytes(json.dumps(value, ensure_ascii=False).encode(), 'application/json; charset=utf-8', status)

    def send_bytes(self, data, mime, status=200, download=None, cache=False):
        self.send_response(status)
        self.send_header('Content-Type', mime)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'private, max-age=31536000, immutable' if cache else 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Content-Security-Policy', "default-src 'self'; img-src 'self' blob: data:; style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'")
        if download:
            self.send_header('Content-Disposition', "attachment; filename*=UTF-8''"+urllib.parse.quote(download))
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def authorized(self):
        return secrets.compare_digest(self.headers.get('X-Token', ''), TOKEN)

    def do_GET(self):
        route = urllib.parse.urlsplit(self.path)
        query = urllib.parse.parse_qs(route.query)
        if route.path == '/api/ping':
            return self.reply({'app': 'anime-denoise-local', 'pid': os.getpid(), 'version': VERSION, 'busy': bool(ACTIVE)})
        if route.path.startswith('/api/'):
            if not self.authorized():
                return self.reply({'error': '頁面已過期，請重新開啟本機工具'}, 403)
            with LOCK:
                if route.path == '/api/state':
                    return self.reply({'jobs': sorted(JOBS.values(), key=lambda j: j['created'], reverse=True), 'active': ACTIVE, 'api': API_VERSION})
                id = route.path.removeprefix('/api/job/')
                if id in JOBS:
                    return self.reply(JOBS[id])
            return self.reply({'error': '找不到處理紀錄'}, 404)
        if route.path in STATIC:
            name = STATIC[route.path]
            content = (ROOT / 'web' / name).read_bytes()
            if name == 'index.html':
                content = content.replace(b'__TOKEN__', TOKEN.encode())
            mime = {'html': 'text/html', 'js': 'text/javascript', 'css': 'text/css'}[name.rsplit('.', 1)[1]]
            return self.send_bytes(content, mime + '; charset=utf-8')
        match = re.fullmatch(r'/image/([a-f0-9]{32})/(original|result|base|mask|thumb)', route.path)
        if match:
            id, kind = match.groups()
            with LOCK:
                job = dict(JOBS.get(id) or {})
            if job:
                folder = DATA / id
                path = {
                    'original': folder / 'original.png',
                    'mask': folder / 'protection.png',
                    'thumb': folder / 'thumb.png',
                    'result': OUTPUT / job.get('output', 'missing'),
                    'base': OUTPUT / (job.get('base_output') or job.get('output') or 'missing'),
                }[kind]
                if kind == 'thumb' and not path.is_file():
                    try:
                        with Image.open(folder / 'original.png') as image:
                            image.thumbnail((192, 192), Image.Resampling.LANCZOS)
                            image.save(path)
                    except OSError:
                        pass
                if path.is_file():
                    download = download_name(job) if query.get('download') == ['1'] else None
                    return self.send_bytes(path.read_bytes(), 'image/png', download=download, cache=kind in ('original', 'thumb'))
        self.reply({'error': '找不到內容'}, 404)

    def do_POST(self):
        global ACTIVE
        if not self.authorized() or self.headers.get('Origin', self.server.base_url) != self.server.base_url:
            return self.reply({'error': '請從本機工具頁面操作'}, 403)
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if length <= 0 or length > 32*1024*1024:
                return self.reply({'error': '單張圖片上限為 32 MB'}, 413)
            data = self.rfile.read(length)
            if self.path == '/api/upload':
                with Image.open(io.BytesIO(data)) as im:
                    if im.width*im.height > 16_777_216 or min(im.size) < 16:
                        raise ValueError('圖片需至少 16 × 16，最多 1,677 萬像素。')
                    if getattr(im, 'is_animated', False):
                        raise ValueError('請使用靜態圖片；動畫請先匯出要處理的影格。')
                    image = ImageOps.exif_transpose(im).convert('RGBA' if 'A' in im.getbands() or 'transparency' in im.info else 'RGB')
                id = secrets.token_hex(16)
                name = urllib.parse.unquote(self.headers.get('X-Filename', 'image.png'))[:150] or 'image.png'
                folder = DATA / id
                folder.mkdir()
                image.save(folder / 'original.png')
                job = dict(id=id, name=name, width=image.width, height=image.height, alpha=image.mode == 'RGBA',
                           created=time.time(), status='ready', strength=.001, downscale=.6)
                with LOCK:
                    JOBS[id] = job
                    save(job)
                return self.reply(job)
            payload = json.loads(data)
            id = payload.get('id')
            if self.path == '/api/protection':
                # Saves the brush mask; when a result exists, also refreshes the blended output file.
                with LOCK:
                    job = JOBS.get(id)
                    if not job:
                        return self.reply({'error': '找不到這張圖片'}, 404)
                    if ACTIVE == id:
                        return self.reply({'error': '這張圖片正在去噪，完成後再調整保護區'}, 409)
                    size = (job['width'], job['height'])
                mask, amount = read_protection(payload, size)
                with LOCK:
                    if ACTIVE == id:
                        return self.reply({'error': '這張圖片正在去噪，完成後再調整保護區'}, 409)
                    mask.save(DATA / id / 'protection.png')
                    job.update(preserve=amount, protected=bool(mask.getbbox()), mask_version=time.time_ns())
                    if job.get('output'):
                        job.setdefault('base_output', job['output'])
                        job['output'] = apply_protection(job)
                    save(job)
                    return self.reply(job)
            if self.path == '/api/process':
                strength = float(payload.get('strength', .001))
                downscale = float(payload.get('downscale', .6))
                if not math.isfinite(downscale) or not .1 <= downscale <= 1:
                    raise ValueError('預縮比例需介於 10%～100%。')
                if not math.isfinite(strength) or not .0001 <= strength <= .01:
                    raise ValueError('平滑強度需介於 0.0001～0.01。')
                with LOCK:
                    if ACTIVE:
                        return self.reply({'error': '另一張圖片仍在處理，請稍候'}, 409)
                    if id not in JOBS:
                        return self.reply({'error': '請先加入圖片'}, 404)
                    job = JOBS[id]
                    if 'protection_mask' in payload:
                        mask, amount = read_protection(payload, (job['width'], job['height']))
                        mask.save(DATA / id / 'protection.png')
                        job.update(preserve=amount, protected=bool(mask.getbbox()), mask_version=time.time_ns())
                    # Keep the previous result visible while the new one is computed; remember which
                    # parameters that visible result was made with.
                    if job.get('output') and 'result_strength' not in job:
                        job.update(result_strength=job.get('strength', .001), result_downscale=job.get('downscale', .6))
                    job.update(status='running', started=time.time(), percent=3, stage='準備處理', strength=strength, downscale=downscale, error=None)
                    ACTIVE = id
                    save(job)
                    POOL.submit(run_job, id, strength, downscale)
                    return self.reply(job)
            if self.path == '/api/delete':
                if not isinstance(id, str) or not re.fullmatch(r'[a-f0-9]{32}', id):
                    raise ValueError('紀錄編號錯誤')
                with LOCK:
                    if ACTIVE == id:
                        return self.reply({'error': '處理中的圖片無法移除'}, 409)
                    JOBS.pop(id, None)
                shutil.rmtree(DATA / id, ignore_errors=True)
                return self.reply({'ok': True})
            if self.path == '/api/reveal':
                with LOCK:
                    job = JOBS.get(id) or {}
                    target = OUTPUT / job['output'] if job.get('output') else None
                if os.name == 'nt':
                    if target and target.is_file():
                        subprocess.Popen(f'explorer /select,"{target}"')
                    else:
                        os.startfile(OUTPUT)
                return self.reply({'ok': True})
            if self.path == '/api/shutdown':
                with LOCK:
                    if ACTIVE:
                        return self.reply({'error': '圖片處理完成後才能結束工具'}, 409)
                self.reply({'ok': True})
                threading.Thread(target=self.server.shutdown, daemon=True).start()
                return
            self.reply({'error': '找不到操作'}, 404)
        except (ValueError, UnidentifiedImageError, Image.DecompressionBombError, OSError) as exc:
            self.reply({'error': str(exc) if not isinstance(exc, UnidentifiedImageError) else '無法讀取圖片，請使用 PNG、JPEG、WebP 或 BMP。'}, 400)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-browser', action='store_true')
    parser.add_argument('--port', type=int, default=0)
    args = parser.parse_args()
    session = DATA / 'server.json'
    if not args.no_browser and session.exists():
        try:
            old = json.loads(session.read_text())
            request = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            info = json.loads(request.open(old['url']+'/api/ping', timeout=1).read())
            if info.get('app') == 'anime-denoise-local':
                if info.get('version') == VERSION or info.get('busy'):
                    # Same code (or still processing an image): just bring the page back.
                    webbrowser.open(old['url'])
                    return
                # Outdated server left running from before an update: replace it.
                os.kill(int(info['pid']), signal.SIGTERM)
                time.sleep(.5)
        except Exception:
            pass
    logging.basicConfig(filename=DATA/'error.log', level=logging.ERROR, encoding='utf-8')
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    server.base_url = 'http://127.0.0.1:'+str(server.server_port)
    session.write_text(json.dumps({'url': server.base_url, 'pid': os.getpid()}), encoding='utf-8')
    if not args.no_browser:
        webbrowser.open(server.base_url)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        POOL.shutdown(wait=True)

if __name__ == '__main__':
    main()
