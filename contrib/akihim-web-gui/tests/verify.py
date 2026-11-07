"""End-to-end check against a running GUI server (start it first).

Uploads a small transparent image, runs the default denoise on the GPU/CPU, and checks that the
output keeps the original size and exact alpha channel, and that the model matches upstream's SHA-256.
Note: this adds a "transparent-test.png" entry to the GUI's history.
"""
from pathlib import Path
import sys, json, os, urllib.request, io, time, re, hashlib
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from PIL import Image, ImageChops
import engine

data_dir = Path(os.environ.get('ANIME_DENOISE_DATA', root / '.data'))
output_dir = Path(os.environ.get('ANIME_DENOISE_OUTPUT', root / 'outputs'))
url = json.loads((data_dir / 'server.json').read_text())['url']
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
token = re.search(r'name="token" content="([^"]+)"', opener.open(url).read().decode())[1]

def post(route, body, filename=None):
    data = body if isinstance(body, bytes) else json.dumps(body).encode()
    headers = {'X-Token': token}
    if filename:
        headers['X-Filename'] = filename
    return json.load(opener.open(urllib.request.Request(url + route, data=data, headers=headers)))

image = Image.new('RGBA', (128, 96))
for y in range(image.height):
    for x in range(image.width):
        image.putpixel((x, y), (40 + x, 100 + y, 170, (x * 2) % 256))
buffer = io.BytesIO()
image.save(buffer, format='PNG')
job = post('/api/upload', buffer.getvalue(), 'transparent-test.png')
post('/api/process', {'id': job['id'], 'strength': .001})
end = time.time() + 300
while time.time() < end:
    result = json.load(opener.open(urllib.request.Request(url + '/api/job/' + job['id'], headers={'X-Token': token})))
    if result['status'] != 'running':
        break
    time.sleep(.5)
assert result['status'] == 'done', result
output = Image.open(output_dir / result['output'])
assert output.size == image.size and output.mode == 'RGBA'
assert ImageChops.difference(image.getchannel('A'), output.getchannel('A')).getbbox() is None
model = engine.MODEL_DIR / engine.MODEL_NAME
assert hashlib.sha256(model.read_bytes()).hexdigest() == engine.upstream.SHA256[model.name]
print('Verified original-size RGBA output, exact alpha preservation, and upstream model SHA-256.')
