"""GUI wrapper around CXP-2024's unchanged recommended pipeline (denoise.py)."""
from pathlib import Path
import gc, sys, urllib.error
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parent
# Use upstream denoise.py: from a vendored copy if present, otherwise the repository root (contrib/akihim-web-gui/../../denoise.py).
for folder in (ROOT / 'vendor', ROOT.parent, ROOT.parent.parent):  # repo root is two levels up in contrib/
    if (folder / 'denoise.py').is_file():
        sys.path.insert(0, str(folder))
        break
import denoise as upstream  # noqa: E402

# Model folder: a local ./models copy if it already holds the weights, otherwise upstream's
# download cache (~/.cache/gpt-anime-denoise); the weights are downloaded there on first use.
MODEL_NAME = upstream.MODELS['esrgan'][0]
MODEL_DIR = ROOT / 'models' if (ROOT / 'models' / MODEL_NAME).is_file() else Path.home() / '.cache' / 'gpt-anime-denoise'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL = None
original_upscale = upstream.upscale

def safe_upscale(model, image, device):
    try:
        return original_upscale(model, image, device)
    except torch.cuda.OutOfMemoryError:
        pass
    gc.collect()
    torch.cuda.empty_cache()
    # Same network with overlapping tiles if a large input exhausts GPU memory.
    width, height = image.size
    result = Image.new('RGB', (width * 4, height * 4))
    tile, border = 192, 32
    for y in range(0, height, tile):
        for x in range(0, width, tile):
            x2, y2 = min(x + tile, width), min(y + tile, height)
            box = (max(0, x-border), max(0, y-border), min(width, x2+border), min(height, y2+border))
            patch = original_upscale(model, image.crop(box), device)
            patch = patch.crop(((x-box[0])*4, (y-box[1])*4, (x2-box[0])*4, (y2-box[1])*4))
            result.paste(patch, (x*4, y*4))
    return result

upstream.upscale = safe_upscale

def process(source, destination, strength, progress, downscale=.6):
    global MODEL
    if MODEL is None:
        if not (MODEL_DIR / MODEL_NAME).is_file():
            progress('首次使用：下載模型中（約 18 MB）', 5)
        progress('載入動漫去噪模型', 10)
        try:
            MODEL = upstream.load_model('esrgan', MODEL_DIR, DEVICE)
        except SystemExit as exc:   # upstream exits when the downloaded file fails its SHA-256 check
            raise RuntimeError('模型下載不完整（校驗不符），請重試。') from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f'模型下載失敗，請檢查網路後重試。（{exc.reason}）') from None
    else:
        progress('載入動漫去噪模型', 10)
    with Image.open(source) as opened:
        image = opened.copy()
    alpha = image.getchannel('A') if image.mode == 'RGBA' else None
    progress(f'預縮 {downscale:.0%}・重建乾淨色塊', 25)
    rgb = image.convert('RGB')
    small = rgb.resize((max(1, round(image.width*downscale)), max(1, round(image.height*downscale))), Image.Resampling.BOX)
    restored = upstream.upscale(MODEL, small, DEVICE).resize(image.size, Image.Resampling.LANCZOS)
    progress('L0 平滑・整理細碎紋理', 65)
    result = upstream.to_img(upstream.l0_smooth(upstream.to_arr(restored), strength))
    if alpha is not None:
        result.putalpha(alpha)
    progress('儲存去噪結果', 95)
    temporary = Path(str(destination) + '.tmp')
    result.save(temporary, format='PNG')
    temporary.replace(destination)
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()
    return torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'CPU'
