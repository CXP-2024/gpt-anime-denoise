"""Blend preserved original pixels into a completed denoise result."""
import base64, io, math
from PIL import Image

def read_protection(payload, size):
    amount = float(payload.get('preserve', .8))
    if not math.isfinite(amount) or not 0 <= amount <= 1:
        raise ValueError('保留原圖程度需介於 0%～100%')
    encoded = payload.get('protection_mask', '')
    if not encoded:
        return Image.new('L', size, 0), amount
    if not isinstance(encoded, str) or not encoded.startswith('data:image/png;base64,'):
        raise ValueError('保護遮罩格式錯誤')
    try:
        data = base64.b64decode(encoded.split(',', 1)[1], validate=True)
        with Image.open(io.BytesIO(data)) as image:
            if image.size != size:
                raise ValueError('保護遮罩尺寸需與原圖一致')
            mask = image.convert('L')
    except (OSError, base64.binascii.Error) as exc:
        raise ValueError('無法讀取保護遮罩') from exc
    return mask, amount

def blend(original, cleaned, mask, amount, destination):
    with Image.open(original) as image:
        source = image.copy()
    with Image.open(cleaned) as image:
        result = image.convert(source.mode)
    weights = mask.point([round(i * amount) for i in range(256)])
    merged = Image.composite(source, result, weights)
    if source.mode == 'RGBA':
        merged.putalpha(source.getchannel('A'))
    temp = destination.with_suffix('.tmp')
    merged.save(temp, format='PNG')
    temp.replace(destination)
