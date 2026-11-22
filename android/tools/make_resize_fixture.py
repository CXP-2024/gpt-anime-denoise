
import numpy as np
from PIL import Image
from pathlib import Path
g = Path(r"F:\gpt-anime-denoise\android\tools\golden")
src = Image.open(g / "in_01ee25d8.png").convert("RGB")
W, H = src.size
# Python round() = banker's
sw, sh = round(W * 0.6), round(H * 0.6)
small = src.resize((sw, sh), Image.BOX)
back = small.resize((W, H), Image.LANCZOS)
src.save(g / "in_resize.png")
small.save(g / "resize_boxed.png")
back.save(g / "resize_back.png")
print("src", src.size, "-> box", small.size, "-> lanczos back", back.size)
