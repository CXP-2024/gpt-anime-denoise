import sys,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from protection import blend,read_protection
from PIL import Image
with tempfile.TemporaryDirectory() as directory:
    root=Path(directory);original=root/'original.png';cleaned=root/'clean.png';out=root/'out.png'
    image=Image.new('RGBA',(3,1),(200,100,50,73));image.save(original)
    Image.new('RGBA',(3,1),(100,50,0,200)).save(cleaned)
    mask=Image.new('L',(3,1));mask.putdata([0,128,255])
    blend(original,cleaned,mask,1,out)
    result=Image.open(out)
    assert result.getpixel((0,0))==(100,50,0,73)
    assert result.getpixel((2,0))==(200,100,50,73)
    assert result.getpixel((1,0))==(150,75,25,73)
    blend(original,cleaned,mask,.8,out)
    assert Image.open(out).getpixel((2,0))==(180,90,40,73)
    try:read_protection({'preserve':1.1},(3,1))
    except ValueError:pass
    else:raise AssertionError('Invalid preserve accepted')
print('PASS: unpainted output, 100% original, soft-edge blending, 80% blend, exact original alpha, bounds validation.')
