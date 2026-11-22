
import onnx
p = r"F:\gpt-anime-denoise\android\tools\models\realesrgan_x4plus_anime_6b.onnx"
m = onnx.load(p)
convs = [n for n in m.graph.node if n.op_type == "Conv"]
k3 = 0
for c in convs:
    for a in c.attribute:
        if a.name == "kernel_shape" and tuple(a.ints) == (3,3):
            k3 += 1
print("total convs:", len(convs), "3x3 convs:", k3)
print("=> receptive radius in input px (3x3 only, ignoring pixel-shuffle):", k3)
