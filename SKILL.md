---
name: gpt-anime-denoise
description: Remove the "GPT-flavoured" noise (fish-scale texture 鱼鳞纹, fragmented detail, fragmented shading) that GPT image models (gpt-image-2/2.5/2.6, ChatGPT, Codex image_gen) leave in anime, illustration and TV-anime style images. Use after generating or receiving any GPT-made anime image that will be delivered, or when the user mentions 鱼鳞纹, 鳞片噪点, GPT 风味噪点, GPT 图噪点, 碎片化细节, 碎片化阴影, fish-scale artifacts, or cleaning GPT anime images.
---

# 去除 GPT 二次元图的风味噪点

GPT 生图在二次元 / 番剧画风里会留下细密的鱼鳞状纹理。人眼在原尺寸下能看出来，但放大、高通滤波和频谱分析都不容易检测到——**不要自己判断"这张图没有噪点"**，每张要交付的 GPT 二次元图都跑一遍。

## 用法

```bash
python <skill目录>/denoise.py <图片或文件夹> [...]           # 默认方案：ESRGAN 缩 60% + L0 0.001
python <skill目录>/denoise.py <图片> --all                    # 7 种候选 + 4×2 原尺寸对比图
python <skill目录>/denoise.py <图片> --method apisr          # 指定单一方法
```

- 依赖：`torch`、`spandrel`、`numpy`、`pillow`、`psutil`（`requirements.txt`）。模型第一次运行时自动下载到 `~/.cache/gpt-anime-denoise`，有 SHA-256 校验；已有模型目录用 `--models` 指定。
- 设备自动选：NVIDIA 用 `cuda`，Apple Silicon（M 系列 Mac）用 `mps`，其余用 `cpu`；要指定就加 `--device cuda|mps|cpu`。
- 输出在 `<图片目录>/<文件名>_denoise/`，尺寸和输入一致，文件名 `<文件名>_<方法>.png`。
- L0 强度用 `--l0` 调（默认 0.001，只是推荐值）：纹理残留就调大到 0.0015–0.002，淡阴影/腮红被抹平就调小到 0.0005–0.0007。
- 方法：`esrgan60_l0`（默认）、`esrgan60`、`cugan60_l0`、`cugan60`、`apisr`、`l0_0.001`、`l0_0.0005`。

## 需要精细调节时：用本地网页界面（GUI）

命令行适合整张、批量处理。需要下面这些精细控制时，请使用本地网页界面，不要靠反复改命令行参数去凑：

- **边看边调强度**：预缩比例和 L0 用滑条调，松手就重跑（只改 L0 约 0.3 秒）；拖分割线对比原图和结果，滚轮放大看细节。
- **局部保护**：用保护笔刷涂，或用磁性套索圈住人物 / 背景，被抹掉的腮红、细纹理、小字按比例混回原图；反选可以只给背景或只给人物去噪。
- **上色修补**：眼睛高光里的黑点、被抹掉的细线这类小地方，用上色笔直接补颜色，按住 Alt / Ctrl 单击取色。

启动：

```bash
python <skill目录>/denoise_app.py      # 自动在浏览器打开 http://127.0.0.1:7865/
```

Windows 上也可以双击 `<skill目录>/去噪工具.bat`（还没装依赖时会先运行一键安装）。图片只在本机处理，结果自动存到 `<skill目录>/outputs/`（环境变量 `ANIME_DENOISE_OUTPUT` 可改位置）。用法和快捷键见网页右上角「说明」或 README「图形界面」一节。这些调节要靠人看图判断，agent 负责帮用户启动、说明怎么用。

## 交付规则

1. 默认交付 `esrgan60_l0`。
2. 用户想自己挑时跑 `--all`，把 `<文件名>_sheet.png`（原尺寸并排）发给用户；**给原尺寸并排图，不要发放大的局部**，放大后反而看不出差别。
3. 不要改用扩散重绘、ControlNet、导向/双边滤波来去噪：会重画眼睛或把画面抹糊（见 README「试过但效果不好的方法」）。
4. 写实照片风格的图也交付默认的 `esrgan60_l0`。RealESRGAN_x4plus、realesr-general-x4v3、SCUNet、DRUNet 都测过，效果都不如默认（见 README「写实风格的图」），不要再换成这些模型。
5. 用户对默认结果的某些局部不满意（细节被抹掉、只想处理背景或人物、个别小瑕疵），推荐用上面的本地网页界面精细调节，而不是一遍遍换方法重跑命令行。
