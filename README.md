# gpt-anime-denoise

去掉 GPT 生图（gpt-image-2 / 2.5 / 2.6、ChatGPT、Codex 内置 image_gen）在**二次元 / 番剧画风**图里留下的**鱼鳞状噪点**。

Removes the fish-scale texture that OpenAI's GPT image models leave in anime and TV-anime style images, without redrawing the picture.

**默认推荐：ESRGAN 缩 60% + L0 0.001** —— 对动漫与番剧画风有奇效。写实风格的图也用这个默认方案：我们试过几种专门为照片训练的去噪 / 超分模型，都不如它（见[写实风格的图](#写实风格的图)）。

**视频讲解（B 站）：** https://www.bilibili.com/video/BV1EC4B6rEMs/

> 如果您用本项目做出了满意的作品，麻烦在简介里 credit 本项目 [gpt-anime-denoise](https://github.com/CXP-2024/gpt-anime-denoise) 和作者 B 站 UP 主 [hihihihi202](https://space.bilibili.com/3546568969947566)。

## 交流群

QQ 群：**1130507071**，欢迎来做技术交流。群里聊 AI 视频、AI 建模、AI 绘画等内容，主打 agent base（dsh、Codex、Claude Code）+ 本地模型。

## 目录

- [效果](#效果)
  - [粉彩白底插画](#粉彩白底插画)
  - [电影感宽幅场景](#电影感宽幅场景)
  - [番剧截帧质感](#番剧截帧质感)
  - [八种方法对比](#八种方法对比)
- [图形界面](#图形界面)
  - [主要功能](#主要功能)
  - [局部保护笔刷](#局部保护笔刷)
  - [磁性套索与反选：只去背景或只去人物](#磁性套索与反选只去背景或只去人物)
  - [外观与快捷键](#外观与快捷键)
- [一键安装依赖](#一键安装依赖)
- [命令行用法](#命令行用法)
- [原理](#原理)
  - [试过但效果不好的方法](#试过但效果不好的方法)
  - [写实风格的图](#写实风格的图)
- [作为 Claude Code / Codex skill 使用](#作为-claude-code--codex-skill-使用)
- [致谢](#致谢)
- [许可](#许可)

## 效果

左边去噪前，右边默认方案去噪后。**请点开看原尺寸**，缩小后鳞纹看不出来。

### 粉彩白底插画

![furina before/after](examples/furina_before_after.webp)

### 电影感宽幅场景

![pool before/after](examples/pool_before_after.webp)

### 番剧截帧质感

![twins before/after](examples/twins_before_after.webp)

这张缩小看不明显，**请注意看刘海部分**。下面是左边角色右眼和刘海的局部，最近邻放大 3 倍（无损，没有插值平滑）：去噪前刘海的发丝色块里布满细碎的鳞状起伏，去噪后变成干净的平涂，发丝线条和高光的位置不变。

![twins bangs zoom](examples/twins_bangs_zoom.webp)

### 八种方法对比

每张都是原尺寸并排：原图、L0 0.0005、L0 0.001、APISR、CUGAN 缩 60%、CUGAN 缩 60% + L0、ESRGAN 缩 60%、**ESRGAN 缩 60% + L0（默认）**。用 `--all` 可以对自己的图生成同样的对比图，挑最合适的一种。

![furina sheet](examples/furina_sheet.webp)
![pool sheet](examples/pool_sheet.webp)
![twins sheet](examples/twins_sheet.webp)

## 图形界面

不想敲命令的话，用本机网页工具（依赖和命令行版一样，见下面「命令行用法」的安装那一行）：

```bash
python denoise_app.py      # 自动在浏览器打开 http://127.0.0.1:7865/
```

Windows 上也可以双击 `去噪工具.bat`：它依次找环境变量 `DENOISE_PYTHON`、本文件夹的 `.venv`、`py` 启动器和 PATH 里的 Python，用第一个装好了依赖的；都没有就在窗口里运行一键安装（见下面「一键安装依赖」），装完自动打开。图片只在本机处理，不会上传；结果自动存到 `outputs/`（可用环境变量 `ANIME_DENOISE_OUTPUT` 改位置）。

![GUI 总览](examples/gui_compare.webp)

### 主要功能

- **拖进来就去噪**：拖入、点击选择或 Ctrl+V 粘贴，可以一次放多张，底部缩略图排队依次处理；RTX 5080 上一张约 1 秒。
- **三档强度 + 微调**：轻柔（80% · 0.0006）/ 推荐（60% · 0.001）/ 强力（45% · 0.0018），下面的「微调参数」可以单独调预缩比例和 L0；松手就重跑，只改 L0 时复用超分结果，约 0.3 秒。正在处理时改参数，旧任务会立刻中止，直接按新参数重来。每张图记住自己的参数。
- **对比查看**：对比 / 原图 / 结果三种视图；拖分割线比较，或按 `X` 在全屏原图和全屏结果之间直接切换（没有分割线）；滚轮缩放（200% 以上按原像素显示），拖动平移，按住 Z 临时看原图。
- **局部保护**：用保护笔刷涂，或用磁性套索圈，涂过、圈过的地方按比例混回原图，不用重跑模型；配合反选，可以只给背景或只给人物去噪。详见下面两节。
- **保存**：右边显示的就是下载的 PNG，尺寸和原图一样，透明背景会保留；同时自动存一份到 `outputs/`，左下角点文件名就能在文件夹里定位。

### 局部保护笔刷

去噪有时会把皮肤的柔和阴影、腮红排线、细纹理、小字也一起抹平。用笔刷涂在这些地方，涂过的地方按比例混回原图，实时生效，不用重跑模型。下面两张都是同一张图上真实的涂抹：

![GUI 局部保护笔刷](examples/gui_mask.webp)

![局部保护前后对比](examples/gui_mask_crops.webp)

- 画面上方浮出工具条：画笔 / 橡皮 / 套索三种工具随时切换，大小（默认 14 px，按住空格转滚轮可以细调）、恢复原图比例（默认 90%）、柔边（默认 10 px，只向外渐变，0 是硬边）、撤销 / 重做、显示范围、反选、清除。
- 每一笔有自己的强度：「恢复原图」设的是接下来这几笔的比例（100% 就是完全不动），同一张图里可以有 90% 的地方和 60% 的地方；淡红色越深表示混回原图越多，柔边的渐变也直接显示出来。在涂过的地方再涂，会改成新的比例。
- 淡红范围默认「涂完自动隐藏」：画的时候只显示这一笔，松手后约 0.7 秒淡出；工具条上的眼睛按钮可改成一直显示，任何时候按住 M 都能看到全部已涂的地方。左边可以切到「未保护结果」，直接对比加保护前后。
- 去噪前也可以先涂；重新去噪后保护区会自动套用。

### 磁性套索与反选：只去背景或只去人物

想只给背景去噪、人物保持原样（或反过来），不用一笔一笔涂，用磁性套索圈一圈就行。下图是沿着人物轮廓圈到一半的样子：

![GUI 磁性套索](examples/gui_lasso.webp)

- **磁性套索**（按 `L`，或点左侧「磁性套索」/ 工具条上的「套索」）：和保护笔刷一样，圈住的地方保持原图，只是用圈选代替涂抹。在人物边缘单击一下当起点，然后沿着轮廓移动鼠标，不用按住，选框会自动吸附到附近最明显的边缘上，并且隔一段自动加一个锚点。拐角或边缘不明显的地方单击一下，手动加锚点；`Backspace` 删除上一个锚点。
- **闭合**：回到起点单击、双击或按回车。还没闭合时再按 `L`、点「完成」或换成画笔 / 橡皮，也会自动沿边缘闭合并填好，和笔刷画完就生效一样；按 `Esc` 则放弃这次选框。圈住的地方按当前「恢复原图」比例填成保护区，柔边照常生效。
- **减选**：按住 Alt 闭合，或者用右键闭合，就会把圈住的地方从保护区里减掉。
- **反选**（按 `I`，或左侧 / 工具条上的「反选」按钮）：保护区和去噪区互换，强度保持不变，可以撤销。

![套索与反选的结果](examples/gui_lasso_result.webp)

左边是圈住人物以后：淡红色是保护区，人物保持原图，只有背景去噪。右边是再按 `I` 反选：只给人物去噪，背景保持原图。

吸附优先用去噪后的结果找边缘（鱼鳞纹已经去掉，轮廓更干净）；还没去噪完时用原图找。透明背景图片的透明边缘也会被当作轮廓。

### 外观与快捷键

![GUI 深色外观](examples/gui_dark.webp)

- **外观**：跟随系统 / 浅色 / 深色。
- **快捷键**：`B` 开始 / 结束涂抹（橡皮状态下先切回画笔），`E` 橡皮，`L` 开始 / 结束磁性套索，`I` 反选，`Esc` 结束涂抹或套索（没闭合的选框会放弃），按住空格转滚轮或 `[` `]` 调画笔大小，右键拖动擦除，按住空格拖动画面，`1` `2` `3` 切换视图，`X` 全屏原图 / 结果互切，`←` `→` 切换图片，`Ctrl+S` 下载，`Ctrl+Z` 撤销（加 Shift 重做），`?` 打开说明。

## 一键安装依赖

```bash
python setup_env.py                      # 识别显卡，显示方案和下载量，确认后装到本文件夹的 .venv
python setup_env.py --kind cpu           # 也可以手动指定：auto / cuda / rocm / mps / cpu
```

只用 Python 标准库，装之前会先问。全部装在 `.venv`（和 AMD 需要的 `.denoise-tools`）里，不改系统，删掉这两个文件夹就是卸载。

| 显卡 | 装什么 | 需要你准备的 |
|---|---|---|
| NVIDIA | CUDA 版 PyTorch，自动选驱动能跑的最新 CUDA 版本 | 只要显卡驱动，不用装 CUDA Toolkit（PyTorch 自带 CUDA 运行库） |
| AMD（Windows） | AMD 官方的 ROCm 7.2.1 for Windows 版 PyTorch，支持 RX 7000 / 9000、Radeon PRO W7000、Ryzen AI Max | 较新的 AMD 驱动（AMD 为 7.2.1 指定 26.2.2）。这套只有 Python 3.12 的包，电脑上没有 3.12 时会用 uv 下一份独立的，不改 PATH 和注册表 |
| Apple Silicon（M 系列 Mac） | PyPI 上的普通 PyTorch wheel，自带 Metal（MPS） | — 不用装别的 |
| 其他 | CPU 版 | — |

装完会实际导入 torch 检查能不能用上显卡；驱动太旧时环境照样能用（CPU），并提示去更新驱动。

设备在运行时自动选，需要指定就加 `--device`：NVIDIA 用 `cuda`，Apple Silicon 用 `mps`，其余用 `cpu`（三平台同一套命令）。Metal 没有 float64，所以 L0 在 MPS 上跑单精度：作者在一张约 79 万像素的图上实测，和 CPU 结果相比只有 3 个像素差 2 级，其余完全一样。

## 命令行用法

```bash
pip install -r requirements.txt          # 先按 pytorch.org 装好对应 CUDA 的 torch（或用上面的 setup_env.py）
python denoise.py image.png              # 默认方案，输出在 image_denoise/
python denoise.py image.png --all        # 7 种候选 + 一张 4×2 对比图
python denoise.py ./folder --out ./clean # 批量
python denoise.py image.png --l0 0.002   # 自行调节 L0 强度
```

**L0 的值可以按图自行调节。** 0.001 只是我们推荐的值，不是唯一答案：
- 纹理还残留 → 调大（如 0.0015、0.002）；
- 淡阴影、腮红、细渐变被抹平或出现色阶 → 调小（如 0.0007、0.0005）；
- 实测 0.01 已经会把腮红抹掉，一般在 0.0005–0.002 之间找。

`--l0` 作用于所有带 `_l0` 的方法（默认方案、`cugan60_l0`），`--all` 的对比图也会按这个值标注。设成 0 就不做 L0，和网页工具把 L0 拖到 0 一样。

支持 PNG、JPEG、WebP、BMP、GIF（静态）、TIFF、AVIF。透明背景、EXIF 方向和 RGB 图的色彩配置（ICC）会保留，16 位图会正确转成 8 位，CMYK 按内嵌配置转成 RGB；输出都是 PNG。

模型在第一次运行时自动下载到 `~/.cache/gpt-anime-denoise`（共约 100 MB，带 SHA-256 校验），也可以用 `--models` 指定已有目录。有 NVIDIA 显卡时自动用 GPU，Apple Silicon 用 MPS，其余（集显、Intel Mac）用 CPU，结果一样，只是慢。

实测速度（1024×1536，约 157 万像素，和 Codex 默认出的 16:9 图 1672×941 差不多大；默认方案）：

| 设备 | 超分（缩 60% 后放大） | L0 平滑 | 合计 |
|---|---|---|---|
| RTX 5080 16GB | 0.6–1.4 秒 | 0.3–0.5 秒 | 约 1–2 秒 |
| Apple M4 Pro（MPS） | 2.6 秒 | 0.3 秒 | 约 3 秒 |
| 只用 CPU（Ryzen 7 9850X3D） | 13 秒 | 15 秒 | 约 28 秒 |

- 有 NVIDIA 显卡时 L0 也在显卡上跑（torch.fft，双精度），和 CPU 版逐像素一样；超分用全精度 FP32（关掉了 TF32），整体结果和 CPU 版只有约 0.7% 的像素差 1–5 级。
- 命令行每次启动要加载 Python 和 torch，单张总耗时约 5 秒；批量处理时只加载一次。
- 显存 / 内存够时超分整图一次处理（这个尺寸显卡峰值约 4.3 GB）。不够时自动把同样的计算按行分段做，每段都带上足够的相邻行，结果和整图一次处理逐像素一样，没有接缝，也不会溢出到共享内存：
  - 显卡只剩 3 GB 可用时，这个尺寸峰值约 1.1 GB，0.9 秒；2400×3600 的大图峰值约 1.2 GB，约 7 秒。
  - 没有显卡、内存只剩 2 GB 可用时，峰值约 2.2 GB（整图一次要 7.8 GB），慢约 2 秒。

## 原理

默认方案分两步：

1. **先缩到 60% 再用动漫超分放回原尺寸。** 鳞纹是很细很密的纹样，缩小时几个像素被平均成一个，纹样基本消失；Real-ESRGAN anime 6B 放大时按它学过的干净动漫图重新补细节，所以 GPT 的纹理不会被带回来。
2. **再做一次轻度 L0 平滑（λ = 0.001）。** L0 梯度最小化（Xu et al. 2011）让整张图里"颜色有变化的像素"尽量少：色块内部的小起伏被压成一块平色，真正的线条和色块边缘因为跳变大而原样保留，不会变软。它是纯数学优化，不会编造细节，眼睛、腮红的位置和颜色都不变。λ 越大抹得越平，太大（如 0.01）会把淡渐变和腮红抹掉；默认 0.001，可用 `--l0` 调节。

其他候选：

| 方法 | 说明 |
|---|---|
| L0 0.0005 / 0.001 | 只做 L0 平滑，改动最小 |
| APISR | 直接 2x 超分（不先缩小），本身已经很干净，不需要再叠 L0 |
| CUGAN 缩 60%（+ L0） | Real-CUGAN 降噪 3 档，比 ESRGAN 抹得更平 |
| ESRGAN 缩 60%（+ L0） | 默认方案 |

### 试过但效果不好的方法

- [Larryvrh/gpt-image-2-artifact-cleaner](https://github.com/Larryvrh/gpt-image-2-artifact-cleaner)：专门针对 GPT Image 2，但 α 开到 1.0 也只改善一点。
- 2x-AnimeSharpV4：几乎不改动画面。
- 导向滤波、双边滤波：会把画面抹糊。
- 1x-DitherDeleterV3-Smooth、1x-DeSharpen：都不如单用 L0。
- 扩散模型低强度重绘（img2img）、ControlNet：能去掉纹理，但会重画眼睛等小细节，角色设计会变。
- 频域陷波：这种纹理在频谱上没有明显的周期尖峰，没有可挖的频点。

### 写实风格的图

GPT 出的写实图也有这种细碎纹理，画面细节越多、灯光越复杂越明显，比如雨夜街道、霓虹城市、夜市。简单的写实图（干净背景的人像、单色墙面）本身纹理很少，不适合拿来测试。

我们用 Codex 生成了 4 张高细节夜景（雨夜小巷、未来都市、夜市、钟表铺），对比默认方案和 4 个为照片训练的开源模型：

| 方法 | 做法 | 平坦区纹理减少（4 张平均） | 色调偏移 |
|---|---|---|---|
| **默认方案** | 动漫超分 ESRGAN，缩 60% + L0 0.001 | **37%** | 3.7 |
| [RealESRGAN_x4plus](https://github.com/xinntao/Real-ESRGAN) | Real-ESRGAN 的照片模型，缩 60% / 80% 后放大 | 29% / 20% | 4.3 / 3.6 |
| [realesr-general-x4v3](https://github.com/xinntao/Real-ESRGAN) | Real-ESRGAN 的通用小模型，缩 60%，降噪强度 1.0 / 0.5 | 24% / 23% | 2.5 / 2.0 |
| [SCUNet](https://github.com/cszn/SCUNet) | 针对真实噪声的盲去噪，GAN / PSNR 两个版本，原尺寸 | 13% / 12% | 2.1 / 0.8 |
| [DRUNet](https://github.com/cszn/DPIR) | 可以指定噪声等级的去噪网络，σ = 15，原尺寸 | 27% | 0.2 |

- **平坦区纹理**：画面里比较平坦的区域还剩多少高频起伏。真实的细节也会算进去，所以这个数只能作参考，最终要看原尺寸的图。
- **色调偏移**：大半径模糊后和原图的平均色差（0–255），0 表示颜色完全没变。

结论：
- 照片模型去掉的纹理都比默认方案少，SCUNet 几乎原样留下了桌面那种颗粒质感。DRUNet 颜色最保真，但整体发软；general-x4v3 会把边缘锐化。
- 看原尺寸对比，这几种都不如默认方案，所以没有另做写实模式。
- 默认方案在写实图上会把皮肤这类大片柔和渐变压平一些。介意的话用「轻柔」档，或者用保护笔刷涂在脸上。

下面是雨夜小巷那张的原尺寸对比（9 格：原图、默认方案、7 种照片模型的设置）：

![photoreal sheet](examples/photoreal_sheet.webp)

## 作为 Claude Code / Codex skill 使用

把整个目录放进 `.claude/skills/gpt-anime-denoise/`（或 Codex 的 skills 目录），`SKILL.md` 会告诉 agent 在 GPT 出完二次元图后跑这一步。

## 致谢

- [@AkihiM](https://github.com/AkihiM) 在 [PR #1](https://github.com/CXP-2024/gpt-anime-denoise/pull/1) 里做了第一版网页界面。现在网页工具里的强度预设、浮动笔刷工具条、多图队列、处理进度、深色外观、自动保存、保留透明通道和显存不足时分块超分等设计，都借鉴了这个 PR 的思路，感谢。PR 已合并，原版界面完整保留在 [`contrib/akihim-web-gui/`](contrib/akihim-web-gui/)，也可以直接运行。
- [@juanhanyu](https://github.com/juanhanyu) 在 [PR #2](https://github.com/CXP-2024/gpt-anime-denoise/pull/2) 里加了一键安装依赖（`setup_env.py`，NVIDIA CUDA / AMD ROCm / CPU）和能自动找 Python 的 `去噪工具.bat`，在 [PR #3](https://github.com/CXP-2024/gpt-anime-denoise/pull/3) 里解决了 AMD 显卡上每种新尺寸都要卡约 90 秒的问题，感谢。
- [@a705876883-arch](https://github.com/a705876883-arch) 在 [PR #4](https://github.com/CXP-2024/gpt-anime-denoise/pull/4) 里给网页工具加了磁性套索和反选，可以只给背景或只给人物去噪，感谢。
- [@SugarCatCat](https://github.com/SugarCatCat) 在 [PR #5](https://github.com/CXP-2024/gpt-anime-denoise/pull/5) 里加了 Apple Silicon（M 系列 Mac）的 Metal 加速和对应的一键安装，感谢。

## 许可

- 代码：MIT（见 `LICENSE`）。
- 模型不随仓库分发，脚本从原作者发布地址下载，使用时遵守各自的许可：[Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)（BSD-3-Clause）、[APISR](https://github.com/Kiteretsu77/APISR)（GPL-3.0）、[Real-CUGAN](https://github.com/bilibili/ailab)。
- 示例图由 GPT 生成，只作效果演示：
  - 鲸娘 DeepSeek 角色设计来自 [ZipZipPipe《AI娘化》](https://www.pixiv.net/artworks/148186519)（作者声明 CC BY-NC-SA 4.0）。
  - 芙宁娜是《原神》角色（© HoYoverse），属于同人图。
  - 番剧截帧那张以 TV 动画《わたしが恋人になれるわけないじゃん、ムリムリ！》第 4 话官方先行场面图为构图参考（© みかみてれん・竹嶋えく／集英社・わたなれ製作委員会），画面里右下角的小字是生成时产生的乱码。
  - 写实夜景对比图是 Codex 按文字描述生成的原创场景。
  - 示例图不适用代码的 MIT 许可，不可商用。
