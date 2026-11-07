# gpt-anime-denoise

去掉 GPT 生图（gpt-image-2 / 2.5 / 2.6、ChatGPT、Codex 内置 image_gen）在**二次元 / 番剧画风**图里留下的**鱼鳞状噪点**。

Removes the fish-scale texture that OpenAI's GPT image models leave in anime and TV-anime style images, without redrawing the picture.

**默认推荐：ESRGAN 缩 60% + L0 0.001** —— 对动漫与番剧画风有奇效。

**视频讲解（B 站）：** https://www.bilibili.com/video/BV1EC4B6rEMs/

> 如果您用本项目做出了满意的作品，麻烦在简介里 credit 本项目 [gpt-anime-denoise](https://github.com/CXP-2024/gpt-anime-denoise) 和作者 B 站 UP 主 [hihihihi202](https://space.bilibili.com/3546568969947566)。

## 交流群

QQ 群：**1130507071**，欢迎来做技术交流。群里聊 AI 视频、AI 建模、AI 绘画等内容，主打 agent base（dsh、Codex、Claude Code）+ 本地模型。

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

Windows 上也可以双击 `去噪工具.bat`（默认用 PATH 里的 `pythonw`，需要时在文件里改成自己的 Python）。图片只在本机处理，不会上传。

![GUI 滑动对比](examples/gui_compare.webp)

- **拖进来就去噪**：拖入图片、点击选择或 Ctrl+V 粘贴，自动按默认方案处理，RTX 5080 上一张约 1 秒。
- **滑动对比**：拖中间的分割线比较原图和结果；滚轮缩放，200% 以上按原像素显示不做平滑；拖动平移，双击适应画面，按住空格临时看原图。
- **调参即时重跑**：预缩比例（10–100%）和 L0 强度（0–0.005）松手就重跑；只改 L0 时复用超分结果，约 0.3 秒。
- **下载**：右边显示的就是下载的 PNG，尺寸和原图一样。

![GUI 保护区域遮罩](examples/gui_mask.webp)

- **保护区域（遮罩）**：腮红排线、细纹理、小字这类不想被抹平的地方，用画笔涂上（上图红色）。涂过的地方按比例混回原图，边缘自动柔化，实时生效，不用重跑模型。
  - 原图占比默认 80%；拉到 100% 就是完全不动。边缘柔化 0–150 px，设成 0 是硬边。
  - 「左边放：未加遮罩」可以直接对比加遮罩前后；红色默认只在画的时候显示，按住 M 临时查看。
  - 快捷键：B 画笔、E 橡皮、V 浏览、`[` `]` 调画笔大小、Ctrl+Z 撤销。

## 命令行用法

```bash
pip install -r requirements.txt          # 先按 pytorch.org 装好对应 CUDA 的 torch
python denoise.py image.png              # 默认方案，输出在 image_denoise/
python denoise.py image.png --all        # 7 种候选 + 一张 4×2 对比图
python denoise.py ./folder --out ./clean # 批量
python denoise.py image.png --l0 0.002   # 自行调节 L0 强度
```

**L0 的值可以按图自行调节。** 0.001 只是我们推荐的值，不是唯一答案：
- 纹理还残留 → 调大（如 0.0015、0.002）；
- 淡阴影、腮红、细渐变被抹平或出现色阶 → 调小（如 0.0007、0.0005）；
- 实测 0.01 已经会把腮红抹掉，一般在 0.0005–0.002 之间找。

`--l0` 作用于所有带 `_l0` 的方法（默认方案、`cugan60_l0`），`--all` 的对比图也会按这个值标注。

模型在第一次运行时自动下载到 `~/.cache/gpt-anime-denoise`（共约 100 MB，带 SHA-256 校验），也可以用 `--models` 指定已有目录。有 NVIDIA 显卡时自动用 GPU，没有就用 CPU。

实测速度（1024×1536，约 157 万像素，和 Codex 默认出的 16:9 图 1672×941 差不多大；默认方案）：

| 设备 | 超分（缩 60% 后放大） | L0 平滑 | 合计 |
|---|---|---|---|
| RTX 5080 16GB | 0.6–1.4 秒 | 0.3–0.5 秒 | 约 1–2 秒 |
| 只用 CPU（Ryzen 7 9850X3D） | 13 秒 | 15 秒 | 约 28 秒 |

- 有 NVIDIA 显卡时 L0 也在显卡上跑（torch.fft，双精度），输出和 CPU 版逐像素一样。显存不够时 L0 会自动退回 CPU。
- 命令行每次启动要加载 Python 和 torch，单张总耗时约 5 秒；批量处理时只加载一次。
- GPU 超分目前整图一次处理，这个尺寸峰值显存约 8.7 GB。显存不到 10 GB 的显卡处理大图可能爆显存，这时加 `--device cpu` 改用 CPU。

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

## 作为 Claude Code / Codex skill 使用

把整个目录放进 `.claude/skills/gpt-anime-denoise/`（或 Codex 的 skills 目录），`SKILL.md` 会告诉 agent 在 GPT 出完二次元图后跑这一步。

## 许可

- 代码：MIT（见 `LICENSE`）。
- 模型不随仓库分发，脚本从原作者发布地址下载，使用时遵守各自的许可：[Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)（BSD-3-Clause）、[APISR](https://github.com/Kiteretsu77/APISR)（GPL-3.0）、[Real-CUGAN](https://github.com/bilibili/ailab)。
- 示例图由 GPT 生成，只作效果演示：
  - 鲸娘 DeepSeek 角色设计来自 [ZipZipPipe《AI娘化》](https://www.pixiv.net/artworks/148186519)（作者声明 CC BY-NC-SA 4.0）。
  - 芙宁娜是《原神》角色（© HoYoverse），属于同人图。
  - 番剧截帧那张以 TV 动画《わたしが恋人になれるわけないじゃん、ムリムリ！》第 4 话官方先行场面图为构图参考（© みかみてれん・竹嶋えく／集英社・わたなれ製作委員会），画面里右下角的小字是生成时产生的乱码。
  - 示例图不适用代码的 MIT 许可，不可商用。
