# 二次元去噪工具 · Android

把 [gpt-anime-denoise](../README.md) 的默认方案（缩 60% + 超分 + L0 平滑）做成了手机 App。
图片只在本机处理，不联网、不上传。

## 安装

```bat
cd android
build.bat release    :: 编译 release 包（21 MB）并安装
build.bat            :: 编译 debug 包并安装（**仅供调试，慢 7 倍**）
build.bat selftest   :: 真机自检（debug 包）
```

**务必安装 release 版。** debug 包的热循环（FFT、L0）由 ART 解释执行，比 AOT 编译的
release 版慢 7 倍；L0 在 1 MP 图上就要 25 秒而不是 3.4 秒。这不是算法差异，是构建差异。

## 实测性能（Redmi Note 12 Turbo，2160×3840 = 8.3 MP，release 版）

| 阶段 | 耗时 |
|---|---|
| 超分 v3 | 11.0 s |
| L0 平滑 | 32.5 s |
| **总计** | **124 s** |

## 与桌面版的一致性

每一段都对着 `denoise.py` 验证（`app/src/test/`），并在真机复验：

| 环节 | 结果 |
|---|---|
| BOX 缩 60% | 与 Pillow **逐字节相同** |
| LANCZOS 放大 | 与 Pillow **逐字节相同** |
| L0（float64 路径） | 与 numpy **逐字节相同** |
| L0（大图 float32 快速路径） | 最大差 2 级（8-bit 下不可见） |
| 整条管线（512²，6B 参考） | v3 模型的预期差异，管线本身无错 |

## 手机版的取舍

**模型**：手机端只用 Real-ESRGAN animevideov3（约为桌面 6B 模型 1/9 的计算量）。
在真机上对比：90% 像素有 1-2 级差异、均值 3.5 级，集中在线条上——略软，但色彩、
构图、眼睛位置完全一致。这是让 8.3 MP 在两分钟内完成的必要取舍。

**大图管线**：8.3 MP 的整图 L0 需要约 1 GB 工作集，会被系统低内存守护进程杀死。
所以手机版把图片切成水平条带流过整条管线（每条带的中间结果用完即弃）：

- 条带的缩放用的是与整图相同的系数表，**分带结果与整图缩放逐字节相同**（有测试）；
- L0 在每个条带内再分块，实测分块解与整图解 1.6% 像素差超过 1 级；
- L0 用 float32 存储、beta 在 1e4 截断（约 9 次迭代而非 22），合计最大 2 级漂移。

**构建**：release 用 R8 压缩 + ART AOT 编译。debug 包解释执行是当初性能问题的根源。

## 硬件加速：实测后全部放弃

| 路径 | 实测 |
|---|---|
| CPU (ORT) | **唯一能用的**，release 版 AOT 后已够快 |
| NNAPI | Google 已弃用；ORT 的 Android 包根本没编译它 |
| XNNPACK EP | 官方包没编译，第三方包上有，但比纯 CPU 慢 |
| QNN (Adreno GPU) | **SIGSEGV**，崩在 ORT 内部 createSession |
| ExecuTorch Vulkan | 128² tile 可用但快不了多少；256² 及以上原生 abort |
| ExecuTorch XNNPACK | 与 ONNX Runtime CPU 相当，无收益 |
| 天玑 NPU | NeuroPilot SDK 不对普通开发者开放 |

## 已知限制

- 只做默认方案；三档强度 + 预缩/L0 微调。
- L0 是全局频域求解，半分辨率做会改变谱结构（实测 22% 像素差 >1 级，不可接受），
  所以大图走的是分块而非降分辨率。
- 写实照片不在测试范围内，与桌面版一致。

## 目录

```
app/src/main/java/com/animedenoise/
  core/        纯 JVM 图像算法（FFT、L0、PIL 复刻缩放、分块），电脑上可直接单测
  sr/          ONNX Runtime 封装、分块推理、后端选择
  image/       解码 / 编码 / 保存到相册
  ui/          Compose 界面
  SelfTest.kt  真机自检（adb am start --ez selftest true 触发）
tools/         模型导出、golden fixture、画质对比与验证脚本
```
