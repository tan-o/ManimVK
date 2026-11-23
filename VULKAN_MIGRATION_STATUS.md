# Manim Vulkan 迁移状态

基线：Manim Community v0.22.1，commit `9aa10de80ebd1192254d63a79db8f559c70f92be`。

当前状态：Windows 端默认使用 Vulkan 渲染，`Scene / Mobject / Camera / Animation / CLI` 上层 API 保持不变；官方组件和 `example_scenes/opengl.py` 中的官方 OpenGL 示例已验证。旧 OpenGL 桥已从 CLI 和 `manim/renderer/vulkan` 的渲染入口删除，不提供旧的 GL 回退路径。

## 渲染架构

- Rust 扩展 `native/manim_vk`：`ash` 管理 Vulkan 设备、交换链和 `VkImage`，`skia-safe` 的 Ganesh Vulkan 后端负责 2D 光栅化。
- Python 端 `manim/renderer/skia/`：保留经典 `Camera`/`Mobject` 绘制流程，只替换 Cairo 的 raster target。
- `SkiaRenderer`：默认优先选择 `DISCRETE_GPU`（当前机器为 NVIDIA RTX 3050 Ti Laptop），可通过 `MANIM_VK_DEVICE` 覆盖设备选择。
- `manim/_config/default.cfg` 的默认 `renderer = vulkan`；`CAIRO` 仍可显式使用，`OPENGL` 由兼容 facade 映射到经典类。
- `Mesh/Object3D` 以逐三角形 Skia path 绘制；HSB、`design_3` 全屏 shader 在 native 侧以 SkSL `RuntimeEffect` 执行。

## 性能第 3 点：减少每帧回读与 Python 开销

已落地：

1. `SkiaRenderTarget.commit_into()`：GPU 回读直接写入 C-contiguous NumPy 数组，去掉 `Vec<u8>` → Python `bytes` → `np.frombuffer` 的中间对象和额外拷贝。
2. `SkiaRenderer._render_camera()` 不再每帧无条件 `commit()`；普通单相机帧留在 GPU surface，只有 MultiCamera 读取 `sub_target.pixels` 前显式提交。
3. `SkiaDrawingContext` 的普通矢量批次不再 `commit()`；背景、点云、图片参与 CPU 合成前才显式提交。`draw_meshes` 同样移除无条件提交。
4. `Manager._wants_frame_pixels()` 针对 Skia 只在 `video`、`image_sequence`、`still` 输出时要求 CPU 像素；无输出和纯预览场景不再回读。
5. `SkiaRenderTarget.read_pixels()` 使用 `_last_readback` 保存直接读取结果，避免旧路径 `commit()` 后再 `copy()` 的第二次整帧拷贝；`mark_cpu_dirty()`、`commit()`、`clear()` 已处理快照与可复用缓冲的同步。
6. `VMobject.get_subpath_split_indices_from_points()` 增加单轮廓快速路径：相邻锚点完全相等时直接返回 `[[0, n_pts]]`，覆盖 Circle、矩形及多数仿射变换后的 Text 字形。

实测（Windows，NVIDIA RTX 3050 Ti Laptop，关闭缓存）：

| 场景 | 旧路径 | 当前路径 | 结果 |
|---|---:|---:|---|
| 1080p60，10 个 Circle，2s | 2.67–2.69s | 2.31–2.43s | 约 12% 下降 |
| 480p15，1000 个 Circle，2.5s | 约 2.99s | 约 2.87–3.05s | 基本持平，瓶颈在 Python 矢量遍历 |
| 480p，38 帧直接回读 | — | 约 0.119s | `commit_into` 直接写入 |

结论：稀疏/高分辨率场景的回读收益明显；对象数很多的 2D 场景主要受 Python 逐对象绘制限制，不再继续做 Text 同色批处理（实测反而更慢，已撤销）。

## 验证结果

- `tests/module`：1060 passed，1 skipped。
- `tests/opengl`：164 passed，1 skipped，4 xfailed。
- `tests/test_camera.py`：40 passed。
- 聚焦 renderer、manager、vectorized mobject、camera：115 passed。
- `tests/test_scene_rendering` 全量 xdist：223 passed，3 failed，1 skipped，1 xfailed，1 xpassed。失败全部位于 `tests/test_scene_rendering/test_parallel_encoding.py`，是 xdist 下 partial movie 文件落盘可见性的时序问题；同一文件串行运行 `46 passed`，与本次性能改动无直接关系。
- 官方 `InlineFullScreenQuad`、`NamedFullScreenQuad`、`SurfaceExample` 输出与旧图逐像素一致；`PointCloudCreateProbe`、`OpenGLImageZoo` 等合成路径输出正常。
- `StaticScene`（Circle + Text + MathTex）文本、公式、半透明圆正常；与旧图仅有抗锯齿/绘制顺序级微差。

## 已知限制

- 任意第三方 GLSL 没有通用转译器；目前只有 HSB、`design_3` 和简单 uniform 色有对应 native 实现。
- 原生窗口已能显示 Vulkan 预览，但键盘/鼠标事件到 `Scene.on_key_*`、`on_mouse_*` 的队列尚未完整接线。
- plain `Scene` 的 OpenGL 3D 投影仍是 `_euler_angles + 焦距 20` 近似；`Mesh` 顶点色为重心细分近似，`OpenGLTexturedSurface` 为逐面图片采样，Mesh 尚未引入深度缓冲。
- 严格逐像素 `1e-5` 指标尚未完全达标；`tests/graphical_units` 因本机 Cairo 1.16.0 低于要求的 1.18 全部 skip。
- `manim/renderer/vulkan` 仍保留少量兼容别名/窗口代码；是否彻底删除目录、以及程序化 `config.renderer=OPENGL` 是否统一路由到 Skia，尚未决策。

## 构建与运行

Windows 构建环境：

```powershell
$env:SKIA_BINARIES_URL = 'file://C:/skia/skia-binaries.tar.gz'
$env:LIB = 'C:\Users\Feysen\miniconda3\libs;' + $env:LIB
Set-Location native\manim_vk
maturin build --release
```

安装最新 wheel：

```powershell
uv pip install --python .venv\Scripts\python.exe --no-deps --reinstall `
  native\manim_vk\target\wheels\manim_vk-0.1.0-cp311-abi3-win_amd64.whl
```

运行：

```powershell
.\.venv\Scripts\python.exe -m manim -ql .\example_scenes\opengl.py InlineFullScreenQuad
```

不传 `--renderer` 时走默认 Vulkan；需要 Cairo 时显式传 `--renderer=cairo`。

## 相关验证脚本

- `bench_skia_1080.py`：1080p60 回读对照，`PERF_OLD=1` 可切换旧路径。
- `bench_skia.py`：480p15 1000 个 Circle 对照，`PERF_OLD=1` 可切换旧路径。
- `profile_skia.py`：生成 `media/perf_profile.prof`。
- `media/perf_tests_module.log`、`media/perf_tests_opengl.log`、`media/perf_tests_scene.log`：回归日志。
- `media/perf_tests_parallel_serial.log`：parallel encoding 串行日志。
