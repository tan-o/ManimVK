# Manim Vulkan / Skia 迁移状态

基线：Manim Community v0.22.1，git 基线 `c774867`。默认渲染器为 Vulkan/Skia；Cairo 仅作为显式参照后端保留。旧 moderngl 后端退役，`--renderer opengl` 与 `config.renderer = "opengl"` 统一归一化为 Vulkan。

## P0 结果

| 项 | 状态 | 证据 |
|---|---|---|
| P0-1 `Mobject.get_image()` | 完成 | 按 `config.renderer` 分派；默认走 `SkiaRenderer`，Cairo 仅在显式请求时加载。 |
| P0-2 图片 / 点云 / 背景 / MultiCamera | 完成 | 图片、点云、背景填充、`MultiCamera` 子视图均由 Skia GPU 绘制/合成；子视图不再 `commit()` 读回，父 target 直接采样子 surface。 |
| P0-3 Cairo / moderngl 依赖剥离 | 完成 | 干净 `.venv-clean` 未安装 cairo、moderngl：`import manim` 输出 `False False False True`，默认场景成功出 PNG。`SkiaRenderer.update_frame` 已移除运行时 `manim.renderer.opengl.shader` import。 |
| P0-4 渲染入口收敛 | 完成 | `Scene` 只在显式 `renderer="cairo"` 时构造 `CairoRenderer`，其余走 `SkiaRenderer`；`tests/opengl` 作为退役套件整体 skip，Vulkan 回归在 `tests/test_vulkan`。 |
| P0-5 自定义 shader | 完成 | 新增 `SkSLMobject` / `draw_sksl_quad`；非法 SkSL 返回 `RuntimeError: SkSL compile failed: ...`，已用缺右花括号样例验证。 |

## P1 结果

| 项 | 状态 | 说明 |
|---|---|---|
| P1-1 Vulkan 回归基线 | 完成 | `tests/test_vulkan/test_vulkan_baseline.py`：vector/text/surface/image/point-cloud 五张 Vulkan 专属基线；另有 MultiCamera、设备失败、跨线程检查。无 Vulkan 设备时 skip。 |
| P1-2 颜色 / alpha 数值报告 | 完成 | `VULKAN_COLOR_DIFF_REPORT.md`：不透明渐变 RGB MAE 0.107、PSNR 53.16 dB；透明场景 RGB MAE 4.79、alpha MAE 0.113、差异像素 6.4%，并给出阈值。 |
| P1-3 预览 adapter 统一 / 失败可读 | 完成 | 离屏设备先创建，名称传给 preview；native 通过 `WGPU_ADAPTER_NAME` 选择同一 adapter。实测 preview `adapter_name()` 与离屏 `NVIDIA GeForce RTX 3050 Ti Laptop GPU` 一致，窗口错误可由 `NativeWindow.error()` 读取。 |
| P1-4 fullscreen / position / monitor | 完成 | `NativeWindow` 新增对应参数；Windows 上全屏 + `UR` 位置冒烟通过，无原生错误。 |
| P1-5 交互预览 | 完成（明确拒绝） | `interactive_embed()` 在 Skia 下 warning 后返回；`embed()` 保留只读预览。 |
| P1-6 闭包 / DirectContext 复用 | 完成 | `Device` 持有 `GpuShared`（一个 `BackendContext` + 一个 `DirectContext`）；每个 target 只 clone 引用；`Box::leak` 每个 Device 只有 1 处。 |
| P1-7 跨线程保护 | 完成 | `SkiaRenderTarget` 记录创建线程，跨线程访问抛可读 `RuntimeError`；`NativeWindow` 保持 `unsendable`。 |
| P1-8 unwrap / expect / panic | 完成 | `native/manim_vk/src/*.rs` 扫描零命中；设备/窗口/着色器错误均映射为 Python 异常。 |
| P1-9 无 GPU / 无驱动提示 | 完成（提示路径） | 设备创建失败抛出带驱动/lavapipe/SwiftShader/`MANIM_VK_DEVICE` 指引的 `RuntimeError`；新增 monkeypatch 回归测试。无 GPU 容器实测未做。 |
| P1-10 构建与发布 | 部分完成 | 移除 skia `gl`、`pdf` feature；Windows x64 wheel 可重建、安装。Linux/macOS wheel 未构建，三平台 CI 未添加；macOS 还需 MoltenVK。 |
| P1-11 文档 | 完成 | 新增 `docs/source/guides/vulkan_backend.rst`、`native/manim_vk/README.md`，README 增加 Vulkan 节，`deep_dive.rst` 更新。 |
| P1-12 仓库清理 / 可回退 | 完成 | 根目录 probe/bench/demo 移入 `tools/audit/`，日志/SPV/coverage 删除；`.gitignore` 补 `*.spv`、`*.whl`；迁移改动已形成单一提交，工作区干净；提交体积约 21.2 MB。 |

## 测试与验证

- `tests/test_vulkan`：10 passed（5 基线 + MultiCamera + SkSL + 2 robustness + 默认 `Mobject.get_image()` 分派）。
- `tests/test_camera.py`：40 passed。
- `tests/module`：1059 passed，2 skipped（最终复跑；此前的 3 个 LaTeX/MiKTeX 环境失败在本轮未重现）。
- `tests/opengl`：169 项被标记为退役套件 skip（历史用例保留，不再断言旧 OpenGL 类）。
- 干净 venv：`cairo=False`、`moderngl=False`、`manim.renderer.opengl=False`、`manim.renderer.cairo=True`；默认 `entry_probe` 输出 `SkiaRenderer Camera RendererType.VULKAN` 并出 PNG。
- 预览：adapter 选择和全屏/位置参数在 Windows 本机冒烟通过。

## 已知限制

- `interactive_embed()` 不支持鼠标/键盘事件；预览为只读窗口。
- 自定义 GLSL 不编译；请改用 `SkSLMobject`。SkSL uniform 目前仅支持 float / float 向量。
- 预构建 native wheel 只有 Windows x64；Linux/macOS 未纳入 CI 产物。
- 无 GPU 容器的 lavapipe/SwiftShader 回退未实测，仅验证了错误提示路径。
- 颜色差异报告是诊断阈值，不是与 Cairo 的像素级 1:1 对齐承诺。

## 构建与运行

```powershell
$env:SKIA_BINARIES_URL = 'file://C:/skia/skia-binaries.tar.gz'
$env:LIB = 'C:\Users\Feysen\miniconda3\libs;' + $env:LIB
Set-Location native\manim_vk
maturin build --release
uv pip install --python ..\..\.venv\Scripts\python.exe --no-deps --reinstall `
  target\wheels\manim_vk-0.1.0-cp311-abi3-win_amd64.whl
```

默认 `manim -qm scene.py Scene` 走 Vulkan；`manim --renderer=cairo -qm scene.py Scene` 使用 Cairo 参照后端。
