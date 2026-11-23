# ManimVK Vulkan 迁移 — 未完成清单（基线版）

- 生成时间：2026-11-23 08:25（本机时间）
- 仓库：`C:\Users\Feysen\Work\Deepseek\ManimVK`
- 用途：交给另一个会话逐项处理。**本文是唯一清单，不再追加。**
- 写法约定：每项都带 `文件:行号` 证据、影响、建议做法、验收标准。不要凭本文之外的猜测扩大范围。

---

## 0. 一句话结论

默认出片路径已经是 Vulkan，实测能出片（`Vulkan render device: NVIDIA`，720p30 的 1.5 秒场景 10.7 秒渲完）。
但“把原来所有渲染全部换成 Vulkan”还差 **5 个硬项（P0）** 和 **12 个收尾项（P1）**；其中 **3 项需要先定口径**，否则做完还会扯皮。

---

## 1. 先定口径（否则“全换”没有验收标准）

| 口径 | 含义 | 影响哪些项 |
|---|---|---|
| A. 画面全部由 Vulkan 出 | 不看安装依赖、不看死代码 | P0-1、P0-2、P0-5 |
| B. 运行时不再依赖 Cairo / OpenGL | `import manim` 和 pip 安装都不需要 cairo、moderngl | P0-3、P0-4、P1-12 |
| C. 输出与 Cairo 视觉一致 | 不要求像素级相同（做不到，见第 6 节） | P1-1、P1-2 |

建议按 **A + B** 为目标，C 用基准图比对来兜底。

---

## 2. 当前架构（一句话）

```python
# manim/renderer/skia/renderer.py:35
class SkiaRenderer(CairoRenderer):
```

Vulkan 不是独立实现，而是**继承 `CairoRenderer`，只替换 render target 和绘制上下文**：

```text
Scene → Manager → SkiaRenderer.update_frame
      → CairoRenderer.update_frame (继承)
      → SkiaDrawingContext.draw
      → SkiaRenderTarget.get_context
      → manim_vk.SkiaTarget（Skia Ganesh + Vulkan）
      → commit_into → numpy frame → av 编码
```

这意味着：**生命周期、帧调度、文件写出、`get_image()` 全部来自 Cairo 类**。功能能跑，但“代码去 Cairo 化”没有完成。

---

## 3. 已确认完成（不要重做）

- [x] 默认渲染器已是 Vulkan：`manim/_config/default.cfg` 里 `renderer = vulkan`
- [x] 命令行 `--renderer=opengl` 被改写成 `vulkan`：`manim/cli/render/commands.py:104-110`
- [x] 默认出片实测走独显：日志 `Vulkan render device: NVIDIA`
- [x] VMobject / Text / Surface / Mesh / 动画 / 相机 已走 GPU（Skia `draw_path` / `draw_vertices`）
- [x] 原生扩展可导入：`.venv\Lib\site-packages\manim_vk\manim_vk.pyd`，wheel 5.8 MB（cp311-abi3, win_amd64）
- [x] 静帧（`-s`）与视频（mp4）都能出

**注意：以下测试数字不代表 Vulkan**
- `tests/module` 1060 passed、`tests/opengl` 164 passed —— 这些是在 **Cairo 或真 OpenGL** 下跑的，原因见 P1-1。

---

## 4. P0 必须处理（5 项）

### P0-1 `Mobject.get_image() / save_image() / show()` 仍然用 Cairo 渲染

- **证据**：`manim/mobject/mobject.py:1013-1021`

  ```python
  from manim.renderer.cairo import CairoRenderer
  with CairoRenderer(camera=Camera() if camera is None else camera) as renderer:
      renderer.render_mobjects([self])
      return renderer.get_image()
  ```

- **影响**：这是公开 API。Vulkan 场景里调用 `mobject.save_image()` / `.get_image()` / `.show()`，出来的图仍是 Cairo 画的；且必须装了 cairo 才能用。
- **建议做法**：提供一个 Skia 版的“单 mobject 出图”路径（可直接复用 `SkiaRenderer` + 一个临时 target，类似 `_get_scene_image` 的做法，`manim/renderer/skia/renderer.py:239-247`），按 `config.renderer` 分派。
- **验收**：
  - `import` 一个 Mobject，`save_image()` 在默认 Vulkan 配置下不触发 `CairoRenderer`（可在 `CairoRenderer.__init__` 打断点/打日志验证）。
  - 同一 Mobject 分别用两个后端出图，尺寸一致、位置一致、肉眼一致。

### P0-2 图片 / 点云 / 背景色叠加 / 多相机仍在 CPU 合成

- **证据**：`manim/renderer/skia/rendering.py:91-101`

  ```python
  def _display_point_cloud_mobjects(...):
      self.target.commit()          # GPU → CPU
      super()._display_point_cloud_mobjects(pmobjects)   # numpy 合成
      self.target.mark_cpu_dirty()  # 下帧再 CPU → GPU
  ```

  同类：`_display_image_mobjects`（97-101）、`_display_background_colored_vmobjects`（69-89，scratch + numpy 乘法叠加）、多相机子图（`skia/renderer.py:160-164` 取 `sub_target.pixels` 作为 CPU 像素）。

- **影响**：`ImageMobject`、`PMobject`（点云/DotCloud）、带背景色的 VMobject、`MultiCamera` 这几类内容，实际像素不是 Vulkan 画的，而是 GPU 读回后 numpy 混合再上传。严格按“全部渲染走 Vulkan”口径，这几类没换。
- **两条路，需先选**：
  1. **算完成**：承认“合成在 CPU、光栅在 GPU”，在文档里写明，不再改。
  2. **真换**：`ImageMobject` 用 Skia 画 image（`Canvas::draw_image`），`PMobject` 用 Skia `draw_points`/`draw_vertices`，MultiCamera 用 Skia 图层合成。工作量大，但能去掉每帧两次 GPU↔CPU 往返。
- **验收**：打开 `bench_skia_1080.py` 类场景（含 ImageMobject + 点云），确认 `commit`/`mark_cpu_dirty` 不再被这几类调用；或明确在文档标注“CPU 合成”。

### P0-3 `import manim` 硬依赖 pycairo + moderngl，安装依赖也没摘

- **证据**：
  - `pyproject.toml:33` `moderngl-window>=2.0.0`、`:34` `moderngl>=5.7.0`、`:38` `pycairo>=1.14`
  - `manim/__init__.py:83` `from .renderer.cairo import *`
  - 实测：`python -c "import manim"` 之后，`cairo`、`moderngl`、`manim.renderer.opengl` 全部在 `sys.modules` 里（True/True/True）
  - `manim/scene/scene.py` 顶部直接 import `..renderer.opengl.renderer`、`..renderer.opengl.shader`（后者第 10 行 `import moderngl`）
- **影响**：没装 Cairo 原生库、没装 moderngl 的机器，`import manim` 直接失败；pip 安装仍会拉这两个包。**这是 B 口径的核心阻断项。**
- **建议做法**：
  1. 把 `CairoRenderer`、`OpenGLRenderer` 改成惰性导入（只在 `config.renderer` 真的等于 cairo/opengl 时导入）。
  2. `pyproject` 把 `pycairo`、`moderngl`、`moderngl-window` 移到 optional extra（如 `[project.optional-dependencies] legacy`）。
  3. 清理 `scene.py` 顶部对 opengl/shader 的 import，改为函数内惰性导入。
- **验收**：新建干净 venv，只装不含 cairo/moderngl 的依赖，`import manim` 成功，默认场景能出片；`python -c "import sys; import manim; print('cairo' in sys.modules, 'moderngl' in sys.modules)"` 输出 `False False`。

### P0-4 两条非 Vulkan 渲染入口仍然可达

- **证据**：
  - `--renderer=cairo`：`manim/scene/scene.py:232-238`，显式 `CairoRenderer(...)`；注释写明“compatibility tests and reference renders”。
  - Python 层 `config.renderer = "opengl"`：`manim/scene/scene.py:224-229` 直接 `renderer = OpenGLRenderer()`。实测 `config.renderer='opengl'` → 实例化 `OpenGLRenderer`（真 moderngl 路径）。只有 **CLI** 的 `--renderer=opengl` 会被改写，Python 赋值绕过改写。
- **影响**：目标若是“只有一个渲染后端”，这两条都破坏前提；且 `tests/opengl` 正是走第二条，所以测试也没在测 Vulkan。
- **需决定**：
  - `cairo` 是否保留为“参照渲染器”：保留就要写进文档、并从默认安装依赖中剥离；不保留就删入口。
  - `config.renderer='opengl'` 是改写成 vulkan（与 CLI 一致），还是保留真 GL。
- **验收**：`grep -rn "CairoRenderer(\|OpenGLRenderer(" manim/` 只剩定义处；或所有入口都被显式分派并有测试。

### P0-5 自定义 shader：GLSL 被静默忽略，SkSL 也没有加载入口

- **证据**：`manim/renderer/skia/mesh.py:1-8` 文档串明说

  > “The Vulkan build deliberately does not keep a second GL pipeline; instead it consumes the mesh's vertex attributes … leaving arbitrary fragment-shader authoring as a separate feature.”

  实现上只认三种：顶点色三角面（`draw_vertices`）、`u_color` 单一 uniform（`mesh.py:53-61`）、两个硬编码全屏 shader（`InlineFullScreenQuad` 的 HSB、`design_3`，`mesh.py:196-218` + `lib.rs:990/1046`）。
- **影响**：用户给的 `Shader` / 自定义 `shader_folder` / 任意 fragment 着色器，不会报错，而是**静默画成顶点色或纯色**——比崩溃更难发现。用户已决定“不做 GLSL 转译、要写就写 SkSL”，但**目前写 SkSL 也没有入口**。
- **建议做法**（按用户决定的最小版本）：加一个 `RuntimeEffect` 加载入口——接受 SkSL 字符串（或 sidecar 文件），编译成 `skia_safe::RuntimeEffect`，绑定 uniform，挂到 Mobject 的绘制路径上。不要做 GLSL→SPIR-V。
- **验收**：写一个最小 SkSL（如纯色渐变/噪声），从 Python 传入，能渲染出图；shader 语法错误时抛出可读异常，而不是静默忽略。

---

## 5. P1 收尾项（12 项）

### P1-1 测试基线全在 Cairo，没有任何 Vulkan 回归

- **证据**：
  - `tests/conftest.py:92` 全局 fixture：`manim.config.renderer = "cairo"`
  - `tests/conftest.py:121-127` `using_opengl_renderer` 设 `config.renderer = "opengl"` → 实例化真 `OpenGLRenderer`（已实测）
  - 整个 `tests/` 目录搜 `vulkan`：**0 次**
  - `tests/test_graphical_units/control_data`：**239 张** Cairo 基线图
  - CI：`.github/workflows/ci.yml:162` `uv run python -m pytest`；`:166` 再跑一遍 `--ignore-glob="*opengl*"`（仍是 cairo）
- **影响**：所有“测试全绿”都与 Vulkan 无关。改坏 Vulkan 画面，CI 也发现不了。
- **建议做法**：
  1. 新增 `tests/test_vulkan/`（或 marker `@pytest.mark.vulkan`），固定 5-8 个场景：圆形/文字/Surface/ImageMobject/点云/多相机/3D/透明。
  2. 生成 Vulkan 自己的基线图（不要和 Cairo 基线混用）。
  3. CI 单独 job：Windows + Vulkan（GPU runner 或 lavapipe），跑这套基线。
- **验收**：故意改坏一个 shader/矩阵，Vulkan 基线测试必须失败；Cairo 测试保持通过。

### P1-2 颜色 / alpha 没有数值验证

- **证据**：Skia surface 用 `ColorType::RGBA8888`（`lib.rs:772-777`），Vulkan image 用 `R8G8B8A8_UNORM`（`lib.rs:257`），读回用 `AlphaType::Unpremul`（`lib.rs:1145-1150`、`1192-1197`）；全仓无 sRGB / ColorSpace 处理。
- **影响**：alpha 读回是 unpremul，理论上和 PIL/编码器一致；但半透明边缘、渐变、暗部是否与 Cairo 一致，**没有证据**。这不是确定的 bug，是未验证风险。
- **建议做法**：同一场景分别用 Cairo / Vulkan 出 PNG，做逐像素 MAE/PSNR 统计（允许差异但要有阈值），重点看透明背景和渐变。
- **验收**：给出一份差异报告（均值、最大差、差异像素占比），并定阈值。

### P1-3 预览窗口另开一套 Vulkan 设备，失败时静默

- **证据**：
  - 离屏：ash 自建 instance/device（`lib.rs:80-206`），受 `MANIM_VK_DEVICE` 控制（`lib.rs:114`）
  - 预览：wgpu 另建 instance，`Backends::VULKAN` + `PowerPreference::HighPerformance`（`window.rs:71-79`），**不读 `MANIM_VK_DEVICE`**
  - 窗口/表面创建失败只 `eprintln!`（`window.rs:63`、`:88`），不抛 Python 异常
- **影响**：同一进程两块 Vulkan 设备；有双显卡时，预览可能落在另一块 GPU 上。窗口起不来时用户只看到“没有窗口”，程序继续跑，没有错误提示。
- **建议做法**：统一 adapter 选择（把离屏选中的 physical device / UUID 传给 wgpu）；失败时抛出 Python 异常或写日志 warning。
- **验收**：在双显卡机器上，日志里离屏设备与预览适配器名称一致；拔掉窗口依赖时能看到明确报错。

### P1-4 预览窗口忽略 `fullscreen` / `window_position` / `window_monitor`

- **证据**：`_WindowSettings` 仍有这三个字段（`manim/renderer/vulkan/window_settings.py:18-31`），config 也仍然暴露（`manim/_config/utils.py:318-321`）；但 `NativeWindow::new` 只接收 `width, height, title`（`window.rs:155-156`），`manim/renderer/vulkan/window.py:53-57` 只传这三个。
- **影响**：这三个 CLI/config 选项在 Vulkan 下**静默失效**，既不报错也不生效。
- **建议做法**：要么在 Rust winit 里实现 monitor 选择、全屏、初始位置；要么在 Python 侧检测到非默认值时明确警告“Vulkan 预览暂不支持”。
- **验收**：`--fullscreen` 真的全屏；`--window-position=UR` 真的在右上；不支持时有 warning。

### P1-5 预览窗口没有键鼠输入，`interactive_embed` 在 Vulkan 下不可交互

- **证据**：`window.rs:111-133` 的 `window_event` 只处理 `CloseRequested` / `Resized` / `RedrawRequested`，没有任何键盘/鼠标事件转发；`manim/scene/scene.py:1470-1477` 的 `interactive_embed` 允许 Vulkan 并设置 `interactive_mode = True`。
- **影响**：Vulkan 下的“交互式预览”只是个静态窗口，鼠标拖动、按键回调都不工作；但代码路径允许进入，用户会以为能用。
- **建议做法**：短期在 `interactive_embed` 里对 Vulkan 明确拒绝或警告；长期把 winit 事件通过 PyO3 回调转给 Scene（`mouse_point`、`key_to_function_map`）。
- **验收**：Vulkan 下调用 `interactive_embed()` 要么给出明确不支持提示，要么真的能响应鼠标/键盘。

### P1-6 每个 render target 泄漏一个闭包，并重复创建 DirectContext

- **证据**：`lib.rs:756-758`

  ```rust
  let get_proc = Box::leak(Box::new(move |of| skia_get_proc(&inner_for_proc, of)));
  ```

  每个 `SkiaTarget::new` 都会 `Box::leak`；此外每次都 `gpu::direct_contexts::make_vulkan`（`lib.rs:770`）。而 `SkiaRenderTarget` 会为每个子相机、每个 scratch target 各建一个（`skia/renderer.py:61-121`、`skia/target.py:92-100`）。
- **影响**：长时间/多场景渲染时内存只增不减；多相机场景 GPU 上下文数量翻倍。
- **建议做法**：`get_proc` 改成 `Arc`/静态缓存，只 leak 一次；DirectContext 按 Device 复用，target 只建 surface。
- **验收**：循环渲染 50 个含 MultiCamera 的场景，进程 RSS 不随场景数线性增长；`Box::leak` 全局只剩 1 处。

### P1-7 `SkiaTarget` / `NativeWindow` 是 `unsendable`，跨线程会 panic

- **证据**：`lib.rs:737` `#[pyclass(unsendable)] struct SkiaTarget`；`window.rs:141` `#[pyclass(unsendable, name = "NativeWindow")]`。
- **影响**：任何把 renderer/target 挪到别的线程的用法（自定义线程、asyncio executor、某些 notebook 内核）会直接 panic，而不是抛异常。目前 Manim 用多进程并行渲染，碰不到，但这是隐性约束。
- **建议做法**：确认单线程假设并在文档写明；或在入口加 `threading.get_ident()` 校验，给出清晰错误。
- **验收**：跨线程调用时返回可读的 `RuntimeError`，而不是进程 abort。

### P1-8 原生层 36 处 `unwrap/expect/panic`

- **证据**：`native/manim_vk/src/*.rs` 合计 36 处（`Select-String "expect\(|unwrap\(|panic!|PyRuntimeError"` 统计）。
- **影响**：部分是无害常量（如 `CString::new("manim_vk").unwrap()`），但热路径上的 `unwrap` 会把可恢复错误变成 abort，Python 侧收不到异常。
- **建议做法**：逐个分级；热路径改 `map_err(PyRuntimeError)`；常量级保留并加注释。
- **验收**：`grep` 清单里每个热路径都有错误映射；故意触发一次设备丢失，得到 Python 异常而非崩溃。

### P1-9 没有 GPU / 没有 Vulkan 驱动时的兜底未验证

- **证据**：`lib.rs:107-110` 只返回 `"no Vulkan physical device found"`；`lib.rs:144` 给 CPU 设备权重 200，理论上可选 lavapipe，但需要外部 ICD。
- **影响**：无独显/无驱动的机器（CI、服务器、部分虚拟机）直接失败；没有软件渲染回退，也没测试。
- **建议做法**：明确支持策略：要么文档要求 Vulkan 驱动，要么检测失败时提示安装 lavapipe/SwiftShader；补一个无 GPU 的 CI job。
- **验收**：在关闭 GPU 的容器里跑 smoke，给出可读提示或成功回退。

### P1-10 构建与发布未收口

- **证据**：wheel 只有 `manim_vk-0.1.0-cp311-abi3-win_amd64.whl`；`Cargo.toml` 的 `skia-safe` features 仍包含 `"gl"`、`"pdf"`（`native/manim_vk/Cargo.toml:17`）；`ash` 用 `[patch.crates-io]` 指到 `third_party/ash-0.37.3`。
- **影响**：Linux/macOS 没有 wheel；`gl` feature 让 Skia 仍编进 GL 后端（与“去 GL 化”目标相悖，也增加链接面）；干净机器重建未验证。
- **建议做法**：去掉不需要的 skia features（先确认 textlayout/jpeg 必需）；写清 Rust 工具链版本；CI 出三平台 wheel；验证 `pip install` 后 `import manim_vk`。
- **验收**：干净容器 `maturin build` 成功；三平台 wheel 可安装；`ldd`/依赖检查无意外 GL 库。

### P1-11 文档仍停留在 Cairo/OpenGL 时代

- **证据**：`README.md` 无渲染器说明；`docs/source/guides/deep_dive.rst:263` 仍写“instantiates a CairoRenderer or an OpenGLRenderer”；`MANIM_VK_DEVICE` 在 README/docs 中出现 **0 次**。
- **影响**：用户不知道默认已是 Vulkan、不知道如何选设备、不知道哪些功能受限（交互、全屏、自定义 shader）。
- **建议做法**：加一节 “Vulkan backend”：默认行为、`MANIM_VK_DEVICE`、已知限制（CPU 合成、无自定义 GLSL、预览限制）。
- **验收**：文档里能查到上述全部信息。

### P1-12 没有 git，仓库 1.89 GB，根目录 53 个临时文件

- **证据**：`Test-Path .git` → `False`；仓库总体积 1.89 GB；根目录匹配 `*.log|.coverage|probe|bench_|check_|smoke|_after` 的文件 53 个（含 `.coverage.*`、`cargo_*.log`、`tests_*_after.log`、20 个 probe/bench 脚本）。
- **影响**：改坏了没有回退点；AI 多轮修改无法对比；仓库臃肿。
- **建议做法**：`git init` + 首次提交（当前状态）；`.gitignore` 排除 `.venv/`、`media/`、`*.log`、`.coverage*`、`__pycache__`；把 probe/bench 归到 `tools/audit/` 或删除。
- **验收**：`git status` 干净；仓库 < 200 MB（不含 .venv/media）；根目录只剩项目文件。

---

## 6. 明确不做 / 口径修正（不是欠账）

- **不做“与 Cairo 像素级 1:1 对齐”**。Cairo 与 Skia 的抗锯齿、字形栅格化、渐变插值都不同，1e-5 不现实。改为“同场景视觉一致 + 差异统计在阈值内”（见 P1-2）。
- **不做通用 GLSL → SPIR-V 转译**（用户已决定）。若需要自定义效果，写 SkSL，并补 P0-5 的加载入口。
- **Cairo 作为参照渲染器**：代码注释明确保留用于兼容测试/参照出图。**是否保留由口径 B 决定**：保留就从默认依赖里剥离（P0-3），不保留就连入口一起删（P0-4）。
- **`Target` 类（非 Skia）**：`lib.rs:242` 定义的 `Target` 在 `manim/` 中已无人调用（`create_target` 的搜索结果全是动画层的同名方法），属于死代码，随 P1-12 一并清理即可，不单列。

---

## 7. 建议执行顺序（有依赖关系）

1. **P1-12 git init + 基线提交**（必须先做，后面每步都能回退）
2. **P1-1 Vulkan 基准测试**（先有“测量尺”，否则改对改错都不知道）
3. **P0-1 `Mobject.get_image` 走 Skia**（改动最小、收益最直接）
4. **P0-3 依赖惰性化**（`import manim` 不再拉 cairo/moderngl）
5. **P0-4 收敛入口**（按口径决定 cairo/opengl 去留）
6. **P0-2 CPU 合成**（工作量大，先定“算不算完成”）
7. **P0-5 SkSL 加载入口**（用户明确要做才做）
8. P1-3/4/5 预览相关 → P1-6/7/8/9 稳定性 → P1-10/11 发布与文档

---

## 8. 验收矩阵（改完必须覆盖的输出形态）

| 形态 | 现状 | 验收方式 |
|---|---|---|
| 视频 mp4 | 通 | 默认渲染出片 + 设备日志 |
| 静帧 `-s` / png | 通 | 出图尺寸/内容 |
| 透明背景 `--transparent` | 未验证 | 出 PNG 检查 alpha |
| 多相机 `MultiCamera` | CPU 合成 | 与单相机版本对比 |
| 3D Surface / ThreeDScene | 通（手动验证过） | 旋转/光照截图 |
| ImageMobject | CPU 合成 | 与原图对比 |
| PMobject / 点云 | CPU 合成 | 点数/颜色/位置 |
| Text / MathTex / Typst | VMobject 路径 | 中英文与公式 |
| 预览窗口 `-p` | 通（无输入） | 打开窗口、缩放、关闭 |
| 交互 `interactive_embed` | 无输入 | 鼠标/键盘事件 |
| 自定义 shader | 静默忽略 | P0-5 验收 |

---

## 9. 本次核查范围（避免下个会话重复劳动）

**逐文件读完的**：
- `manim/renderer/skia/renderer.py`（243 行，全文）
- `manim/renderer/skia/target.py`（175 行，全文）
- `manim/renderer/skia/rendering.py`（86 行，全文）
- `manim/renderer/vulkan/renderer.py`（13 行，全文）
- `manim/renderer/vulkan/window.py`（103 行，全文）
- `manim/renderer/vulkan/window_settings.py`（全文）
- `native/manim_vk/src/window.rs`（262 行，全文）
- `native/manim_vk/src/lib.rs` 的第 80-280、700-1147、1140-1235 段（设备选择、SkiaTarget、绘制方法、commit/readback/load）
- `manim/mobject/mobject.py:998-1033`（`get_image` 段）
- `tests/conftest.py:78-127`
- `manim/cli/render/commands.py:92-112`
- `pyproject.toml` 依赖段、`native/manim_vk/Cargo.toml`
- `manim/opengl/__init__.py` 前 70 行（确认 facade 重绑定）
- `manim/scene/scene.py` 的渲染器分派段（215-250）、`get_image` 段、`interactive_embed` 段

**只做调用点/符号扫描、没有逐行读的**：
- 动画/场景业务逻辑（`animation/*`、`mobject/*` 的几何与变换）——与渲染后端替换无关
- `tests/` 的用例内容（只统计了 renderer、skip/xfail、基线数量）
- `docs/` 全文（只查了 renderer 相关段落）

**本次实测过的**：
- 默认渲染一次 1.5 秒 720p30 场景：成功，`Vulkan render device: NVIDIA`，10.7 秒
- `config.renderer` 三值分派：`vulkan → SkiaRenderer`、`cairo → CairoRenderer`、`opengl → OpenGLRenderer`
- `import manim` 后 `sys.modules`：`cairo`、`moderngl`、`manim.renderer.opengl` 均为 True
- `git status`：not a git repository

---

## 10. 快速自检命令（改完跑这几条）

```powershell
# 1. 默认路径必须走 Vulkan
.\.venv\Scripts\python.exe -m manim -qm --disable_caching smoke.py Smoke
# 期望日志：Vulkan render device: NVIDIA

# 2. 依赖是否真的摘掉（干净 venv 里）
.\.venv\Scripts\python.exe -c "import sys, manim; print('cairo' in sys.modules, 'moderngl' in sys.modules)"
# 目标：False False

# 3. 是否还有绕过 Vulkan 的入口
Select-String -Path (Get-ChildItem manim -Recurse -Include *.py) -Pattern 'CairoRenderer\(|OpenGLRenderer\('

# 4. 测试里是否有 Vulkan
(Get-ChildItem tests -Recurse -Include *.py,*.cfg | Select-String -Pattern 'vulkan').Count
# 目标：> 0，且基线能失败

# 5. 仓库是否可回退
git status
```

---

**结束。以下不再追加条目；若后续发现新问题，要么归入上述某一项的“验收未通过”，要么属于新需求，另开文档。**
