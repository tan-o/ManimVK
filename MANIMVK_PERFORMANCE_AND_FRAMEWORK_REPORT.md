# ManimVK 性能瓶颈与渲染框架问题核查报告

> **2026-11-23 后续更新**：性能补丁已合并（`d8b8fde`），静态背景每帧上传和 Fourier 箭头 tip 每帧重建已消除。最新 1080p60 100 向量实测为 **28.7s / 600 帧**（本文件下述 252.8s 是补丁前历史数据）。最新拆解、golden 逐帧对照与残余问题见 `MANIMVK_FRAMEWORK_AND_GOLDEN_REPORT.md`。

日期：2026-11-23  
范围：`<repo>`（ManimCE 的 Vulkan/Skia fork）  
基准场景：3b1b `FourierOfPiSymbol` 的 ManimCE 移植版，100 个旋转向量 + 100 个圆 + 轨迹，10.000s  
基准脚本：`%TEMP%\mvk_fourier\fourier_pi_100.py`（仓库外临时脚本）  
本轮没有修改仓库代码；所有实验通过环境变量和运行时 monkeypatch 完成。

---

## 0. 结论摘要

1. **1080p60 下 Vulkan 比同仓库 Cairo 慢，不是“取帧链路”造成的。** 回读 + 编码约 19.6s，只占 7.8%。此前另一份性能 TODO 把 1080p 慢归因于取帧链路，并设 Phase 1 去修回读；本次实测推翻了这个归因。
2. **最大的一块隐藏开销是“每帧把整张静态背景重新上传到 GPU”**，实测 1080p60 为 **73.9s / 600 帧，平均 123ms/帧**。它发生在 `CairoRenderer.update_frame()` 的 `target.set_pixels(self.static_image)` 分支，Skia/Vulkan 目标每次都会执行 `np.copyto` + `tobytes()` + PyO3 `Vec<u8>` 拷贝 + `load_pixels` + `flush_and_submit`。
3. **把这次重复上传关掉后，完整 Vulkan 出片（A/B 计时，画面会残影）从 252.8s 降到 170.4s**，已经低于同仓库 Cairo 的 183.1s；但仍然高于 150s，因为第二块是 Python 场景更新。
4. **第二块是 Python 逐对象更新，1080p60 为 144.7–151.1s，占 57%–59%。** 它不是渲染器开销，而是场景里 100 个 `Vector` 每帧调用 `set_length()`，触发 `Arrow.scale()` 的 `pop_tips()` → `add_tip()` → 重新计算箭头 tip 几何与曲线长度；`cubic_bezier` 被调用约 95 万次。
5. 因此正确顺序是：**先修静态背景的 GPU 常驻/缓存（可省约 74–82s），再修 Vector 的每帧 tip 重建（目标把 145s 压到 30s 以内）**。只做前者可以到约 170s；两者都做才有希望进入 150s 以内。
6. 如果验收口径只是“低于同仓库 Cairo”，阶段 1 已经够（170.4s < 183.1s）；如果坚持“≤150s”，必须做阶段 2，不能只靠渲染器层优化。

> 与既有清单的关系：仓库里的 `VULKAN_MIGRATION_TODO.md` 是“迁移完整性”清单（5 P0 + 12 P1），负责“有没有换干净”；本报告负责“换完之后为什么慢、框架哪里不对”。两者不替代：F1–F12 是性能与框架问题的收敛版，迁移完整性仍以原清单为准。

### 1080p60 六种口径实测

| 口径 | 帧数 | render 时间 | 说明 |
|---|---:|---:|---|
| 原版 ManimCE 0.22.1 Cairo | 600 | 225.775s | 独立 venv，作为外部基线 |
| ManimVK Cairo | 600 | 183.093s | 同仓库 Cairo |
| ManimVK Vulkan 完整 | 600 | 252.756s | 当前实际最慢 |
| Vulkan `dry_run`（不回读/编码） | 600 | 233.213s | 省 19.543s |
| 再 `NODRAW`（关 Skia 绘制） | 600 | 219.166s | 再省 14.047s |
| 再关重复静态上传（`SKIPSTATIC`） | 600 | 143.051s | 只剩 Python 更新，省 76.115s |
| 完整出片 + `SKIPSTATIC` | 600 | 170.368s | 已低于同仓库 Cairo |

> `SKIPSTATIC` 只是把重复的 `set_pixels` 跳过，**画面会累积动态残影，不能作为最终实现**。它的用途是证明这 74–82s 的确切归属；正确实现应当把静态背景作为 GPU 纹理/图层每帧重绘，而不是跳过恢复。

---

## 1. 计时口径

统一条件：

- 同一脚本、同一机器、串行执行、`disable_caching=True`、`progress_bar=none`；
- `Q=high_quality` 为 1920×1080、60fps；`Q=low_quality` 为 854×480、15fps；
- 箭头参数 `SW=2`、`TIP=0.15`、`TIPR=0.2`；
- `DRY=1` 时 `config.dry_run=True`，不做回读/编码；
- `NODRAW=1` 时把 `SkiaDrawingContext.draw` 置空；
- `NODRAW_CAIRO=1` 时把 `CairoRenderer._draw_frame` 置空；
- `TIMERS=1` 时 monkeypatch 计时以下方法，并区分嵌套调用：
  - `SkiaRenderer.update_frame` / `CairoRenderer.update_frame`
  - `SkiaRenderer._render_camera` / `SkiaRenderer.get_frame` / `_present_frame`
  - `SkiaDrawingContext.draw`
  - `SkiaRenderTarget.set_pixels` / `reset` / `read_pixels`
  - `Scene.update_to_time` / `update_mobjects` / `update_self`

计时器用“嵌套深度”区分 root 与子调用，避免把 `update_to_time` 和 `update_frame` 相加时重复计数。

---

## 2. 1080p60 的 252.8s 到底花在哪

### 2.1 逐段计时（`DRY=1 NODRAW=1 TIMERS=1`）

```
TIMING n=100 coef=0.050s build=0.192s render=219.339s total=219.581s
TSTAT update_to_time 144.736s calls=600
TSTAT update_mobjects 145.083s calls=602
TSTAT update_frame   148.321s calls=1202   # 嵌套：Skia 外层 + Cairo 内层
TSTAT set_pixels      73.872s calls=600
TSTAT _render_camera   0.018s calls=601
TSTAT draw             0.001s calls=601
TSTAT reset            0.016s calls=1
TSTAT get_frame        0.001s calls=1
TSTAT read_pixels      0.001s calls=1
TROOT update_to_time 144.736s
TROOT update_frame    74.172s
```

关键点：

- `update_to_time` 是纯 Python 场景更新，**144.736s**；
- `update_frame` 作为 root 只有 **74.172s**，其中 **`set_pixels` 73.872s**；
- 表里的 `update_frame 148.321s` 是 `SkiaRenderer.update_frame` 与 `CairoRenderer.update_frame` 嵌套计入同一个 key 的结果，不能直接当 root；root 值看 `TROOT`；
- `_render_camera`、`draw`、`reset`、`get_frame`、`read_pixels` 全部接近 0（因为 `NODRAW` 和 `DRY`）；
- 144.736 + 74.172 + 约 0.4s 其他 = 219.3s。

也就是说，之前一直找不到的“74.7s Vulkan 每帧固定开销”，**就是静态背景的重复上传，不是 reset，也不是回读**。

### 2.2 完整出片的构成

用“完整 − 逐级关闭”的差值：

| 构成 | 时间 | 占 252.8s |
|---|---:|---:|
| Python 场景更新 | 144.7s | 57.3% |
| 静态背景重复上传 | 73.9s | 29.2% |
| Skia 实际绘制 | 14.0s | 5.5% |
| 回读 + 编码 + 写文件 | 19.6s | 7.8% |
| 其他/误差 | 0.6s | 0.2% |

这里的“静态背景上传”是直接计时值；“完整出片 + SKIPSTATIC”相对完整出片省了 **82.4s**，说明每次 `load_pixels` 里的 `flush_and_submit` 还会带来额外的 GPU 同步和流水线停顿，实际代价比单独的 Python 计时更大。

### 2.3 为什么 `reset` 计时会误导

`SkiaRenderTarget.reset` 在 600 帧里只调用 **1 次**，总耗时 **0.012s**；`NORESET` 也几乎没有变化。原因是这个场景有静态 mobject（Pi 轮廓），`static_image` 不为 `None`，所以 `CairoRenderer.update_frame()` 走的是：

```python
if self.static_image is not None:
    target.set_pixels(self.static_image)
else:
    target.reset(self.camera)
```

`reset` 分支根本没有被执行。只盯 `reset` 会得出“重置不是瓶颈”的正确结论，但会漏掉真正每帧执行的 `set_pixels`。

### 2.4 480p15 对照

```
TIMING n=100 coef=0.049s build=0.200s render=33.586s
TSTAT update_to_time 29.696s calls=150
TSTAT set_pixels      3.222s calls=150
TSTAT update_frame    3.464s  # root
TROOT update_to_time 29.696s
TROOT update_frame    3.464s
```

- 480p15：静态上传 **3.222s / 150 帧 = 21.5ms/帧**；
- 1080p60：静态上传 **73.872s / 600 帧 = 123.1ms/帧**；
- 每帧像素比 5.06 倍，上传时间约 5.7 倍，基本随像素线性增长，符合“整帧上传”的判断。

这也解释了为什么 480p15 下 Vulkan 反而比原版 Cairo 快约 11.6%：低分辨率时上传成本小，Python 更新占主导；到 1080p，上传成本被放大 5 倍，直接把 Vulkan 拖到 Cairo 后面。

---

## 3. 瓶颈一：静态背景每帧上传 GPU

### 3.1 代码路径

1. `manim/renderer/cairo/renderer.py:361-380` `save_static_frame_data()`
   - 先把静态 mobject 单独渲染一次；
   - `self.static_image = self.get_frame()`，得到一整张 RGBA 背景图。
2. `manim/renderer/cairo/renderer.py:279-282` `update_frame()`
   - 每帧如果 `static_image` 存在，就 `target.set_pixels(self.static_image)`。
3. `manim/renderer/skia/target.py:92-97` `SkiaRenderTarget.set_pixels()`
   - `super().set_pixels(pixels)` → `np.copyto(self._pixels, pixels)`；
   - `self._pixels.tobytes()` 生成一份完整 bytes；
   - `self._native.load_pixels(...)`。
4. `manim/renderer/cairo/target.py:164-173` `_CairoRenderTarget.set_pixels()`
   - 只有 `np.copyto`，纯 CPU 内存拷贝，所以 Cairo 不觉得贵。
5. `native/manim_vk/src/lib.rs:1555-1576` `load_pixels(&mut self, pixels: Vec<u8>)`
   - PyO3 把 Python `bytes` 再转成 Rust `Vec<u8>`（又一次完整拷贝）；
   - 建 `Pixmap`，`surface.write_pixels_from_pixmap`；
   - **每次调用都 `flush_and_submit()`**，同步等待 GPU 提交完成。

### 3.2 每帧实际发生的数据搬运

以 1920×1080、RGBA8 为例，一帧背景 8.29MB：

- `np.copyto`：8.29MB 读 + 8.29MB 写；
- `tobytes()`：再分配 8.29MB 并拷一次；
- PyO3 `Vec<u8>`：再拷一次；
- `write_pixels_from_pixmap`：上传到 GPU；
- `flush_and_submit`：同步。

约 4 次全帧 CPU 拷贝 + 1 次 GPU 上传 + 1 次同步，600 帧就是约 20GB 级别的内存流量。实测 73.9s，折算约 270MB/s 的有效吞吐，瓶颈在拷贝和同步，不在 PCIe 理论带宽。

### 3.3 为什么只在 Skia/Vulkan 出现

Cairo 的 `set_pixels` 只是把同一张 CPU 图复制进 `_pixels`，之后 Cairo 直接在 CPU 上画；它没有“上传到 GPU”这一步。Skia 需要 GPU surface，所以同一份 `static_image` 每帧都要重新上传。**这不是 Cairo 慢，而是 Skia 目标继承了 Cairo 的“每帧恢复整张背景”的语义，却没有做 GPU 侧缓存。**

### 3.4 修复方向

**推荐：把静态背景变成 GPU 常驻图层，而不是每帧上传的 NumPy 图。**

可行实现（按侵入性从低到高）：

1. **静态图纹理缓存**
   - 在 `SkiaRenderTarget` 里增加 `_static_image_texture` / `_static_image_version`；
   - `set_pixels` 只在版本变化时 `load_pixels` 一次，生成 `skia_safe::Image`；
   - 每帧 `update_frame` 先 `canvas.draw_image(static_image, 0, 0)` 把背景画回 surface，再画动态 mobject；
   - 这样每帧只多一次 GPU 全屏 blit，成本通常 <1ms。

2. **双 surface 复制**
   - 保留一个 `background_surface`，`save_static_frame_data` 时把静态帧画进去；
   - 每帧 `surface.copy_from(background_surface)` 或 `canvas.draw_surface(background_surface)`；
   - 适合不想引入 `skia_safe::Image` 生命周期管理的情况。

3. **只清除动态区域**
   - 记录动态 mobject 的包围盒，每帧只恢复受影响区域；
   - 对傅里叶这种全屏旋转向量不划算，因为包围盒接近全屏；不作为首选。

4. **把 `static_image` 的恢复责任上移到渲染器**
   - `SkiaRenderer.update_frame` 目前只是 `super().update_frame()`，没有机会介入；
   - 可以在 Skia 侧覆写 `update_frame`，先画 GPU 背景，再调用 `CairoRenderer` 的绘制逻辑，避免走 `set_pixels` 分支。

**必须配套的失效条件：**

- 每个 play 开始时 `save_static_frame_data` 会生成新的 `static_image`，版本要变；
- `static_image` 内容被原地修改时，仅靠 `id()` 不够，建议用显式 version/token；
- `clear()`、`close()`、`switch` 到别的 camera、`background_image` 变化时要释放/重建纹理；
- `MultiCamera` 子 target 各自有 `SkiaRenderTarget`，静态纹理缓存要么按 target 存，要么共享同一个 GPU 资源。

**验收：**

- `set_pixels` 在每个 play 中调用次数 ≤1；
- 1080p60 `set_pixels` 总耗时 ≤0.2s；
- 完整 Vulkan 出片 ≤176s（保守）或 ≤170s（与 A/B 上限一致）；
- 与当前 Vulkan 成片逐帧像素差为 0（GPU blit 不应改变像素）。

---

## 4. 瓶颈二：Python 逐对象更新，Vector 每帧重建箭头 tip

### 4.1 计时事实

- 1080p60：`update_to_time` **144.736s / 600 帧 = 241ms/帧**；
- 1080p60 另一次运行：**151.111s / 600 帧 = 252ms/帧**；
- 480p15：`update_to_time` **29.696s / 150 帧 = 198ms/帧**；
- 场景只有 100 个 `Vector` + 100 个 `Circle` + 1 个 `TracedPath`，更新成本与分辨率无关。

### 4.2 cProfile 热点（480p15，`DRY=1 NODRAW=1`）

cProfile 本身有约 1.7 倍放大，所以绝对值不能直接当生产耗时；看调用次数和相对占比。

| 函数 | 调用次数 | 累计时间 | 说明 |
|---|---:|---:|---|
| `fourier_pi_100.py:150(update_vector)` | 15,200 | 47.882s | 场景自定义 updater |
| `line.py:277(set_length)` | 15,200 | 45.203s | 每帧每个 Vector 调一次 |
| `line.py:610(Arrow.scale)` | 46,000/15,400 | 44.809s | `set_length` 进入箭头缩放 |
| `arc.py:114(add_tip)` | 15,400 | 28.226s | 每帧重新加 tip |
| `arc.py:208(reset_endpoints_based_on_tip)` | 15,400 | 16.748s | 重新定位 tip 端点 |
| `mobject.py:2106(put_start_and_end_on)` | 30,700 | 14.121s | tip 定位链 |
| `vectorized_mobject.py:1627(point_from_proportion)` | 30,800 | 15.554s | 弧长参数化 |
| `vectorized_mobject.py:1607(get_curve_functions_with_lengths)` | 123,200 | 14.920s | 曲线长度 |
| `bezier.py:146(cubic_bezier)` | 954,800 | 8.821s | 单点最多 |
| `tips.py:118(base)` | 30,800 | 15.606s | tip 几何 |
| `arc.py:178(position_tip)` | 15,400 | 11.228s | tip 定位 |
| `tips.py:172(tip_angle)` | 15,400 | 8.126s | 每帧算角度 |

### 4.3 机制

场景 updater 是：

```python
v.set_length(max(mag, 1e-9))
v.set_angle(phase + ...)
v.shift(v.center_func() - v.get_start())
```

`Vector` 继承 `Arrow`，`Arrow.set_length()` 继承 `Line.set_length()`：

```python
def set_length(self, length):
    return self.scale(length / self.get_length())
```

`Arrow.scale()` 的默认路径（`line.py:610-659`）：

1. `has_tip()` / `has_start_tip()`；
2. `old_tips = self.pop_tips()`；
3. `super().scale(factor)` 缩放箭杆；
4. `_set_stroke_width_from_length()`；
5. `self.add_tip(tip=old_tips[0])` 重新加回；
6. `add_tip` 里重新计算 `tip_length`、`tip_angle`、`reset_endpoints_based_on_tip`、`position_tip`；
7. 这些步骤又走 `point_from_proportion` / `get_curve_functions_with_lengths` / `cubic_bezier`。

结果：**100 个 Vector × 600 帧 = 60,000 次完整的 tip 拆除与重建**，每次 tip 重建又做多次弧长采样。`cubic_bezier` 95 万次调用就是这么来的。

这不是 Skia/Vulkan 的问题；同一份场景在 Cairo 上也会付出这 144s。它只是被 Vulkan 的 74s 上传掩盖了，修掉上传后就会成为最大项。

### 4.4 修复方向

**A. 场景层：不要每帧调用 `Vector.set_length()`**

- 傅里叶向量长度每帧本来就变，但箭头 tip 的视觉尺寸应当基本恒定；`Arrow.scale()` 每帧重建 tip 是纯浪费。
- 方案 1：用 `Line` 画箭杆，箭头用一个预制的 `ArrowTriangleFilledTip` 作为独立子对象；每帧只更新箭杆两端和 tip 的平移/旋转，不改 tip 几何。
- 方案 2：自定义 `FastVector`，缓存 `tip_length`、`stroke_width`，直接对 `points` 数组做“起点 + 单位方向 × 长度”的批量赋值；不调用 `Arrow.scale`、`add_tip`、`pop_tips`。
- 方案 3：如果必须保留 `Vector` 外观，先把 100 个向量建成固定长度，再用 `apply_points_function` 做仿射变换；但 `Arrow.scale(scale_tips=True)` 会连 tip 一起缩放，视觉不等价，需要确认。

**B. 数组层：100 个向量批量算，再写回**

- 每帧用 NumPy 一次性算 100 个 `(start, end, angle)`；
- 再把结果写入每个 mobject 的 `points`，避免每个对象内部的 `np.linalg.norm`、`normalize`、`rotation_matrix` 重复分配；
- 预计能把 Python 更新从 145s 降到 20–40s，具体取决于 `set_points` 和 family 递归成本。

**C. Rust 层：VectorChain**

- 如果目标是 ≤150s 且还要留余量，最终可能需要把“100 个向量的端点、箭头、圆半径”做成一个原生对象；
- Rust 侧每帧只接收 `t`，内部算所有点，一次性提交给 Skia；
- 但这会改变场景 API 和 updater 语义，属于较大的架构改动，不建议在修完 A/B 之前做。

**验收：**

- 同一 1080p60 场景 `update_to_time` ≤30s；
- 画面与当前 Vulkan 成片逐帧对比，箭头粗细、tip 长度、轨迹位置一致；
- 不改变 `Scene.play()`、updater 时序、`TracedPath` 行为。

---

## 5. 其余开销

### 5.1 Skia 绘制约 14s

- `NODRAW` 前后差 14.047s，600 帧约 23.4ms/帧；
- 当前 200 个 mobject，`SkiaDrawingContext.draw` 走 `it.groupby(type)` 后逐对象绘制；
- 这不是当前主矛盾；修完上传和更新后，它会变成约 8% 的项，再考虑按样式/路径批量提交。

### 5.2 回读 + 编码约 19.6s

- `get_frame` / `read_pixels` 在完整出片路径每帧一次；
- `commit_into` 已经直接写入 NumPy buffer，避免了额外 Python 拷贝，这一点做得对；
- 但 `surface.read_pixels` 是同步的，且 `flush_and_submit` 与编码串行；
- 19.6s 是 Phase 1 的上限，不是当前主瓶颈；如果后续要压到 150s 以内，可以用 PBO/双缓冲让回读与下一帧绘制重叠，收益上限约 10–15s。

### 5.3 `reset` / `clear` 不是问题

- 600 帧只调用一次 reset，12ms；
- `NORESET` 无变化；
- 不要把 reset 当作 74s 的来源。

---

## 6. 渲染框架问题清单（本轮核实）

以下按“对当前性能/正确性的影响”排序。行号以当前仓库为准。

### F1. 静态背景按“CPU 图 + 每帧上传”建模（P0，已实测）

- 证据：
  - `manim/renderer/cairo/renderer.py:279-282` 每帧 `set_pixels(static_image)`；
  - `manim/renderer/skia/target.py:92-97` 每次 `np.copyto` + `tobytes` + `load_pixels`；
  - `manim/renderer/cairo/target.py:164-173` Cairo 侧只是 CPU copy；
  - `native/manim_vk/src/lib.rs:1555-1576` `Vec<u8>` 再拷贝 + 每次 `flush_and_submit`。
- 影响：1080p60 73.9s；图像/点云/多相机场景会成倍放大。
- 修复：GPU 常驻静态纹理/背景 surface；按版本失效；每帧 GPU blit，而不是上传。

### F2. `SkiaRenderer` 继承 `CairoRenderer`，静态图语义没有 Skia 覆写（P0）

- 证据：`manim/renderer/skia/renderer.py:35` `class SkiaRenderer(CairoRenderer)`；`:236` 直接 `super().update_frame(...)`。
- 影响：Skia 目标被迫复用 Cairo 的“整帧背景恢复”逻辑，无法在 Skia 层做 GPU 缓存。
- 修复：在 `SkiaRenderer.update_frame` 中显式处理静态背景；或把“背景恢复”抽象成 target 能力，由 target 决定 CPU copy / GPU blit。

### F3. `set_pixels` 的 Python/Rust 接口每帧多次全帧拷贝（P0/P1）

- 证据：`target.py:92-97` 的 `np.copyto` + `tobytes`；`lib.rs:1555` 参数是 `Vec<u8>`，PyO3 需要把 `bytes` 转成 Vec；`lib.rs:1573-1575` 每次 flush。
- 影响：即使只上传一次，单次 1080p 也要约 0.1s；对逐帧 CPU 合成路径是灾难。
- 修复：`load_pixels` 接受 `PyBuffer<u8>` / `memoryview`，直接在 NumPy 缓冲区上建 Pixmap；`flush_and_submit` 延迟到帧边界；加 `set_pixels_if_changed(version)`。

### F4. 没有 Vulkan 回归/性能基线（P1，正确性风险）

- 证据：`tests/conftest.py:92` 强制 `config.renderer="cairo"`；`tests/` 中 “vulkan” 出现 0 次；`tests/test_graphical_units/control_data` 有 239 张 Cairo 基线。
- 影响：所有测试通过不代表 Vulkan 正确或不变慢；这次 74s 上传没有任何测试能发现。
- 修复：测试参数化 `renderer`；增加 Vulkan 基线帧（至少静态背景 + 动态对象 + 图像对象）；增加性能 smoke（见第 8 节）。

### F5. 背景色/图像填充路径仍有逐对象全图拷贝（P1）

- 证据：`manim/renderer/skia/rendering.py:123-133` 每个 background-colored VMobject 都执行 `np.ascontiguousarray(background).tobytes()`，再 `target.fill_path_with_image(...)`；`:161-170` 图像 mobject 走原生路径，点云走 `draw_point_pixels`，这两条已经不是 CPU 合成；多相机在 `skia/renderer.py:155-183` 已改成子 target + GPU 纹理采样，不要误判成 CPU 合成。
- 影响：背景色对象多时，每个对象每帧复制一整张背景图；这是和 F1 同类的“全帧拷贝”，只是入口不同。傅里叶场景没有 background-colored 对象，所以本次 73.9s 不包含它。
- 修复：`fill_path_with_image` 接受 GPU 纹理/缓存的 `SkImage`，而不是每对象 `tobytes()`；背景图按 version 缓存。

### F6. 跨 target 有共享 `DirectContext`，但没有静态纹理资源缓存（P1）

- 证据：`native/manim_vk/src/lib.rs:43-45` `GpuShared { backend, context }` 是 Device 级；`lib.rs:813-821` 每个 `SkiaTarget` 只是 `gpu.context.clone()`（引用计数 +1），**不是每 target 新建 DirectContext**；`lib.rs:745` 的 `Box::leak` 是每个 Device 一次的函数指针 trampoline，不是每 target 泄漏。
- 影响：跨 target 共享 GPU 纹理在技术上可行，但当前没有任何缓存层；每个 target 有自己的 `SkSurface`，静态背景仍然各自上传。多相机场景会重复付出上传成本。
- 修复：把静态背景纹理缓存挂在 `GpuShared` 或 `SkiaRenderTarget` 的资源池上，按 `(像素尺寸, version)` 复用；target 关闭时释放引用，避免把“Device 级共享”误做成“target 级私有”。


### F7. `SkiaTarget` / `NativeWindow` `#[pyclass(unsendable)]`（P2）

- 证据：`native/manim_vk/src/lib.rs:793-794`、`native/manim_vk/src/window.rs:222-223`。
- 影响：渲染线程绑定；不能把上传/回读放到后台线程，限制异步流水线。
- 修复：短期保持单线程；若要做异步回读，需要重新设计线程模型，不要直接跨线程调用。

### F8. 预览窗口是独立 wgpu Vulkan 实例（P2）

- 证据：`native/manim_vk/src/window.rs:5-8` 明确“presentation separate”；`:149-152` `PixelsBuilder` + `wgpu::Backends::VULKAN` 单独请求 adapter；`:111-133` 事件处理范围有限。
- 影响：与 Skia Ganesh 的 Vulkan 设备不共享；预览需要额外拷贝；无键鼠输入。
- 修复：长期统一到一个 Vulkan 设备/交换链；短期不影响离屏渲染性能。

### F9. `update_frame` 每帧重建对象列表（P2）

- 证据：`skia/renderer.py:221-234` 每帧 `list_update` + 逐对象 `isinstance` 分流。
- 影响：当前约 0.3–0.5s/600 帧，不是主项；对象数上千时会放大。
- 修复：场景对象变更时维护 `classic_mobjects` / `mesh_roots` 缓存，只增量更新。

### F10. 绘制路径逐对象提交（P2）

- 证据：`skia/rendering.py:57-64` 按类型 `groupby` 后调用 `display_funcs`，每个 mobject 单独构建路径。
- 影响：当前 14s；对象多时线性增长。
- 修复：同色/同 stroke 的路径批量合并；或把静态层预合成为一张纹理（与 F1 一并解决）。

### F11. 回读是同步全帧，未与编码重叠（P1）

- 证据：`lib.rs:1519-1521` `commit_into` 先 `flush_and_submit`；`:1547` `surface.read_pixels` 同步。
- 影响：19.6s；不是当前主项，但 150s 目标下需要挤出来。
- 修复：PBO/双缓冲；readback 与 ffmpeg 编码流水线化；不要在每帧 `load_pixels` 里 flush。

### F12. 没有静态层失效契约（P1，正确性风险）

- 证据：`manim/renderer/cairo/renderer.py:361-380` `save_static_frame_data` 每次 play 生成新 `static_image`；`update_frame` 只判断 `is not None`。
- 影响：如果做缓存，必须知道何时失效；否则会画出上一 play 的背景。
- 修复：给 `static_image` 增加 version/token；`save_static_frame_data`、`clear`、`close`、camera 切换时递增；target 按 token 缓存。

---

## 7. 已验证的死路 / 不要重复做的事

1. **不要再把 `SkiaRenderTarget.reset` 当主因。** 600 帧只调 1 次，12ms；`NORESET` 无变化。
2. **不要再把“取帧链路”当 1080p 主因。** 回读+编码 19.6s，占 7.8%；即使全去掉也到不了 150s。
3. **不要用旧 cProfile 里的 `nt.stat` 当渲染热点。** 那是 importlib 导入期，不是每帧。
4. **不要只做 `SKIPSTATIC` 就交付。** 它会留下动态残影；必须用 GPU 背景重绘。
5. **不要把 `update_frame` 的 148s 当成“Skia 绘制 148s”。** 它是嵌套计时：外层 Skia + 内层 Cairo + `set_pixels`；root 只有 74s，其中 73.9s 是上传。
6. **不要以为 `Vector` 的 144s 是 Vulkan 特有。** Cairo 同样要付；它是场景/Arrow 设计问题。

---

## 8. 建议的修复顺序与验收

### 阶段 1：静态背景 GPU 化（预计 252.8s → 170–176s）

1. 在 `SkiaRenderTarget` 增加静态背景纹理缓存（`skia_safe::Image` 或 background surface）。
2. `set_pixels` 改为 `set_static_pixels(pixels, version)`，只在 version 变化时上传。
3. `SkiaRenderer.update_frame` 每帧先 GPU blit 背景，再画动态对象；不再走 `CairoRenderer` 的 `set_pixels` 分支。
4. 失效：`save_static_frame_data`、`clear`、`close`、camera 切换、background image 变化。

验收：

- `set_pixels` 每 play ≤1 次；
- 1080p60 完整 Vulkan ≤176s；
- 与当前 Vulkan 成片逐帧像素差 0；
- 480p15 不退化，Cairo 路径不受影响。

### 阶段 2：Vector 更新去 tip 重建（预计 170s → 120–140s）

1. 先做场景层 `FastVector`：箭杆 + 预制 tip，分开更新。
2. 再用 NumPy 批量算 100 个端点/角度。
3. 如果还不够，再评估 Rust `VectorChain`。

验收：

- `update_to_time` 从 145–151s 降到 ≤30s；
- 画面一致；
- 不改变 updater 时序和 `TracedPath` 结果。

### 阶段 3：回读/绘制优化（可选，冲 150s 以内）

- PBO/双缓冲回读；
- 同样式路径批量；
- 目标：1080p60 完整 Vulkan ≤150s，且不低于 Cairo 的画面质量。

### 建议增加的观测点

在 `SkiaRenderTarget` 里加计数器（环境变量开启，默认关闭）：

- `static_upload_count`
- `static_upload_bytes`
- `dynamic_upload_count`
- `readback_count`
- `flush_count`

性能 smoke 测试（不进默认 CI，单独 job）：

- 480p15，100 Vector + 100 Circle，`dry_run`；
- 断言 `static_upload_count == 1`；
- 断言 `update_to_time` 不超过阈值；
- 1080p60 完整出片，记录总时长，允许 ±10% 波动。

---

## 9. 可以直接发给另一个 AI 的问题清单

1. 在 Skia Ganesh + Vulkan 的 `SkiaTarget` 上，如何把 `static_image` 变成 GPU 常驻背景，并在每帧以最低成本恢复？用 `skia_safe::Image` + `canvas.draw_image`，还是 background `SkSurface` + `draw_surface`？两者在 Ganesh 下是否有隐式拷贝/同步？
2. 现有 `CairoRenderer.update_frame()` 的 `static_image` 语义如何最小改动地上移到 `SkiaRenderer`，同时不破坏 Cairo 路径、`MultiCamera` 和 `save_static_frame_data`？
3. `load_pixels(&mut self, pixels: Vec<u8>)` 如何改成接受 `PyBuffer<u8>` / `memoryview`，避免 `tobytes` 和 PyO3 Vec 拷贝？`Pixmap::new` 能否直接借用 NumPy 缓冲区？生命周期/对齐有什么坑？
4. `flush_and_submit` 能否从每次 `load_pixels` 移到帧边界？如果静态背景只在 play 开始时上传一次，是否可以完全不 flush，等第一次 draw 时再提交？
5. 对 100 个 `Vector` 的每帧更新，怎样在不改变视觉的前提下避免 `Arrow.scale()` 的 `pop_tips`/`add_tip`？是自定义 `FastVector`，还是改 `Arrow.scale` 增加“固定 tip 尺寸”的快路径？哪种对 ManimCE 兼容性最好？
6. 如果要把 100 个向量的端点/箭头/圆批量更新移到 Rust，最小可用的对象 API 应该是什么？每帧传 `t` 和系数数组，Rust 返回 points，还是让 Rust 直接持有 `SkiaTarget` 并绘制？后者会不会和 `#[pyclass(unsendable)]` 冲突？
7. 如何在现有测试体系里加 Vulkan 基线，而不把 239 张 Cairo 基线全部重做？能否只对“静态背景 + 动态对象 + 图像对象 + 多相机”四类做少量 golden？
8. 回读 19.6s 中，PBO/双缓冲在 Skia Ganesh 上怎么落地？`surface.read_pixels` 有没有异步/回调接口？如果没有，是否只能用 `GrDirectContext::submit` + 双 surface 轮换？
9. `SkiaTarget::new` 的 `Box::leak` 和每 target 一个 `DirectContext`，在多相机场景下应该如何共享资源？有没有官方推荐的 `GrDirectContext` 共享/缓存方式？
10. 对 1080p60、600 帧、200 个 mobject 的场景，修完静态背景后合理的目标应该是多少？170s 是否已经接近 Python 更新的下限，还是还有未发现的 GPU 同步点？

---

## 10. 复现命令

```powershell
# 1080p60 完整 Vulkan
$env:Q='high_quality'; $env:R='vulkan'; $env:SW='2'; $env:TIP='0.15'; $env:TIPR='0.2'
& '<repo>\.venv\Scripts\python.exe' `
  "$env:TEMP\mvk_fourier\fourier_pi_100.py"

# 只测 Python 更新 + 静态上传（无绘制、无出片）
$env:DRY='1'; $env:NODRAW='1'; $env:TIMERS='1'

# 证明静态上传成本（画面会残影，仅计时）
$env:SKIPSTATIC='1'
```

关键原始数据：

```text
1080p60 Vulkan 完整：252.756s
1080p60 Vulkan dry：233.213s
1080p60 Vulkan dry+NODRAW：219.166s
1080p60 dry+NODRAW+TIMERS：
  update_to_time 144.736s
  set_pixels      73.872s / 600 calls
  update_frame root 74.172s
1080p60 dry+NODRAW+SKIPSTATIC：143.051s
1080p60 完整+SKIPSTATIC：170.368s
480p15 dry+NODRAW+TIMERS：
  update_to_time 29.696s / 150 calls
  set_pixels      3.222s / 150 calls
```

---

## 11. 附：本轮没有改动仓库

- 只修改了临时基准脚本 `%TEMP%\mvk_fourier\fourier_pi_100.py`，增加 `TIMERS`、`SKIPSTATIC` 等实验分支；
- 仓库内没有提交、没有重命名、没有删除；
- 仓库当前没有 `.git`，无法用 git diff 证明；如需复核，可对比上述文件行号与当前源码。

如果另一台机器上还有别的项目在改同一目录，请以这份报告的行号为“核查线索”，不要直接让另一个 AI 大范围重构；先做 F1（静态背景 GPU 化）这一处，收益最大、风险最小。
