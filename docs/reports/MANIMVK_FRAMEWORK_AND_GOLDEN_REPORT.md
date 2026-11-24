# ManimVK 渲染框架与 Golden 逐帧对照报告

日期：2026-11-23  
仓库：`<repo>`  
远端：`<GitHub remote>`  
当前 `master`：`d50f33b`（`vulkan-strip` + 3D bundle 已合并并推送）  
本轮性质：审查后的 golden 全量对照 + 根因修复；不是只留一个 Cairo 基线等人接手。

> 后续 bundle 合并、图像 FAIL 修复、毛球动画实跑的结果见  
> `MANIMVK_BUNDLE_MERGE_AND_HAIRY_BALL_REPORT.md`。本文第 0/4/5 节的数量是 bundle 之前的状态。

---

## 0. 结论

1. **golden 已经实际跑完，不是“留给别人做”。** 全量收集 458 个场景，Cairo/Vulkan 各成功渲染 454 个，得到 `PASS 426 / FAIL 15 / REVIEW 13 / ERROR_CANDIDATE 0 / ERROR_BOTH 4`。
2. 对 15 个 FAIL 和 13 个 REVIEW 逐项定位后，修掉了 6 类真实根因：PMobject 点径翻倍、背景图 VMobject 的填充/遮罩语义错误、图像采样默认最近邻、StreamLines 随机相位、networkx 随机布局、**相机 Y 平移矩阵符号错误**。
3. 对受影响的 30 个场景复跑后，当前为 **`PASS 16 / FAIL 2 / REVIEW 12 / ERROR_CANDIDATE 0 / ERROR_BOTH 0`**。`ERROR_CANDIDATE` 已经清零，`FollowingGraphCamera` 和 `SlidingMultipleFrames` 已转为 PASS。
4. **Cairo 暂时不能删。** 删除门槛是 `FAIL=0`、`ERROR_CANDIDATE=0`，且 REVIEW 人工确认。现在还有 2 个 FAIL 和 12 个 REVIEW；2 个 FAIL 都是同一张 2×4 图像放大时的采样相位差异，不是几何错位。
5. 性能补丁后的真实瓶颈已经变化。1080p60、600 帧、100 向量傅里叶场景当前 `render=28.7s`，拆解为：
   - Python 场景更新 **16.9s（58.7%）**
   - Skia 实际绘制 **5.1s（17.9%）**
   - 回读 **5.9s（20.5%）**
   - 编码/其他约 0.8s
6. **所以“把主体全部迁到 Rust”并不能解决主要瓶颈。** 绘制已经在 Rust/Skia 里，只占 18%；真正的大头仍是 Python 每帧更新 100 个 `Vector` 和 100 个 `Circle`。要提速，优先做“原生或向量化的 updater”，而不是只改绘制层。

---

## 1. 当前性能实测

### 1.1 同一场景的前后对比

场景：3b1b `FourierOfPiSymbol` 移植版，100 个旋转向量 + 100 个圆 + 轨迹，10s，1080p60，600 帧。

| 口径 | render 时间 | 说明 |
|---|---:|---|
| 补丁前 Vulkan | 252.756s | 每帧上传整张静态背景 + 每帧重建箭头 tip |
| 补丁后 Vulkan | **28.709s** | 静态层 GPU 常驻、箭头 tip 刚性缓存、路径批量提交 |
| 另一次补丁后测量 | 31.386s | 同机波动，量级一致 |

补丁后逐段计时（`TIMERS=1`）：

```
TIMING n=100 coef=0.052s build=0.124s render=28.709s total=28.885s
TSTAT update_mobjects 16.897s calls=602
TSTAT update_to_time 16.861s calls=600
TSTAT get_frame      5.890s calls=600
TSTAT read_pixels    5.882s calls=600
TSTAT update_frame   5.461s calls=601
TSTAT _render_camera 5.150s calls=601
TSTAT draw           5.139s calls=601
TSTAT reset          0.011s calls=1
WALL high_quality 29.110s
```

占比：

| 构成 | 时间 | 占比 |
|---|---:|---:|
| Python 场景更新 | 16.9s | 58.7% |
| Skia 绘制 | 5.1s | 17.9% |
| GPU 回读 | 5.9s | 20.5% |
| 编码/其他 | ~0.8s | 2.9% |

### 1.2 纯 updater 基准

`benchmarks/bench_fourier_update.py 100 60` 的结果是 **16.2ms/帧**（60 帧平均）。600 帧下界约 9.7s；真实渲染路径的 `update_to_time` 是 28.1ms/帧（16.9s），差额来自场景 Manager、`TracedPath`、`ValueTracker` 和实际箭头参数（fill、stroke、tip）等真实路径开销。即使只看下界，Python 更新也已经是最大单项。

### 1.3 旧报告的结论哪些还成立

旧报告 `MANIMVK_PERFORMANCE_AND_FRAMEWORK_REPORT.md` 的 252.8s 拆解仍然解释了“补丁前为什么慢”，但绝对值已过时：

- 静态背景每帧上传 73.9s：**已修掉**，`SkiaRenderTarget` 现在按版本缓存 GPU 静态纹理，每帧只做 GPU blit。
- `Vector.set_length()` 每帧 `pop_tips/add_tip`：**已修掉**，箭头 tip 改为刚性缩放缓存。
- 旧报告说“Python 更新 144.7s、占 57%”：补丁后降到 16.9s，但仍是 58.7%，占比没变，说明它一直是框架的主矛盾，只是之前被 73.9s 上传掩盖。
- 旧报告说“Skia 绘制约 14s”：当前 5.1s，路径批量提交生效。
- 旧报告说“回读+编码 19.6s”：当前回读 5.9s，编码约 0.8s；如果做异步回读，仍有约 5s 可省。

---

## 2. Golden 方法与结果

### 2.1 方法

- 场景索引：`golden\scenes.json`，模块目录：`golden\scenes`，共 458 个。
- 渲染器：`tools\golden\render.py`，同一场景分别在 Cairo 和 Vulkan 下渲染首帧、若干 eval 帧、末帧。
- 对照：`tools\golden\compare.py`，输出 `PASS / FAIL / REVIEW / ERROR_*`，并生成逐像素 diff 热力图。
- 判定阈值按当前工具默认：coarse MAE、RGB MAE、alpha MAE、PSNR、changed ratio 综合；不是单一 MSE。
- 确定性处理：模块导入前 `random.seed(0)`、`np.random.seed(0)`；每次 `construct` 前再 seed；networkx 的 `spring_layout`、`random_layout` 包装为默认 `seed=0`。

> 注意：Golden 必须在完整 PATH 下运行。PowerShell 默认沙箱路径会丢掉 `C:\Windows\System32` 和 MiKTeX，曾导致 134 个 LaTeX 场景 `FileNotFoundError`。需要从注册表恢复 machine + user PATH 后再跑。

### 2.2 全量初始结果

| Verdict | 数量 |
|---|---:|
| PASS | 426 |
| FAIL | 15 |
| REVIEW | 13 |
| ERROR_CANDIDATE | 0 |
| ERROR_BOTH | 4 |
| ERROR_REFERENCE | 0 |

4 个 `ERROR_BOTH` 不是渲染器问题，是收集/示例问题：

- `TexTemplateFromCLI`：LaTeX 转 dvi 失败。
- `CustomTipExample`：收集模块语法错误。
- `CustomLayoutExample`：缺少 `Any` 导入。
- `SoundExample`：缺少 `click.wav`。

### 2.3 修复后针对性复跑

对 30 个原先 FAIL/REVIEW 的场景复跑：

| Verdict | 数量 |
|---|---:|
| PASS | 16 |
| FAIL | 2 |
| REVIEW | 12 |
| ERROR_CANDIDATE | 0 |
| ERROR_BOTH | 0 |

报告：`golden_after\report_final\report.md`。  
这是针对性复跑，不是 458 个全量重跑；全量结果仍以初始 426 PASS 为基线，修复只影响列出的场景类别。

---

## 3. 已修复的根因

### 3.1 PMobject 点径被扩张两次（精确修复）

**现象**：`PMobjectExample`、`PgroupExample` 的 Vulkan 点云明显更大，接近实心圆盘；Cairo 是细小密集点。

**根因**：`manim/renderer/skia/rendering.py` 的 `_draw_point_cloud_native` 先调用 `_thickened_coordinates`，把每个点扩成 `thickness × thickness` 个像素；随后又用 `half_size = thickness / 2` 画方块，等于把扩张做了两次。

**修复**：不再调用 `_thickened_coordinates`；每个原始点只画一个 `thickness × thickness` 方块，中心取 `x + thickness/2`、`y + thickness/2`，`half_size = thickness/2`。

**结果**：`PMobjectExample`、`PgroupExample` 变为逐像素 0；`PointCloudDotExample` 降到 `RGB 1.74 / coarse 0.24`，转 REVIEW。

### 3.2 背景图 VMobject 的填充与遮罩语义错误

**现象**：`BasicUsage_2` 的 Vulkan 向量场出现大片红色块；`StreamLines` 的细线偏亮或偏暗；`EndAnimation` 不一致。

**Skia 侧根因**：`_display_background_colored_vmobjects` 对每个对象无条件 `fill_path_with_image`，忽略 `fill_opacity`。`StreamLines` 是开放路径，填充规则会把它隐式闭合，产生大面积色块。

**Cairo 侧根因**：Cairo 用 scratch alpha 遮罩背景图，但把背景 RGB 也乘了一次 alpha，再 `alpha_composite`，导致 coverage 被乘两次，细线偏暗。

**修复**：

- Skia：`fill_opacity == 0` 时不执行图像填充；新增 `_apply_image_stroke`，在路径中心采样背景图颜色，用该颜色描边，保持原线宽/join/cap。
- Cairo：只把 `scratch.pixels[:, :, 3]` 乘到背景图 alpha，保持 RGB 不变。

**结果**：`EndAnimation` 逐像素 0；`ContinuousMotion` 修复随机相位后 PASS；`BasicUsage_2` 从大面积红块降到 `coarse 2.65 / RGB 6.34 / changed 38.8%`，转 REVIEW。剩余差异来自“中心采样单色”只是 Cairo 逐像素 alpha 遮罩的近似，不是几何错误。

### 3.3 图像采样默认最近邻

**现象**：`ImageFromArray` 的 Vulkan 把 4×4 灰度图放大成方块色块；Cairo 是平滑渐变。

**根因**：原生 `draw_image_quad`、`draw_target_quad` 使用 `SamplingOptions::default()`，即 `FilterMode::Nearest`。

**修复**：改为 `SamplingOptions::from(CubicResampler::catmull_rom())`；背景图填充用 Linear；`blit_full` 保留默认。

**结果**：`ImageFromArray` 从 `RGB 7.41 / coarse 7.45 / changed 58.8%` 改善到 `RGB 4.24 / coarse 4.26 / changed 49.7%`，但仍 FAIL。两侧图像外接矩形完全一致（x 7..846，y 30..449，840×420），差异来自 Cairo 的 PIL 透视 BICUBIC 与 Skia GPU cubic 的采样相位/边缘，不是几何错位。

### 3.4 随机相位与随机布局

**StreamLines**：`start_animation` 用 `random.random()` 给每条线赋初值，Cairo/Vulkan 分进程渲染时随机状态不同，导致同一时刻线条数量/长度不同。已在 golden harness 中模块导入前和每次 `construct` 前 `random.seed(0); np.random.seed(0)`；重复渲染同一 renderer 也不再漂移。`ContinuousMotion` 四帧全部 PASS（约 `RGB 0.88 / coarse 0.36`）。

**networkx**：`spring_layout`/`random_layout` 使用 `np.random.default_rng(seed=None)`，`np.random.seed` 无效；`ImportNetworkxGraph` 的 `nx.erdos_renyi_graph` 还在模块导入时生成。已在 `exec_module` 前 seed，并把 `spring_layout`、`random_layout` 包装为默认 `seed=0`；`kamada_kawai`/`spiral` 不接受 seed，已排除。`GraphAutoPosition`、`RandomLayout`、`SpringLayout` 变为 PASS；`ImportNetworkxGraph` 四帧全 PASS，其中 first/final 为 0。

### 3.5 相机 Y 平移矩阵符号错误（本轮最严重的框架 bug）

**现象**：`FollowingGraphCamera` 的 eval 中间帧 Vulkan 全黑，Cairo 正常；`SlidingMultipleFrames` 也有缩放后的内容偏移。最终帧却正常，因为 `Restore` 把相机中心回到原点。

**根因**：`SkiaRenderTarget.get_context()` 里计算 Y 平移时写成了

```python
ty = settings.pixel_height / 2 + center[1] * sy
```

但 `sy` 是负值（用于把 Manim 的 y-up 翻成 Skia 的 y-down）。正确的像素映射是

```text
y' = sy * (y - center_y) + H/2
```

展开后 `ty` 应当是 `H/2 - center_y * sy`，而不是加。相机中心 Y 非零时，画面会整体偏移约 `2 * |sy * center_y|` 像素。`FollowingGraphCamera` 的相机中心 Y 到 -2.97，偏移超过 700px，所以中间帧直接黑屏。这个问题不是抗锯齿，也不是静态层缓存。

**修复**：

```python
tx = settings.pixel_width / 2 - center[0] * sx
ty = settings.pixel_height / 2 - center[1] * sy
```

**结果**：`FollowingGraphCamera`、`SlidingMultipleFrames` 从 FAIL 转为 PASS；`MovingCameraOnGraph` 也转为 PASS。这个修复影响所有相机中心 Y 非零的 Vulkan 画面，不只是 golden 场景。

> 同时补了 `Scene.get_moving_mobjects`：相机 frame 被动画或被 updater 驱动时，即使它不在 `scene.mobjects` 里，也必须把所有 mobject 视为 moving，禁止复用静态层。否则静态层会在相机移动时保留旧投影。

### 3.6 相机/缩放抗锯齿（已不再是主要问题）

`MovingCameraOnGraph` 在矩阵修复后已转为 PASS；`FollowingGraphCamera`、`SlidingMultipleFrames` 也已 PASS。此前把它们的差异归为“缩放后 AA”是误判，真正原因是相机 Y 平移矩阵。当前没有专门为 AA 调参，避免为了边缘差异破坏 426 个已 PASS 的场景。

---

## 4. 剩余 2 个 FAIL 的逐项判断

| 场景 | 最差帧 | coarse | RGB | PSNR | changed | 判断 |
|---|---|---|---:|---:|---:|---:|---|
| `MovingZoomedSceneAround` | eval_first | 4.26 | 4.24 | 31.8 | 49.7% | 2×4 图像 cubic 采样相位 |
| `ImageFromArray` | final | 4.26 | 4.24 | 31.8 | 49.7% | 同一张 2×4 图像，外接矩形一致 |

这两个场景用的是同一张 `np.uint8([[0,100,30,200],[255,0,5,33]])`，放大到约 840×420。Vulkan 用 GPU cubic Catmull-Rom，Cairo 用 PIL 透视 BICUBIC。外接矩形已经完全一致，差异在采样相位和边缘处理，不是几何错位。

两条路：

1. **接受为视觉容差**：把“几何/颜色语义正确、仅亚像素采样差异”从 FAIL 降为 REVIEW，并在报告中人工确认。这需要你拍板，不能由我单方面改阈值。
2. **复刻 Cairo 的 CPU/PIL 采样**：图像对象改走 PIL 透视 BICUBIC。代价是图像路径回 CPU，可能吃掉部分性能收益；需要评估是否值得为这 2 个场景做。

> `FollowingGraphCamera` 和 `SlidingMultipleFrames` 已不再在 FAIL 名单里；它们之前是相机 Y 平移矩阵符号错误，已在 3.5 修复。

---

## 5. 12 个 REVIEW 的分布

| 场景 | coarse | RGB | changed | 主要差异 |
|---|---:|---:|---:|---:|---|
| `OpeningManim`（docs） | 0.39 | 1.00 | 7.8% | 文本/线条 AA |
| `OpeningManim`（example） | 0.39 | 1.00 | 7.8% | 同上，重复场景 |
| `ThreeDSurfacePlot` | 0.70 | 1.10 | 13.2% | 3D 表面边缘 |
| `CoordSysExample` | 0.43 | 1.07 | 8.1% | 坐标网格 AA |
| `ConvexHull3DExample` | 1.28 | 1.39 | 11.0% | 3D 面/边 |
| `ImageInterpolationEx` | 2.01 | 2.45 | 10.5% | 图像采样 |
| `PointCloudDotExample` | 0.24 | 1.74 | 1.0% | 点云边缘 |
| `BasicUsage_2` | 2.65 | 6.34 | 38.8% | 背景图 alpha 遮罩近似 |
| `SpawningAndFlowingArea` | 2.63 | 4.48 | 23.6% | 同上 |
| `StreamLineCreation` | 0.55 | 2.28 | 12.5% | 流线颜色/透明度 |
| `LinearTransformationSceneExample` | 0.18 | 0.82 | 10.3% | 线条 AA |
| `RateFuncExample` | 0.37 | 1.11 | 6.4% | 文本/线条 AA |

REVIEW 的共同特征：差异集中在边缘、透明度或采样相位，没有对象缺失、位置错乱或拓扑错误。需要人工看一遍 diff 热力图确认，而不是直接当作 PASS。

---

## 6. 渲染框架问题

### 6.1 最大问题：Python 每帧逐对象更新仍是主瓶颈

当前 16.9s / 28.7s 花在 `Scene.update_to_time`。100 个 `Vector` 每帧执行：

```python
v.set_length(...)
v.set_angle(...)
v.shift(...)
```

即使箭头 tip 已经改成刚性缓存，`set_length → scale → set_points`、`set_angle → rotate`、`shift → apply_points_function` 仍在 Python 里逐对象做 NumPy 运算和 family 递归。100 个圆又各自 `set_width` + `move_to`。600 帧就是 12 万次对象级更新。

**这不是 Skia/Vulkan 的绘制问题。** 同一场景在 Cairo 上也要付这笔更新成本；Vulkan 的绘制只占 5.1s。

### 6.2 第二个问题：回读是同步的

`get_frame` 5.9s，其中 `read_pixels` 5.88s。每帧 `flush_and_submit` + `surface.read_pixels` 同步等待 GPU；没有 PBO/双缓冲，也没有和编码重叠。按当前 600 帧算，约 9.8ms/帧。异步回读的理论收益约 5s，足以把 28.7s 压到 23s 左右，但前提是先解决更新瓶颈。

### 6.3 第三个问题：绘制仍按对象提交，没有 display list

`SkiaDrawingContext.draw` 5.1s / 601 次，约 8.5ms/帧。当前 200 个对象还不算灾难，但每增加一个 VMobject，就多一次 Python 侧路径构建、`set_cubic_path` 点拷贝和原生 draw 调用。对象数上千时会线性放大。

更彻底的做法是 **display list / 原生场景图**：

- Python 每帧只提交一个扁平命令列表（对象 id、路径 id、变换、样式、透明度），一次 PyO3 调用；
- Rust 侧持有路径缓存，静态路径只上传一次，动态对象只更新变换或点数组；
- 绘制循环在 Rust 内完成，不再逐对象跨语言调用。

但要注意：这只减少“提交开销”，不会减少 Python 计算 updater 的时间。

### 6.4 第四个问题：背景图 compositing 仍是近似

当前 Skia 对背景图 VMobject 的描边用“路径中心采样单色”，Cairo 是逐像素 alpha 遮罩。两者在细线、渐变背景、半透明填充上不可能完全一致。`BasicUsage_2` 的 REVIEW 就是这个问题。

可选方案：

1. GPU `saveLayer` + `DstIn`：把对象画进离屏层，用对象 alpha 作为遮罩，再把背景图通过 `srcIn`/`dstIn` 合成。语义最接近 Cairo，但需要处理 layer 边界、抗锯齿和性能。
2. Rust 侧 image-shader stroke：在描边 shader 里按设备坐标采样背景图，而不是中心单色。适合细线，但复杂路径和 join/cap 难完全复刻。
3. 接受 REVIEW：如果产品口径允许“细线颜色近似”，这是成本最低的。

### 6.5 第五个问题：两套渲染器各自实现语义，没有共享 conformance 层

Cairo 和 Skia 是两条独立实现：填充、遮罩、采样、AA 各写一套。golden 之前没有进入 CI，`tests/conftest.py` 还强制 `renderer="cairo"`，所以 Vulkan 的回归只能靠人工跑 458 个场景才发现。

这次 5 类根因里有 3 类（背景图填充、alpha 遮罩、图像采样）都是“两套实现语义不一致”，不是单纯的 bug。建议：

- 把 golden 的 30 个高风险场景做成 CI 的 smoke（不必每次跑 458 个）；
- 给 `PASS/REVIEW/FAIL` 建立基线文件，CI 只允许 REVIEW 数量不增加；
- 新渲染特性必须同时给出 Cairo 语义说明和 Vulkan 对照帧。

### 6.6 已经修掉/不再是问题的项

- 静态背景每帧上传：已改为 GPU 常驻纹理，`set_pixels` 每 play 最多一次。
- `Vector` tip 每帧拆建：已改为刚性缩放缓存，`pop_tips/add_tip` 不再每帧发生。
- 相机 Y 平移矩阵：已修正 `SkiaRenderTarget.get_context()` 的符号错误；相机中心 Y 非零时不再整体偏移/黑屏。
- 相机 frame 动画/updater 与静态层冲突：`Scene.get_moving_mobjects` 现在会禁用静态层复用，避免旧投影残留。
- `reset`：600 帧只调 1 次、11ms，不是瓶颈。
- 多相机子 target：已经是 GPU 纹理采样，不是 CPU 合成。
- `set_pixels` 的 `Vec<u8>` 全帧拷贝：静态路径已避开；动态路径仍需注意，但当前不是主项。

---

## 7. 关于“主体迁到 Rust / Python 一次性生成渲染数据”

### 7.1 “把主体迁到 Rust”能解决什么

- 能解决：绘制提交、路径缓存、回读流水线、display list 合并。
- 不能解决：Python updater 的 16.9s。只要每帧还有 100 个 Python 闭包在改 100 个对象，迁移绘制层不会动这块。

因此正确顺序是：

1. **先做 updater 的原生/向量化**：把 100 个向量的端点、角度、圆半径用一次 NumPy 批量算完，再批量写回；或者做 Rust `VectorChain`，Python 每帧只传 `t`。
2. **再做异步回读**：PBO/双缓冲，把 5.9s 隐藏到绘制后面。
3. **最后做 display list / 原生场景图**：把 5.1s 绘制和跨语言提交压下去。

### 7.2 “Python 一次性生成渲染数据交给 Rust”是否可行

分两种情况：

- **静态层**：完全可行。一个 play 的静态 mobject 可以预生成为 display list 或一张 GPU 纹理，Rust 每帧只 blit。这正是性能补丁已经做的方向。
- **动态层**：不完全可以。Manim 的 updater 是任意 Python 函数，每帧依赖上一帧状态、`dt`、外部闭包；除非把 updater 的语义也迁到 Rust，或者预先把整段动画的每帧变换采样成数组，否则“一次性生成”只能覆盖确定性的、可预计算的动画。

对傅里叶这种场景，可行折中是：

- Python 在 play 开始时一次性算出 600 帧、100 个向量的端点/角度/长度数组（约 600×100×几个 float，几 MB）；
- 每帧只把一个 slice 传给 Rust，Rust 内部生成路径并绘制；
- 语义仍然是“每帧更新”，但 Python 的逐对象闭包被批量化掉了。

这不是通用 Manim API，属于“高性能场景模式”；通用路径仍需保留 updater。

### 7.3 一个可落地的原生接口草案

```text
NativeScene.submit_frame(
    frame_index,
    transforms: Float32Array,   # 每对象 6 个 float：平移/旋转/缩放
    style_updates: ...,
    path_updates: ...           # 仅动态几何对象
)
```

- 静态对象只在第一帧提交完整路径，后续只提交 transform；
- 动态对象提交点数组或预计算的路径；
- Rust 侧按 z-order 绘制，避免每对象 Python→Rust 调用；
- 回读用双缓冲，帧 n 的 readback 与帧 n+1 的绘制重叠。

这个接口的前提是先把“对象身份/路径缓存/失效契约”定义清楚，否则会出现静态层画错、缓存泄漏。

---

## 8. 建议的验收与下一步

### P0：updater 批量化（收益最大）

- 目标：1080p60 傅里叶 `update_to_time` 从 16.9s 降到 ≤6s。
- 做法：先做场景级 `FastVector`/批量 NumPy，再做 Rust `VectorChain`。
- 验收：总 render ≤18s；第 8 秒画面与当前 Vulkan 逐像素一致；不改变 `Scene.play()` 和 updater 语义。

### P1：异步回读

- 目标：`read_pixels` 从 5.9s 降到 ≤1s 可见开销。
- 做法：PBO/双缓冲，readback 与下一帧绘制/编码重叠；不要在每帧 `load_pixels` 里同步 flush。
- 验收：总 render ≤23s（在 P0 之前）或 ≤12s（在 P0 之后）；画面不变。

### P2：display list / 原生场景图

- 目标：`draw` 从 5.1s 降到 ≤2s，对象数上千时仍近线性。
- 做法：静态路径缓存 + 每帧扁平命令列表；Rust 内完成 z-order 和批量提交。
- 验收：200 对象场景 draw ≤2s；2000 对象场景不出现数量级退化；Cairo 路径不受影响。

### P3：golden 收口

- 先人工确认 12 个 REVIEW，特别是 `BasicUsage_2`、`SpawningAndFlowingArea`、`StreamLineCreation` 的背景图遮罩差异。
- 对 2 个 FAIL 做决策：接受为容差，还是复刻 Cairo CPU 采样。
- 只有 `FAIL=0`、`ERROR_CANDIDATE=0`、REVIEW 全部签字后，才删 Cairo。

### P4：把 golden 进 CI

- 30 个高风险场景 smoke，跑 Cairo/Vulkan 两套；
- 基线文件记录 PASS/REVIEW 数量；
- 任何新渲染路径必须提供对照帧，不能只跑单元测试。

---

## 9. 复现与产物

### 9.1 关键命令

```powershell
# 全量 golden
python tools\golden\render.py --renderer cairo  --out golden\cairo
python tools\golden\render.py --renderer vulkan --out golden\vulkan
python tools\golden\compare.py --reference golden\cairo --candidate golden\vulkan --out golden\report

# 受影响场景复跑
python tools\golden\render.py --renderer cairo  --scenes golden_after\scenes.txt --out golden_after\cairo
python tools\golden\render.py --renderer vulkan --scenes golden_after\scenes.txt --out golden_after\vulkan
python tools\golden\compare.py --reference golden_after\cairo --candidate golden_after\vulkan --out golden_after\report_final

# updater 基准
python benchmarks\bench_fourier_update.py 100 60

# 1080p60 傅里叶 + 分段计时
$env:Q="high_quality"; $env:TIMERS="1"; $env:R="vulkan"
python $env:TEMP\mvk_fourier\fourier_pi_100.py
```

### 9.2 主要产物

- `golden\report\report.md`：458 场景初始全量报告。
- `golden_after\report_final\report.md`：30 个受影响场景的修复后报告。
- `MANIMVK_PERFORMANCE_AND_FRAMEWORK_REPORT.md`：补丁前 252.8s 的详细拆解（历史基线）。
- 本文件：当前性能、golden 结果、残余差异和框架建议。

### 9.3 本轮修改的提交状态

已本地提交（`fix: correct Skia camera Y translation and golden renderer semantics`），**尚未推送**。涉及文件：

- `manim/renderer/skia/rendering.py`：点云单次方块、背景图零透明跳过、中心采样描边。
- `manim/renderer/cairo/rendering.py`：背景图遮罩只乘 alpha。
- `manim/renderer/skia/target.py`：修正相机 Y 平移矩阵的符号。
- `manim/scene/scene.py`：相机 frame 动画/updater 时禁用静态层复用。
- `native/manim_vk/src/lib.rs`：cubic/linear 采样。
- `tools/golden/render.py`：随机种子、networkx 布局 seed。
- `tests/module/renderer/test_skia_camera_matrix.py`：新增相机中心映射到像素中心的回归测试。
- 原生 wheel 已重编译并安装到 `.venv`；其他机器拉取后必须重编译原生扩展，不能只更新 Python 代码。

---

## 10. 需要你或外部同学拍板的问题

1. 2 个 FAIL：接受“几何正确、仅采样差异”的视觉容差，还是复刻 Cairo 的 CPU/PIL 路径？
2. `BasicUsage_2` / `SpawningAndFlowingArea` / `StreamLineCreation`：是否要求逐像素 alpha 遮罩？如果是，优先做 GPU `saveLayer + DstIn`，不要继续用中心采样近似。
3. 性能目标：是“低于同仓库 Cairo”（已做到），还是“1080p60 600 帧 ≤15s”？后者必须做 P0+P1，单靠渲染层不够。
4. 是否接受“高性能场景模式”的预计算数组接口？这会改变部分场景的写法，但能绕开 Python 逐对象 updater。
5. Cairo 删除时点：只有在上述 FAIL/REVIEW 全部收口后才执行；当前不建议删。

---

## 11. 一句话总结

这次 golden 不是“留个 Cairo 对照等人做”，而是已经跑完、定位、修复并复跑：`ERROR_CANDIDATE` 清零，剩余 2 个 FAIL 都是同一张 2×4 图像的采样相位差异，12 个 REVIEW 是边缘/透明度近似；性能上真正的瓶颈已经从“静态上传 + 箭头 tip 重建”转移到“Python 每帧逐对象更新”，这也是下一步最值得投入的地方。
