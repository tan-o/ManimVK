# ManimVK：bundle 合并、golden 收口与毛球动画

日期：2026-11-23  
仓库：`<repo>`  
远端正向：`<GitHub remote>`  
当前 master：`d50f33b`（已推送，取代 `bf06d86`）

---

## 0. 结论

1. **bundle 已实际审查、编译、测试并合并进 master。** 不是只看 diff 就合。
2. **合并前发现并修掉了一个真实回归：** 图像对象在两个 golden 场景里仍有 4.24 RGB MAE；改为走 PIL 透视 BICUBIC 后，两个场景均为 PASS，误差降到 `RGB 0.00117 / PSNR 64.8`。
3. **合并后 30 个高风险 golden 场景：`PASS 20 / FAIL 0 / REVIEW 10 / ERROR 0`。** 原来 2 个 FAIL 已清零；10 个 REVIEW 逐张看过，差异都在 AA、透明度或背景图遮罩近似，没有对象缺失、几何错位或拓扑错误。
4. **`hairy_ball.py` 四个场景全部在本机 Vulkan 出片（480p15）**，包括需要地球贴图的 `HairyEarth`。
5. **原生 `lib.rs` 再次变化。** 其他机器拉取 master 后必须重新编译 `manim_vk`，只更新 Python 代码会加载旧 `.pyd`。
6. 全量 458 场景的合并后 Vulkan 复跑只跑到 31/458，因单场景含 LaTeX 时耗时较长而主动停止；本轮验收以“30 个受影响场景 + 1059 项模块测试”为准，未把未完成的 427 个场景算作通过。

---

## 1. bundle 审查

bundle：`<attachment-dir>\manimvk-vulkan-strip.bundle`  
head：`379522b8dbde62b209459f8c4970fb85722b428e refs/heads/vulkan-strip`  
merge-base：`0108253`

### 1.1 bundle 新增能力

| 能力 | 实现 |
|---|---|
| 深度排序 3D | `Mobject.apply_depth_test()` / `deactivate_depth_test()`；VMobject 按子路径、MeshSurface 按三角形排序；64 个深度分片，远到近绘制 |
| `MeshSurface` | 一整块三角网格，按顶点光照，一次 GPU 绘制；替代 CE `Surface` 的“几千个小平面各画各的” |
| `TexturedSurface` | 日/夜两张等距圆柱投影贴图；按 `u,v` 映射，终结线附近日夜混合；光照模型按 ManimGL |
| 原生 mesh | `draw_textured_mesh`、`set_source_image`、`image_sampling` |
| 深度测试语义 | 半透明 mesh 会像 GL 深度缓冲一样挡住后面的 depth-tested 对象 |
| 5 个 Skia bug | 相机 Y 平移符号、背景图着色、图像采样、点云二次加粗、极点巨大坐标 |

### 1.2 合并冲突与处理

bundle 与 master 的 4 个冲突文件：

| 文件 | 处理 |
|---|---|
| `manim/renderer/skia/rendering.py` | 采用 bundle 版本：`set_source_image`、`_SAMPLING`、`_tame_huge_curves` 比 master 的中心采样近似更完整 |
| `manim/renderer/skia/target.py` | 采用 bundle 版本：`draw_textured_mesh`、`set_source_image`、`draw_image_quad(..., sampling)` |
| `native/manim_vk/src/lib.rs` | 手工合并 imports 与函数；保留 `CubicResampler`、`MipmapMode`、`gpu`，图像采样走 `Self::image_sampling(sampling)` |
| `tools/golden/render.py` | 去掉顶层 `_seed`，统一在 worker 内 `seed()`；保留 networkx 的 `spring_layout`/`random_layout` seed 包装 |

合并后全仓库检查，无 `<<<<<<<` / `>>>>>>>` 冲突标记。

**没有丢掉 master 侧的 `c400cc5` TeX 修复。** `tex_mobject.py`、`typst_mobject.py` 里已经没有 `should_subdivide_sharp_curves` / `should_remove_null_curves`；完整 PATH 下 `Tex('x')`、`NumberPlane()` 都能构造，不再是 `TypeError`。

### 1.3 原生扩展

bundle 改了 `native/manim_vk/src/lib.rs`，必须重新编译：

```powershell
$env:PATH = '%USERPROFILE%\.cargo\bin;' + $env:PATH
$env:SKIA_BINARIES_URL = 'file:///C:/skia/skia-binaries.tar.gz'
$env:LIB = '%USERPROFILE%\miniconda3\libs;' + $env:LIB
uv run --with maturin maturin build --release
uv pip install --python .venv\Scripts\python.exe --no-deps --reinstall `
  native\manim_vk\target\wheels\manim_vk-0.1.0-cp311-abi3-win_amd64.whl
```

本机结果：`Finished release profile in 23.51s`，wheel 安装成功，`from manim import MeshSurface, TexturedSurface` 可用。

> 这一步不是“可选”。`lib.rs` 的函数签名和 bundle 的 Python 调用是配套的；只拉 Python 代码、不重编译，会在 `set_source_image` / `draw_textured_mesh` 处报缺失符号。

### 1.4 合并提交

```
d50f33b fix: rasterize ImageMobject with PIL perspective like Cairo
1c80139 Merge bundle vulkan-strip: depth-sorted 3D, MeshSurface/TexturedSurface
5ee0f2c docs: finalize golden and framework report
```

`master` 已 fast-forward 到 `d50f33b` 并推送：

```
bf06d86..d50f33b  master -> master
```

---

## 2. 继续修：图像对象的最后 2 个 FAIL

### 2.1 问题

`ImageFromArray` 和 `MovingZoomedSceneAround` 使用同一张 2×4 灰度图放大到约 840×420。外接矩形完全一致，但 Vulkan 与 Cairo 的采样相位不同：

```
coarse 4.26 / RGB 4.24 / PSNR 31.8 / changed 49.7%
```

bundle 的 `image_sampling()` 已按 `resampling_algorithm` 选 Catmull-Rom cubic，但仍不是 PIL 的 BICUBIC 透视变换，所以 FAIL 没消失。

### 2.2 修复

在 `manim/renderer/skia/rendering.py` 中，普通 `ImageMobject` 不再直接走 GPU 采样，而是：

1. 用与 Cairo 完全相同的 PIL `Image.transform(PERSPECTIVE, BICUBIC)` 做透视栅格化；
2. 得到全帧 RGBA；
3. 以 1:1 设备坐标上传，GPU 只做一次无重采样的 SrcOver 合成。

`ImageMobjectFromCamera` 仍走 GPU `draw_target_quad`，不把相机画面拉回 CPU。

### 2.3 结果

| 场景 | 修复前 | 修复后 |
|---|---|---|
| `ImageFromArray` | FAIL，RGB 4.24 | **PASS**，RGB 0.00117，PSNR 64.8 |
| `MovingZoomedSceneAround` | FAIL，RGB 4.24 | **PASS**，RGB 0.00117，PSNR 64.8 |

30 个高风险场景总账从 `PASS 16 / FAIL 2 / REVIEW 12` 变为：

```
PASS 20 / FAIL 0 / REVIEW 10 / ERROR_CANDIDATE 0 / ERROR_BOTH 0
```

---

## 3. 合并后 30 个高风险场景

参考：`golden_after\cairo`  
候选：`golden_bundle\vulkan`  
报告：`golden_bundle\report\report.md`

| Verdict | 数量 |
|---|---:|
| PASS | 20 |
| FAIL | 0 |
| REVIEW | 10 |
| ERROR_CANDIDATE | 0 |
| ERROR_BOTH | 0 |

10 个 REVIEW：

| 场景 | coarse | RGB | changed | 主要差异 |
|---|---:|---:|---:|---|
| `docs_source_examples__OpeningManim` | 0.39 | 1.00 | 7.8% | 文本/线条 AA |
| `example_scenes__basic__OpeningManim` | 0.39 | 1.00 | 7.8% | 同上，重复场景 |
| `ThreeDSurfacePlot` | 0.70 | 1.10 | 13.2% | 3D 表面边缘 |
| `CoordSysExample` | 0.43 | 1.07 | 8.1% | 坐标网格 AA |
| `ConvexHull3DExample` | 1.17 | 1.24 | 9.6% | 3D 面/边光照 |
| `BasicUsage_2` | 2.22 | 5.58 | 37.1% | 背景图逐像素遮罩 approximation |
| `SpawningAndFlowingArea` | 1.42 | 3.18 | 22.0% | 同上 |
| `StreamLineCreation` | 0.55 | 2.28 | 12.5% | 流线颜色/透明度 |
| `LinearTransformationSceneExample` | 0.18 | 0.82 | 10.3% | 线条 AA |
| `RateFuncExample` | 0.37 | 1.11 | 6.4% | 文本/线条 AA |

`BasicUsage_2`、`SpawningAndFlowingArea` 已比 bundle 合并前明显改善（RGB 6.34→5.58、4.48→3.18），但 Skia 目前仍用“中心采样单色”近似 Cairo 的逐像素 alpha 遮罩。这是已知的语义近似，不是几何错误。

**Cairo 删除门槛：** `FAIL=0`、`ERROR_CANDIDATE=0` 已满足；REVIEW 人工确认已逐张看过，未发现对象缺失或位置错乱。按门槛可以进入“删 Cairo”阶段，但这是独立的大改，建议单独提交，不混在本轮 bundle 合并里。

---

## 4. `hairy_ball.py` 实跑

脚本：`<attachment-dir>\hairy_ball.py`  
输出：`<repo>\renders\hairy_ball\videos\hairy_ball\480p15`

| 场景 | 状态 | 输出 | 预览 |
|---|---|---|---|
| `SingleNullPointHairyBall` | ✅ 480p15，2 animations | `SingleNullPointHairyBall.mp4` | `renders\hairy_ball\hairy_frame_5s.png` |
| `SingleNullPointHairyBallRevealed` | ✅ 480p15，2 animations | `SingleNullPointHairyBallRevealed.mp4` | `renders\hairy_ball\revealed_frame_8s.png` |
| `PairsOfNullPoints` | ✅ 480p15，11 animations | `PairsOfNullPoints.mp4` | `renders\hairy_ball\pairs_frame_8s.png` |
| `HairyEarth` | ✅ 480p15，14 animations，78s | `HairyEarth.mp4` | `renders\hairy_ball\earth_frame_36s.png`、`earth_frame_45s.png`、`earth_frame_60s.png` |

### 4.1 地球贴图

`HairyEarth` 需要两张等距圆柱投影图。已放在脚本同目录：

```
EarthTextureMap.jpg       5400×2700   NASA Blue Marble
NightEarthTextureMap.jpg  3600×1800   NASA Black Marble 2012
```

这两张图是 NASA 公有领域素材。`TexturedSurface` 通过 `config.assets_dir` 和 `ImageMobject` 的扩展名搜索找到它们；如果换机器，把两张图放在脚本旁边或 `assets/` 下即可。

### 4.2 画面

- 5 万根毛的球：蓝色毛发从北极秃点向外梳开，`???` 黑盖在北极，随后转开。
- `PairsOfNullPoints`：2000 个采样点的球面箭头场、坐标轴、两阶段流线；蓝色箭头朝外，球背面被深度测试正确遮挡。
- `HairyEarth`：地球日夜贴图 + 5000 条流线 + 蓝白矢量场；低清预览中地球轮廓、日夜纹理和流线都可见。

### 4.3 已知差异

- 半透明球会挡住它后面所有开了深度的对象。`PairsOfNullPoints` 里坐标轴藏在球内的那一段会被球挡掉；ManimGL 只挡“后加入场景”的对象。这是 bundle 文档里写明的语义差异，不是崩溃。
- `HairyEarth` 的球面光照按顶点计算（101×51 网格），终结线附近有少量块状过渡；1080p60 正式片尚未跑。
- 脚本是 480p15 低清验证；正式出片用 `-pqh` 或 `-p1080p60`。

---

## 5. 测试与构建

### 5.1 测试

| 范围 | 结果 |
|---|---|
| `tests\module` 全量 | **1059 passed, 1 skipped**，218s |
| `tests\module\renderer` | 13 passed, 52s |
| `tests\module\mobject\test_image.py` + `tests\test_vulkan\test_vulkan_baseline.py` | 7 passed, 49s |

bundle 新增的 `tests\module\renderer\test_depth_pieces.py` 12 项全部通过。

### 5.2 复现命令

```powershell
# 恢复完整 PATH（MiKTeX、Git、Rust 都在这里）
$env:PATH = (Get-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager\Environment').Path + ';' + (Get-ItemProperty 'HKCU:\Environment').Path

# 30 个高风险 golden
python tools\golden\render.py --scenes golden --renderer vulkan --out golden_bundle --only "<30 个 id 用 | 连接>" --jobs 3
python tools\golden\compare.py --reference golden_after\cairo --candidate golden_bundle\vulkan --out golden_bundle\report

# 毛球
manim -ql --renderer vulkan --media_dir renders\hairy_ball hairy_ball.py SingleNullPointHairyBall
manim -ql --renderer vulkan --media_dir renders\hairy_ball hairy_ball.py PairsOfNullPoints
manim -ql --renderer vulkan --media_dir renders\hairy_ball hairy_ball.py HairyEarth
```

---

## 6. 剩余事项

1. **全量 458 场景合并后复跑没有完成**：跑到 31/458 后主动停止；当前确定数据是 30 个受影响场景 20 PASS / 0 FAIL / 10 REVIEW，以及 1059 项模块测试全绿。若要把 458 全部签收，需要再跑一次约 1 小时的完整任务。
2. **10 个 REVIEW 签字**：目前人工看过，均为 AA/透明度/背景遮罩近似；是否接受为产品容差需要你最终拍板。
3. **删 Cairo**：门槛已到 `FAIL=0 / ERROR=0`；建议作为单独提交，保留 golden 对照基线到删除前最后一刻。
4. **1080p60 毛球正式片**：低清已跑通，正式片没有跑；`HairyEarth` 预计 10~15 分钟/场景（本机 480p15 78s 片长约 4 分钟渲染）。
5. **原生重编译**：任何拉取 master 的机器都必须重建 `.pyd`；这是本轮最容易漏的一步。
