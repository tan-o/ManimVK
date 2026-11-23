# ManimVK 待办清单

状态基准：`master` 已合并 3D bundle、图像采样修复和验收报告（`a60b142`）。

本文件只列“还没完成”的事项；已完成的历史记录见根目录各份 `MANIMVK_*_REPORT.md`。

---

## P0：验收收口

- [ ] **全量 458 场景在 bundle 合并后重跑 Vulkan**。上次只跑到 31/458，不能把剩余 427 个算作通过。
- [ ] **处理 4 个旧全量中的示例/收集错误**：
  - `TexTemplateFromCLI`：LaTeX → dvi 失败
  - `CustomTipExample`：收集出的示例不是合法 Python
  - `CustomLayoutExample`：缺 `Any` 导入
  - `SoundExample`：缺 `click.wav`
- [ ] **给 10 个 REVIEW 落基线并签字**。当前 30 个受影响场景为 `PASS 20 / FAIL 0 / REVIEW 10 / ERROR 0`；REVIEW 主要集中在：
  - `BasicUsage_2`
  - `SpawningAndFlowingArea`
  - `StreamLineCreation`
  - `OpeningManim`（两份）
  - `ThreeDSurfacePlot`
  - `CoordSysExample`
  - `ConvexHull3DExample`
  - `LinearTransformationSceneExample`
  - `RateFuncExample`
- [ ] **完整 `tests/` 套件复跑**。目前确定的是 `tests/module`：1059 passed、1 skipped；其余套件尚未全量签收。
- [ ] **决定是否删除 Cairo 参考渲染器**。删除门槛 `FAIL=0`、`ERROR_CANDIDATE=0` 已满足；REVIEW 签字和单独提交还未完成。

## P1：功能与素材

- [ ] **把 `hairy_ball.py` 和两张地球贴图纳入仓库**。目前脚本和 `EarthTextureMap.jpg` / `NightEarthTextureMap.jpg` 仍在附件目录，换机器或清临时目录后 `HairyEarth` 会缺素材。
- [ ] **跑 `HairyEarth` 的 1080p90 正式片**。当前只有 480p15；1080p90 只完成 `SingleNullPointHairyBall`（17 秒、1530 帧）。
- [ ] **确认半透明球遮挡语义**。`PairsOfNullPoints` 中球内坐标轴会被整段挡掉；ManimGL 只挡后加入场景的对象。这是 bundle 已知差异，尚未修。
- [ ] **实现交互预览**（swapchain + 输入事件）。目前只有离线渲染；bundle 文档将其列为下一轮。
- [ ] **明确 Mesh/GLSL 支持范围**。当前只支持 SkSL；GLSL Mesh/Shader 是明确不支持，不是遗漏，但需要在文档中保持说明。

## P2：性能

- [ ] **updater 批量化**。1080p60 傅里叶仍约 `render=28.7s`，其中 Python `update_to_time` 16.9s，占 58.7%。
- [ ] **在真机上验收异步 YUV 回读**。已实现：视频帧走 Skia `asyncRescaleAndReadPixelsYUV420`，GPU 上转 BT.709 I420（每像素 1.5 字节而不是 4），第 N 帧在第 N+1 帧画完、提交前取回，回读和下一帧的 Python 工作重叠；编码器直接吃 YUV，省掉 swscale（沙箱 1080p 实测编码线程 20.4→13.2 ms/帧）。沙箱只有 CPU 模拟器，需要在 GPU 上量 `read_pixels` 的 5.9s 降到多少，并抽查颜色（成片已打 BT.709 标签）。
- [ ] **display list / 原生场景图**。绘制仍按对象逐个提交，`draw≈5.1s`；对象数上千会线性放大。
- [ ] **90fps 编码调优**。目前默认 `libx264 crf23`（输入已是 GPU 转好的 YUV）；沙箱 1080p 扁平画面 `preset=veryfast` 再快约 25%，体积反而略小。需要在真机上实测 `veryfast`、`h264_nvenc` 和 `max-inflight-encoders`，并记录画质/体积取舍。

## P3：工程化

- [ ] **golden 进 CI**。把 30 个高风险场景做成 smoke，保留 PASS/REVIEW 基线，禁止 REVIEW 数量增长。
- [ ] **Cairo/Skia 共享 conformance 层**。背景图逐像素遮罩、AA、采样仍是两套实现；`BasicUsage_2` 等 REVIEW 的根因在这里。
- [ ] **CI 里对 `native/manim_vk` 做 `DOCS_RS=1 cargo clippy`**。不需要 Skia 二进制就能类型检查（见 `native/manim_vk/README.md`），可以挡住编译不过的提交。
- [ ] **原生扩展版本检查**。`lib.rs` 变更后所有机器都必须重编译 `.pyd`；需要版本号校验或预编译 wheel，避免加载旧二进制。
- [ ] **整理未跟踪产物**。`golden*`、`renders` 等测试输出不应进入仓库；需要明确 `.gitignore` 和归档策略。

---

## 建议顺序

1. 先跑全量 458，修 4 个示例错误。
2. 给 10 个 REVIEW 落基线，决定 Cairo 去留。
3. 把毛球脚本和贴图纳入仓库，补 1080p90 正式片。
4. 再做 updater 批量和异步回读；没有稳定 golden 基线前不要大改渲染路径。
