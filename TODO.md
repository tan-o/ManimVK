# ManimVK 待办清单

本文件只列“还没完成”的事项；已完成的历史记录见根目录各份 `MANIMVK_*_REPORT.md` 和提交记录。

验收方式：每次推到 `rust-core-preview`，GitHub Actions（lavapipe 上的真实 Skia/Vulkan）跑完整 `tests/`、渲染一段视频、渲染全部示例场景；全绿才快进合并到 `master`。

---

## 功能与素材

- [ ] **把 `hairy_ball.py` 和两张地球贴图纳入仓库**。目前脚本和 `EarthTextureMap` / `NightEarthTextureMap` 仍在仓库外，换机器后 `HairyEarth` 会缺素材。
- [ ] **跑 `HairyEarth` 的 1080p90 正式片**，并记录每段动画的 it/s。
- [ ] **确认半透明球遮挡语义**。`PairsOfNullPoints` 中球内坐标轴会被整段挡掉；ManimGL 只挡后加入场景的对象。
- [ ] **实现交互预览**（swapchain + 输入事件）。目前只有离线渲染；需要在有 GPU 的机器上开发和验收。
- [ ] **`TexTemplateFromCLI`**：示例要求命令行传 `--tex_template`，场景渲染工具还不支持按场景传参（`tools/scenes/known_errors.json` 里唯一的已知错误）。

## 性能

- [ ] **在真机上量深度排序场景的 GPU 侧**。Python 侧已经是每帧一次原生调用（display list）；5 万箭头的稠密场每帧仍有约 10 万次 `drawPath`，Ganesh 能否合批要在 GPU 上看。如果 GPU 侧成为瓶颈，下一步是在同一深度切片内把同一对象的片段合成一条路径。
- [ ] **updater**。场景自己的 updater（例如 `SphereVectorField.update_vectors`）现在是 3D 场景每帧的最大头；ManimCE 语义要求逐个调用，只能在具体 mobject 里向量化或下沉到 `manim_core`。
- [ ] **在真机上验收异步 YUV 回读**，抽查 BT.709 颜色。
- [ ] **90fps 编码调优**。默认 `libx264 crf23`；在真机上比较 `veryfast`、`h264_nvenc` 和 `max-inflight-encoders`。

## 工程化

- [ ] **预编译 wheel**。`manim_core` / `manim_vk` 改接口后每台机器都要重编（`manim/_native.py` 会拒绝加载旧版本）；发布时应提供 wheel。
