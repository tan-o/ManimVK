# 3b1b 傅里叶场景在 ManimVK 上的渲染性能报告

日期：2026-11-23  
环境：Windows / RTX 3050 Ti Laptop / Python 3.13.2（ManimVK）、Python 3.12.9（原版 ManimCE）  
结论一句话：**这个 100 向量傅里叶场景，480p15 下 Vulkan 比原版 Cairo 快约 11%，1080p60 下反而慢约 12%。提速的真正瓶颈是逐对象 Python 更新和每对象跨语言调用，不是 GPU 填充率。**

---

## 1. 测试对象与来源

- 3b1b 原文件：`3b1b/videos/_2019/diffyq/part2/fourier_series.py`
- 关键类：`FourierCirclesScene`、`FourierOfPiSymbol`（原版默认 `n_vectors=101`）
- 原版是 ManimGL 写法，依赖 `manim_imports_ext`、`CONFIG`、`OldTex`、`ParametricCurve`，不能直接在 ManimCE 上执行。
- 本测试按原版结构移植为 ManimCE 场景，参数：
  - `n_vectors = 100`（按你的要求取 100，不是原版 101）
  - `slow_factor = 0.1`，跑一个完整周期，视频时长 10.000s
  - 100 个旋转向量 + 100 个圆 + 末端 `TracedPath`
  - Pi 字形：`Tex(r"$\pi$")`，高度 6，取第一个有顶点的子对象，居中

移植时有两处等价替换：

1. 系数计算不再调用 1 万次 `point_from_proportion`，改为把 cubic Bezier 段细分后按弧长重采样，再用向量化的 Riemann 和求系数。系数计算实测约 **0.05s**。
2. 3b1b 原版的“路径拖尾淡出”依赖逐点 stroke width；ManimCE 不支持逐点描边宽度，这里改用 `TracedPath` 记录末端轨迹，视觉等价，但不是逐像素复刻。

---

## 2. 渲染耗时

统一设置：`format=mp4`、`disable_caching=True`、`progress_bar=none`，同一脚本、同一台机器、同一时间段串行执行，避免并发干扰。

| 渲染路径 | 分辨率/帧率 | 帧数 | 渲染耗时 | 墙钟耗时 | 单帧 |
|---|---:|---:|---:|---:|---:|
| 原版 ManimCE 0.22.1 Cairo | 1080p60 | 600 | 225.8s | 226.4s | 0.376s |
| ManimVK Cairo | 1080p60 | 600 | 183.1s | 183.7s | 0.305s |
| ManimVK Vulkan | 1080p60 | 600 | 252.8s | 253.1s | 0.421s |
| 原版 ManimCE 0.22.1 Cairo | 480p15 | 150 | 44.0s | 44.6s | 0.293s |
| ManimVK Vulkan | 480p15 | 150 | 38.9s | 39.2s | 0.259s |

### 2.1 无输出 / 无绘制拆分（补充实验）

为了判断“慢在取帧链路”还是“慢在场景更新/绘制”，又跑了四组隔离实验。方法是用 `dry_run=True` 关掉回读与编码，用 monkeypatch 分别关掉 Skia 绘制、Cairo 绘制、Skia target reset。属于工程近似，不是官方 profiler，但足够看量级。

| 配置（1080p60，600 帧） | 渲染耗时 | 说明 |
|---|---:|---|
| Vulkan 完整 | 252.8s | 正常出片 |
| Vulkan，无输出（dry_run） | 233.2s | 关掉回读 + 编码 |
| Vulkan，无输出 + 无绘制 | 219.2s | 再关掉 Skia 路径绘制 |
| Vulkan，无输出 + 无绘制 + 无 target reset | 221.0s | 与上一行基本无差 |
| Cairo，无输出 + 无绘制 | 144.5s | 只剩场景更新与帧循环 |
| Cairo 完整 | 183.1s | 正常出片 |

折算到 1080p60 的构成：

| 组成 | 耗时 | 占比 |
|---|---:|---:|
| 场景更新 / Python 帧循环（Cairo 无绘制基线） | ≈144.5s | 57% |
| Vulkan 路径每帧固定开销（target/清屏/提交一类） | ≈74.7s | 30% |
| Skia 实际绘制（建路径 + 填充描边） | ≈14.0s | 6% |
| 回读 + 编码 + 写文件 | ≈19.6s | 8% |
| 合计 | 252.8s | 100% |

关键结论：

- **回读 + 编码总共只占 19.6s（7.8%）**，不是 1080p 下 Vulkan 落后的主因。
- **实际 Skia 绘制只占 14.0s（5.5%）**，所以“每帧一次 draw_batch”对傅里叶这个场景最多省 14s，不要指望它单独把 252.8s 打到 150s。
- **最大头是 144.5s 的场景更新**，其次是 Vulkan 路径 74.7s 的每帧固定开销；后者在 480p15 只有约 5.1s（34ms/帧），到 1080p60 变成 125ms/帧，明显随像素数增长，但具体归属（清屏、提交、同步、target 生命周期）还需要 0.3 的分段计时确认。
- 因此“主要原因是取帧链路，而不是 Python 更新”这个判断**不成立**；至少要先补 0.3 的分段数据，再决定先修哪一段。

换算：

- 1080p60：Vulkan 比原版 Cairo **慢 11.9%**；比同仓库 Cairo **慢 38.1%**。
- 480p15：Vulkan 比原版 Cairo **快 11.6%**。

说明：

- “原版 ManimCE”指在独立 venv 中安装的 `manim 0.22.1`，不是 ManimVK 的代码；版本和实现差异会影响绝对值。两条 Cairo 曲线相差约 23%（225.8s vs 183.1s），说明这个差异不可忽略，因此**同仓库 Cairo 对比更能隔离渲染器差异**。
- 原版 1080p60 的输出视频在后续 ManimVK Cairo 跑同名路径时被覆盖，只剩耗时数据；ManimVK 的两条视频保留。
- 单次测量，未做多次取中位数；机器为笔记本，长时间渲染可能有温度降频，绝对值有 ±5% 左右波动。

---

## 3. 箭头样式调整

原版参数：`stroke_width=6`、`tip_length=0.35`、`max_tip_length_to_length_ratio=0.35`。  
调整后：`stroke_width=2`、`tip_length=0.15`、`max_tip_length_to_length_ratio=0.2`。

效果：箭杆明显变细，末端 100 个箭头堆叠形成的白色团块从约 0.5 单位缩到约 0.2 单位；轨迹和圆不变。

1080p60 下粗细对总耗时几乎无影响（253.4s → 252.8s），因为瓶颈不在填充率。

---

## 4. 为什么 1080p60 下 Vulkan 反而慢

这个场景每帧要更新 200 个对象：

- 100 个 `Vector`：每个 updater 执行 `set_length`、`set_angle`、`shift`
- 100 个 `Circle`：每个 updater 执行 `set_width`、`move_to`
- 1 个 `TracedPath`：每帧追加点

这些全是 Python 闭包，逐对象执行，和渲染器无关。600 帧合计约 12 万次对象级更新。结合 2.1 的拆分：

- 场景更新本身约 144.5s，占 57%，是最大单项；
- 实际 Skia 绘制约 14.0s，只占 5.5%，所以 480p15 下 Vulkan 能赢，是因为固定开销小、draw 占比相对高；
- 1080p60 下多出约 74.7s 的 Vulkan 每帧固定开销（清屏/提交/同步/target 生命周期一类，具体待 0.3 分段确认），再加 19.6s 回读编码，才把 Vulkan 推到 252.8s；
- 说“1080p 下 Vulkan 慢主要是取帧链路”是不准确的：回读+编码合计只有 19.6s，真正的大头是更新和每帧固定开销。

这和之前 1025 个 Circle 的 480p15 基准一致：那里总耗时 2.93s，draw 0.96s、update 1.13s、read 0.09s、other 0.75s；其中建路径一项就 0.76s。**回读早已不是主导项，逐对象 Python 遍历和每对象 PyO3 调用才是。**

---

## 5. 可复现的脚本与命令

脚本：`<tmp>\mvk_fourier\fourier_pi_100.py`

关键环境变量：

```powershell
$env:Q='high_quality'   # 或 low_quality
$env:R='vulkan'         # 或 cairo
$env:SW='2'             # 箭杆粗细
$env:TIP='0.15'         # 箭头长度
$env:TIPR='0.2'         # 箭头长度/向量长度上限
```

运行：

```powershell
& '<repo>\.venv\Scripts\python.exe' fourier_pi_100.py
# 原版 CE:
& '<tmp>\manimce_venv\Scripts\python.exe' fourier_pi_100.py
```

输出：

- ManimVK Vulkan：`%TEMP%\mvk_fourier\media\vulkan\videos\1080p60\FourierPi100.mp4`
- ManimVK Cairo：`%TEMP%\mvk_fourier\media\cairo\videos\1080p60\FourierPi100.mp4`

---

## 6. 架构结论：要不要“全搬 Rust”

方向对，但不能理解成“Python 只写脚本、其余全部搬走”。Manim 的灵活性来自任意 Python 回调：`add_updater`、lambda、自定义 `Animation`、`rate_func`。这些无法整体 Rust 化，否则 API 兼容性直接崩。

合理的拆法是三层：

| 层 | 现在 | 建议 |
|---|---|---|
| 场景描述 / Mobject 树 / 布局 / TeX | Python | 留在 Python |
| 逐帧动画求值 | Python 对象遍历 | 标准动画编译成时间线，Rust 执行；自定义 updater 保留 Python 逃生口 |
| 路径构建 / 填充描边 / 合成 / 提交 / 回读 / 编码 | 每对象十几次 PyO3 | Rust 批处理，一次提交 |

### 6.1 每帧一次批处理（优先做）

Python 每帧把点、偏移、颜色、线宽、样式拼成扁平数组，只调一次 PyO3：

```text
draw_batch(
  points      float32[M, 2]
  offsets     uint32[N+1]
  sub_offsets uint32[K+1]
  colors      uint8[N, 4]
  widths      float32[N]
  joins/caps  uint8[N]
  fill/stroke uint8[N]
  matrix      float32[6]
)
```

Rust 在一次调用里循环建 `SkPath` 并绘制。现在 1025 对象每帧约 2 万次跨语言调用，批处理可压到 1 次；建路径 0.76s 有望降到 0.2–0.3s，总耗时约 2.9s → 2.4s。

注意：拼数组要用 numpy 预分配、复用缓冲区；如果仍在 Python 里逐对象填 list，只是把 PyO3 开销换成 Python 循环开销。

### 6.2 整片预烘（不推荐作为通用架构）

Python 先算完 600 帧，写成大缓冲，Rust 一次渲染完。可行，但：

- 内存随“帧数 × 对象数 × 点数”增长，这个场景几十 MB，复杂场景可到 GB；
- Python updater 仍要逐帧跑，时间不会消失，只是挪到预烘阶段；
- 失去预览、局部重渲、缓存复用。

### 6.3 编译成时间线（更合适）

Python 只跑一次，输出静态几何 + 每个对象的动画通道关键帧 + 相机关键帧；Rust 按帧号插值、建路径、绘制。标准动画完全不需要 Python 每帧参与，内存小、可 seek，自定义 updater 标记为 `dynamic` 回退 Python。

### 6.4 批处理解决不了的两件事

1. **逐对象 Python updater**：这个傅里叶场景 200 个对象全是闭包更新，绘制批处理救不了，必须改批量更新接口或声明式动画。
2. **整帧回读/编码**：1080p60 下 Vulkan 慢的主因之一，批处理绘制不会减少回读；要压这块得考虑零拷贝或直接把帧送编码器。

---

## 7. 建议的执行顺序

1. Rust 批量路径构建 + 一次提交（收益最大、接口清晰）。
2. 子路径切分缓存、颜色插值提前判等、跨对象批量变换（纯 Python/numpy，风险低）。
3. 标准动画编译成时间线，Rust 执行；自定义 updater 保留 Python 回调。
4. 最后再考虑动画调度整体数据化——那接近重写 Manim 运行时，不属于“渲染器迁移”。

---

## 8. 附：本场景的判定

- **如果目标是“常见 1080p60 成片”**：当前 Vulkan 路径不占优，Cairo 更快；继续迁移的收益主要在 480p 或更小的预览/草稿渲染。
- **如果目标是“大量对象的稠密场景”**：先做 draw_batch 和逐对象调用压缩，否则换 Vulkan 只是把 Python 瓶颈换成 GPU 提交开销。
- **如果目标是“交互式预览”**：Vulkan 的价值不在 mp4 成片，而在窗口呈现；但当前预览窗口仍是独立 wgpu 实例、没有输入事件，这部分还没打通。
