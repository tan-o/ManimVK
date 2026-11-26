"""山间小芽：一段约 13 秒的扁平风格短片（原创角色）。

层叠的远山、流云和薄雾，竹林在风里摇；一只茶芽小精灵从草丛里冒出来，
追着一片落叶跳到石头上接住它，天色转成黄昏，鸟群飞过。

整段动画由一个时间轴（ValueTracker）驱动，每个物体的形状和位置都是
时间的函数，所以任意一帧都可以单独求值。

    manim -qh spirit_valley.py SpiritValley
"""

from __future__ import annotations

import numpy as np

from manim import (
    DOWN,
    LEFT,
    ORIGIN,
    RIGHT,
    UP,
    Circle,
    Ellipse,
    ManimColor,
    Scene,
    Rectangle,
    ValueTracker,
    VGroup,
    VMobject,
    config,
    interpolate_color,
    linear,
)

DURATION = 13.0
SCALE = 1.35  # 小芽的大小

# 配色：清透的白天，暖色的黄昏。
SKY_DAY = ("#5FAED6", "#CFE8EE", "#F6EDD3")
SKY_DUSK = ("#4A5E9C", "#C98BA0", "#F6B47E")
FAR = ("#A9C2CF", "#D3E0E3")
MID = ("#6F9F92", "#A9C8B3")
NEAR = ("#3E7A5C", "#78AE7E")
GROUND = ("#4F8F4F", "#86B86C")
BODY = "#FFF6E2"
INK = "#2F3437"
SPROUT = "#7BC46A"
SPROUT_DARK = "#4E9A49"
CHEEK = "#F4A3A0"
LEAF_COLORS = ("#E9A43B", "#DB6A3C", "#9CC85C", "#F0C24E")


# ---------------------------------------------------------------------------
# 时间工具


def smooth(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def ramp(t: float, start: float, end: float) -> float:
    """0 before ``start``, 1 after ``end``, smooth in between."""
    return smooth((t - start) / (end - start))


def keyframes(t: float, frames: list[tuple[float, float]]) -> float:
    """Smoothly interpolate ``(time, value)`` pairs."""
    if t <= frames[0][0]:
        return frames[0][1]
    for (t0, v0), (t1, v1) in zip(frames, frames[1:]):
        if t <= t1:
            return v0 + (v1 - v0) * smooth((t - t0) / (t1 - t0))
    return frames[-1][1]


def gradient_colors(colors: tuple[str, ...], dusk: tuple[str, ...], amount: float):
    return [
        interpolate_color(ManimColor(a), ManimColor(b), amount)
        for a, b in zip(colors, dusk)
    ]


# ---------------------------------------------------------------------------
# 形状


def ridge(x_min: float, x_max: float, base: float, height: float, seed: int,
          bottom: float = -6.0) -> VMobject:
    """A mountain ridge: a smooth top line over a flat bottom."""
    rng = np.random.default_rng(seed)
    xs = np.linspace(x_min, x_max, 90)
    ys = np.full_like(xs, base)
    for k in range(1, 5):
        ys += height / k * np.sin(xs * (0.35 * k + rng.uniform(0, 0.2)) + rng.uniform(0, 6))
    top = np.column_stack([xs, ys, np.zeros_like(xs)])
    shape = VMobject(stroke_width=0)
    shape.set_points_smoothly(top)
    shape.add_line_to(np.array([x_max, bottom, 0]))
    shape.add_line_to(np.array([x_min, bottom, 0]))
    shape.add_line_to(top[0])
    return shape


def blob(center, radii: list[tuple[float, float, float]], color, opacity) -> VGroup:
    """A cloud or a mist bank: overlapping ellipses of one colour."""
    group = VGroup(
        *[
            Ellipse(width=2 * w, height=2 * h, stroke_width=0, fill_color=color, fill_opacity=opacity)
            .move_to(np.array(center) + np.array([dx, 0, 0]) + np.array([0, h * 0.4, 0]))
            for dx, w, h in radii
        ]
    )
    return group


def leaf_outline(length: float = 0.32, width: float = 0.16) -> np.ndarray:
    """Cubic Bezier points of a pointed leaf along +x from the origin, for
    ``set_points``: two curves, upper and lower edge."""
    tip = np.array([length, 0, 0])
    upper = [ORIGIN, np.array([0.25 * length, 1.2 * width, 0]), np.array([0.7 * length, 1.0 * width, 0]), tip]
    lower = [tip, np.array([0.7 * length, -1.0 * width, 0]), np.array([0.25 * length, -1.2 * width, 0]), ORIGIN]
    return np.array(upper + lower, dtype=float)


def rotation(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


class Sprite(VGroup):
    """小芽：奶白色的蛋形身体，头顶一对嫩芽。

    Built once around the origin with its feet at y = 0; ``pose`` places a
    fresh copy of the rest shape every frame, so squash and stretch never
    accumulate.
    """

    def __init__(self) -> None:
        super().__init__()
        body = Ellipse(width=1.0, height=0.9, fill_color=BODY, fill_opacity=1,
                       stroke_color=INK, stroke_width=3).move_to(UP * 0.45)
        belly = Ellipse(width=0.55, height=0.35, fill_color="#FFFDF6", fill_opacity=1,
                        stroke_width=0).move_to(UP * 0.28)
        feet = VGroup(*[
            Ellipse(width=0.24, height=0.12, fill_color=BODY, fill_opacity=1,
                    stroke_color=INK, stroke_width=2.5).move_to(np.array([x, 0.04, 0]))
            for x in (-0.2, 0.2)
        ])
        stem = VMobject(stroke_color=SPROUT_DARK, stroke_width=4)
        stem.set_points_as_corners([UP * 0.88, UP * 1.05])
        leaves = VGroup()
        for side in (-1, 1):
            leaf = VMobject(fill_color=SPROUT, fill_opacity=1, stroke_color=SPROUT_DARK, stroke_width=2)
            points = leaf_outline(0.34, 0.13) @ rotation(np.pi / 2 - side * 0.9).T
            leaf.set_points(points + UP * 1.03)
            leaves.add(leaf)
        self.eyes = VGroup(*[
            Ellipse(width=0.09, height=0.13, fill_color=INK, fill_opacity=1, stroke_width=0)
            .move_to(np.array([x, 0.52, 0]))
            for x in (-0.17, 0.17)
        ])
        self.sparkles = VGroup(*[
            Circle(radius=0.018, fill_color=WHITE_HEX, fill_opacity=1, stroke_width=0)
            .move_to(np.array([x + 0.015, 0.55, 0]))
            for x in (-0.17, 0.17)
        ])
        self.cheeks = VGroup(*[
            Ellipse(width=0.16, height=0.08, fill_color=CHEEK, fill_opacity=0.75, stroke_width=0)
            .move_to(np.array([x, 0.4, 0]))
            for x in (-0.3, 0.3)
        ])
        self.happy = VGroup()
        for x in (-0.17, 0.17):
            arc = VMobject(stroke_color=INK, stroke_width=3)
            arc.set_points_smoothly([np.array([x - 0.06, 0.5, 0]), np.array([x, 0.56, 0]), np.array([x + 0.06, 0.5, 0])])
            self.happy.add(arc)
        self.mouth = VMobject(stroke_color=INK, stroke_width=2.5)
        self.mouth.set_points_smoothly([np.array([-0.05, 0.43, 0]), np.array([0, 0.40, 0]), np.array([0.05, 0.43, 0])])
        self.body_parts = VGroup(feet, body, belly)
        self.sprout = VGroup(stem, leaves)
        self.add(self.body_parts, self.sprout, self.cheeks, self.mouth, self.eyes, self.sparkles, self.happy)


WHITE_HEX = "#FFFFFF"


# ---------------------------------------------------------------------------
# 场景


class SpiritValley(Scene):
    def construct(self) -> None:
        clock = ValueTracker(0.0)
        frame = self.camera.frame

        # 天空：竖直渐变，覆盖镜头能走到的范围。
        sky = Rectangle(width=40, height=14, stroke_width=0).move_to(UP * 1.5)
        sky.set_sheen_direction(DOWN)

        def paint_sky(mob):
            # The camera frame is not a scene mobject; the sky, updated
            # first, moves it before anything places itself against it.
            move_camera(frame)
            dusk = ramp(clock.get_value(), 9.5, 12.5)
            mob.set_fill(gradient_colors(SKY_DAY, SKY_DUSK, dusk), opacity=1)

        sky.add_updater(paint_sky)

        # 太阳和光晕：午后偏高，黄昏时沉向山后。
        sun = VGroup(*[
            Circle(radius=r, stroke_width=0, fill_color="#FFF3C4", fill_opacity=o)
            for r, o in ((1.6, 0.10), (1.1, 0.16), (0.75, 0.3), (0.5, 1.0))
        ])

        def place_sun(mob):
            t = clock.get_value()
            dusk = ramp(t, 9.0, 13.0)
            mob.move_to(np.array([3.6 + 0.4 * dusk, 2.6 - 2.1 * dusk, 0]) + frame.get_center() * np.array([0.9, 0.9, 0]))
            mob[-1].set_fill(interpolate_color(ManimColor("#FFF3C4"), ManimColor("#FFC27A"), dusk))

        sun.add_updater(place_sun)

        # 三层山：越远视差越小，黄昏时一起偏暖偏暗。
        layers = []
        for colors, base, height, seed, depth, x_span in (
            (FAR, 0.2, 0.55, 3, 0.85, (-16, 20)),
            (MID, -0.9, 0.45, 7, 0.6, (-16, 20)),
            (NEAR, -2.0, 0.35, 11, 0.35, (-16, 20)),
        ):
            layer = ridge(*x_span, base, height, seed)
            layer.set_sheen_direction(DOWN)
            layer.colors = colors
            layer.depth = depth
            layer.home = layer.get_center().copy()
            layers.append(layer)

        def shade_layer(mob):
            dusk = ramp(clock.get_value(), 9.5, 12.5)
            warm = [interpolate_color(ManimColor(c), ManimColor("#5A4C78"), 0.55 * dusk * (1.25 - mob.depth)) for c in mob.colors]
            mob.set_fill(warm, opacity=1)
            mob.move_to(mob.home + frame.get_center() * np.array([mob.depth, mob.depth * 0.6, 0]))

        for layer in layers:
            layer.add_updater(shade_layer)

        # 云和雾：各自慢慢漂。
        clouds = VGroup(
            blob((-6, 2.9, 0), [(-0.6, 0.7, 0.35), (0.1, 0.9, 0.5), (0.9, 0.6, 0.32)], WHITE_HEX, 0.9),
            blob((1.5, 3.3, 0), [(-0.5, 0.6, 0.28), (0.2, 0.8, 0.42), (0.9, 0.5, 0.25)], WHITE_HEX, 0.85),
            blob((7.5, 2.6, 0), [(-0.4, 0.7, 0.3), (0.4, 0.9, 0.45)], WHITE_HEX, 0.85),
        )
        for index, cloud in enumerate(clouds):
            cloud.home = cloud.get_center().copy()
            cloud.speed = 0.12 + 0.05 * index

        def drift_clouds(group):
            t = clock.get_value()
            dusk = ramp(t, 9.5, 12.5)
            for cloud in group:
                cloud.move_to(cloud.home + RIGHT * cloud.speed * t + frame.get_center() * np.array([0.9, 0.9, 0]))
                cloud.set_fill(interpolate_color(ManimColor(WHITE_HEX), ManimColor("#FAD2B8"), dusk))

        clouds.add_updater(drift_clouds)

        mists = VGroup(
            blob((-3, -0.6, 0), [(-1.5, 2.5, 0.25), (1.0, 2.8, 0.3)], WHITE_HEX, 0.35),
            blob((5, -1.4, 0), [(-1.0, 2.6, 0.22), (1.5, 2.2, 0.26)], WHITE_HEX, 0.3),
        )
        for mist in mists:
            mist.home = mist.get_center().copy()

        def drift_mist(group):
            t = clock.get_value()
            for index, mist in enumerate(group):
                mist.move_to(mist.home + RIGHT * (0.25 * np.sin(0.4 * t + index)) + frame.get_center() * np.array([0.5, 0.3, 0]))

        mists.add_updater(drift_mist)

        # 前景的小坡、石头和草。
        ground = ridge(-16, 20, -2.75, 0.12, 21, bottom=-7)
        ground.set_sheen_direction(DOWN)

        def shade_ground(mob):
            dusk = ramp(clock.get_value(), 9.5, 12.5)
            mob.set_fill([interpolate_color(ManimColor(c), ManimColor("#8C6A5A"), 0.3 * dusk) for c in GROUND], opacity=1)

        ground.add_updater(shade_ground)

        rock = VMobject(fill_color="#8E9C9A", fill_opacity=1, stroke_color=INK, stroke_width=3)
        rock.set_points_smoothly([
            np.array(p) for p in [(2.0, -2.75, 0), (2.15, -2.2, 0), (2.7, -1.95, 0), (3.4, -2.05, 0),
                                  (3.75, -2.5, 0), (3.8, -2.75, 0), (2.0, -2.75, 0)]
        ])
        rock_shine = VMobject(stroke_color="#C7D2CF", stroke_width=4)
        rock_shine.set_points_smoothly([np.array([2.5, -2.15, 0]), np.array([2.9, -2.02, 0]), np.array([3.2, -2.06, 0])])
        rock_top = 2.0 + 0.0  # 小芽落在石头上时脚底的高度（见 pose）

        grass_roots = np.random.default_rng(5).uniform(-9, 9, 140)
        grass_heights = np.random.default_rng(6).uniform(0.25, 0.55, 140)
        grass = VMobject(stroke_color="#3F7D45", stroke_width=3.2, fill_opacity=0)

        def wind(t: float) -> float:
            """Gust strength: a breeze, then a gust from 5.8 s to 9 s."""
            return 0.25 + 0.75 * (ramp(t, 5.8, 6.6) - ramp(t, 8.2, 9.4))

        def sway_grass(mob):
            t = clock.get_value()
            gust = wind(t)
            points = []
            for x, h in zip(grass_roots, grass_heights):
                bend = (0.08 + 0.25 * gust) * np.sin(2.2 * t + 1.7 * x) + 0.18 * gust
                base = np.array([x, -2.78, 0])
                tip = base + np.array([bend * h * 1.4, h, 0])
                mid = base + np.array([0.15 * bend * h, 0.55 * h, 0])
                points.append(np.stack([base, base + (mid - base) * 0.8, mid + (tip - mid) * 0.4, tip]))
            mob.set_points(np.concatenate(points))

        grass.add_updater(sway_grass)

        # 竹子：一节一节，绕根部随风摆。
        bamboo_back, bamboo_front = VGroup(), VGroup()
        for index, (x, height, phase) in enumerate( ((-5.6, 7.5, 0.0), (-4.9, 6.2, 1.3), (5.8, 7.0, 2.1), (6.6, 5.6, 0.7))):
            stalk = VGroup()
            for k in range(int(height / 0.9)):
                stalk.add(
                    Rectangle(width=0.22, height=0.84, stroke_color="#2F6B45", stroke_width=2,
                              fill_color="#5FA36A", fill_opacity=1)
                    .move_to(np.array([x, -2.8 + 0.45 + 0.9 * k, 0]))
                )
            for k, side in ((3, 1), (5, -1), (6, 1)):
                if k * 0.9 < height:
                    leaf = VMobject(fill_color="#4E9A55", fill_opacity=1, stroke_width=0)
                    leaf.set_points(leaf_outline(0.9, 0.16) @ rotation(0.35 * side + (0 if side > 0 else np.pi)).T
                                    + np.array([x, -2.8 + 0.9 * k + 0.1, 0]))
                    stalk.add(leaf)
            stalk.root = np.array([x, -2.8, 0])
            stalk.phase = phase
            stalk.rest = stalk.copy()
            (bamboo_back if index % 2 == 0 else bamboo_front).add(stalk)

        def sway_bamboo(group):
            t = clock.get_value()
            gust = wind(t)
            for stalk in group:
                angle = -(0.015 + 0.05 * gust) * np.sin(1.3 * t + stalk.phase) - 0.05 * gust
                for part, rest in zip(stalk, stalk.rest):
                    part.set_points((rest.points - stalk.root) @ rotation(angle).T + stalk.root)

        bamboo_back.add_updater(sway_bamboo)
        bamboo_front.add_updater(sway_bamboo)

        # 落叶：风起时从竹梢飘下，那片橙色的最后落在小芽头上。
        rng = np.random.default_rng(9)
        falling = VGroup()
        for index in range(14):
            leaf = VMobject(fill_color=LEAF_COLORS[index % 4], fill_opacity=1, stroke_width=0)
            leaf.outline = leaf_outline(0.46, 0.2) - np.array([0.23, 0, 0])
            leaf.set_points(leaf.outline)
            leaf.start = 6.0 + rng.uniform(0, 3.2)
            leaf.origin = np.array([rng.uniform(-6.5, 7.0), 4.6, 0])
            leaf.fall = rng.uniform(0.9, 1.4)
            leaf.flutter = rng.uniform(0.5, 1.0)
            leaf.spin = rng.uniform(1.5, 3.0) * rng.choice([-1, 1])
            falling.add(leaf)
        hero_leaf = falling[0]
        hero_leaf.set_fill(LEAF_COLORS[1])
        hero_leaf.start = 6.2

        # 小芽。
        sprite = Sprite()
        rest = sprite.copy()

        def sprite_pose(t: float):
            """Feet position, squash (x, y scale), eye glance and the face."""
            # 冒出来：3.0–4.2 s 从草里升起，带一点回弹。
            rise = keyframes(t, [(3.0, -1.2), (3.8, 0.12), (4.2, 0.0)])
            x = keyframes(t, [(6.9, -1.2), (7.6, 0.2), (7.8, 0.2), (8.6, 2.8)])
            ground_y = -2.78
            # 两次跳跃：6.9–7.6、7.8–8.6，第二跳落到石头上。
            y = ground_y + rise
            hop = 0.0
            if 6.9 <= t <= 7.6:
                s = (t - 6.9) / 0.7
                hop = 1.1 * 4 * s * (1 - s)
            elif 7.8 <= t <= 8.6:
                s = (t - 7.8) / 0.8
                landing = -1.98 - ground_y
                hop = 1.6 * 4 * s * (1 - s) + landing * s
            elif t > 8.6:
                hop = -1.98 - ground_y
                for bounce in (10.2, 10.75):
                    if bounce <= t <= bounce + 0.4:
                        s = (t - bounce) / 0.4
                        hop += 0.35 * 4 * s * (1 - s)
            y += hop
            # 挤压与拉伸：起跳前压扁，空中拉长，落地再压扁。
            squash = 1.0
            for t0, t1 in ((6.9, 7.6), (7.8, 8.6)):
                squash *= 1 - 0.22 * np.exp(-((t - (t0 - 0.08)) / 0.08) ** 2)  # 蓄力
                squash *= 1 + 0.18 * np.exp(-((t - (t0 + 0.2)) / 0.12) ** 2)  # 腾空
                squash *= 1 - 0.25 * np.exp(-((t - (t1 + 0.05)) / 0.09) ** 2)  # 落地
            squash *= 1 - 0.2 * np.exp(-((t - 4.2) / 0.1) ** 2) + 0.12 * np.exp(-((t - 3.7) / 0.12) ** 2)
            # 接住叶子后开心地蹦两下。
            for bounce in (10.2, 10.75):
                squash *= 1 - 0.16 * np.exp(-((t - bounce) / 0.07) ** 2)
                squash *= 1 + 0.12 * np.exp(-((t - bounce - 0.18) / 0.1) ** 2)
            # 呼吸。
            squash *= 1 + 0.025 * np.sin(2 * np.pi * t / 1.6)
            glance = keyframes(t, [(4.4, 0.0), (4.8, -0.05), (5.4, -0.05), (5.7, 0.06), (6.3, 0.06), (6.6, 0.03)])
            blink = 1.0 - (np.exp(-((t - 5.0) / 0.06) ** 2) + np.exp(-((t - 9.9) / 0.06) ** 2))
            happy = ramp(t, 10.0, 10.2)
            return np.array([x, y, 0]), 1 / np.sqrt(squash), squash, glance, max(blink, 0.05), happy

        def place_sprite(mob):
            t = clock.get_value()
            feet, sx, sy, glance, blink, happy = sprite_pose(t)
            mob.become(rest.copy())
            mob.eyes.stretch(blink, 1).shift(RIGHT * glance)
            mob.sparkles.shift(RIGHT * glance).set_opacity(1 if blink > 0.5 and happy < 0.5 else 0)
            mob.eyes.set_opacity(1 - happy)
            mob.happy.set_stroke(opacity=happy)
            # 嫩芽随风和动作轻轻晃。
            wobble = 0.12 * np.sin(5 * t) * wind(t) + 0.25 * (sy - 1)
            mob.sprout.rotate(wobble, about_point=UP * 0.88)
            mob.stretch(sx, 0, about_point=ORIGIN).stretch(sy, 1, about_point=ORIGIN)
            mob.scale(SCALE, about_point=ORIGIN).shift(feet)
            # 冒出来时在地面之后（地面挡住还没升起的部分），站稳后到前面来。
            mob.set_z_index(7.5 if t < 4.15 else 12.5)

        sprite.add_updater(place_sprite)

        def drop_leaves(group):
            t = clock.get_value()
            catch = 9.7
            for leaf in group:
                age = t - leaf.start
                if age < 0:
                    leaf.set_opacity(0)
                    continue
                leaf.set_opacity(1)
                if leaf is hero_leaf:
                    # 从小芽上方飘下，9.7 s 落在它的嫩芽上。
                    feet = sprite_pose(min(t, catch))[0]
                    top = feet + UP * 1.05 * SCALE
                    start = np.array([1.2, 4.4, 0])
                    s = min(age / (catch - leaf.start), 1.0)
                    position = start + (top - start) * s + RIGHT * 0.5 * np.sin(3.4 * age) * (1 - s)
                    angle = 0.6 * np.sin(2.6 * age) * (1 - s) + 0.35 * s
                    if t > catch:
                        position = sprite_pose(t)[0] + UP * 1.05 * SCALE * sprite_pose(t)[2]
                else:
                    position = leaf.origin + np.array(
                        [0.8 * np.sin(leaf.flutter * 3 * age) + 0.9 * age * wind(t), -leaf.fall * age, 0]
                    )
                    angle = leaf.spin * age
                    if position[1] < -2.75:
                        position[1] = -2.75
                        angle = 0.0
                leaf.set_points(leaf.outline @ rotation(angle).T + position)

        falling.add_updater(drop_leaves)

        # 黄昏时飞过的鸟。
        birds = VGroup(*[VMobject(stroke_color=INK, stroke_width=2.5) for _ in range(5)])

        def fly_birds(group):
            t = clock.get_value()
            for index, bird in enumerate(group):
                age = t - (10.3 + 0.18 * index)
                center = np.array([-8 + 2.0 * age - 0.5 * index, 2.6 - 0.25 * index + 0.15 * np.sin(age + index), 0])
                center += frame.get_center() * np.array([0.9, 0.9, 0])
                flap = 0.12 * np.sin(9 * age + index)
                span = 0.18
                bird.set_points_smoothly([
                    center + np.array([-span, flap, 0]),
                    center + np.array([-span / 2, 0.02, 0]),
                    center,
                    center + np.array([span / 2, 0.02, 0]),
                    center + np.array([span, flap, 0]),
                ])
                bird.set_stroke(opacity=1 if age > 0 else 0)

        birds.add_updater(fly_birds)

        # 镜头：从左边的远山慢慢推过来，最后轻轻推近小芽。
        def move_camera(mob):
            t = clock.get_value()
            x = keyframes(t, [(0.0, -5.0), (3.0, -0.8), (6.5, -0.2), (9.0, 1.6), (13.0, 2.7)])
            y = keyframes(t, [(0.0, 0.6), (3.0, -0.6), (9.0, -0.6), (13.0, -0.9)])
            width = keyframes(t, [(0.0, config.frame_width), (3.0, config.frame_width * 0.85),
                                  (9.0, config.frame_width * 0.85), (13.0, config.frame_width * 0.62)])
            mob.move_to(np.array([x, y, 0]))
            mob.set(width=width)


        order = [sky, sun, *layers, birds, clouds, mists, bamboo_back, sprite, ground, grass, rock,
                 rock_shine, falling, bamboo_front]
        for depth, mob in enumerate(order):
            mob.set_z_index(depth)
        sprite.set_z_index(7.5)
        self.add(*order)
        for mob in (sky, sun, *layers, clouds, mists, ground, grass, bamboo_back, bamboo_front, sprite,
                    falling, birds):
            for updater in mob.get_updaters():
                updater(mob)
        self.play(clock.animate.set_value(DURATION), run_time=DURATION, rate_func=linear)
