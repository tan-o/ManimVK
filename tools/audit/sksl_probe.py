from manim import Scene, SkSLMobject


class SkSLProbe(Scene):
    def construct(self):
        shader = SkSLMobject(
            """
            uniform float2 u_resolution;
            half4 main(float2 fragCoord) {
                float2 uv = fragCoord / u_resolution;
                return half4(uv.x, uv.y, 0.5, 1.0);
            }
            """
        )
        self.add(shader)


class SkSLBadProbe(Scene):
    def construct(self):
        # Missing closing brace: must raise a readable SkSL compile error.
        self.add(SkSLMobject("half4 main(float2 p) { return half4(1.0);"))
