//! Surface meshes (`MeshSurface`, `TexturedSurface`), drawn with Skia's
//! custom meshes (`SkMesh`) and tested against an exact depth image.
//!
//! The shading is ManimGL's: `add_light` is `finalize_color.glsl`, an
//! untextured surface is lit per vertex as in `surface/vert.glsl`, and a
//! textured one per pixel, with the day and night images mixed across the
//! terminator, as in `textured_surface/frag.glsl`.
//!
//! Two depth images (see `depth`) hold, at every pixel, the nearest opaque
//! mesh and the nearest mesh of any opacity.  An opaque mesh is tested
//! against the first, a translucent one against the second, so that it
//! shows only its nearest layer, once, over the opaque meshes behind it; a
//! fragment that fails is not drawn, as a depth buffer would discard it.

use pyo3::buffer::PyBuffer;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use skia_safe::gpu;
use skia_safe::mesh::{self, Attribute, Mesh as SkMesh, MeshSpecification, Varying};
use skia_safe::runtime_effect::ChildPtr;
use skia_safe::{
    BlendMode, Blender, Canvas, Data, FilterMode, Image, MipmapMode, Paint, Rect, SamplingOptions,
    Shader, TileMode,
};

use crate::depth;

/// One image of a mesh: texture key, RGBA pixels, width, height.
pub type TextureSpec<'py> = (u64, Bound<'py, PyAny>, u32, u32);

/// One surface mesh, as four groups:
/// - geometry: float32 projected positions `(n, 2)`, scene points `(n, 3)`,
///   unit normals `(n, 3)`, texture coordinates `(n, 2)` in `[0, 1]`, the
///   screen-linear depth key `(n,)` (nearer is larger; `None` without depth
///   testing) and int32 triangles `(t, 3)`;
/// - images: the light and dark images, either may be `None`;
/// - lighting: light position, camera position, shading
///   `(reflectiveness, gloss, shadow)`;
/// - style: straight RGBA colour (multiplied with the images) and opacity.
pub type MeshSpec<'py> = (
    (
        Bound<'py, PyAny>,
        Bound<'py, PyAny>,
        Bound<'py, PyAny>,
        Bound<'py, PyAny>,
        Option<Bound<'py, PyAny>>,
        Bound<'py, PyAny>,
    ),
    (Option<TextureSpec<'py>>, Option<TextureSpec<'py>>),
    ([f32; 3], [f32; 3], [f32; 3]),
    ([f32; 4], f32),
);

/// Bytes per vertex: position (2), point (3), normal (3), uv (2), depth (1).
const STRIDE: usize = 11 * 4;

const LIGHT: &str = r"
float4 add_light(float4 color, float3 point, float3 unit_normal) {
    if (shading == float3(0.0)) { return color; }
    float reflectiveness = shading.x;
    float gloss = shading.y;
    float shadow = shading.z;
    float4 result = color;
    float3 to_camera = normalize(cameraPosition - point);
    float3 to_light = normalize(lightPosition - point);
    float light_to_normal = dot(to_light, unit_normal);
    float bright_factor = max(light_to_normal, 0.0) * reflectiveness;
    float3 light_reflection = reflect(-to_light, unit_normal);
    float light_to_cam = dot(light_reflection, to_camera);
    float shine = gloss * exp(-3.0 * pow(1.0 - light_to_cam, 2.0));
    bright_factor += shine;
    result.rgb = mix(result.rgb, float3(1.0), bright_factor);
    if (light_to_normal < 0.0) {
        result.rgb = mix(result.rgb, float3(0.0), max(-light_to_normal, 0.0) * shadow);
    }
    return result;
}
";

fn vertex_program() -> String {
    format!(
        r"
uniform float4 toDevice;
uniform float3 lightPosition;
uniform float3 cameraPosition;
uniform float3 shading;
uniform float4 baseColor;
uniform float textures;
{LIGHT}
Varyings main(const Attributes a) {{
    Varyings v;
    v.position = a.xy;
    // Packed: Skia meshes allow six varyings, `position` included.
    v.pointDepth = float4(a.point, a.depth);
    v.normal = a.normal;
    v.coords = float4(a.uv, a.xy * toDevice.xy + toDevice.zw);
    // An untextured surface is lit per vertex, as ManimGL's surface shader.
    v.color = textures < 0.5 ? add_light(baseColor, a.point, normalize(a.normal)) : baseColor;
    return v;
}}
"
    )
}

fn fragment_program() -> String {
    format!(
        r"
uniform shader lightImage;
uniform shader darkImage;
uniform shader depthImage;
uniform float3 lightPosition;
uniform float3 cameraPosition;
uniform float3 shading;
uniform float textures;
uniform float2 lightSize;
uniform float2 darkSize;
uniform float depthTest;
uniform float opacity;
{LIGHT}
float2 main(const Varyings v, out float4 color) {{
    float3 point = v.pointDepth.xyz;
    float depth = v.pointDepth.w;
    float2 uv = v.coords.xy;
    if (depthTest > 0.5) {{
        float3 bytes = floor(depthImage.eval(v.coords.zw).rgb * 255.0 + 0.5);
        float nearest = dot(bytes, float3(65536.0, 256.0, 1.0)) / 16777215.0;
        // Behind the nearest mesh of the image: discarded, as by a depth
        // test.  The slack (1/65536 of the depth range, 256 steps of the
        // 24-bit image) keeps a mesh's own pixels, which Vulkan and Skia
        // rasterise with slightly different rounding.
        if (depth < nearest - 1.0 / 65536.0) {{
            color = float4(0.0);
            return uv;
        }}
    }}
    float4 c = v.color;
    if (textures > 0.5) {{
        float3 n = normalize(v.normal);
        c = unpremul(lightImage.eval(uv * (lightSize - 1.0) + 0.5));
        if (textures > 1.5) {{
            float4 dark = unpremul(darkImage.eval(uv * (darkSize - 1.0) + 0.5));
            float dp = dot(normalize(lightPosition - point), n);
            c = mix(dark, c, smoothstep(-0.2, 0.2, dp));
        }}
        if (c.a == 0.0) {{
            color = float4(0.0);
            return uv;
        }}
        c = add_light(c * v.color, point, n);
    }}
    // ManimGL draws the surface with its opacity, not the image's alpha.
    color = float4(saturate(c.rgb) * opacity, opacity);
    return uv;
}}
"
    )
}

/// The mesh specification, compiled once per target.
pub fn specification() -> PyResult<MeshSpecification> {
    use mesh::attribute::Type as A;
    use mesh::varying::Type as V;
    let attributes = [
        Attribute::new(A::Float2, 0, "xy"),
        Attribute::new(A::Float3, 8, "point"),
        Attribute::new(A::Float3, 20, "normal"),
        Attribute::new(A::Float2, 32, "uv"),
        Attribute::new(A::Float, 40, "depth"),
    ];
    // At most six varyings with `position`: the scene point and depth, the
    // normal, the texture and device coordinates, the vertex colour.
    let varyings = [
        Varying::new(V::Float4, "pointDepth"),
        Varying::new(V::Float3, "normal"),
        Varying::new(V::Float4, "coords"),
        Varying::new(V::Float4, "color"),
    ];
    MeshSpecification::make(
        &attributes,
        STRIDE,
        &varyings,
        vertex_program(),
        fragment_program(),
    )
    .map_err(|error| PyValueError::new_err(format!("mesh shader: {error}")))
}

fn f32_rows(array: &Bound<PyAny>, width: usize, name: &str) -> PyResult<Vec<f32>> {
    let buffer = PyBuffer::<f32>::get(array)
        .map_err(|error| PyValueError::new_err(format!("{name}: {error}")))?;
    let shape = buffer.shape();
    let ok = if width == 1 {
        shape.len() == 1
    } else {
        shape.len() == 2 && shape[1] == width
    };
    if !ok {
        return Err(PyValueError::new_err(format!(
            "{name} must be float32 with {width} columns"
        )));
    }
    buffer
        .to_vec(array.py())
        .map_err(|error| PyValueError::new_err(format!("{name}: {error}")))
}

fn read_image(
    spec: &Option<TextureSpec>,
    texture: &mut impl FnMut(&TextureSpec) -> PyResult<Image>,
) -> PyResult<Option<(Image, [f32; 2])>> {
    spec.as_ref()
        .map(|spec| Ok((texture(spec)?, [spec.2 as f32, spec.3 as f32])))
        .transpose()
}

pub struct Mesh {
    /// Interleaved vertices, [`STRIDE`] bytes each.
    vertices: Vec<f32>,
    /// Projected positions, for bounds and the depth pass.
    xy: Vec<[f32; 2]>,
    /// The screen-linear depth key per vertex, if depth tested.
    keys: Option<Vec<f32>>,
    triangles: Vec<[u32; 3]>,
    light: Option<(Image, [f32; 2])>,
    dark: Option<(Image, [f32; 2])>,
    lighting: ([f32; 3], [f32; 3], [f32; 3]),
    color: [f32; 4],
    opacity: f32,
}

impl Mesh {
    /// Read a mesh from Python; `texture` turns an image into a GPU texture.
    pub fn read(
        spec: &MeshSpec,
        mut texture: impl FnMut(&TextureSpec) -> PyResult<Image>,
    ) -> PyResult<Self> {
        let ((xy, point, normal, uv, keys, triangles), (light, dark), lighting, style) = spec;
        let xy = f32_rows(xy, 2, "mesh positions")?;
        let point = f32_rows(point, 3, "mesh points")?;
        let normal = f32_rows(normal, 3, "mesh normals")?;
        let uv = f32_rows(uv, 2, "mesh texture coordinates")?;
        let count = xy.len() / 2;
        let keys = keys
            .as_ref()
            .map(|keys| f32_rows(keys, 1, "mesh depth"))
            .transpose()?;
        if point.len() != 3 * count
            || normal.len() != 3 * count
            || uv.len() != 2 * count
            || keys.as_ref().is_some_and(|keys| keys.len() != count)
        {
            return Err(PyValueError::new_err("mesh vertex arrays differ in length"));
        }
        let buffer = PyBuffer::<i32>::get(triangles)
            .map_err(|error| PyValueError::new_err(format!("mesh triangles: {error}")))?;
        if buffer.shape().len() != 2 || buffer.shape()[1] != 3 {
            return Err(PyValueError::new_err("mesh triangles must be int32 (t, 3)"));
        }
        let corners = buffer
            .to_vec(triangles.py())
            .map_err(|error| PyValueError::new_err(format!("mesh triangles: {error}")))?;
        if corners
            .iter()
            .any(|&corner| corner < 0 || corner as usize >= count)
        {
            return Err(PyValueError::new_err(
                "mesh triangles name missing vertices",
            ));
        }
        let mut vertices = Vec::with_capacity(count * STRIDE / 4);
        for index in 0..count {
            vertices.extend_from_slice(&xy[2 * index..2 * index + 2]);
            vertices.extend_from_slice(&point[3 * index..3 * index + 3]);
            vertices.extend_from_slice(&normal[3 * index..3 * index + 3]);
            vertices.extend_from_slice(&uv[2 * index..2 * index + 2]);
            vertices.push(0.0); // the normalised depth, set in `normalise`
        }
        Ok(Self {
            light: read_image(light, &mut texture)?,
            dark: read_image(dark, &mut texture)?,
            vertices,
            xy: xy.chunks_exact(2).map(|p| [p[0], p[1]]).collect(),
            keys,
            triangles: corners
                .chunks_exact(3)
                .map(|t| [t[0] as u32, t[1] as u32, t[2] as u32])
                .collect(),
            lighting: *lighting,
            color: style.0,
            opacity: style.1,
        })
    }

    pub fn is_opaque(&self) -> bool {
        self.opacity >= 1.0
    }

    pub fn depth_range(&self) -> Option<(f32, f32)> {
        let keys = self.keys.as_ref()?;
        let finite = keys.iter().copied().filter(|key| key.is_finite());
        let low = finite.clone().fold(f32::INFINITY, f32::min);
        let high = finite.fold(f32::NEG_INFINITY, f32::max);
        (low <= high).then_some((low, high))
    }

    /// Normalise the depth keys over the frame's range (see
    /// [`normalised_key`]).
    pub fn normalise(&mut self, range: (f32, f32)) {
        let Some(keys) = &self.keys else { return };
        for (index, &key) in keys.iter().enumerate() {
            self.vertices[index * STRIDE / 4 + 10] = normalised_key(key, range);
        }
    }

    /// The triangles for the depth pass, in device pixels: those of an
    /// opaque mesh, or of any visible mesh with `translucent` (a fully
    /// transparent one hides nothing: ManimGL discards fragments of zero
    /// alpha).
    pub fn depth_vertices(&self, to_device: [f32; 4], translucent: bool, out: &mut Vec<f32>) {
        let included = self.is_opaque() || (translucent && self.opacity > 0.0);
        if !included || self.keys.is_none() {
            return;
        }
        for triangle in &self.triangles {
            for &corner in triangle {
                let [x, y] = self.xy[corner as usize];
                out.push(x * to_device[0] + to_device[2]);
                out.push(y * to_device[1] + to_device[3]);
                out.push(self.vertices[corner as usize * STRIDE / 4 + 10]);
            }
        }
        debug_assert_eq!(out.len() % depth::VERTEX_FLOATS, 0);
    }

    /// The triangles that hide VMobjects (those of `depth_vertices` with
    /// `translucent`), as `(x, y, raw key)` in device pixels, for occlusion
    /// culling.
    pub fn occluder_vertices(&self, to_device: [f32; 4], out: &mut Vec<f32>) {
        let Some(keys) = &self.keys else { return };
        if !(self.is_opaque() || self.opacity > 0.0) {
            return;
        }
        for triangle in &self.triangles {
            for &corner in triangle {
                let [x, y] = self.xy[corner as usize];
                out.push(x * to_device[0] + to_device[2]);
                out.push(y * to_device[1] + to_device[3]);
                out.push(keys[corner as usize]);
            }
        }
    }

    fn image_shader(image: &Option<(Image, [f32; 2])>) -> Shader {
        image
            .as_ref()
            .and_then(|(image, _)| {
                image.to_shader(
                    Some((TileMode::Clamp, TileMode::Clamp)),
                    SamplingOptions::new(FilterMode::Linear, MipmapMode::Linear),
                    None,
                )
            })
            .unwrap_or_else(|| skia_safe::shaders::color(skia_safe::Color::WHITE))
    }

    /// Draw the mesh, tested per pixel against the opaque depth image when
    /// opaque and against the image of all meshes when translucent.
    pub fn draw(
        &self,
        canvas: &Canvas,
        context: &mut gpu::DirectContext,
        frame: &Frame,
    ) -> PyResult<()> {
        if self.triangles.is_empty() || self.opacity <= 0.0 {
            return Ok(());
        }
        let mut data: Vec<f32> = Vec::with_capacity(self.triangles.len() * 3 * STRIDE / 4);
        let (mut low, mut high) = ([f32::INFINITY; 2], [f32::NEG_INFINITY; 2]);
        for triangle in &self.triangles {
            for &corner in triangle {
                let start = corner as usize * STRIDE / 4;
                data.extend_from_slice(&self.vertices[start..start + STRIDE / 4]);
                let [x, y] = self.xy[corner as usize];
                low = [low[0].min(x), low[1].min(y)];
                high = [high[0].max(x), high[1].max(y)];
            }
        }
        let bounds = Rect::new(low[0], low[1], high[0], high[1]);
        let bytes: Vec<u8> = data.iter().flat_map(|value| value.to_ne_bytes()).collect();
        let buffer = gpu::ganesh::meshes::make_vertex_buffer(context, &bytes)
            .ok_or_else(|| PyValueError::new_err("mesh vertex buffer failed"))?;
        let (depth, has_depth) = if self.is_opaque() {
            (&frame.opaque_depth, frame.has_opaque)
        } else {
            (&frame.all_depth, frame.has_any)
        };
        let children = [
            ChildPtr::from(Self::image_shader(&self.light)),
            ChildPtr::from(Self::image_shader(&self.dark)),
            ChildPtr::from(depth.clone()),
        ];
        let spec = frame
            .spec
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("mesh drawn without its shaders"))?;
        let uniforms = frame.uniforms(spec, self, self.keys.is_some() && has_depth);
        let mesh = SkMesh::make(
            spec.clone(),
            mesh::Mode::Triangles,
            buffer,
            data.len() / (STRIDE / 4),
            0,
            Data::new_copy(&uniforms),
            &children,
            bounds,
        )
        .map_err(|error| PyValueError::new_err(format!("mesh: {error}")))?;
        // The paint is the blend's source and the mesh colour its
        // destination: `Dst` keeps the mesh colour, drawn source-over.
        canvas.draw_mesh(&mesh, Blender::mode(BlendMode::Dst), &Paint::default());
        Ok(())
    }
}

/// A depth key mapped to `[1/1024, 1 - 1/1024]` over the frame's mesh key
/// range `(low, high)`, so that `0` stays free for "no mesh" in the depth
/// images (see `depth`); a key outside the range lands outside that span,
/// clamped to `[1/2048, 1]` (still in front of or behind every mesh), and a
/// key that is not finite maps to `0`.
pub fn normalised_key(key: f32, (low, high): (f32, f32)) -> f32 {
    if !key.is_finite() {
        return 0.0;
    }
    let scale = (1.0 - 2.0 / 1024.0) / (high - low).max(f32::MIN_POSITIVE);
    (1.0 / 1024.0 + (key - low) * scale).clamp(1.0 / 2048.0, 1.0)
}

/// What every mesh and VMobject of a display list shares.
pub struct Frame {
    /// The mesh shaders; present whenever the display list has meshes.
    pub spec: Option<MeshSpecification>,
    /// The nearest opaque mesh per device pixel, as a shader (see `depth`).
    pub opaque_depth: Shader,
    pub has_opaque: bool,
    /// The nearest mesh of any opacity per device pixel.
    pub all_depth: Shader,
    pub has_any: bool,
    /// The depth key range the images are normalised over (see
    /// `Mesh::normalise`), to normalise the keys of VMobject runs alike.
    pub key_range: Option<(f32, f32)>,
    pub to_device: [f32; 4],
}

impl Frame {
    fn uniforms(&self, spec: &MeshSpecification, mesh: &Mesh, depth_test: bool) -> Vec<u8> {
        let mut bytes = vec![0u8; spec.uniform_size()];
        let mut put = |name: &str, values: &[f32]| {
            if let Some(uniform) = spec.find_uniform(name) {
                let offset = uniform.offset();
                for (index, value) in values.iter().enumerate() {
                    let at = offset + 4 * index;
                    bytes[at..at + 4].copy_from_slice(&value.to_ne_bytes());
                }
            }
        };
        let (light_position, camera_position, shading) = mesh.lighting;
        let textures = match (&mesh.light, &mesh.dark) {
            (None, _) => 0.0,
            (Some(_), None) => 1.0,
            (Some(_), Some(_)) => 2.0,
        };
        put("toDevice", &self.to_device);
        put("lightPosition", &light_position);
        put("cameraPosition", &camera_position);
        put("shading", &shading);
        put("baseColor", &mesh.color);
        put("textures", &[textures]);
        put(
            "lightSize",
            &mesh.light.as_ref().map_or([1.0; 2], |(_, size)| *size),
        );
        put(
            "darkSize",
            &mesh.dark.as_ref().map_or([1.0; 2], |(_, size)| *size),
        );
        put("depthTest", &[if depth_test { 1.0 } else { 0.0 }]);
        put("opacity", &[mesh.opacity.clamp(0.0, 1.0)]);
        bytes
    }
}
