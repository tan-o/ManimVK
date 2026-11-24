//! A frame's VMobjects and textured meshes, drawn in one call.
//!
//! Python sends each VMobject once per frame, as its projected points, its
//! subpaths and its paints (background stroke, fill, stroke), and each mesh
//! once, as its shaded triangles; then a list of runs in drawing order.  A
//! run draws one object: all of it, or the depth pieces listed for it
//! (subpaths of a VMobject, triangles of a mesh).  A field of 50 000
//! depth-sorted arrows is 100 000 runs of one piece each, interleaved with
//! slices of a translucent sphere; drawing them here costs one call from
//! Python instead of one per run.

use pyo3::buffer::PyBuffer;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use skia_safe::gradient::{Colors as GradientColors, Gradient, Interpolation};
use skia_safe::paint::{Cap, Join, Style};
use skia_safe::vertices::{VertexMode, Vertices};
use skia_safe::{
    BlendMode, Canvas, Color, Color4f, FilterMode, Image, MipmapMode, Paint, PathBuilder, Point,
    Rect, SamplingOptions, TileMode,
};

/// One paint of an object: stroke width (None fills), join, cap, colour stops
/// `(offset, r, g, b, a)` and, for a gradient, its line `(x0, y0, x1, y1)`.
pub type PaintSpec = (
    Option<f32>,
    i32,
    i32,
    Vec<(f32, f32, f32, f32, f32)>,
    Option<(f32, f32, f32, f32)>,
);

/// One VMobject: points `(n, 2+)` float64, subpaths `(k, 3)` int64 of
/// `(start, end, closed)` into the points, and its paints in drawing order.
pub type ObjectSpec<'py> = (Bound<'py, PyAny>, Bound<'py, PyAny>, Vec<PaintSpec>);

/// One textured mesh: texture key, RGBA pixels with their width and height,
/// float32 positions `(n, 2)`, texture coordinates `(n, 2)` and colours
/// `(n, 4)` with `per_triangle` rows per depth piece, its opacity, and the
/// colour that clears what lies under its pieces (see `Mesh::draw`).
pub type MeshSpec<'py> = (
    u64,
    Bound<'py, PyAny>,
    u32,
    u32,
    Bound<'py, PyAny>,
    Bound<'py, PyAny>,
    Bound<'py, PyAny>,
    usize,
    f32,
    Option<(f32, f32, f32, f32)>,
);

pub struct Object {
    points: Vec<Point>,
    subpaths: Vec<(usize, usize, bool)>,
    paints: Vec<Paint>,
}

pub struct Mesh {
    paint: Paint,
    positions: Vec<Point>,
    uvs: Vec<Point>,
    colors: Vec<Color>,
    per_triangle: usize,
    alpha: f32,
    occlude: Option<Color>,
}

pub fn int_rows(array: &Bound<PyAny>, width: usize, name: &str) -> PyResult<Vec<i64>> {
    let buffer = PyBuffer::<i64>::get(array)
        .map_err(|error| PyValueError::new_err(format!("{name}: {error}")))?;
    let shape = buffer.shape();
    let rows_ok = match width {
        1 => shape.len() == 1,
        _ => shape.len() == 2 && shape[1] == width,
    };
    if !rows_ok {
        return Err(PyValueError::new_err(format!(
            "{name} must have {width} int64 columns"
        )));
    }
    buffer
        .to_vec(array.py())
        .map_err(|error| PyValueError::new_err(format!("{name}: {error}")))
}

/// `(k, 3)` int64 rows of `(start, end, closed)`, as `manim_core.cubic_subpaths`
/// returns them.
pub fn subpath_table(array: &Bound<PyAny>) -> PyResult<Vec<(usize, usize, bool)>> {
    Ok(int_rows(array, 3, "subpaths")?
        .chunks_exact(3)
        .map(|row| (row[0] as usize, row[1] as usize, row[2] != 0))
        .collect())
}

fn points(array: &Bound<PyAny>) -> PyResult<Vec<Point>> {
    let buffer = PyBuffer::<f64>::get(array)
        .map_err(|error| PyValueError::new_err(format!("path points: {error}")))?;
    let shape = buffer.shape();
    if shape.len() != 2 || shape[1] < 2 {
        return Err(PyValueError::new_err("path points must have shape (n, 2+)"));
    }
    let stride = shape[1];
    let data = buffer
        .as_slice(array.py())
        .ok_or_else(|| PyValueError::new_err("path points must be C-contiguous"))?;
    Ok(data
        .chunks_exact(stride)
        .map(|row| Point::new(row[0].get() as f32, row[1].get() as f32))
        .collect())
}

fn paint((width, join, cap, stops, gradient): PaintSpec) -> Option<Paint> {
    let first = *stops.first()?;
    let mut paint = Paint::default();
    paint.set_anti_alias(true);
    match gradient {
        Some((x0, y0, x1, y1)) if stops.len() > 1 => {
            let colors: Vec<Color4f> = stops
                .iter()
                .map(|&(_, r, g, b, a)| Color4f::new(r, g, b, a))
                .collect();
            let positions: Vec<f32> = stops.iter().map(|stop| stop.0).collect();
            let gradient = Gradient::new(
                GradientColors::new(&colors, Some(positions.as_slice()), TileMode::Clamp, None),
                Interpolation::default(),
            );
            let shader = skia_safe::gradient::shaders::linear_gradient(
                (Point::new(x0, y0), Point::new(x1, y1)),
                &gradient,
                None,
            )?;
            paint.set_shader(shader);
        }
        _ => {
            let (_, r, g, b, a) = first;
            paint.set_color4f(Color4f::new(r, g, b, a), None);
        }
    }
    match width {
        None => {
            paint.set_style(Style::Fill);
        }
        Some(width) => {
            paint.set_style(Style::Stroke);
            paint.set_stroke_width(width);
            paint.set_stroke_join(join_of(join));
            paint.set_stroke_cap(cap_of(cap));
        }
    }
    Some(paint)
}

pub fn join_of(code: i32) -> Join {
    match code {
        1 => Join::Round,
        2 => Join::Bevel,
        _ => Join::Miter,
    }
}

pub fn cap_of(code: i32) -> Cap {
    match code {
        1 => Cap::Round,
        2 => Cap::Square,
        _ => Cap::Butt,
    }
}

/// Append one subpath: a move to its first point, then a cubic per four
/// points, `a0 h0 h1 a1  a1' h ...`, as Manim stores curves.
pub fn add_subpath(
    path: &mut PathBuilder,
    points: &[Point],
    start: usize,
    end: usize,
    closed: bool,
) {
    if end > points.len() || end < start + 4 {
        return;
    }
    path.move_to(points[start]);
    let mut index = start;
    while index + 4 <= end {
        path.cubic_to(points[index + 1], points[index + 2], points[index + 3]);
        index += 4;
    }
    if closed {
        path.close();
    }
}

fn f32_points(array: &Bound<PyAny>, width: usize, name: &str) -> PyResult<Vec<f32>> {
    let buffer = PyBuffer::<f32>::get(array)
        .map_err(|error| PyValueError::new_err(format!("{name}: {error}")))?;
    let shape = buffer.shape();
    if shape.len() != 2 || shape[1] != width {
        return Err(PyValueError::new_err(format!(
            "{name} must have {width} float32 columns"
        )));
    }
    buffer
        .to_vec(array.py())
        .map_err(|error| PyValueError::new_err(format!("{name}: {error}")))
}

/// Read a VMobject from Python.
pub fn object((points_array, subpaths_array, paints): ObjectSpec) -> PyResult<Object> {
    Ok(Object {
        points: points(&points_array)?,
        subpaths: subpath_table(&subpaths_array)?,
        paints: paints.into_iter().filter_map(paint).collect(),
    })
}

/// Read a mesh from Python; `texture` is its image, from the texture cache.
pub fn mesh(spec: &MeshSpec, texture: &Image) -> PyResult<Mesh> {
    let (_, _, _, _, positions, uvs, colors, per_triangle, alpha, occlude) = spec;
    let positions = f32_points(positions, 2, "mesh positions")?;
    let uvs = f32_points(uvs, 2, "mesh uvs")?;
    let colors = f32_points(colors, 4, "mesh colors")?;
    let count = positions.len() / 2;
    if uvs.len() != count * 2 || colors.len() != count * 4 || *per_triangle == 0 {
        return Err(PyValueError::new_err(
            "a mesh needs matching positions, uvs and colors",
        ));
    }
    if count % *per_triangle != 0 || *per_triangle % 3 != 0 {
        return Err(PyValueError::new_err(
            "mesh pieces must hold whole triangles",
        ));
    }
    let shader = texture
        .to_shader(
            Some((TileMode::Clamp, TileMode::Clamp)),
            SamplingOptions::new(FilterMode::Linear, MipmapMode::Linear),
            None,
        )
        .ok_or_else(|| PyValueError::new_err("mesh texture shader failed"))?;
    let mut paint = Paint::default();
    paint.set_anti_alias(true);
    paint.set_shader(shader);
    Ok(Mesh {
        paint,
        positions: positions
            .chunks_exact(2)
            .map(|p| Point::new(p[0], p[1]))
            .collect(),
        uvs: uvs
            .chunks_exact(2)
            .map(|p| Point::new(p[0], p[1]))
            .collect(),
        colors: colors
            .chunks_exact(4)
            .map(|c| Color4f::new(c[0], c[1], c[2], c[3]).to_color())
            .collect(),
        per_triangle: *per_triangle,
        alpha: *alpha,
        occlude: occlude.map(|(r, g, b, a)| Color4f::new(r, g, b, a).to_color()),
    })
}

impl Object {
    /// Draw all subpaths, or the listed ones; `path` is reused across runs.
    fn draw(&self, canvas: &Canvas, pieces: Option<&[i64]>, path: &mut PathBuilder) {
        if self.paints.is_empty() {
            return;
        }
        path.reset();
        match pieces {
            None => {
                for &(first, last, closed) in &self.subpaths {
                    add_subpath(path, &self.points, first, last, closed);
                }
            }
            Some(pieces) => {
                for &piece in pieces {
                    if let Some(&(first, last, closed)) = self.subpaths.get(piece as usize) {
                        add_subpath(path, &self.points, first, last, closed);
                    }
                }
            }
        }
        let path = path.snapshot();
        for paint in &self.paints {
            canvas.draw_path(&path, paint);
        }
    }
}

impl Mesh {
    /// Draw all triangles, or the listed pieces, blended over each other in
    /// one layer drawn with the mesh's opacity.  Depth pieces of a
    /// translucent mesh first clear what lies under them to `occlude`, as a
    /// depth buffer would hide it.
    fn draw(&self, canvas: &Canvas, pieces: Option<&[i64]>) {
        let selected;
        let (positions, uvs, colors): (&[Point], &[Point], &[Color]) = match pieces {
            None => (&self.positions, &self.uvs, &self.colors),
            Some(pieces) => {
                let per = self.per_triangle;
                let rows: Vec<usize> = pieces
                    .iter()
                    .map(|&piece| piece as usize * per)
                    .filter(|&first| first + per <= self.positions.len())
                    .flat_map(|first| first..first + per)
                    .collect();
                selected = (
                    rows.iter().map(|&i| self.positions[i]).collect::<Vec<_>>(),
                    rows.iter().map(|&i| self.uvs[i]).collect::<Vec<_>>(),
                    rows.iter().map(|&i| self.colors[i]).collect::<Vec<_>>(),
                );
                (&selected.0, &selected.1, &selected.2)
            }
        };
        if positions.is_empty() {
            return;
        }
        let vertices = Vertices::new_copy(VertexMode::Triangles, positions, uvs, colors, None);
        if let (Some(color), Some(_)) = (self.occlude, pieces) {
            if self.alpha < 1.0 {
                let mut clear = Paint::default();
                clear.set_shader(skia_safe::shaders::color(color));
                clear.set_blend_mode(BlendMode::Src);
                canvas.draw_vertices(&vertices, BlendMode::Src, &clear);
            }
        }
        let layered = self.alpha < 1.0;
        if layered {
            canvas.save_layer_alpha(
                None::<Rect>,
                (self.alpha.clamp(0.0, 1.0) * 255.0).round() as u32,
            );
        }
        canvas.draw_vertices(&vertices, BlendMode::Modulate, &self.paint);
        if layered {
            canvas.restore();
        }
    }
}

/// Run kinds in the first column of `runs`.
const VMOBJECT: i64 = 0;
const MESH: i64 = 1;

/// Draw `runs`, `(m, 4)` int64 rows of `(kind, object, start, end)`: the
/// VMobject or mesh `object` with the pieces `parts[start..end]` (indices
/// into its subpaths or triangles), or all of it when `start == end`.
pub fn draw(
    canvas: &Canvas,
    objects: &[Object],
    meshes: &[Mesh],
    runs: &Bound<PyAny>,
    parts: &Bound<PyAny>,
) -> PyResult<()> {
    let runs = int_rows(runs, 4, "runs")?;
    let parts = int_rows(parts, 1, "parts")?;
    let mut path = PathBuilder::new();
    for run in runs.chunks_exact(4) {
        let (kind, index, start, end) = (run[0], run[1] as usize, run[2] as usize, run[3] as usize);
        if end > parts.len() || start > end {
            return Err(PyValueError::new_err("run pieces are out of range"));
        }
        let pieces = (start < end).then(|| &parts[start..end]);
        let missing = || PyValueError::new_err(format!("run names a missing object {index}"));
        match kind {
            VMOBJECT => objects
                .get(index)
                .ok_or_else(missing)?
                .draw(canvas, pieces, &mut path),
            MESH => meshes.get(index).ok_or_else(missing)?.draw(canvas, pieces),
            _ => return Err(PyValueError::new_err(format!("unknown run kind {kind}"))),
        }
    }
    Ok(())
}
