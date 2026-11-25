//! A frame's VMobjects and surface meshes, drawn in one call.
//!
//! Python sends each VMobject once per frame, as its projected points, its
//! subpaths and its paints (background stroke, fill, stroke), and each mesh
//! once (see `meshes`); then a list of runs in drawing order.  A
//! run draws one object: a mesh, all of a VMobject, or the depth pieces
//! listed for it (its curves or subpaths).  A depth-tested run carries the
//! depth key of its nearest piece and is clipped per pixel to where no mesh
//! is nearer, as a depth buffer would hide it.  A field of 50 000
//! depth-sorted arrows is a few hundred runs, one per depth slice and path;
//! drawing them here costs one call from Python instead of one per run.

use pyo3::buffer::PyBuffer;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use skia_safe::gradient::{Colors as GradientColors, Gradient, Interpolation};
use skia_safe::paint::{Cap, Join, Style};
use skia_safe::runtime_effect::ChildPtr;
use skia_safe::{
    gpu, Canvas, ClipOp, Color4f, Data, Paint, PathBuilder, Point, RuntimeEffect, TileMode,
};

use std::cell::Cell;
use std::time::Instant;

use crate::meshes::{normalised_key, Frame, Mesh};
use crate::occlusion::Occlusion;

/// One paint of an object: stroke width (None fills), join, cap, colour stops
/// `(offset, r, g, b, a)` and, for a gradient, its line `(x0, y0, x1, y1)`.
pub type PaintSpec = (
    Option<f32>,
    i32,
    i32,
    Vec<(f32, f32, f32, f32, f32)>,
    Option<(f32, f32, f32, f32)>,
);

/// One VMobject: points `(n, 2+)` float64, rows `(k, 3)` int64 of
/// `(start, end, flags)` into the points, and its paints in drawing order.
/// Drawn whole, every row is a subpath, closed with flag [`CLOSED`]; drawn
/// in depth pieces, a row is a piece (see `Object::draw`).
pub type ObjectSpec<'py> = (Bound<'py, PyAny>, Bound<'py, PyAny>, Vec<PaintSpec>);

/// Row flags: the subpath is closed, the row starts it, the row ends it.
const CLOSED: i64 = 1;
const FIRST: i64 = 2;
const LAST: i64 = 4;

pub struct Object {
    points: Vec<Point>,
    rows: Vec<(usize, usize, i64)>,
    paints: Vec<Paint>,
    /// How far its strokes reach beyond its points, in local units: along
    /// a single curve (half the widest stroke, a square cap's corner), and
    /// where curves join (a miter's tip at the miter limit).
    reach: f32,
    joint_reach: f32,
}

/// Occlusion culling of depth pieces (see `occlusion`), with counts for
/// `take_draw_stats`.
pub struct Cull<'a> {
    pub occlusion: &'a Occlusion,
    /// The frame's mesh key range (see `normalised_key`).
    pub range: Option<(f32, f32)>,
    pub tested: Cell<usize>,
    pub culled: Cell<usize>,
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
            paint.set_stroke_miter(MITER_LIMIT);
            paint.set_stroke_cap(cap_of(cap));
        }
    }
    Some(paint)
}

/// Manim Community draws with Cairo, whose miter limit is 10 (Skia's
/// default is 4): a miter join turns into a bevel only below about 11.5°.
pub const MITER_LIMIT: f32 = 10.0;

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

/// Read a VMobject from Python.
pub fn object((points_array, subpaths_array, paints): ObjectSpec) -> PyResult<Object> {
    let rows = int_rows(&subpaths_array, 3, "subpaths")?
        .chunks_exact(3)
        .map(|row| (row[0] as usize, row[1] as usize, row[2]))
        .collect();
    let paints: Vec<Paint> = paints.into_iter().filter_map(paint).collect();
    let strokes = paints.iter().filter(|paint| paint.style() != Style::Fill);
    let reach = strokes
        .clone()
        .map(|paint| 0.5 * paint.stroke_width() * std::f32::consts::SQRT_2)
        .fold(0.0, f32::max);
    let joint_reach = strokes
        .map(|paint| match paint.stroke_join() {
            Join::Miter => 0.5 * paint.stroke_width() * MITER_LIMIT,
            _ => 0.5 * paint.stroke_width() * std::f32::consts::SQRT_2,
        })
        .fold(reach, f32::max);
    Ok(Object {
        points: points(&points_array)?,
        rows,
        paints,
        reach,
        joint_reach,
    })
}

/// Side of the screen tiles the depth pieces of a run are drawn in, in
/// device pixels.
///
/// Ganesh draws an anti-aliased path on the GPU only while it is small: in
/// its path atlas, up to 256 x 256 pixels of bounds (or, if it has at most
/// 16 384 verbs, triangulated).  A larger path is rasterised on the CPU into
/// a mask texture the size of its bounds and the mask is cached.  The pieces
/// of a depth slice spread over the whole frame, so they are drawn as one
/// path per tile, which leaves up to 64 pixels for pieces reaching out of
/// their tile on each side.
const TILE: f32 = 128.0;

/// Contiguous pieces of one subpath, drawn as one: their range of points,
/// whether the chain closes its subpath, and its bounds in device pixels.
struct Chain {
    start: usize,
    end: usize,
    closed: bool,
    low: [f32; 2],
    high: [f32; 2],
}

impl Object {
    /// Draw all subpaths, or the listed pieces; `path` is reused across
    /// draws, `to_device` maps the points to device pixels.
    ///
    /// Pieces are joined in index order: a piece that starts where the
    /// previous one ended, in the same subpath, continues it, so the curves
    /// of a stroke drawn together keep their joins; a subpath drawn from its
    /// first piece to its last is closed if it is closed.  With `cull`,
    /// chains outside the frame, or wholly behind the meshes at the run's
    /// raw depth key `key`, are left out.  The chains are drawn one path per
    /// screen tile (see [`TILE`]).
    fn draw(
        &self,
        canvas: &Canvas,
        pieces: Option<&[i64]>,
        key: f32,
        to_device: [f32; 4],
        cull: Option<&Cull>,
        path: &mut PathBuilder,
    ) {
        if self.paints.is_empty() {
            return;
        }
        path.reset();
        let Some(pieces) = pieces else {
            for &(first, last, flags) in &self.rows {
                add_subpath(path, &self.points, first, last, flags & CLOSED != 0);
            }
            self.draw_path(canvas, path);
            return;
        };
        let scale = to_device[0].abs().max(to_device[1].abs());
        let mut tiled: Vec<((i32, i32), Chain)> = self
            .chains(pieces, to_device)
            .into_iter()
            .filter(|chain| {
                let Some(cull) = cull else { return true };
                cull.tested.set(cull.tested.get() + 1);
                // Joins only where a chain has more than one curve.
                let joined = chain.end - chain.start > 4 || chain.closed;
                let reach = scale * if joined { self.joint_reach } else { self.reach };
                let low = [chain.low[0] - reach, chain.low[1] - reach];
                let high = [chain.high[0] + reach, chain.high[1] + reach];
                let hidden = cull.occlusion.hidden(low, high, cull.occluding_key(key));
                cull.culled.set(cull.culled.get() + usize::from(hidden));
                !hidden
            })
            .map(|chain| {
                let x = 0.5 * (chain.low[0] + chain.high[0]);
                let y = 0.5 * (chain.low[1] + chain.high[1]);
                (((x / TILE).floor() as i32, (y / TILE).floor() as i32), chain)
            })
            .collect();
        // Stable: in a tile, chains keep their order.
        tiled.sort_by_key(|(tile, _)| *tile);
        for group in tiled.chunk_by(|a, b| a.0 == b.0) {
            path.reset();
            for (_, chain) in group {
                add_subpath(path, &self.points, chain.start, chain.end, chain.closed);
            }
            self.draw_path(canvas, path);
        }
    }

    fn draw_path(&self, canvas: &Canvas, path: &mut PathBuilder) {
        let path = path.snapshot();
        let started = Instant::now();
        for paint in &self.paints {
            canvas.draw_path(&path, paint);
        }
        TIMES.with(|times| times.set((times.get().0 + started.elapsed().as_secs_f64(), times.get().1)));
    }

    /// The listed pieces joined into chains (see [`Object::draw`]).
    fn chains(&self, pieces: &[i64], to_device: [f32; 4]) -> Vec<Chain> {
        let mut order: Vec<usize> = pieces
            .iter()
            .filter_map(|&piece| usize::try_from(piece).ok())
            .filter(|&piece| piece < self.rows.len())
            .collect();
        order.sort_unstable();
        order.dedup();
        let mut chains: Vec<Chain> = Vec::new();
        // Whether the open chain began its subpath, and the flags of its
        // last piece.
        let mut began = false;
        let mut last_flags = 0;
        for piece in order {
            let (start, end, flags) = self.rows[piece];
            if end > self.points.len() || end < start + 4 {
                continue;
            }
            let continues = flags & FIRST == 0
                && chains.last().is_some_and(|chain| chain.end == start && !chain.closed);
            if !continues {
                if let Some(chain) = chains.last_mut() {
                    chain.closed = began && last_flags & LAST != 0 && last_flags & CLOSED != 0;
                }
                began = flags & FIRST != 0;
                chains.push(Chain {
                    start,
                    end: start,
                    closed: false,
                    low: [f32::INFINITY; 2],
                    high: [f32::NEG_INFINITY; 2],
                });
            }
            let chain = chains.last_mut().expect("pushed above");
            chain.end = end;
            for point in &self.points[start..end] {
                let x = point.x * to_device[0] + to_device[2];
                let y = point.y * to_device[1] + to_device[3];
                chain.low = [chain.low[0].min(x), chain.low[1].min(y)];
                chain.high = [chain.high[0].max(x), chain.high[1].max(y)];
            }
            last_flags = flags;
        }
        if let Some(chain) = chains.last_mut() {
            chain.closed = began && last_flags & LAST != 0 && last_flags & CLOSED != 0;
        }
        chains
    }
}

/// The per-pixel depth test of a VMobject run, as a clip: coverage 1 where
/// the run's key is not behind the nearest mesh of the depth image (or no
/// mesh is there), 0 elsewhere.  The slack, 1/1024 of the frame's depth
/// range, keeps a stroke lying on a surface, as ManimGL's mesh lines are,
/// against the curvature between the surface's vertices.
pub const DEPTH_CLIP_SKSL: &str = r"
uniform shader depthImage;
uniform float key;
half4 main(float2 p) {
    float3 bytes = floor(depthImage.eval(p).rgb * 255.0 + 0.5);
    float nearest = dot(bytes, float3(65536.0, 256.0, 1.0)) / 16777215.0;
    return half4(key >= nearest - 1.0 / 1024.0 ? 1.0 : 0.0);
}
";

impl Cull<'_> {
    /// The raw depth key an occluder must exceed to hide every pixel of a
    /// run with raw key `key` from the per-pixel test ([`DEPTH_CLIP_SKSL`]):
    /// the run's normalised key plus that test's slack, mapped back, with a
    /// margin for the 24-bit depth image.  Culling with it only removes what
    /// the test would clip away entirely.  NaN (never occluded) for a run
    /// without a key.
    fn occluding_key(&self, key: f32) -> f32 {
        match self.range {
            Some((low, high)) if key.is_finite() => {
                let scale = (1.0 - 2.0 / 1024.0) / (high - low).max(f32::MIN_POSITIVE);
                low + (normalised_key(key, (low, high)) + 1e-6) / scale
            }
            _ => f32::NAN,
        }
    }
}

thread_local! {
    /// Seconds spent in Skia's `draw_path` (recording, which includes
    /// stroking paths on the CPU) and in drawing meshes, since the last
    /// `take_times`.
    static TIMES: Cell<(f64, f64)> = const { Cell::new((0.0, 0.0)) };
}

/// `(skia path recording, meshes)` seconds since the last call.
pub fn take_times() -> (f64, f64) {
    TIMES.with(|times| times.replace((0.0, 0.0)))
}

/// Run kinds in the first column of `runs`.
const VMOBJECT: i64 = 0;
const MESH: i64 = 1;

/// Draw `runs`, `(m, 4)` int64 rows of `(kind, object, start, end)`: the
/// mesh `object`, or the VMobject `object` with the pieces
/// `parts[start..end]` (indices into its rows), or all of it when
/// `start == end`.  `keys` holds the raw depth key of each run (its
/// nearest part), NaN for a run drawn without the depth test; `clip` is the
/// compiled [`DEPTH_CLIP_SKSL`], and `cull` leaves out pieces that cannot
/// be seen.
#[allow(clippy::too_many_arguments)]
pub fn draw(
    canvas: &Canvas,
    context: &mut gpu::DirectContext,
    objects: &[Object],
    meshes: &[Mesh],
    frame: &Frame,
    clip: &RuntimeEffect,
    cull: Option<&Cull>,
    runs: &Bound<PyAny>,
    parts: &Bound<PyAny>,
    keys: &[f32],
) -> PyResult<()> {
    let runs = int_rows(runs, 4, "runs")?;
    let parts = int_rows(parts, 1, "parts")?;
    if keys.len() * 4 != runs.len() {
        return Err(PyValueError::new_err("one depth key per run is needed"));
    }
    let mut path = PathBuilder::new();
    for (run, &key) in runs.chunks_exact(4).zip(keys) {
        let (kind, index, start, end) = (run[0], run[1] as usize, run[2] as usize, run[3] as usize);
        if end > parts.len() || start > end {
            return Err(PyValueError::new_err("run pieces are out of range"));
        }
        let pieces = (start < end).then(|| &parts[start..end]);
        let missing = || PyValueError::new_err(format!("run names a missing object {index}"));
        match kind {
            VMOBJECT => {
                let object = objects.get(index).ok_or_else(missing)?;
                let tested = match frame.key_range {
                    Some(range) if frame.has_any && key.is_finite() => {
                        Some(normalised_key(key, range))
                    }
                    _ => None,
                };
                match tested {
                    None => object.draw(canvas, pieces, key, frame.to_device, cull, &mut path),
                    Some(normalised) => {
                        let shader = clip
                            .make_shader(
                                Data::new_copy(&normalised.to_ne_bytes()),
                                &[ChildPtr::from(frame.all_depth.clone())],
                                None,
                            )
                            .ok_or_else(|| PyValueError::new_err("depth clip shader failed"))?;
                        canvas.save();
                        // The depth image is in device pixels.
                        let matrix = canvas.local_to_device();
                        canvas.reset_matrix();
                        canvas.clip_shader(shader, ClipOp::Intersect);
                        canvas.set_matrix(&matrix);
                        object.draw(canvas, pieces, key, frame.to_device, cull, &mut path);
                        canvas.restore();
                    }
                }
            }
            MESH => {
                let started = Instant::now();
                meshes.get(index).ok_or_else(missing)?.draw(canvas, context, frame)?;
                TIMES.with(|times| {
                    times.set((times.get().0, times.get().1 + started.elapsed().as_secs_f64()))
                });
            }
            _ => return Err(PyValueError::new_err(format!("unknown run kind {kind}"))),
        }
    }
    Ok(())
}
