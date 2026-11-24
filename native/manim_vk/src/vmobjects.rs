//! Draw many VMobjects, or depth-sorted pieces of them, in one call.
//!
//! Python sends each VMobject once per frame, as its projected points, its
//! subpaths and its paints (background stroke, fill, stroke), and then a list
//! of runs in drawing order.  A run draws one object: all its subpaths, or
//! the depth pieces listed for it.  A field of 50 000 depth-sorted arrows is
//! 100 000 runs of one piece each; drawing them here costs one call from
//! Python instead of 100 000.

use pyo3::buffer::PyBuffer;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use skia_safe::gradient::{Colors as GradientColors, Gradient, Interpolation};
use skia_safe::paint::{Cap, Join, Style};
use skia_safe::{Canvas, Color4f, Paint, PathBuilder, Point, TileMode};

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

struct Object {
    points: Vec<Point>,
    subpaths: Vec<(usize, usize, bool)>,
    paints: Vec<Paint>,
}

fn int_rows(array: &Bound<PyAny>, width: usize, name: &str) -> PyResult<Vec<i64>> {
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

/// Draw `runs`, `(m, 3)` int64 rows of `(object, start, end)`: object
/// `objects[object]` with the subpaths `parts[start..end]` (indices into its
/// subpaths), or with all of them when `start == end`.
pub fn draw(
    canvas: &Canvas,
    objects: Vec<ObjectSpec>,
    runs: &Bound<PyAny>,
    parts: &Bound<PyAny>,
) -> PyResult<()> {
    let objects = objects
        .into_iter()
        .map(|(points_array, subpaths_array, paints)| {
            Ok(Object {
                points: points(&points_array)?,
                subpaths: subpath_table(&subpaths_array)?,
                paints: paints.into_iter().filter_map(paint).collect(),
            })
        })
        .collect::<PyResult<Vec<Object>>>()?;
    let runs = int_rows(runs, 3, "runs")?;
    let parts = int_rows(parts, 1, "parts")?;
    let mut path = PathBuilder::new();
    for run in runs.chunks_exact(3) {
        let (index, start, end) = (run[0] as usize, run[1] as usize, run[2] as usize);
        let object = objects
            .get(index)
            .ok_or_else(|| PyValueError::new_err(format!("run names object {index}")))?;
        if object.paints.is_empty() {
            continue;
        }
        if end > parts.len() || start > end {
            return Err(PyValueError::new_err("run pieces are out of range"));
        }
        path.reset();
        if start == end {
            for &(first, last, closed) in &object.subpaths {
                add_subpath(&mut path, &object.points, first, last, closed);
            }
        } else {
            for &piece in &parts[start..end] {
                if let Some(&(first, last, closed)) = object.subpaths.get(piece as usize) {
                    add_subpath(&mut path, &object.points, first, last, closed);
                }
            }
        }
        let path = path.snapshot();
        for paint in &object.paints {
            canvas.draw_path(&path, paint);
        }
    }
    Ok(())
}
