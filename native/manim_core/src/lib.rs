//! CPU kernels for Mobject geometry.
//!
//! Manim's per-frame cost is dominated by many small NumPy calls on short point
//! arrays (a few dozen points per mobject), each with a fixed dispatch cost.
//! These functions do one mobject family's work in a single call.  They take
//! point arrays of shape `(n, 3)` and return new arrays: Manim assigns new
//! point arrays rather than mutating shared ones, and so do callers of these.
//! The crate has no GPU dependency, so it builds anywhere.

use numpy::ndarray::{Array2, ArrayD, ArrayView2, Axis, Zip};
use numpy::PyArrayLike1;
use numpy::{
    AllowTypeChange, IntoPyArray, PyArray2, PyArrayDyn, PyArrayLike2, PyArrayLikeDyn,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub mod geometry;
pub mod shapes;

type Points<'py> = PyArrayLike2<'py, f64, AllowTypeChange>;

fn check_points(points: &ArrayView2<f64>) -> PyResult<()> {
    if points.ncols() != 3 {
        return Err(PyValueError::new_err(format!(
            "points must have 3 columns, not {}",
            points.ncols()
        )));
    }
    Ok(())
}

/// Map every array through `p -> (p - about) @ matrix.T + about + shift`.
///
/// With no matrix this is a translation by `shift`.  This is the work of
/// `Mobject.shift`, `scale`, `rotate`, `stretch` and `apply_matrix` on all
/// family members at once.
#[pyfunction]
#[pyo3(signature = (arrays, matrix=None, about=[0.0; 3], shift=[0.0; 3]))]
fn transform_points<'py>(
    py: Python<'py>,
    arrays: Vec<Points<'py>>,
    matrix: Option<[[f64; 3]; 3]>,
    about: [f64; 3],
    shift: [f64; 3],
) -> PyResult<Vec<Bound<'py, PyArray2<f64>>>> {
    let mut out = Vec::with_capacity(arrays.len());
    for array in &arrays {
        let points = array.as_array();
        check_points(&points)?;
        let mut result = Array2::<f64>::zeros(points.raw_dim());
        match matrix {
            None => {
                Zip::from(result.rows_mut())
                    .and(points.rows())
                    .for_each(|mut r, p| {
                        for k in 0..3 {
                            r[k] = p[k] + shift[k];
                        }
                    });
            }
            Some(m) => {
                let offset = [
                    about[0] + shift[0],
                    about[1] + shift[1],
                    about[2] + shift[2],
                ];
                Zip::from(result.rows_mut())
                    .and(points.rows())
                    .for_each(|mut r, p| {
                        let d = [p[0] - about[0], p[1] - about[1], p[2] - about[2]];
                        for k in 0..3 {
                            r[k] = m[k][0] * d[0] + m[k][1] * d[1] + m[k][2] * d[2] + offset[k];
                        }
                    });
            }
        }
        out.push(result.into_pyarray(py));
    }
    Ok(out)
}

/// The bounding box `(low, high)` of the arrays, or None if they are empty.
///
/// With `curve_size` n > 0 only anchors count: the first and last point of
/// each n-point curve, as `VMobject.get_points_defining_boundary` uses; an
/// array of a single point counts as that point.  With 0 every point counts.
#[pyfunction]
#[pyo3(signature = (arrays, curve_size=0))]
fn bounds(arrays: Vec<Points<'_>>, curve_size: usize) -> PyResult<Option<([f64; 3], [f64; 3])>> {
    let mut low = [f64::INFINITY; 3];
    let mut high = [f64::NEG_INFINITY; 3];
    let mut any = false;
    for array in &arrays {
        let points = array.as_array();
        check_points(&points)?;
        let count = points.nrows();
        for (i, p) in points.axis_iter(Axis(0)).enumerate() {
            if curve_size > 0 && count > 1 {
                let r = i % curve_size;
                if r != 0 && r != curve_size - 1 {
                    continue;
                }
            }
            any = true;
            for k in 0..3 {
                // NaN never wins a comparison, as with NumPy's fmin it is skipped.
                if p[k] < low[k] {
                    low[k] = p[k];
                }
                if p[k] > high[k] {
                    high[k] = p[k];
                }
            }
        }
    }
    Ok(if any { Some((low, high)) } else { None })
}

/// `(1 - alpha) * start + alpha * end`, elementwise, as `manim.utils.bezier.interpolate`.
#[pyfunction]
fn lerp<'py>(
    py: Python<'py>,
    start: PyArrayLikeDyn<'py, f64, AllowTypeChange>,
    end: PyArrayLikeDyn<'py, f64, AllowTypeChange>,
    alpha: f64,
) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    if start.shape() != end.shape() {
        return Err(PyValueError::new_err(format!(
            "shapes differ: {:?} and {:?}",
            start.shape(),
            end.shape()
        )));
    }
    let a = start.as_array();
    let b = end.as_array();
    let mut result = ArrayD::<f64>::zeros(a.raw_dim());
    Zip::from(&mut result)
        .and(&a)
        .and(&b)
        .for_each(|r, &x, &y| *r = (1.0 - alpha) * x + alpha * y);
    Ok(result.into_pyarray(py))
}

/// Subpaths of a cubic Bezier path as `(k, 3)` int64 rows of
/// `(start, end, closed)` point ranges.
///
/// See `geometry::cubic_subpaths`; this is
/// `VMobject.get_subpath_split_indices_from_points` plus the closing test the
/// renderers apply, in one call.  Curves reaching beyond `limit` are drawn as
/// straight lines between their clamped anchors, so the returned points may
/// differ from the input: they are returned too.
#[pyfunction]
#[pyo3(signature = (points, curve_size=4, n_dims=2, atol=1e-6, limit=f64::INFINITY))]
fn cubic_subpaths<'py>(
    py: Python<'py>,
    points: Points<'py>,
    curve_size: usize,
    n_dims: usize,
    atol: f64,
    limit: f64,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<i64>>)> {
    let view = points.as_array();
    if view.ncols() < n_dims || n_dims == 0 || curve_size < 2 {
        return Err(PyValueError::new_err("bad points, curve_size or n_dims"));
    }
    let mut owned = view.to_owned();
    if limit.is_finite() {
        geometry::tame_huge_curves(&mut owned, curve_size, limit);
    }
    let subpaths = geometry::cubic_subpaths(owned.view(), curve_size, n_dims, atol);
    let table = Array2::from_shape_fn((subpaths.len(), 3), |(row, column)| {
        let (start, end, closed) = subpaths[row];
        [start as i64, end as i64, closed as i64][column]
    });
    Ok((owned.into_pyarray(py), table.into_pyarray(py)))
}

/// A copy of `points` with curves beyond `limit` as clamped straight lines;
/// see `geometry::tame_huge_curves`.
#[pyfunction]
#[pyo3(signature = (points, curve_size=4, limit=1e5))]
fn tame_huge_curves<'py>(
    py: Python<'py>,
    points: Points<'py>,
    curve_size: usize,
    limit: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    if curve_size < 2 {
        return Err(PyValueError::new_err("curve_size must be at least 2"));
    }
    let mut owned = points.as_array().to_owned();
    geometry::tame_huge_curves(&mut owned, curve_size, limit);
    Ok(owned.into_pyarray(py))
}

fn same_rows(a: &ArrayView2<f64>, b: &ArrayView2<f64>) -> PyResult<()> {
    check_points(a)?;
    check_points(b)?;
    if a.nrows() != b.nrows() {
        return Err(PyValueError::new_err("starts and ends differ in length"));
    }
    Ok(())
}

/// Straight segments, one cubic each: points of shape `(4n, 3)`.
#[pyfunction]
fn line_points<'py>(py: Python<'py>, starts: Points<'py>, ends: Points<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let (a, b) = (starts.as_array(), ends.as_array());
    same_rows(&a, &b)?;
    Ok(shapes::lines(a, b).into_pyarray(py))
}

/// Circles in the xy-plane, `segments` cubic arcs each, as `Circle` lays them out.
#[pyfunction]
#[pyo3(signature = (centers, radii, segments=8))]
fn circle_points<'py>(
    py: Python<'py>,
    centers: Points<'py>,
    radii: PyArrayLike1<'py, f64, AllowTypeChange>,
    segments: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let (c, r) = (centers.as_array(), radii.as_array());
    check_points(&c)?;
    if c.nrows() != r.len() || segments == 0 {
        return Err(PyValueError::new_err("one radius per center and at least one segment"));
    }
    Ok(shapes::circles(c, r, segments).into_pyarray(py))
}

/// Arrow shafts `(4n, 3)` and filled triangle tips `(12n, 3)`; see `shapes::arrows`.
#[pyfunction]
#[pyo3(signature = (starts, ends, tip_length=0.35, tip_width=0.35, max_tip_ratio=0.25))]
fn arrow_points<'py>(
    py: Python<'py>,
    starts: Points<'py>,
    ends: Points<'py>,
    tip_length: f64,
    tip_width: f64,
    max_tip_ratio: f64,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)> {
    let (a, b) = (starts.as_array(), ends.as_array());
    same_rows(&a, &b)?;
    if tip_length <= 0.0 {
        return Err(PyValueError::new_err("tip_length must be positive"));
    }
    let (shafts, tips) = shapes::arrows(a, b, tip_length, tip_width, max_tip_ratio);
    Ok((shafts.into_pyarray(py), tips.into_pyarray(py)))
}

/// The interface level `manim._native` expects: bump it, there and here,
/// whenever a function is added or changes its signature or behaviour.
const API_VERSION: u32 = 2;

#[pymodule]
fn manim_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("API_VERSION", API_VERSION)?;
    m.add_function(wrap_pyfunction!(transform_points, m)?)?;
    m.add_function(wrap_pyfunction!(bounds, m)?)?;
    m.add_function(wrap_pyfunction!(lerp, m)?)?;
    m.add_function(wrap_pyfunction!(cubic_subpaths, m)?)?;
    m.add_function(wrap_pyfunction!(tame_huge_curves, m)?)?;
    m.add_function(wrap_pyfunction!(line_points, m)?)?;
    m.add_function(wrap_pyfunction!(circle_points, m)?)?;
    m.add_function(wrap_pyfunction!(arrow_points, m)?)?;
    Ok(())
}
