//! Pure Rust geometry shared by the Python bindings and the renderer crate.

use numpy::ndarray::{Array2, ArrayView2};

/// `numpy.isclose`'s default relative tolerance, used throughout Manim.
pub const RTOL: f64 = 1.0e-5;

/// `numpy.isclose(a, b, rtol=RTOL, atol=atol)`: NaN is never close, and
/// infinities are close only to the same infinity.
#[inline]
pub fn is_close(a: f64, b: f64, atol: f64) -> bool {
    if a == b {
        return true;
    }
    let diff = (a - b).abs();
    diff.is_finite() && diff <= atol + RTOL * b.abs()
}

fn rows_close(points: &ArrayView2<f64>, i: usize, j: usize, n_dims: usize, atol: f64) -> bool {
    (0..n_dims).all(|k| is_close(points[[i, k]], points[[j, k]], atol))
}

/// Split a path of `curve_size`-point Bezier curves into subpaths.
///
/// A new subpath starts wherever a curve's start anchor is not close (in the
/// first `n_dims` coordinates) to the previous curve's end anchor.  Subpaths
/// shorter than one curve are dropped, and each is marked closed when its
/// first and last points coincide, as the renderers decide.
pub fn cubic_subpaths(
    points: ArrayView2<f64>,
    curve_size: usize,
    n_dims: usize,
    atol: f64,
) -> Vec<(usize, usize, bool)> {
    let count = points.nrows();
    let mut result = Vec::new();
    if count < curve_size {
        return result;
    }
    let mut start = 0;
    let push = |start: usize, end: usize, result: &mut Vec<(usize, usize, bool)>| {
        if end - start >= curve_size {
            // Python compares (first, last) with the last point as reference.
            let closed = rows_close(&points, start, end - 1, n_dims.min(2), atol);
            result.push((start, end, closed));
        }
    };
    let mut boundary = curve_size;
    while boundary < count {
        if !rows_close(&points, boundary, boundary - 1, n_dims, atol) {
            push(start, boundary, &mut result);
            start = boundary;
        }
        boundary += curve_size;
    }
    push(start, count, &mut result);
    result
}

/// Replace every curve with a coordinate beyond `limit` by the straight line
/// between its anchors clamped to `limit` (see the Skia renderer: single
/// precision paths break down on huge coordinates such as a plotted pole).
pub fn tame_huge_curves(points: &mut Array2<f64>, curve_size: usize, limit: f64) {
    let count = points.nrows();
    if count % curve_size != 0 {
        return;
    }
    let dims = points.ncols();
    for curve in 0..count / curve_size {
        let first = curve * curve_size;
        let rows = first..first + curve_size;
        let huge = rows
            .clone()
            .any(|i| (0..dims).any(|k| points[[i, k]].abs() > limit));
        if !huge {
            continue;
        }
        let last = first + curve_size - 1;
        for k in 0..dims {
            let a = points[[first, k]].clamp(-limit, limit);
            let b = points[[last, k]].clamp(-limit, limit);
            for (step, i) in rows.clone().enumerate() {
                let t = step as f64 / (curve_size - 1) as f64;
                points[[i, k]] = a + t * (b - a);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::ndarray::array;

    #[test]
    fn splits_where_anchors_differ_and_marks_closed() {
        let p = array![
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 0.0],
            [5.0, 5.0, 0.0], [6.0, 5.0, 0.0], [6.0, 6.0, 0.0], [7.0, 7.0, 0.0],
        ];
        let s = cubic_subpaths(p.view(), 4, 2, 1e-6);
        assert_eq!(s, vec![(0, 8, true), (8, 12, false)]);
    }

    #[test]
    fn nan_is_never_close_and_infinity_matches_itself() {
        assert!(!is_close(f64::NAN, f64::NAN, 1e-6));
        assert!(is_close(f64::INFINITY, f64::INFINITY, 1e-6));
        assert!(!is_close(f64::INFINITY, -f64::INFINITY, 1e-6));
    }

    #[test]
    fn huge_curves_become_clamped_lines() {
        let mut p = array![[0.0, 0.0, 0.0], [1.0, 1e15, 0.0], [2.0, 1e15, 0.0], [3.0, 1e15, 0.0]];
        tame_huge_curves(&mut p, 4, 10.0);
        assert_eq!(p.row(0).to_vec(), vec![0.0, 0.0, 0.0]);
        assert_eq!(p.row(3).to_vec(), vec![3.0, 10.0, 0.0]);
        assert!((p[[1, 1]] - 10.0 / 3.0).abs() < 1e-12);
    }
}
