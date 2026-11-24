//! Pure Rust geometry shared by the Python bindings and the renderer crate.

use numpy::ndarray::{Array2, ArrayView2};
use rayon::prelude::*;

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

/// Point index ranges `[start, end)` of the subpaths of a path of
/// `curve_size`-point Bezier curves, as
/// `VMobject.get_subpath_split_indices_from_points` returns them: a new
/// subpath starts wherever a curve's start anchor is not close (in the first
/// `n_dims` coordinates, with the start anchor as `numpy.isclose`'s
/// reference) to the previous curve's end anchor.
pub fn split_indices(
    points: ArrayView2<f64>,
    curve_size: usize,
    n_dims: usize,
    atol: f64,
) -> Vec<(usize, usize)> {
    let count = points.nrows();
    if count < curve_size {
        return Vec::new();
    }
    let mut result = Vec::new();
    let mut start = 0;
    let mut boundary = curve_size;
    while boundary < count {
        let joined =
            (0..n_dims).all(|k| is_close(points[[boundary - 1, k]], points[[boundary, k]], atol));
        if !joined {
            result.push((start, boundary));
            start = boundary;
        }
        boundary += curve_size;
    }
    result.push((start, count));
    result
}

/// [`split_indices`] without subpaths shorter than one curve, each marked
/// closed when its first and last points coincide in x and y, as the
/// renderers decide (the last point is the reference).
pub fn cubic_subpaths(
    points: ArrayView2<f64>,
    curve_size: usize,
    n_dims: usize,
    atol: f64,
) -> Vec<(usize, usize, bool)> {
    split_indices(points, curve_size, n_dims, atol)
        .into_iter()
        .filter(|&(start, end)| end - start >= curve_size)
        .map(|(start, end)| {
            let closed = (0..n_dims.min(2))
                .all(|k| is_close(points[[start, k]], points[[end - 1, k]], atol));
            (start, end, closed)
        })
        .collect()
}

/// Project points as `ThreeDCamera.project_points`: rotate `p - center` by
/// `rotation`, then scale x and y by `zoom` times the perspective factor
/// `d / (d - z)` (`1e6` behind the eye), or `exp(z / d)` in front of the
/// screen with `exponential`.
pub fn project_points(
    points: ArrayView2<f64>,
    center: [f64; 3],
    rotation: [[f64; 3]; 3],
    focal_distance: f64,
    zoom: f64,
    exponential: bool,
) -> Array2<f64> {
    let mut out = Array2::<f64>::zeros((points.nrows(), 3));
    for (mut row, point) in out.rows_mut().into_iter().zip(points.rows()) {
        let p = [
            point[0] - center[0],
            point[1] - center[1],
            point[2] - center[2],
        ];
        let q: [f64; 3] = std::array::from_fn(|i| {
            rotation[i][0] * p[0] + rotation[i][1] * p[1] + rotation[i][2] * p[2]
        });
        let z = q[2];
        let factor = if exponential && z >= 0.0 {
            (z / focal_distance).exp()
        } else if !exponential && focal_distance - z < 0.0 {
            1e6
        } else {
            focal_distance / (focal_distance - z)
        };
        let scale = factor * zoom;
        row[0] = q[0] * scale;
        row[1] = q[1] * scale;
        row[2] = z;
    }
    out
}

/// The mean of the points of each `[start, end)` range, in parallel: the
/// centers of a VMobject's subpaths, its depth pieces.
pub fn range_means(points: ArrayView2<f64>, ranges: &[(usize, usize)]) -> Array2<f64> {
    let dims = points.ncols();
    let mut means = Array2::<f64>::zeros((ranges.len(), dims));
    let rows = means.as_slice_mut().expect("a new array is contiguous");
    rows.par_chunks_mut(dims.max(1))
        .zip(ranges.par_iter())
        .for_each(|(row, &(start, end))| {
            for point in points.slice(numpy::ndarray::s![start..end, ..]).rows() {
                for (total, value) in row.iter_mut().zip(point) {
                    *total += value;
                }
            }
            let count = (end - start).max(1) as f64;
            row.iter_mut().for_each(|total| *total /= count);
        });
    means
}

/// The drawing order of depth pieces: by depth slice (`depth_slices` slices
/// between the nearest and farthest finite depth, far to near), then by
/// group, then by depth, then by index (so equal keys keep their order).
/// Pieces with an infinite depth go after every slice, by group and index.
pub fn depth_order(depth: &[f64], group: &[i64], depth_slices: usize) -> Vec<usize> {
    let finite = || depth.iter().copied().filter(|d| d.is_finite());
    let low = finite().fold(f64::INFINITY, f64::min);
    let high = finite().fold(f64::NEG_INFINITY, f64::max);
    let span = (high - low).max(1e-9);
    let slice = |d: f64| {
        if d.is_finite() {
            (((d - low) / span * depth_slices as f64) as usize).min(depth_slices - 1)
        } else {
            depth_slices
        }
    };
    let mut keys: Vec<(usize, i64, f64, usize)> = depth
        .par_iter()
        .zip(group.par_iter())
        .enumerate()
        .map(|(index, (&d, &g))| (slice(d), g, if d.is_finite() { d } else { 0.0 }, index))
        .collect();
    keys.par_sort_unstable_by(|a, b| {
        a.0.cmp(&b.0)
            .then(a.1.cmp(&b.1))
            .then(a.2.total_cmp(&b.2))
            .then(a.3.cmp(&b.3))
    });
    keys.into_iter().map(|key| key.3).collect()
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
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.2, 0.0],
            [0.0, 0.0, 0.0],
            [5.0, 5.0, 0.0],
            [6.0, 5.0, 0.0],
            [6.0, 6.0, 0.0],
            [7.0, 7.0, 0.0],
        ];
        let s = cubic_subpaths(p.view(), 4, 2, 1e-6);
        assert_eq!(s, vec![(0, 8, true), (8, 12, false)]);
    }

    #[test]
    fn splits_keep_short_tails_like_python() {
        let p = array![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 1.0, 0.0],
            [0.0, 0.5, 0.0],
        ];
        assert_eq!(split_indices(p.view(), 4, 3, 1e-6), vec![(0, 4), (4, 6)]);
        assert_eq!(
            split_indices(p.slice(numpy::ndarray::s![..3, ..]), 4, 3, 1e-6),
            vec![]
        );
    }

    #[test]
    fn projection_scales_by_perspective() {
        let p = array![[1.0, 2.0, 10.0], [1.0, 1.0, 30.0]];
        let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let out = project_points(p.view(), [0.0; 3], identity, 20.0, 2.0, false);
        assert_eq!(out.row(0).to_vec(), vec![4.0, 8.0, 10.0]);
        assert_eq!(out.row(1).to_vec(), vec![2e6, 2e6, 30.0]);
    }

    #[test]
    fn depth_order_sorts_by_slice_group_then_depth() {
        let depth = [3.0, 0.0, f64::INFINITY, 1.0, 0.5];
        let group = [0, 1, 0, 0, 0];
        // Two slices: [0, 1.5) holds 0.0, 1.0 and 0.5; [1.5, 3] holds 3.0.
        assert_eq!(depth_order(&depth, &group, 2), vec![4, 3, 1, 0, 2]);
    }

    #[test]
    fn range_means_average_rows() {
        let p = array![[0.0, 0.0], [2.0, 4.0], [1.0, 1.0]];
        let means = range_means(p.view(), &[(0, 2), (2, 3)]);
        assert_eq!(means.row(0).to_vec(), vec![1.0, 2.0]);
        assert_eq!(means.row(1).to_vec(), vec![1.0, 1.0]);
    }

    #[test]
    fn nan_is_never_close_and_infinity_matches_itself() {
        assert!(!is_close(f64::NAN, f64::NAN, 1e-6));
        assert!(is_close(f64::INFINITY, f64::INFINITY, 1e-6));
        assert!(!is_close(f64::INFINITY, -f64::INFINITY, 1e-6));
    }

    #[test]
    fn huge_curves_become_clamped_lines() {
        let mut p = array![
            [0.0, 0.0, 0.0],
            [1.0, 1e15, 0.0],
            [2.0, 1e15, 0.0],
            [3.0, 1e15, 0.0]
        ];
        tame_huge_curves(&mut p, 4, 10.0);
        assert_eq!(p.row(0).to_vec(), vec![0.0, 0.0, 0.0]);
        assert_eq!(p.row(3).to_vec(), vec![3.0, 10.0, 0.0]);
        assert!((p[[1, 1]] - 10.0 / 3.0).abs() < 1e-12);
    }
}
