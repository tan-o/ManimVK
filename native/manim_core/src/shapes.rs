//! Point arrays of many simple shapes at once, for Manim's batch mobjects.
//!
//! Each shape is one or more straight or circular cubic Bezier curves with the
//! same point layout as the corresponding single mobject (`Line`, `Circle`,
//! an `Arrow`'s shaft and filled triangle tip), so a batch draws the same
//! picture as N separate mobjects, in one path.

use numpy::ndarray::{Array2, ArrayView1, ArrayView2};

fn put_line(out: &mut Array2<f64>, row: usize, a: [f64; 3], b: [f64; 3]) {
    for (step, t) in [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0].iter().enumerate() {
        for k in 0..3 {
            out[[row + step, k]] = a[k] + t * (b[k] - a[k]);
        }
    }
}

fn row3(view: &ArrayView2<f64>, i: usize) -> [f64; 3] {
    [view[[i, 0]], view[[i, 1]], view[[i, 2]]]
}

/// One straight cubic per row of `starts`/`ends`: shape `(4n, 3)`.
pub fn lines(starts: ArrayView2<f64>, ends: ArrayView2<f64>) -> Array2<f64> {
    let n = starts.nrows();
    let mut out = Array2::zeros((4 * n, 3));
    for i in 0..n {
        put_line(&mut out, 4 * i, row3(&starts, i), row3(&ends, i));
    }
    out
}

/// Circles in the xy-plane as `segments` cubic arcs each, starting at angle 0
/// and running counterclockwise, as `Circle(num_components=segments + 1)`.
pub fn circles(centers: ArrayView2<f64>, radii: ArrayView1<f64>, segments: usize) -> Array2<f64> {
    let n = centers.nrows();
    let step = std::f64::consts::TAU / segments as f64;
    let handle = 4.0 / 3.0 * (step / 4.0).tan();
    // Unit-circle template, shared by every circle.
    let mut template = Vec::with_capacity(4 * segments);
    for s in 0..segments {
        let (a0, a1) = (s as f64 * step, (s + 1) as f64 * step);
        let (c0, s0, c1, s1) = (a0.cos(), a0.sin(), a1.cos(), a1.sin());
        template.push([c0, s0]);
        template.push([c0 - handle * s0, s0 + handle * c0]);
        template.push([c1 + handle * s1, s1 - handle * c1]);
        template.push([c1, s1]);
    }
    let mut out = Array2::zeros((template.len() * n, 3));
    for i in 0..n {
        let (cx, cy, cz, r) = (centers[[i, 0]], centers[[i, 1]], centers[[i, 2]], radii[i]);
        for (j, [x, y]) in template.iter().enumerate() {
            let row = i * template.len() + j;
            out[[row, 0]] = cx + r * x;
            out[[row, 1]] = cy + r * y;
            out[[row, 2]] = cz;
        }
    }
    out
}

/// Arrows from `starts` to `ends`: straight shafts that stop at the tips, and
/// filled triangular tips as three straight cubics each, laid out like
/// `ArrowTriangleFilledTip` (tip point, then the left and right base corners).
///
/// A tip is `tip_length` long and `tip_width` wide, shrunk to at most
/// `max_tip_ratio` of its arrow's length; tips lie in the plane spanned by the
/// arrow and the z axis' normal (the xy-plane for flat arrows).
pub fn arrows(
    starts: ArrayView2<f64>,
    ends: ArrayView2<f64>,
    tip_length: f64,
    tip_width: f64,
    max_tip_ratio: f64,
) -> (Array2<f64>, Array2<f64>) {
    let n = starts.nrows();
    let mut shafts = Array2::zeros((4 * n, 3));
    let mut tips = Array2::zeros((12 * n, 3));
    for i in 0..n {
        let a = row3(&starts, i);
        let b = row3(&ends, i);
        let d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let length = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        if length == 0.0 {
            put_line(&mut shafts, 4 * i, a, a);
            for c in 0..3 {
                put_line(&mut tips, 12 * i + 4 * c, a, a);
            }
            continue;
        }
        let u = [d[0] / length, d[1] / length, d[2] / length];
        // Perpendicular: z × u, or y × u for arrows along z.
        let mut side = [-u[1], u[0], 0.0];
        let mut side_len = (side[0] * side[0] + side[1] * side[1]).sqrt();
        if side_len < 1e-12 {
            side = [u[2], 0.0, -u[0]];
            side_len = (side[0] * side[0] + side[2] * side[2]).sqrt();
        }
        let scale = tip_length.min(max_tip_ratio * length);
        let half = tip_width * scale / tip_length.max(1e-300) / 2.0 / side_len;
        let base = [b[0] - u[0] * scale, b[1] - u[1] * scale, b[2] - u[2] * scale];
        let left = [base[0] + side[0] * half, base[1] + side[1] * half, base[2] + side[2] * half];
        let right = [base[0] - side[0] * half, base[1] - side[1] * half, base[2] - side[2] * half];
        put_line(&mut shafts, 4 * i, a, base);
        put_line(&mut tips, 12 * i, b, left);
        put_line(&mut tips, 12 * i + 4, left, right);
        put_line(&mut tips, 12 * i + 8, right, b);
    }
    (shafts, tips)
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::ndarray::{array, Array1};

    #[test]
    fn circle_matches_manim_layout() {
        let c = circles(array![[1.0, 0.0, 0.0]].view(), Array1::from(vec![2.0]).view(), 8);
        assert_eq!(c.nrows(), 32);
        assert!((c[[0, 0]] - 3.0).abs() < 1e-12);
        assert!((c[[1, 1]] - 0.530433).abs() < 1e-6);
        assert!((c[[31, 0]] - 3.0).abs() < 1e-12 && c[[31, 1]].abs() < 1e-12);
    }

    #[test]
    fn arrow_tip_is_shrunk_for_short_arrows() {
        let (shaft, tip) = arrows(array![[0.0, 0.0, 0.0]].view(), array![[1.0, 0.0, 0.0]].view(), 0.35, 0.35, 0.25);
        assert!((shaft[[3, 0]] - 0.75).abs() < 1e-12);
        assert_eq!(tip.row(0).to_vec(), vec![1.0, 0.0, 0.0]);
        assert!((tip[[3, 1]] - 0.125).abs() < 1e-12);
    }
}
