//! Occlusion culling with Intel's Masked Occlusion Culling
//! (`third_party/masked_occlusion_culling`, the algorithm of Unity's Burst
//! Occlusion Culling).
//!
//! Each frame the meshes that hide VMobjects (every visible mesh, see
//! `depth`) are rasterised on the CPU into a conservative hierarchical depth
//! buffer; a run's pieces whose screen bounds lie wholly behind it are not
//! drawn at all, instead of being drawn and then clipped away per pixel.
//! The test only says "occluded" when every pixel of the bounds is behind a
//! mesh, so the frame is the same as without culling.
//!
//! The library stores `1 / w` as depth; ManimVK's depth keys are
//! `1 / (focal_distance - z)`, the same quantity, so `w = 1 / key`.
//! It is built for x86 (SSE4.1, AVX2, picked at run time); elsewhere
//! nothing is culled.

/// Pixels added around a piece's bounds: anti-aliasing and the difference
/// between this rasteriser's coverage and the GPU's.
const MARGIN: f32 = 2.0;

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
mod ffi {
    use std::ffi::c_void;
    extern "C" {
        pub fn moc_create() -> *mut c_void;
        pub fn moc_destroy(moc: *mut c_void);
        pub fn moc_set_resolution(moc: *mut c_void, width: u32, height: u32);
        pub fn moc_set_near_clip_plane(moc: *mut c_void, near: f32);
        pub fn moc_clear(moc: *mut c_void);
        pub fn moc_render_triangles(
            moc: *mut c_void,
            vertices: *const f32,
            triangles: *const u32,
            count: i32,
        );
        pub fn moc_test_rect(
            moc: *const c_void,
            xmin: f32,
            ymin: f32,
            xmax: f32,
            ymax: f32,
            wmin: f32,
        ) -> i32;
    }
}

/// The occlusion buffer of one render target.
pub struct Occlusion {
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    raw: std::ptr::NonNull<std::ffi::c_void>,
    /// The device size, and the buffer size (a multiple of 8 by 4, as the
    /// library needs) the device pixels are mapped into.
    device: [f32; 2],
    buffer: [f32; 2],
    has_occluders: bool,
}

// The library object is only touched through `&mut self` or `&self` of the
// render target that owns it, which is not shared between threads.
unsafe impl Send for Occlusion {}

impl Occlusion {
    pub fn new(width: u32, height: u32) -> Option<Self> {
        let buffer = [width.div_ceil(8) * 8, height.div_ceil(4) * 4];
        #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
        {
            let raw = std::ptr::NonNull::new(unsafe { ffi::moc_create() })?;
            unsafe {
                ffi::moc_set_resolution(raw.as_ptr(), buffer[0], buffer[1]);
                ffi::moc_set_near_clip_plane(raw.as_ptr(), 1e-6);
            }
            Some(Self {
                raw,
                device: [width as f32, height as f32],
                buffer: [buffer[0] as f32, buffer[1] as f32],
                has_occluders: false,
            })
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "x86")))]
        {
            let _ = buffer;
            None
        }
    }

    /// Start a frame: no occluders.
    pub fn clear(&mut self) {
        self.has_occluders = false;
        #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
        unsafe {
            ffi::moc_clear(self.raw.as_ptr())
        };
    }

    /// Rasterise occluding triangles: `vertices` are `(x, y, key)` in device
    /// pixels and raw depth keys, three per triangle.
    ///
    /// They are rasterised nearest first, as occluders should be: the
    /// buffer keeps a conservative depth per tile, and a far triangle (the
    /// back of a sphere) covering a tile before the near ones would leave
    /// the tile at the far depth.
    pub fn add_triangles(&mut self, vertices: &[f32]) {
        let triangles: Vec<&[f32]> = vertices.chunks_exact(9).collect();
        if triangles.is_empty() {
            return;
        }
        let nearest = |triangle: &[f32]| triangle[2].max(triangle[5]).max(triangle[8]);
        let mut order: Vec<usize> = (0..triangles.len()).collect();
        order.sort_unstable_by(|&a, &b| nearest(triangles[b]).total_cmp(&nearest(triangles[a])));
        let corners = triangles.len() * 3;
        let mut clip = Vec::with_capacity(corners * 3);
        for vertex in order.iter().flat_map(|&index| triangles[index].chunks_exact(3)) {
            let w = 1.0 / vertex[2].max(f32::MIN_POSITIVE);
            let [x, y] = self.ndc([vertex[0], vertex[1]]);
            clip.extend_from_slice(&[x * w, y * w, w]);
        }
        let indices: Vec<u32> = (0..corners as u32).collect();
        self.has_occluders = true;
        #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
        unsafe {
            ffi::moc_render_triangles(
                self.raw.as_ptr(),
                clip.as_ptr(),
                indices.as_ptr(),
                (corners / 3) as i32,
            )
        };
    }

    /// Whether nothing of the device-pixel bounds `low..high` can be seen:
    /// outside the frame, or behind the occluders everywhere at the raw
    /// depth key `key` (that of its nearest point).
    pub fn hidden(&self, low: [f32; 2], high: [f32; 2], key: f32) -> bool {
        let low = [low[0] - MARGIN, low[1] - MARGIN];
        let high = [high[0] + MARGIN, high[1] + MARGIN];
        if high[0] < 0.0 || high[1] < 0.0 || low[0] > self.device[0] || low[1] > self.device[1] {
            return true;
        }
        if !self.has_occluders || key.is_nan() || key <= 0.0 {
            return false;
        }
        // Device y points down, NDC y up.
        let [xmin, ymax] = self.ndc(low);
        let [xmax, ymin] = self.ndc(high);
        #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
        {
            let result =
                unsafe { ffi::moc_test_rect(self.raw.as_ptr(), xmin, ymin, xmax, ymax, 1.0 / key) };
            // OCCLUDED; VIEW_CULLED is handled above.
            result == 1
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "x86")))]
        {
            let _ = (xmin, ymin, xmax, ymax);
            false
        }
    }

    fn ndc(&self, [x, y]: [f32; 2]) -> [f32; 2] {
        [2.0 * x / self.buffer[0] - 1.0, 1.0 - 2.0 * y / self.buffer[1]]
    }
}

impl Drop for Occlusion {
    fn drop(&mut self) {
        #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
        unsafe {
            ffi::moc_destroy(self.raw.as_ptr())
        };
    }
}
