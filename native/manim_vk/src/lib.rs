//! Native Vulkan core for Manim.
//!
//! `Device` owns the Vulkan instance/device and one shared Skia Ganesh
//! context; `SkiaTarget` is a GPU render target that draws Manim's classic
//! mobjects (paths, images, point clouds, SkSL effects) and reads frames back.

mod window;

use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::CString;
use std::rc::Rc;
use std::sync::Arc;

use ash::vk;
use pyo3::buffer::PyBuffer;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use skia_safe::gradient::{Colors as GradientColors, Gradient, Interpolation};
use skia_safe::paint::{Cap, Join, Style};
use skia_safe::runtime_effect::RuntimeShaderBuilder;
use skia_safe::vertices::{VertexMode, Vertices};
use skia_safe::{
    gpu, AlphaType, BlendMode, ClipOp, Color, Color4f, ColorType, CubicResampler, Data, FilterMode,
    Image, ImageInfo as SkImageInfo, Matrix, MipmapMode, Paint, PathBuilder, Point, Rect,
    RuntimeEffect, SamplingOptions, TileMode, M44,
};

struct DeviceInner {
    entry: ash::Entry,
    instance: ash::Instance,
    physical_device: vk::PhysicalDevice,
    device: ash::Device,
    queue: vk::Queue,
    queue_family: u32,
    device_name: String,
}

/// One Skia Ganesh context shared by every target created from the same
/// ``Device``.  ``context`` only holds a reference-counted pointer, so each
/// target can clone it without creating a second Vulkan context.
struct GpuShared {
    backend: Box<gpu::vk::BackendContext<'static>>,
    context: gpu::DirectContext,
}

impl Drop for DeviceInner {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
        let _ = &self.entry;
    }
}

fn vk_err(context: &str, result: vk::Result) -> PyErr {
    PyRuntimeError::new_err(format!("{context}: {result:?}"))
}

#[pyclass(unsendable)]
struct Device {
    inner: Arc<DeviceInner>,
    gpu: RefCell<Option<Rc<GpuShared>>>,
}

#[pymethods]
impl Device {
    #[new]
    #[pyo3(signature = (application_name = "Manim", validation = false))]
    fn new(application_name: &str, validation: bool) -> PyResult<Self> {
        let entry = unsafe { ash::Entry::load() }
            .map_err(|error| PyRuntimeError::new_err(format!("load Vulkan loader: {error}")))?;

        let app_name = CString::new(application_name)
            .map_err(|_| PyValueError::new_err("application name contains a NUL byte"))?;
        let engine_name = CString::new("manim_vk")
            .map_err(|_| PyValueError::new_err("engine name contains a NUL byte"))?;
        let app_info = vk::ApplicationInfo::default()
            .application_name(&app_name)
            .engine_name(&engine_name)
            .api_version(vk::make_api_version(0, 1, 3, 0));

        let mut layer_names: Vec<CString> = Vec::new();
        if validation {
            layer_names.push(
                CString::new("VK_LAYER_KHRONOS_validation").map_err(|_| {
                    PyValueError::new_err("validation layer name contains a NUL byte")
                })?,
            );
        }
        let layer_refs: Vec<*const std::ffi::c_char> =
            layer_names.iter().map(|name| name.as_ptr()).collect();
        let instance_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_layer_names(&layer_refs);
        let instance = unsafe { entry.create_instance(&instance_info, None) }
            .map_err(|result| vk_err("vkCreateInstance", result))?;

        let physical_devices = unsafe { instance.enumerate_physical_devices() }
            .map_err(|result| vk_err("vkEnumeratePhysicalDevices", result))?;
        if physical_devices.is_empty() {
            unsafe { instance.destroy_instance(None) };
            return Err(PyRuntimeError::new_err("no Vulkan physical device found"));
        }

        // Prefer a discrete GPU by default.  ``MANIM_VK_DEVICE`` can contain
        // a case-insensitive substring to force a particular adapter.
        let requested_device = std::env::var("MANIM_VK_DEVICE")
            .ok()
            .map(|name| name.to_lowercase());
        let mut selected: Option<(i32, vk::PhysicalDevice, u32)> = None;
        for physical_device in physical_devices {
            let properties = unsafe { instance.get_physical_device_properties(physical_device) };
            let name = String::from_utf8_lossy(
                &properties
                    .device_name
                    .iter()
                    .map(|&c| c as u8)
                    .collect::<Vec<u8>>(),
            )
            .trim_end_matches('\0')
            .to_string();
            let families =
                unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
            let Some((index, _)) = families
                .iter()
                .enumerate()
                .find(|(_, family)| family.queue_flags.contains(vk::QueueFlags::GRAPHICS))
            else {
                continue;
            };

            let score = if requested_device
                .as_ref()
                .is_some_and(|needle| name.to_lowercase().contains(needle))
            {
                1000
            } else {
                match properties.device_type {
                    vk::PhysicalDeviceType::DISCRETE_GPU => 500,
                    vk::PhysicalDeviceType::INTEGRATED_GPU => 400,
                    vk::PhysicalDeviceType::VIRTUAL_GPU => 300,
                    vk::PhysicalDeviceType::CPU => 200,
                    _ => 100,
                }
            };
            if selected
                .as_ref()
                .map(|(best_score, _, _)| score > *best_score)
                .unwrap_or(true)
            {
                selected = Some((score, physical_device, index as u32));
            }
        }
        let (_, physical_device, queue_family) = selected.ok_or_else(|| {
            unsafe { instance.destroy_instance(None) };
            PyRuntimeError::new_err("no Vulkan queue family supports graphics")
        })?;
        let properties = unsafe { instance.get_physical_device_properties(physical_device) };
        let device_name = String::from_utf8_lossy(
            &properties
                .device_name
                .iter()
                .map(|&c| c as u8)
                .collect::<Vec<u8>>(),
        )
        .trim_end_matches('\0')
        .to_string();

        let priorities = [1.0f32];
        let queue_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family)
            .queue_priorities(&priorities);
        let queue_infos = [queue_info];
        let device_info = vk::DeviceCreateInfo::default().queue_create_infos(&queue_infos);
        let device = unsafe { instance.create_device(physical_device, &device_info, None) }
            .map_err(|result| vk_err("vkCreateDevice", result))?;
        let queue = unsafe { device.get_device_queue(queue_family, 0) };

        Ok(Self {
            inner: Arc::new(DeviceInner {
                entry,
                instance,
                physical_device,
                device,
                queue,
                queue_family,
                device_name,
            }),
            gpu: RefCell::new(None),
        })
    }

    fn device_name(&self) -> String {
        self.inner.device_name.clone()
    }

    fn api_version(&self) -> String {
        let properties = unsafe {
            self.inner
                .instance
                .get_physical_device_properties(self.inner.physical_device)
        };
        let version = properties.api_version;
        format!(
            "{}.{}.{}",
            vk::api_version_major(version),
            vk::api_version_minor(version),
            vk::api_version_patch(version)
        )
    }

    fn create_skia_target(&self, width: u32, height: u32) -> PyResult<SkiaTarget> {
        if width == 0 || height == 0 {
            return Err(PyValueError::new_err("target dimensions must be positive"));
        }
        let gpu = self.gpu_shared()?;
        SkiaTarget::new(self.inner.clone(), gpu, width, height)
    }
}

impl Device {
    /// Build the shared Skia Ganesh context on first use.  The ``get_proc``
    /// closure is leaked once per device (not once per target) and the
    /// backend keeps that pointer alive for the process lifetime.
    fn gpu_shared(&self) -> PyResult<Rc<GpuShared>> {
        if let Some(shared) = self.gpu.borrow().as_ref() {
            return Ok(shared.clone());
        }
        let shared = make_gpu_shared(self.inner.clone())?;
        *self.gpu.borrow_mut() = Some(shared.clone());
        Ok(shared)
    }
}

fn make_gpu_shared(inner: Arc<DeviceInner>) -> PyResult<Rc<GpuShared>> {
    use ash::vk::Handle;
    let inner_for_proc = inner.clone();
    // Leak exactly one trampoline per Device.  Skia's BackendContext stores a
    // raw pointer to it, so it must outlive every target created from this
    // device; one device per process is the normal Manim lifecycle.
    let get_proc = Box::leak(Box::new(move |of| skia_get_proc(&inner_for_proc, of)));
    let backend = Box::new(unsafe {
        gpu::vk::BackendContext::new_builder(
            inner.instance.handle().as_raw() as _,
            inner.physical_device.as_raw() as _,
            inner.device.handle().as_raw() as _,
            // Skia's graphics queue index is the queue *family* index.
            (inner.queue.as_raw() as _, inner.queue_family as usize),
            &*get_proc,
            Some((1, 3, 0).into()),
        )
        .build()
    });
    let context = gpu::direct_contexts::make_vulkan(&backend, None)
        .ok_or_else(|| PyRuntimeError::new_err("Skia make_vulkan returned None"))?;
    Ok(Rc::new(GpuShared { backend, context }))
}

fn skia_get_proc(inner: &DeviceInner, of: gpu::vk::GetProcOf) -> *const std::ffi::c_void {
    use ash::vk::Handle;
    unsafe {
        match of {
            gpu::vk::GetProcOf::Instance(instance, name) => {
                match inner
                    .entry
                    .get_instance_proc_addr(vk::Instance::from_raw(instance as _), name)
                {
                    Some(function) => function as *const std::ffi::c_void,
                    None => std::ptr::null(),
                }
            }
            gpu::vk::GetProcOf::Device(device, name) => {
                match inner
                    .instance
                    .get_device_proc_addr(vk::Device::from_raw(device as _), name)
                {
                    Some(function) => function as *const std::ffi::c_void,
                    None => std::ptr::null(),
                }
            }
        }
    }
}

/// A Skia Ganesh surface backed by the same Vulkan device as `Device`.
#[pyclass(unsendable)]
struct SkiaTarget {
    inner: Arc<DeviceInner>,
    _gpu: Rc<GpuShared>,
    width: u32,
    height: u32,
    surface: skia_safe::Surface,
    context: gpu::DirectContext,
    path: PathBuilder,
    paint: Paint,
    matrix: M44,
    matrix_sx: f32,
    matrix_sy: f32,
    matrix_tx: f32,
    matrix_ty: f32,
    /// GPU snapshot of the static mobjects of the current play.
    static_layer: Option<Image>,
    /// Camera background image, uploaded once per image.
    background_layer: Option<Image>,
    /// Background-image fills keyed by the caller's cache key (the id of a
    /// cached, read-only image-file array), so each file is uploaded once.
    fill_images: HashMap<u64, Image>,
    /// Mipmapped mesh textures keyed by the caller's cache key.
    mesh_images: HashMap<u64, Image>,
}

impl SkiaTarget {
    fn new(inner: Arc<DeviceInner>, gpu: Rc<GpuShared>, width: u32, height: u32) -> PyResult<Self> {
        // Cloning a DirectContext only bumps Skia's reference count; all
        // targets share the one Ganesh context created for this Device.
        let mut context = gpu.context.clone();
        let info = SkImageInfo::new(
            (width as i32, height as i32),
            ColorType::RGBA8888,
            AlphaType::Premul,
            None,
        );
        let surface = gpu::surfaces::render_target(
            &mut context,
            gpu::Budgeted::Yes,
            &info,
            None,
            gpu::SurfaceOrigin::TopLeft,
            None,
            false,
            None,
        )
        .ok_or_else(|| PyRuntimeError::new_err("Skia render_target returned None"))?;
        Ok(Self {
            inner,
            _gpu: gpu,
            width,
            height,
            surface,
            context,
            path: PathBuilder::new(),
            paint: {
                let mut paint = Paint::default();
                paint.set_anti_alias(true);
                paint
            },
            matrix: M44::new_identity(),
            matrix_sx: 1.0,
            matrix_sy: -1.0,
            matrix_tx: 0.0,
            matrix_ty: 0.0,
            static_layer: None,
            background_layer: None,
            fill_images: HashMap::new(),
            mesh_images: HashMap::new(),
        })
    }

    /// Replace every pixel of the surface with `image` at device scale.
    fn blit_full(surface: &mut skia_safe::Surface, image: &Image) {
        let mut paint = Paint::default();
        paint.set_blend_mode(BlendMode::Src);
        let canvas = surface.canvas();
        canvas.save();
        canvas.reset_matrix();
        canvas.draw_image_with_sampling_options(
            image,
            (0, 0),
            SamplingOptions::default(),
            Some(&paint),
        );
        canvas.restore();
    }

    /// Build an RGBA8 unpremultiplied raster image from a C-contiguous buffer.
    /// Drawing it on the Ganesh canvas uploads it once; Skia caches the texture
    /// for as long as the returned `Image` is alive.
    fn raster_image(pixels: &Bound<PyAny>, width: u32, height: u32) -> PyResult<Image> {
        let buffer = PyBuffer::<u8>::get(pixels)
            .map_err(|error| PyValueError::new_err(format!("pixel buffer: {error}")))?;
        let expected = (width * height * 4) as usize;
        if buffer.item_count() != expected {
            return Err(PyValueError::new_err(format!(
                "pixel buffer has {} bytes, expected {expected}",
                buffer.item_count()
            )));
        }
        let slice = buffer
            .as_slice(pixels.py())
            .ok_or_else(|| PyValueError::new_err("pixel buffer must be C-contiguous"))?;
        // SAFETY: ReadOnlyCell<u8> is repr(transparent) over u8 and the buffer
        // stays alive (held by `buffer`) for the duration of the copy below.
        let bytes = unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, expected) };
        let info = SkImageInfo::new(
            (width as i32, height as i32),
            ColorType::RGBA8888,
            AlphaType::Unpremul,
            None,
        );
        Image::from_raster_data(&info, Data::new_copy(bytes), (width * 4) as usize)
            .ok_or_else(|| PyRuntimeError::new_err("Skia Image::from_raster_data failed"))
    }

    /// Image sampling for `ImageMobject.resampling_algorithm`: 0 nearest,
    /// 1 bilinear, 2 bicubic (Catmull-Rom, as PIL's bicubic).
    fn image_sampling(sampling: u8) -> SamplingOptions {
        match sampling {
            0 => SamplingOptions::new(FilterMode::Nearest, MipmapMode::None),
            1 => SamplingOptions::new(FilterMode::Linear, MipmapMode::None),
            _ => SamplingOptions::from(CubicResampler::catmull_rom()),
        }
    }

    /// Copy a C-contiguous float32 buffer of `width` columns into a Vec.
    fn f32_rows(values: &Bound<PyAny>, width: usize, what: &str) -> PyResult<Vec<f32>> {
        let buffer = PyBuffer::<f32>::get(values)
            .map_err(|error| PyValueError::new_err(format!("{what}: {error}")))?;
        if buffer.item_count() % width != 0 {
            return Err(PyValueError::new_err(format!(
                "{what} must have {width} columns"
            )));
        }
        buffer
            .to_vec(values.py())
            .map_err(|error| PyValueError::new_err(format!("{what}: {error}")))
    }
}

#[pymethods]
impl SkiaTarget {
    #[getter]
    fn width(&self) -> u32 {
        self.width
    }

    #[getter]
    fn height(&self) -> u32 {
        self.height
    }

    fn clear_rgba(&mut self, red: f32, green: f32, blue: f32, alpha: f32) {
        let color = Color::from_argb(
            (alpha * 255.0) as u8,
            (red * 255.0) as u8,
            (green * 255.0) as u8,
            (blue * 255.0) as u8,
        );
        self.surface.canvas().clear(color);
    }

    /// Set the user-to-device matrix. `sy` is the y scale; callers pass a
    /// negative value to flip Manim's y-up coordinates into Skia's y-down space.
    fn set_matrix(&mut self, sx: f32, sy: f32, tx: f32, ty: f32) {
        let mut matrix = M44::new_identity();
        matrix.set_row_major(&[
            sx, 0.0, 0.0, tx, 0.0, sy, 0.0, ty, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ]);
        self.matrix = matrix;
        self.matrix_sx = sx;
        self.matrix_sy = sy;
        self.matrix_tx = tx;
        self.matrix_ty = ty;
        self.surface.canvas().set_matrix(&self.matrix);
    }

    fn new_path(&mut self) {
        self.path.reset();
    }

    fn new_sub_path(&mut self) {
        // Cairo's new_sub_path only resets the current point; the following
        // move_to() starts the actual contour in Skia.
    }

    fn move_to(&mut self, x: f32, y: f32) {
        self.path.move_to(Point::new(x, y));
    }

    fn line_to(&mut self, x: f32, y: f32) {
        self.path.line_to(Point::new(x, y));
    }

    fn curve_to(&mut self, x1: f32, y1: f32, x2: f32, y2: f32, x3: f32, y3: f32) {
        self.path
            .cubic_to(Point::new(x1, y1), Point::new(x2, y2), Point::new(x3, y3));
    }

    fn close_path(&mut self) {
        self.path.close();
    }

    /// Replace the current path with a whole cubic-Bezier path in one call.
    ///
    /// `points` is a C-contiguous float64 array of shape (n, k), k >= 2, in
    /// scene coordinates (the canvas matrix maps them to pixels).  Each entry
    /// of `subpaths` is (start, end, closed): points start..end form anchors
    /// and handles a0 h0 h1 a1 a1' h ... with four points per curve, exactly
    /// as Cairo's move_to/curve_to loop consumes them.
    fn set_cubic_path(
        &mut self,
        points: &Bound<PyAny>,
        subpaths: Vec<(usize, usize, bool)>,
    ) -> PyResult<()> {
        let buffer = PyBuffer::<f64>::get(points)
            .map_err(|error| PyValueError::new_err(format!("path points: {error}")))?;
        let shape = buffer.shape().to_vec();
        if shape.len() != 2 || shape[1] < 2 {
            return Err(PyValueError::new_err("path points must have shape (n, 2+)"));
        }
        let (count, stride) = (shape[0], shape[1]);
        let data = buffer
            .as_slice(points.py())
            .ok_or_else(|| PyValueError::new_err("path points must be C-contiguous"))?;
        let at = |index: usize| {
            Point::new(
                data[index * stride].get() as f32,
                data[index * stride + 1].get() as f32,
            )
        };
        self.path.reset();
        for (start, end, closed) in subpaths {
            if end > count || end < start + 4 {
                continue;
            }
            self.path.move_to(at(start));
            let mut index = start;
            while index + 4 <= end {
                self.path
                    .cubic_to(at(index + 1), at(index + 2), at(index + 3));
                index += 4;
            }
            if closed {
                self.path.close();
            }
        }
        Ok(())
    }

    /// Keep the surface as it is now as this play's static background.  The
    /// snapshot stays on the GPU: no flush, no readback, no upload.
    fn capture_static_layer(&mut self) {
        self.static_layer = Some(self.surface.image_snapshot());
    }

    fn clear_static_layer(&mut self) {
        self.static_layer = None;
    }

    /// Overwrite the surface with the static background.  Returns false when
    /// none is captured, so the caller clears to the camera background.
    fn restore_static_layer(&mut self) -> bool {
        match self.static_layer.as_ref() {
            Some(image) => {
                Self::blit_full(&mut self.surface, image);
                true
            }
            None => false,
        }
    }

    /// Upload a full-target camera background image (once per image).
    fn set_background_layer(&mut self, pixels: &Bound<PyAny>) -> PyResult<()> {
        self.background_layer = Some(Self::raster_image(pixels, self.width, self.height)?);
        Ok(())
    }

    /// Overwrite the surface with the camera background image.
    fn draw_background_layer(&mut self) -> PyResult<()> {
        let image = self
            .background_layer
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("no background layer was set"))?;
        Self::blit_full(&mut self.surface, image);
        Ok(())
    }

    fn set_source_rgba(&mut self, red: f32, green: f32, blue: f32, alpha: f32) {
        self.paint.set_shader(None);
        self.paint
            .set_color4f(Color4f::new(red, green, blue, alpha), None);
    }

    /// `stops` is a list of (offset, red, green, blue, alpha), with offset in 0..=1.
    fn set_source_linear_gradient(
        &mut self,
        x0: f32,
        y0: f32,
        x1: f32,
        y1: f32,
        stops: Vec<(f32, f32, f32, f32, f32)>,
    ) {
        if stops.is_empty() {
            return;
        }
        let mut colors = Vec::with_capacity(stops.len());
        let mut positions = Vec::with_capacity(stops.len());
        for (offset, red, green, blue, alpha) in stops {
            colors.push(Color4f::new(red, green, blue, alpha));
            positions.push(offset);
        }
        let gradient_colors =
            GradientColors::new(&colors, Some(positions.as_slice()), TileMode::Clamp, None);
        let gradient = Gradient::new(gradient_colors, Interpolation::default());
        if let Some(shader) = skia_safe::gradient::shaders::linear_gradient(
            (Point::new(x0, y0), Point::new(x1, y1)),
            &gradient,
            None,
        ) {
            self.paint.set_shader(shader);
        }
    }

    fn set_line_width(&mut self, width: f32) {
        self.paint.set_stroke_width(width);
    }

    fn set_line_join(&mut self, join: i32) {
        self.paint.set_stroke_join(match join {
            1 => Join::Round,
            2 => Join::Bevel,
            _ => Join::Miter,
        });
    }

    fn set_line_cap(&mut self, cap: i32) {
        self.paint.set_stroke_cap(match cap {
            1 => Cap::Round,
            2 => Cap::Square,
            _ => Cap::Butt,
        });
    }

    fn fill_preserve(&mut self) {
        let path = self.path.snapshot();
        self.paint.set_style(Style::Fill);
        self.surface.canvas().draw_path(&path, &self.paint);
    }

    fn stroke_preserve(&mut self) {
        let path = self.path.snapshot();
        self.paint.set_style(Style::Stroke);
        self.surface.canvas().draw_path(&path, &self.paint);
    }

    /// Draw a raster RGBA image into an arbitrary device-space quad.  The
    /// image is sampled by Skia's GPU pipeline, so no host-side compositing is
    /// required for ``ImageMobject``.
    /// `sampling` is 0 (nearest), 1 (bilinear) or 2 (bicubic).
    #[allow(clippy::too_many_arguments)]
    fn draw_image_quad(
        &mut self,
        pixels: Vec<u8>,
        image_width: u32,
        image_height: u32,
        quad: Vec<(f32, f32)>,
        alpha: f32,
        sampling: u8,
    ) -> PyResult<()> {
        if quad.len() != 4 {
            return Err(PyValueError::new_err("image quad needs four corners"));
        }
        if image_width == 0
            || image_height == 0
            || pixels.len() != (image_width * image_height * 4) as usize
        {
            return Err(PyValueError::new_err(
                "image pixel buffer has the wrong size",
            ));
        }
        let info = SkImageInfo::new(
            (image_width as i32, image_height as i32),
            ColorType::RGBA8888,
            AlphaType::Unpremul,
            None,
        );
        let image = Image::from_raster_data(
            &info,
            Data::new_copy(pixels.as_slice()),
            (image_width * 4) as usize,
        )
        .ok_or_else(|| PyRuntimeError::new_err("Skia Image::from_raster_data failed"))?;
        let shader = image
            .to_shader(
                Some((TileMode::Clamp, TileMode::Clamp)),
                Self::image_sampling(sampling),
                None,
            )
            .ok_or_else(|| PyRuntimeError::new_err("Skia image shader failed"))?;

        let (w, h) = (image_width as f32, image_height as f32);
        let tex = [
            Point::new(0.0, 0.0),
            Point::new(w, 0.0),
            Point::new(w, h),
            Point::new(0.0, h),
        ];
        let corners = [
            Point::new(quad[0].0, quad[0].1),
            Point::new(quad[1].0, quad[1].1),
            Point::new(quad[2].0, quad[2].1),
            Point::new(quad[3].0, quad[3].1),
        ];
        let positions = [
            corners[0], corners[1], corners[2], corners[0], corners[2], corners[3],
        ];
        let tex_coords = [tex[0], tex[1], tex[2], tex[0], tex[2], tex[3]];
        let colors = [Color::WHITE; 6];
        let vertices = Vertices::new_copy(
            VertexMode::Triangles,
            &positions,
            &tex_coords,
            &colors,
            None,
        );
        let canvas = self.surface.canvas();
        canvas.reset_matrix();
        self.paint.set_shader(shader);
        self.paint.set_style(Style::Fill);
        self.paint.set_color(Color::WHITE);
        self.paint.set_alpha((alpha.clamp(0.0, 1.0) * 255.0) as u8);
        canvas.draw_vertices(&vertices, BlendMode::Modulate, &self.paint);
        self.paint.set_shader(None);
        self.paint.set_alpha(255);
        canvas.set_matrix(&self.matrix);
        Ok(())
    }

    /// Draw another Skia target's surface into a device-space quad without
    /// reading its pixels back to the CPU.  Used by MultiCamera composition.
    fn draw_target_quad(
        &mut self,
        source: &mut SkiaTarget,
        quad: Vec<(f32, f32)>,
        alpha: f32,
    ) -> PyResult<()> {
        if quad.len() != 4 {
            return Err(PyValueError::new_err("target quad needs four corners"));
        }
        // Both surfaces belong to the one DirectContext, which orders the
        // source's pending draws before this read; no flush is needed.
        let image = source.surface.image_snapshot();
        let shader = image
            .to_shader(
                Some((TileMode::Clamp, TileMode::Clamp)),
                SamplingOptions::default(),
                None,
            )
            .ok_or_else(|| PyRuntimeError::new_err("Skia target image shader failed"))?;
        let (w, h) = (source.width as f32, source.height as f32);
        let tex = [
            Point::new(0.0, 0.0),
            Point::new(w, 0.0),
            Point::new(w, h),
            Point::new(0.0, h),
        ];
        let corners = [
            Point::new(quad[0].0, quad[0].1),
            Point::new(quad[1].0, quad[1].1),
            Point::new(quad[2].0, quad[2].1),
            Point::new(quad[3].0, quad[3].1),
        ];
        let positions = [
            corners[0], corners[1], corners[2], corners[0], corners[2], corners[3],
        ];
        let tex_coords = [tex[0], tex[1], tex[2], tex[0], tex[2], tex[3]];
        let colors = [Color::WHITE; 6];
        let vertices = Vertices::new_copy(
            VertexMode::Triangles,
            &positions,
            &tex_coords,
            &colors,
            None,
        );
        let canvas = self.surface.canvas();
        canvas.reset_matrix();
        self.paint.set_shader(shader);
        self.paint.set_style(Style::Fill);
        self.paint.set_color(Color::WHITE);
        self.paint.set_alpha((alpha.clamp(0.0, 1.0) * 255.0) as u8);
        canvas.draw_vertices(&vertices, BlendMode::Modulate, &self.paint);
        self.paint.set_shader(None);
        self.paint.set_alpha(255);
        canvas.set_matrix(&self.matrix);
        Ok(())
    }

    /// Compile a user-supplied SkSL runtime effect and draw it over a
    /// device-space quad.  Syntax errors are reported as Python exceptions
    /// instead of silently falling back to a flat colour.
    fn draw_sksl_quad(
        &mut self,
        source: String,
        uniforms: Vec<(String, Vec<f32>)>,
        quad: Vec<(f32, f32)>,
    ) -> PyResult<()> {
        if quad.len() != 4 {
            return Err(PyValueError::new_err("SkSL quad needs four corners"));
        }
        let effect = RuntimeEffect::make_for_shader(source.as_str(), None)
            .map_err(|error| PyRuntimeError::new_err(format!("SkSL compile failed: {error}")))?;
        let mut builder = RuntimeShaderBuilder::new(effect);
        let mut has_resolution = false;
        for (name, values) in uniforms {
            if name == "u_resolution" {
                has_resolution = true;
            }
            builder
                .set_uniform_float(name.as_str(), values.as_slice())
                .map_err(|error| {
                    PyValueError::new_err(format!("SkSL uniform {name}: {error:?}"))
                })?;
        }
        if !has_resolution {
            // Most full-screen effects declare u_resolution.  Ignoring the
            // error keeps shaders that do not need it valid.
            let _ =
                builder.set_uniform_float("u_resolution", &[self.width as f32, self.height as f32]);
        }
        let shader = builder
            .make_shader(&Matrix::new_identity())
            .ok_or_else(|| PyRuntimeError::new_err("SkSL make_shader returned None"))?;
        let mut clip = PathBuilder::new();
        clip.move_to(Point::new(quad[0].0, quad[0].1));
        for corner in &quad[1..] {
            clip.line_to(Point::new(corner.0, corner.1));
        }
        clip.close();
        let clip = clip.snapshot();
        let canvas = self.surface.canvas();
        canvas.reset_matrix();
        canvas.save();
        canvas.clip_path(&clip, ClipOp::Intersect, true);
        self.paint.set_shader(shader);
        self.paint.set_style(Style::Fill);
        self.paint.set_color(Color::WHITE);
        self.paint.set_alpha(255);
        canvas.draw_rect(
            Rect::new(0.0, 0.0, self.width as f32, self.height as f32),
            &self.paint,
        );
        canvas.restore();
        self.paint.set_shader(None);
        canvas.set_matrix(&self.matrix);
        Ok(())
    }

    /// Use a full-target background image, tinted by an RGBA colour, as the
    /// source of the following fills and strokes (Cairo's `set_source` with an
    /// image pattern).  The local matrix maps the path's scene coordinates onto
    /// the image's device pixels, keeping the mask and background on the GPU.
    /// `key` identifies an immutable pixel source (a cached image file) whose
    /// upload is kept; pass None for pixels that may change between calls.
    #[pyo3(signature = (pixels, red, green, blue, alpha, key=None))]
    fn set_source_image(
        &mut self,
        pixels: &Bound<PyAny>,
        red: f32,
        green: f32,
        blue: f32,
        alpha: f32,
        key: Option<u64>,
    ) -> PyResult<()> {
        let image = match key {
            Some(key) => match self.fill_images.get(&key) {
                Some(image) => image.clone(),
                None => {
                    let image = Self::raster_image(pixels, self.width, self.height)?;
                    self.fill_images.insert(key, image.clone());
                    image
                }
            },
            None => Self::raster_image(pixels, self.width, self.height)?,
        };
        // The shader's local matrix takes image pixels to the path's scene
        // coordinates: the inverse of the canvas matrix.
        let local = Matrix::new_all(
            self.matrix_sx,
            0.0,
            self.matrix_tx,
            0.0,
            self.matrix_sy,
            self.matrix_ty,
            0.0,
            0.0,
            1.0,
        )
        .invert()
        .ok_or_else(|| PyValueError::new_err("the canvas matrix is not invertible"))?;
        let image_shader = image
            .to_shader(
                Some((TileMode::Clamp, TileMode::Clamp)),
                SamplingOptions::default(),
                Some(&local),
            )
            .ok_or_else(|| PyRuntimeError::new_err("Skia background shader failed"))?;
        let tint = skia_safe::shaders::color(Color4f::new(red, green, blue, alpha).to_color());
        self.paint.set_shader(skia_safe::shaders::blend(
            BlendMode::Modulate,
            image_shader,
            tint,
        ));
        self.paint.set_color(Color::WHITE);
        Ok(())
    }

    /// Draw textured triangles through the current canvas matrix.
    ///
    /// `positions` and `uvs` are float32 (n, 2) arrays, `uvs` in texture
    /// pixels, and `colors` float32 (n, 4), three rows per triangle.  Each
    /// texel is multiplied by its vertex colour, and the triangles blend over
    /// each other in order inside one layer composited with `alpha`.  With
    /// `occlude` (an RGBA colour) the triangles first replace what is under
    /// them with that colour.  The texture is uploaded with mipmaps once per
    /// `key`.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (key, pixels, image_width, image_height, positions, uvs, colors, alpha, occlude=None))]
    fn draw_textured_mesh(
        &mut self,
        key: u64,
        pixels: &Bound<PyAny>,
        image_width: u32,
        image_height: u32,
        positions: &Bound<PyAny>,
        uvs: &Bound<PyAny>,
        colors: &Bound<PyAny>,
        alpha: f32,
        occlude: Option<(f32, f32, f32, f32)>,
    ) -> PyResult<()> {
        let image = match self.mesh_images.get(&key) {
            Some(image) => image.clone(),
            None => {
                let raster = Self::raster_image(pixels, image_width, image_height)?;
                let image = raster.with_default_mipmaps().unwrap_or(raster);
                self.mesh_images.insert(key, image.clone());
                image
            }
        };
        let positions = Self::f32_rows(positions, 2, "mesh positions")?;
        let uvs = Self::f32_rows(uvs, 2, "mesh uvs")?;
        let colors = Self::f32_rows(colors, 4, "mesh colors")?;
        let count = positions.len() / 2;
        if uvs.len() != count * 2 || colors.len() != count * 4 || count % 3 != 0 {
            return Err(PyValueError::new_err(
                "mesh needs matching positions, uvs and colors, three per triangle",
            ));
        }
        if count == 0 {
            return Ok(());
        }
        let points: Vec<Point> = positions
            .chunks_exact(2)
            .map(|p| Point::new(p[0], p[1]))
            .collect();
        let texs: Vec<Point> = uvs
            .chunks_exact(2)
            .map(|p| Point::new(p[0], p[1]))
            .collect();
        let vertex_colors: Vec<Color> = colors
            .chunks_exact(4)
            .map(|c| Color4f::new(c[0], c[1], c[2], c[3]).to_color())
            .collect();
        let vertices =
            Vertices::new_copy(VertexMode::Triangles, &points, &texs, &vertex_colors, None);
        let shader = image
            .to_shader(
                Some((TileMode::Clamp, TileMode::Clamp)),
                SamplingOptions::new(FilterMode::Linear, MipmapMode::Linear),
                None,
            )
            .ok_or_else(|| PyRuntimeError::new_err("Skia mesh texture shader failed"))?;
        let mut paint = Paint::default();
        paint.set_anti_alias(true);
        paint.set_shader(shader);
        let canvas = self.surface.canvas();
        if let Some((red, green, blue, opacity)) = occlude {
            let mut clear = Paint::default();
            clear.set_shader(skia_safe::shaders::color(
                Color4f::new(red, green, blue, opacity).to_color(),
            ));
            clear.set_blend_mode(BlendMode::Src);
            canvas.draw_vertices(&vertices, BlendMode::Src, &clear);
        }
        let layered = alpha < 1.0;
        if layered {
            canvas.save_layer_alpha(None::<Rect>, (alpha.clamp(0.0, 1.0) * 255.0).round() as u32);
        }
        canvas.draw_vertices(&vertices, BlendMode::Modulate, &paint);
        if layered {
            canvas.restore();
        }
        Ok(())
    }

    /// Draw point-cloud pixels directly into the target.  ``points`` contains
    /// (x, y, r, g, b, a) tuples in device pixel coordinates, and ``half_size``
    /// is the half extent of the square used for a point.
    fn draw_point_pixels(&mut self, points: Vec<(f32, f32, u8, u8, u8, u8)>, half_size: f32) {
        if points.is_empty() {
            return;
        }
        let half = half_size.max(0.5);
        let mut positions = Vec::with_capacity(points.len() * 6);
        let mut colors = Vec::with_capacity(points.len() * 6);
        for (x, y, r, g, b, a) in points {
            let color = Color::from_argb(a, r, g, b);
            let corners = [
                (x - half, y - half),
                (x + half, y - half),
                (x + half, y + half),
                (x - half, y + half),
            ];
            for index in [0, 1, 2, 0, 2, 3] {
                positions.push(Point::new(corners[index].0, corners[index].1));
                colors.push(color);
            }
        }
        let tex_coords = vec![Point::new(0.0, 0.0); positions.len()];
        let vertices = Vertices::new_copy(
            VertexMode::Triangles,
            &positions,
            &tex_coords,
            &colors,
            None,
        );
        let canvas = self.surface.canvas();
        canvas.reset_matrix();
        self.paint.set_shader(None);
        self.paint.set_style(Style::Fill);
        self.paint.set_color(Color::WHITE);
        self.paint.set_alpha(255);
        self.paint.set_anti_alias(false);
        canvas.draw_vertices(&vertices, BlendMode::Modulate, &self.paint);
        self.paint.set_anti_alias(true);
        canvas.set_matrix(&self.matrix);
    }

    /// Flush Ganesh and copy the RGBA surface back to host memory.
    fn commit(&mut self) -> PyResult<Vec<u8>> {
        self.context
            .flush_and_submit()
            .map_err(|error| PyRuntimeError::new_err(format!("Skia flush failed: {error:?}")))?;
        let info = SkImageInfo::new(
            (self.width as i32, self.height as i32),
            ColorType::RGBA8888,
            AlphaType::Unpremul,
            None,
        );
        let mut pixels = vec![0u8; (self.width * self.height * 4) as usize];
        if !self
            .surface
            .read_pixels(&info, &mut pixels, (self.width * 4) as usize, (0, 0))
        {
            return Err(PyRuntimeError::new_err("Skia surface read_pixels failed"));
        }
        Ok(pixels)
    }

    /// Flush Ganesh and write the RGBA surface directly into a writable
    /// Python buffer (normally a C-contiguous NumPy frame array).  This avoids
    /// the temporary Rust vector and the second Python-side copy done by
    /// ``commit()``.
    fn commit_into(&mut self, target: &Bound<PyAny>) -> PyResult<()> {
        self.context
            .flush_and_submit()
            .map_err(|error| PyRuntimeError::new_err(format!("Skia flush failed: {error:?}")))?;
        let buffer = PyBuffer::<u8>::get(target)
            .map_err(|error| PyRuntimeError::new_err(format!("readback buffer: {error}")))?;
        if buffer.readonly() {
            return Err(PyValueError::new_err("readback buffer is read-only"));
        }
        let expected = (self.width * self.height * 4) as usize;
        if buffer.item_count() != expected {
            return Err(PyValueError::new_err(format!(
                "readback buffer has {} bytes, expected {expected}",
                buffer.item_count()
            )));
        }
        let py = target.py();
        let slice = buffer
            .as_mut_slice(py)
            .ok_or_else(|| PyValueError::new_err("readback buffer must be C-contiguous"))?;
        let data = unsafe { std::slice::from_raw_parts_mut(slice.as_ptr() as *mut u8, expected) };
        let info = SkImageInfo::new(
            (self.width as i32, self.height as i32),
            ColorType::RGBA8888,
            AlphaType::Unpremul,
            None,
        );
        if !self
            .surface
            .read_pixels(&info, data, (self.width * 4) as usize, (0, 0))
        {
            return Err(PyRuntimeError::new_err("Skia surface read_pixels failed"));
        }
        Ok(())
    }

    /// Upload host RGBA pixels into the GPU surface.  Reads the caller's
    /// C-contiguous buffer in place; the draw is submitted with the frame.
    fn load_pixels(&mut self, pixels: &Bound<PyAny>) -> PyResult<()> {
        let buffer = PyBuffer::<u8>::get(pixels)
            .map_err(|error| PyValueError::new_err(format!("pixel buffer: {error}")))?;
        let expected = (self.width * self.height * 4) as usize;
        if buffer.item_count() != expected {
            return Err(PyValueError::new_err("pixel buffer has the wrong size"));
        }
        let slice = buffer
            .as_slice(pixels.py())
            .ok_or_else(|| PyValueError::new_err("pixel buffer must be C-contiguous"))?;
        // SAFETY: see raster_image; Skia only reads from the slice.
        let bytes = unsafe { std::slice::from_raw_parts_mut(slice.as_ptr() as *mut u8, expected) };
        let info = SkImageInfo::new(
            (self.width as i32, self.height as i32),
            ColorType::RGBA8888,
            AlphaType::Unpremul,
            None,
        );
        let pixmap = skia_safe::Pixmap::new(&info, bytes, (self.width * 4) as usize)
            .ok_or_else(|| PyRuntimeError::new_err("Skia Pixmap::new failed"))?;
        self.surface.write_pixels_from_pixmap(&pixmap, (0, 0));
        Ok(())
    }
}

#[pymodule]
fn manim_vk(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<Device>()?;
    module.add_class::<SkiaTarget>()?;
    module.add_class::<window::NativeWindow>()?;
    module.add("__version__", "0.1.0")?;
    Ok(())
}
