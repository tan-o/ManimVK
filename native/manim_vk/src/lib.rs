//! Native Vulkan core for Manim.
//!
//! `Device` owns the Vulkan instance/device and one shared Skia Ganesh
//! context; `SkiaTarget` is a GPU render target that draws Manim's classic
//! mobjects (paths, images, point clouds, SkSL effects) and reads frames back.

mod depth;
mod display_list;
mod meshes;
mod textures;
mod window;

use std::cell::RefCell;
use std::collections::VecDeque;
use std::ffi::CString;
use std::rc::Rc;
use std::sync::Arc;

use ash::vk;
use pyo3::buffer::PyBuffer;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use skia_safe::image::{AsyncReadResult, RescaleGamma, RescaleMode};
use skia_safe::paint::Style;
use skia_safe::runtime_effect::RuntimeShaderBuilder;
use skia_safe::vertices::{VertexMode, Vertices};
use skia_safe::{
    gpu, AlphaType, BlendMode, ClipOp, Color, Color4f, ColorType, CubicResampler, Data, FilterMode,
    IRect, Image, ImageInfo as SkImageInfo, Matrix, MipmapMode, Paint, PathBuilder, Point, Rect,
    RuntimeEffect, SamplingOptions, TileMode, YUVColorSpace, M44,
};
use textures::TextureCache;

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
    /// Never read: the context points into it, so it lives as long.
    _backend: Box<gpu::vk::BackendContext<'static>>,
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
    Ok(Rc::new(GpuShared {
        _backend: backend,
        context,
    }))
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
    /// Never read: keeps the Vulkan device alive while the surface exists.
    _inner: Arc<DeviceInner>,
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
    /// Textures of keyed, immutable pixel arrays: image files, ImageMobject
    /// sources and mesh images.
    textures: TextureCache,
    /// The surface mesh shaders, compiled on first use.
    mesh_spec: Option<skia_safe::mesh::MeshSpecification>,
    /// The Vulkan pass that gives meshes their exact depth, made on first
    /// use; it holds the device itself and frees its objects before it.
    depth_pass: Option<depth::DepthPass>,
    /// The last transient background image (see `set_source_image`).
    transient_image: Option<(u64, Image)>,
    /// YUV reads in flight, oldest first, by ticket.
    yuv_reads: VecDeque<(u64, YuvSlot)>,
    next_ticket: u64,
}

/// Where Skia's callback leaves a YUV read: unset until it is called, then
/// the result, which is None when the read failed.
type YuvSlot = Rc<RefCell<Option<Option<AsyncReadResult>>>>;

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
            _inner: inner,
            _gpu: gpu,
            width,
            height,
            surface,
            context,
            path: PathBuilder::new(),
            paint: {
                let mut paint = Paint::default();
                paint.set_anti_alias(true);
                paint.set_stroke_miter(display_list::MITER_LIMIT);
                paint
            },
            matrix: M44::new_identity(),
            matrix_sx: 1.0,
            matrix_sy: -1.0,
            matrix_tx: 0.0,
            matrix_ty: 0.0,
            static_layer: None,
            background_layer: None,
            textures: TextureCache::default(),
            transient_image: None,
            mesh_spec: None,
            depth_pass: None,
            yuv_reads: VecDeque::new(),
            next_ticket: 0,
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
        skia_safe::images::raster_from_data(&info, Data::new_copy(bytes), (width * 4) as usize)
            .ok_or_else(|| PyRuntimeError::new_err("Skia images::raster_from_data failed"))
    }

    /// The texture for keyed pixels, or a one-off upload without a key.
    fn texture(
        &mut self,
        key: Option<u64>,
        pixels: &Bound<PyAny>,
        width: u32,
        height: u32,
    ) -> PyResult<Image> {
        match key {
            Some(key) => self.textures.get(&mut self.context, key, false, || {
                Self::raster_image(pixels, width, height)
            }),
            None => Self::raster_image(pixels, width, height),
        }
    }

    /// Draw `image` stretched over the device-space quad `corners` (top-left,
    /// top-right, bottom-right, bottom-left) with the canvas's current matrix.
    fn draw_image_vertices(
        &mut self,
        image: &Image,
        corners: &[Point],
        alpha: f32,
        sampling: u8,
        blend: BlendMode,
    ) -> PyResult<()> {
        let shader = image
            .to_shader(
                Some((TileMode::Clamp, TileMode::Clamp)),
                Self::image_sampling(sampling),
                None,
            )
            .ok_or_else(|| PyRuntimeError::new_err("Skia image shader failed"))?;
        let (w, h) = (image.width() as f32, image.height() as f32);
        let tex = [
            Point::new(0.0, 0.0),
            Point::new(w, 0.0),
            Point::new(w, h),
            Point::new(0.0, h),
        ];
        let order = [0, 1, 2, 0, 2, 3];
        let positions = order.map(|index| corners[index]);
        let tex_coords = order.map(|index| tex[index]);
        let vertices = Vertices::new_copy(
            VertexMode::Triangles,
            &positions,
            &tex_coords,
            &[Color::WHITE; 6],
            None,
        );
        let mut paint = Paint::default();
        paint.set_anti_alias(true);
        paint.set_shader(shader);
        paint.set_alpha_f(alpha.clamp(0.0, 1.0));
        paint.set_blend_mode(blend);
        self.surface
            .canvas()
            .draw_vertices(&vertices, BlendMode::Modulate, &paint);
        Ok(())
    }

    /// Image sampling for `ImageMobject.resampling_algorithm`: 0 nearest,
    /// 1 bilinear, 2 bicubic (Catmull-Rom, as PIL's resize).
    fn image_sampling(sampling: u8) -> SamplingOptions {
        match sampling {
            0 => SamplingOptions::new(FilterMode::Nearest, MipmapMode::None),
            1 => SamplingOptions::new(FilterMode::Linear, MipmapMode::None),
            _ => SamplingOptions::from(CubicResampler::catmull_rom()),
        }
    }
}

#[pymethods]
impl SkiaTarget {
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
        self.matrix = M44::row_major(&[
            sx, 0.0, 0.0, tx, 0.0, sy, 0.0, ty, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ]);
        self.matrix_sx = sx;
        self.matrix_sy = sy;
        self.matrix_tx = tx;
        self.matrix_ty = ty;
        self.surface.canvas().set_matrix(&self.matrix);
    }

    /// Replace the current path with a whole cubic-Bezier path in one call.
    ///
    /// `points` is a C-contiguous float64 array of shape (n, k), k >= 2, in
    /// scene coordinates (the canvas matrix maps them to pixels).  `subpaths`
    /// is `(k, 3)` int64 rows of (start, end, closed): points start..end form
    /// anchors and handles a0 h0 h1 a1 a1' h ... with four points per curve.
    fn set_cubic_path(&mut self, points: &Bound<PyAny>, subpaths: &Bound<PyAny>) -> PyResult<()> {
        let subpaths = display_list::subpath_table(subpaths)?;
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
        let points: Vec<Point> = data
            .chunks_exact(stride)
            .take(count)
            .map(|row| Point::new(row[0].get() as f32, row[1].get() as f32))
            .collect();
        self.path.reset();
        for (start, end, closed) in subpaths {
            display_list::add_subpath(&mut self.path, &points, start, end, closed);
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

    /// Draw a frame's VMobjects and textured meshes, whole or as depth
    /// pieces, in one call; see `display_list::draw`.  The canvas matrix maps
    /// their scene points.  Mesh textures, with mipmaps, are kept per key.
    fn draw_display_list<'py>(
        &mut self,
        objects: Vec<display_list::ObjectSpec<'py>>,
        meshes: Vec<meshes::MeshSpec<'py>>,
        runs: &Bound<'py, PyAny>,
        parts: &Bound<'py, PyAny>,
    ) -> PyResult<()> {
        let objects = objects
            .into_iter()
            .map(display_list::object)
            .collect::<PyResult<Vec<_>>>()?;
        let mut prepared = Vec::with_capacity(meshes.len());
        for spec in &meshes {
            prepared.push(meshes::Mesh::read(spec, |(key, pixels, width, height)| {
                self.textures.get(&mut self.context, *key, true, || {
                    Self::raster_image(pixels, *width, *height)
                })
            })?);
        }
        let to_device = [
            self.matrix_sx,
            self.matrix_sy,
            self.matrix_tx,
            self.matrix_ty,
        ];
        let range = prepared
            .iter()
            .filter_map(meshes::Mesh::depth_range)
            .reduce(|a, b| (a.0.min(b.0), a.1.max(b.1)));
        let mut depth_vertices = Vec::new();
        if let Some(range) = range {
            for mesh in &mut prepared {
                mesh.normalise(range);
                mesh.depth_vertices(to_device, &mut depth_vertices);
            }
        }
        let has_depth = !depth_vertices.is_empty();
        let depth = if has_depth {
            // Skia's recorded work, which may read the previous depth image,
            // is submitted before the pass that overwrites it.
            self.context.flush_and_submit().map_err(|error| {
                PyRuntimeError::new_err(format!("Skia flush failed: {error:?}"))
            })?;
            if self.depth_pass.is_none() {
                self.depth_pass = Some(depth::DepthPass::new(
                    self._inner.clone(),
                    self.width,
                    self.height,
                )?);
            }
            let pass = self.depth_pass.as_mut().expect("created above");
            pass.render(&depth_vertices)?;
            pass.image(&mut self.context)?
                .to_raw_shader(
                    Some((TileMode::Clamp, TileMode::Clamp)),
                    SamplingOptions::new(FilterMode::Nearest, MipmapMode::None),
                    None,
                )
                .ok_or_else(|| PyRuntimeError::new_err("depth image shader failed"))?
        } else {
            skia_safe::shaders::color(Color::TRANSPARENT)
        };
        if self.mesh_spec.is_none() && !prepared.is_empty() {
            self.mesh_spec = Some(meshes::specification()?);
        }
        let frame = meshes::Frame {
            spec: self.mesh_spec.clone(),
            depth,
            has_depth,
            to_device,
        };
        display_list::draw(
            self.surface.canvas(),
            &mut self.context,
            &objects,
            &prepared,
            &frame,
            runs,
            parts,
        )
    }

    fn set_line_width(&mut self, width: f32) {
        self.paint.set_stroke_width(width);
    }

    fn set_line_join(&mut self, join: i32) {
        self.paint.set_stroke_join(display_list::join_of(join));
    }

    fn set_line_cap(&mut self, cap: i32) {
        self.paint.set_stroke_cap(display_list::cap_of(cap));
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

    /// Draw an RGBA image into an arbitrary device-space quad, sampled by
    /// Skia's GPU pipeline.  `key` names immutable pixels whose texture is
    /// kept across frames (see `textures`); None uploads `pixels` for this
    /// draw only.  `sampling` is 0 (nearest), 1 (bilinear) or 2 (bicubic).
    ///
    /// `mix`, `(key, pixels, weight)` for a second image of the same size,
    /// blends it over the first: inside a layer drawn with `alpha`, the
    /// first image is added with weight `1 - weight` and the second with
    /// `weight`, a premultiplied lerp of the two.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (key, pixels, image_width, image_height, quad, alpha, sampling, mix=None))]
    fn draw_image_quad(
        &mut self,
        key: Option<u64>,
        pixels: &Bound<PyAny>,
        image_width: u32,
        image_height: u32,
        quad: Vec<(f32, f32)>,
        alpha: f32,
        sampling: u8,
        mix: Option<(Option<u64>, PyObject, f32)>,
    ) -> PyResult<()> {
        if quad.len() != 4 {
            return Err(PyValueError::new_err("image quad needs four corners"));
        }
        let corners: Vec<Point> = quad.iter().map(|&(x, y)| Point::new(x, y)).collect();
        let image = self.texture(key, pixels, image_width, image_height)?;
        let other = match &mix {
            Some((mix_key, mix_pixels, weight)) => Some((
                self.texture(
                    *mix_key,
                    mix_pixels.bind(pixels.py()),
                    image_width,
                    image_height,
                )?,
                weight.clamp(0.0, 1.0),
            )),
            None => None,
        };
        let canvas = self.surface.canvas();
        let depth = canvas.save();
        canvas.reset_matrix();
        let result = match other {
            None => self.draw_image_vertices(&image, &corners, alpha, sampling, BlendMode::SrcOver),
            Some((other, weight)) => {
                self.surface
                    .canvas()
                    .save_layer_alpha_f(Rect::bounds(&corners), alpha.clamp(0.0, 1.0));
                self.draw_image_vertices(&image, &corners, 1.0 - weight, sampling, BlendMode::Plus)
                    .and_then(|()| {
                        self.draw_image_vertices(
                            &other,
                            &corners,
                            weight,
                            sampling,
                            BlendMode::Plus,
                        )
                    })
            }
        };
        // Also pops the mix layer, if there is one.
        self.surface.canvas().restore_to_count(depth);
        result
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
        // The source's frame is complete once it is composited: it is never
        // read back itself, so its textures age here.
        source.textures.end_frame();
        // Both surfaces belong to the one DirectContext, which orders the
        // source's pending draws before this read; no flush is needed.
        let image = source.surface.image_snapshot();
        let shader = image
            .to_shader(
                Some((TileMode::Clamp, TileMode::Clamp)),
                SamplingOptions::from(CubicResampler::catmull_rom()),
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
    /// `key` names an immutable pixel source (a cached image file) whose
    /// texture is kept; pass None for pixels that may change between calls.
    ///
    /// With `transient`, `key` names pixels that are only valid for a while
    /// (one batch of a frame): the upload is kept in a single slot and reused
    /// while the same key comes back, instead of being cached for good.
    #[pyo3(signature = (pixels, red, green, blue, alpha, key=None, transient=false))]
    #[allow(clippy::too_many_arguments)]
    fn set_source_image(
        &mut self,
        pixels: &Bound<PyAny>,
        red: f32,
        green: f32,
        blue: f32,
        alpha: f32,
        key: Option<u64>,
        transient: bool,
    ) -> PyResult<()> {
        let image = match (key, transient) {
            (Some(key), true) => match &self.transient_image {
                Some((cached_key, image)) if *cached_key == key => image.clone(),
                _ => {
                    let image = Self::raster_image(pixels, self.width, self.height)?;
                    self.transient_image = Some((key, image.clone()));
                    image
                }
            },
            (Some(key), false) => {
                let (width, height) = (self.width, self.height);
                self.textures.get(&mut self.context, key, false, || {
                    Self::raster_image(pixels, width, height)
                })?
            }
            (None, _) => Self::raster_image(pixels, self.width, self.height)?,
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
                SamplingOptions::new(FilterMode::Linear, MipmapMode::None),
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

    /// Draw point-cloud pixels directly into the target: ``centers`` is an
    /// ``(n, 2)`` float32 array of device pixel coordinates, ``colors`` an
    /// ``(n, 4)`` uint8 RGBA array, and ``half_size`` the half extent of the
    /// square drawn for a point.
    fn draw_point_pixels(
        &mut self,
        centers: &Bound<'_, PyAny>,
        colors: &Bound<'_, PyAny>,
        half_size: f32,
    ) -> PyResult<()> {
        let centers = PyBuffer::<f32>::get(centers)?;
        let rgba = PyBuffer::<u8>::get(colors)?;
        if centers.shape().len() != 2
            || centers.shape()[1] != 2
            || rgba.shape().len() != 2
            || rgba.shape()[1] != 4
            || rgba.shape()[0] != centers.shape()[0]
        {
            return Err(PyValueError::new_err(
                "point pixels need (n, 2) float32 centers and (n, 4) uint8 colors",
            ));
        }
        let py = colors.py();
        let centers = centers.to_vec(py)?;
        let rgba = rgba.to_vec(py)?;
        if centers.is_empty() {
            return Ok(());
        }
        let half = half_size.max(0.5);
        let count = centers.len() / 2;
        let mut positions = Vec::with_capacity(count * 6);
        let mut colors = Vec::with_capacity(count * 6);
        for (center, c) in centers.chunks_exact(2).zip(rgba.chunks_exact(4)) {
            let (x, y) = (center[0], center[1]);
            let color = Color::from_argb(c[3], c[0], c[1], c[2]);
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
        Ok(())
    }

    /// Start reading the surface as I420 video without waiting for the GPU;
    /// returns a ticket for `collect_yuv420_into`.
    ///
    /// Skia converts on the GPU (`asyncRescaleAndReadPixelsYUV420`, as
    /// Chrome's video capture does): BT.709 limited range, 1.5 instead of 4
    /// bytes per pixel, and the encoder skips its own RGB-to-YUV conversion.
    /// The read is ordered before later draws, so the next frame can be
    /// drawn while this one is in flight.  The surface's colours are read as
    /// they are: use it for opaque frames.
    fn request_yuv420(&mut self) -> PyResult<u64> {
        if !self.width.is_multiple_of(2) || !self.height.is_multiple_of(2) {
            return Err(PyValueError::new_err(
                "YUV 4:2:0 frames need an even width and height",
            ));
        }
        self.textures.end_frame();
        let slot: YuvSlot = Rc::new(RefCell::new(None));
        let callback_slot = slot.clone();
        let (width, height) = (self.width as i32, self.height as i32);
        self.surface.async_rescale_and_read_pixels_yuv420(
            YUVColorSpace::Rec709_Limited,
            None,
            IRect::from_wh(width, height),
            (width, height),
            RescaleGamma::Src,
            RescaleMode::Nearest,
            move |read| *callback_slot.borrow_mut() = Some(read),
        );
        self.context
            .flush_and_submit()
            .map_err(|error| PyRuntimeError::new_err(format!("Skia flush failed: {error:?}")))?;
        self.next_ticket += 1;
        self.yuv_reads.push_back((self.next_ticket, slot));
        Ok(self.next_ticket)
    }

    /// Wait for a read started by `request_yuv420` and copy it into a
    /// writable, C-contiguous uint8 buffer of `width * height * 3 / 2`
    /// bytes: the Y plane, then the half-size U and V planes.
    ///
    /// Collect before the drawing of a later frame is submitted: a read that
    /// is not done yet is waited for with a CPU sync on the work submitted
    /// so far.
    fn collect_yuv420_into(&mut self, ticket: u64, target: &Bound<PyAny>) -> PyResult<()> {
        let index = self
            .yuv_reads
            .iter()
            .position(|(pending, _)| *pending == ticket)
            .ok_or_else(|| PyValueError::new_err(format!("no YUV read {ticket} in flight")))?;
        let (_, slot) = self.yuv_reads.remove(index).expect("index is in range");
        self.context.check_async_work_completion();
        if slot.borrow().is_none() {
            self.context.submit(gpu::SyncCpu::Yes);
            self.context.check_async_work_completion();
        }
        let read = slot
            .borrow_mut()
            .take()
            .flatten()
            .ok_or_else(|| PyRuntimeError::new_err("Skia YUV 4:2:0 readback failed"))?;
        let (width, height) = (self.width as usize, self.height as usize);
        let buffer = PyBuffer::<u8>::get(target)
            .map_err(|error| PyRuntimeError::new_err(format!("readback buffer: {error}")))?;
        if buffer.readonly() {
            return Err(PyValueError::new_err("readback buffer is read-only"));
        }
        let expected = width * height * 3 / 2;
        if buffer.item_count() != expected {
            return Err(PyValueError::new_err(format!(
                "readback buffer has {} bytes, expected {expected}",
                buffer.item_count()
            )));
        }
        let slice = buffer
            .as_mut_slice(target.py())
            .ok_or_else(|| PyValueError::new_err("readback buffer must be C-contiguous"))?;
        // SAFETY: Cell<u8> is repr(transparent) over u8; the buffer is held by
        // `buffer` and nothing else writes to it during this call.
        let data = unsafe { std::slice::from_raw_parts_mut(slice.as_ptr() as *mut u8, expected) };
        let mut offset = 0;
        for (plane, (plane_width, plane_height)) in [
            (width, height),
            (width / 2, height / 2),
            (width / 2, height / 2),
        ]
        .into_iter()
        .enumerate()
        {
            let stride = read.row_bytes(plane);
            // SAFETY: Skia's result holds `plane_height` rows of `stride`
            // bytes for this plane while `read` is alive.
            let source = unsafe {
                std::slice::from_raw_parts(read.data(plane) as *const u8, stride * plane_height)
            };
            for (row, line) in source.chunks_exact(stride).enumerate() {
                let start = offset + row * plane_width;
                data[start..start + plane_width].copy_from_slice(&line[..plane_width]);
            }
            offset += plane_width * plane_height;
        }
        Ok(())
    }

    /// Flush Ganesh and write the RGBA surface directly into a writable
    /// Python buffer (normally a C-contiguous NumPy frame array).  This avoids
    /// the temporary Rust vector and the second Python-side copy done by
    /// ``commit()``.
    fn commit_into(&mut self, target: &Bound<PyAny>) -> PyResult<()> {
        self.textures.end_frame();
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
}

/// The interface level `manim._native` expects: bump it, there and here,
/// whenever a method is added or changes its signature or behaviour.
const API_VERSION: u32 = 5;

#[pymodule]
fn manim_vk(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<Device>()?;
    module.add_class::<SkiaTarget>()?;
    module.add_class::<window::NativeWindow>()?;
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    module.add("API_VERSION", API_VERSION)?;
    Ok(())
}
