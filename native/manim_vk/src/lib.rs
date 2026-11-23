//! Minimal native Vulkan core for Manim.
//!
//! Phase P1 of the migration plan: create a device, allocate an offscreen
//! RGBA8 target, clear it, and read the pixels back.  This module is
//! deliberately small; render passes and pipelines arrive in later phases.

mod window;

use std::ffi::CString;
use std::sync::Arc;

use ash::vk;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use skia_safe::{
    AlphaType, BlendMode, Color, Color4f, ColorType, Font, FontMgr, FontStyle, ImageInfo as SkImageInfo, Matrix,
    Paint, PathBuilder, Point, Rect, RuntimeEffect, Shader, TileMode, M44, gpu,
};
use skia_safe::runtime_effect::RuntimeShaderBuilder;
use skia_safe::vertices::{VertexMode, Vertices};
use skia_safe::gradient::{Colors as GradientColors, Gradient, Interpolation};
use skia_safe::paint::{Cap, Join, Style};

struct DeviceInner {
    entry: ash::Entry,
    instance: ash::Instance,
    physical_device: vk::PhysicalDevice,
    device: ash::Device,
    queue: vk::Queue,
    queue_family: u32,
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    fence: vk::Fence,
    device_name: String,
}

impl Drop for DeviceInner {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_fence(self.fence, None);
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
        let _ = &self.entry;
    }
}

fn vk_err(context: &str, result: vk::Result) -> PyErr {
    PyRuntimeError::new_err(format!("{context}: {result:?}"))
}

fn find_memory_type(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    type_bits: u32,
    required: vk::MemoryPropertyFlags,
) -> PyResult<u32> {
    let properties = unsafe { instance.get_physical_device_memory_properties(physical_device) };
    for index in 0..properties.memory_type_count {
        let supported = (type_bits & (1 << index)) != 0;
        let flags = properties.memory_types[index as usize].property_flags;
        if supported && flags.contains(required) {
            return Ok(index);
        }
    }
    Err(PyRuntimeError::new_err(
        "no Vulkan memory type satisfies the requested properties",
    ))
}

#[pyclass]
struct Device {
    inner: Arc<DeviceInner>,
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
        let engine_name = CString::new("manim_vk").unwrap();
        let app_info = vk::ApplicationInfo::default()
            .application_name(&app_name)
            .engine_name(&engine_name)
            .api_version(vk::make_api_version(0, 1, 3, 0));

        let mut layer_names: Vec<CString> = Vec::new();
        if validation {
            layer_names.push(CString::new("VK_LAYER_KHRONOS_validation").unwrap());
        }
        let layer_refs: Vec<*const std::ffi::c_char> = layer_names.iter().map(|name| name.as_ptr()).collect();
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
                &properties.device_name.iter().map(|&c| c as u8).collect::<Vec<u8>>(),
            )
            .trim_end_matches('\0')
            .to_string();
            let families = unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
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
            &properties.device_name.iter().map(|&c| c as u8).collect::<Vec<u8>>(),
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

        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(queue_family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let command_pool = unsafe { device.create_command_pool(&pool_info, None) }
            .map_err(|result| vk_err("vkCreateCommandPool", result))?;
        let allocate = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let command_buffers = unsafe { device.allocate_command_buffers(&allocate) }
            .map_err(|result| vk_err("vkAllocateCommandBuffers", result))?;
        let fence_info = vk::FenceCreateInfo::default();
        let fence = unsafe { device.create_fence(&fence_info, None) }
            .map_err(|result| vk_err("vkCreateFence", result))?;

        Ok(Self {
            inner: Arc::new(DeviceInner {
                entry,
                instance,
                physical_device,
                device,
                queue,
                queue_family,
                command_pool,
                command_buffer: command_buffers[0],
                fence,
                device_name,
            }),
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

    fn create_target(&self, width: u32, height: u32) -> PyResult<Target> {
        if width == 0 || height == 0 {
            return Err(PyValueError::new_err("target dimensions must be positive"));
        }
        Target::new(self.inner.clone(), width, height)
    }

    fn create_skia_target(&self, width: u32, height: u32) -> PyResult<SkiaTarget> {
        if width == 0 || height == 0 {
            return Err(PyValueError::new_err("target dimensions must be positive"));
        }
        SkiaTarget::new(self.inner.clone(), width, height)
    }
}

#[pyclass]
struct Target {
    inner: Arc<DeviceInner>,
    width: u32,
    height: u32,
    image: vk::Image,
    image_memory: vk::DeviceMemory,
    readback_buffer: vk::Buffer,
    readback_memory: vk::DeviceMemory,
    readback_size: u64,
    layout: vk::ImageLayout,
}

impl Target {
    fn new(inner: Arc<DeviceInner>, width: u32, height: u32) -> PyResult<Self> {
        let format = vk::Format::R8G8B8A8_UNORM;
        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D { width, height, depth: 1 })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(
                vk::ImageUsageFlags::COLOR_ATTACHMENT
                    | vk::ImageUsageFlags::TRANSFER_DST
                    | vk::ImageUsageFlags::TRANSFER_SRC,
            )
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let image = unsafe { inner.device.create_image(&image_info, None) }
            .map_err(|result| vk_err("vkCreateImage", result))?;

        let requirements = unsafe { inner.device.get_image_memory_requirements(image) };
        let memory_type = find_memory_type(
            &inner.instance,
            inner.physical_device,
            requirements.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        let allocate = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type);
        let image_memory = unsafe { inner.device.allocate_memory(&allocate, None) }
            .map_err(|result| vk_err("vkAllocateMemory(image)", result))?;
        unsafe { inner.device.bind_image_memory(image, image_memory, 0) }
            .map_err(|result| vk_err("vkBindImageMemory", result))?;

        let readback_size = u64::from(width) * u64::from(height) * 4;
        let buffer_info = vk::BufferCreateInfo::default()
            .size(readback_size)
            .usage(vk::BufferUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let readback_buffer = unsafe { inner.device.create_buffer(&buffer_info, None) }
            .map_err(|result| vk_err("vkCreateBuffer", result))?;
        let buffer_requirements = unsafe { inner.device.get_buffer_memory_requirements(readback_buffer) };
        let readback_type = find_memory_type(
            &inner.instance,
            inner.physical_device,
            buffer_requirements.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        let readback_allocate = vk::MemoryAllocateInfo::default()
            .allocation_size(buffer_requirements.size)
            .memory_type_index(readback_type);
        let readback_memory = unsafe { inner.device.allocate_memory(&readback_allocate, None) }
            .map_err(|result| vk_err("vkAllocateMemory(readback)", result))?;
        unsafe { inner.device.bind_buffer_memory(readback_buffer, readback_memory, 0) }
            .map_err(|result| vk_err("vkBindBufferMemory", result))?;

        Ok(Self {
            inner,
            width,
            height,
            image,
            image_memory,
            readback_buffer,
            readback_memory,
            readback_size,
            layout: vk::ImageLayout::UNDEFINED,
        })
    }


    /// P2 smoke test: draw through Skia's Ganesh Vulkan backend directly into
    /// this target's VkImage, then read the image back through Vulkan.
    fn draw_skia_test(&mut self) -> PyResult<Vec<u8>> {
        use ash::vk::Handle;

        let width = self.width as i32;
        let height = self.height as i32;
        let mut cpu_pixels = vec![0u8; (width * height * 4) as usize];

        {
            let get_proc = |of: gpu::vk::GetProcOf| unsafe {
                match of {
                    gpu::vk::GetProcOf::Instance(instance, name) => match self
                        .inner
                        .entry
                        .get_instance_proc_addr(vk::Instance::from_raw(instance as _), name)
                    {
                        Some(function) => function as *const std::ffi::c_void,
                        None => std::ptr::null(),
                    },
                    gpu::vk::GetProcOf::Device(device, name) => match self
                        .inner
                        .instance
                        .get_device_proc_addr(vk::Device::from_raw(device as _), name)
                    {
                        Some(function) => function as *const std::ffi::c_void,
                        None => std::ptr::null(),
                    },
                }
            };

            let backend_context = unsafe {
                gpu::vk::BackendContext::new_builder(
                    self.inner.instance.handle().as_raw() as _,
                    self.inner.physical_device.as_raw() as _,
                    self.inner.device.handle().as_raw() as _,
                    (self.inner.queue.as_raw() as _, 0usize),
                    &get_proc,
                    Some((1, 3, 0).into()),
                )
                .build()
            };

            let mut context = gpu::direct_contexts::make_vulkan(&backend_context, None)
                .ok_or_else(|| PyRuntimeError::new_err("Skia make_vulkan returned None"))?;

            let mut image_info = unsafe {
                gpu::vk::ImageInfo::new(
                    self.image.as_raw() as _,
                    gpu::vk::Alloc::default(),
                    gpu::vk::ImageTiling::OPTIMAL,
                    gpu::vk::ImageLayout::UNDEFINED,
                    gpu::vk::Format::R8G8B8A8_UNORM,
                    1,
                    self.inner.queue_family,
                    None,
                    None,
                    None,
                )
            };
            image_info.image_usage_flags = (vk::ImageUsageFlags::COLOR_ATTACHMENT
                | vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST)
                .as_raw() as _;

            let backend_rt =
                gpu::ganesh::vk::backend_render_targets::make_vk((width, height), &image_info);
            let mut surface = gpu::surfaces::wrap_backend_render_target(
                &mut context,
                &backend_rt,
                gpu::SurfaceOrigin::TopLeft,
                ColorType::RGBA8888,
                None,
                None,
            )
            .ok_or_else(|| PyRuntimeError::new_err("Skia wrap_backend_render_target returned None"))?;

            let canvas = surface.canvas();
            canvas.clear(Color::BLACK);
            let mut paint = Paint::default();
            paint.set_color(Color::RED);
            paint.set_anti_alias(true);
            canvas.draw_rect(Rect::from_xywh(40.0, 40.0, 200.0, 100.0), &paint);
            paint.set_color(Color::from_argb(255, 30, 160, 255));
            canvas.draw_circle((400.0, 180.0), 70.0, &paint);
            paint.set_color(Color::WHITE);
            let font_mgr = FontMgr::new();
            let typeface = font_mgr
                .match_family_style("DejaVu Sans", FontStyle::normal())
                .or_else(|| font_mgr.match_family_style("Noto Sans", FontStyle::normal()))
                .ok_or_else(|| PyRuntimeError::new_err("no usable system typeface"))?;
            let font = Font::new(typeface, 40.0);
            canvas.draw_str("Skia/Vulkan", (60.0, 300.0), &font, &paint);

            context
                .flush_and_submit()
                .map_err(|error| PyRuntimeError::new_err(format!("Skia flush failed: {error:?}")))?;

            let dst_info = SkImageInfo::new(
                (width, height),
                ColorType::RGBA8888,
                AlphaType::Unpremul,
                None,
            );
            if !surface.read_pixels(&dst_info, &mut cpu_pixels, (width * 4) as usize, (0, 0)) {
                return Err(PyRuntimeError::new_err("Skia surface read_pixels failed"));
            }
        }

        self.layout = vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL;
        let pixels = self.readback_current()?;
        // Keep the Skia-side CPU copy available for callers that want to compare
        // the two readback paths; for now the Vulkan readback is authoritative.
        let _ = cpu_pixels;
        Ok(pixels)
    }

    fn readback_current(&mut self) -> PyResult<Vec<u8>> {
        let device = &self.inner.device;
        unsafe {
            device
                .reset_command_buffer(
                    self.inner.command_buffer,
                    vk::CommandBufferResetFlags::empty(),
                )
                .map_err(|result| vk_err("vkResetCommandBuffer", result))?;
            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            device
                .begin_command_buffer(self.inner.command_buffer, &begin)
                .map_err(|result| vk_err("vkBeginCommandBuffer", result))?;

            let range = vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            };
            let to_read = vk::ImageMemoryBarrier::default()
                .old_layout(self.layout)
                .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                .image(self.image)
                .subresource_range(range);
            device.cmd_pipeline_barrier(
                self.inner.command_buffer,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[to_read],
            );

            let copy = vk::BufferImageCopy::default()
                .buffer_offset(0)
                .buffer_row_length(0)
                .buffer_image_height(0)
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image_extent(vk::Extent3D {
                    width: self.width,
                    height: self.height,
                    depth: 1,
                });
            device.cmd_copy_image_to_buffer(
                self.inner.command_buffer,
                self.image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                self.readback_buffer,
                &[copy],
            );
            device
                .end_command_buffer(self.inner.command_buffer)
                .map_err(|result| vk_err("vkEndCommandBuffer", result))?;

            let command_buffers = [self.inner.command_buffer];
            let submit = vk::SubmitInfo::default().command_buffers(&command_buffers);
            device
                .queue_submit(self.inner.queue, &[submit], self.inner.fence)
                .map_err(|result| vk_err("vkQueueSubmit", result))?;
            device
                .wait_for_fences(&[self.inner.fence], true, u64::MAX)
                .map_err(|result| vk_err("vkWaitForFences", result))?;
            device
                .reset_fences(&[self.inner.fence])
                .map_err(|result| vk_err("vkResetFences", result))?;

            let ptr = device
                .map_memory(
                    self.readback_memory,
                    0,
                    self.readback_size,
                    vk::MemoryMapFlags::empty(),
                )
                .map_err(|result| vk_err("vkMapMemory", result))?;
            let bytes = std::slice::from_raw_parts(ptr as *const u8, self.readback_size as usize)
                .to_vec();
            device.unmap_memory(self.readback_memory);
            self.layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
            Ok(bytes)
        }
    }

    fn clear_and_read(
        &mut self,
        red: f32,
        green: f32,
        blue: f32,
        alpha: f32,
    ) -> PyResult<Vec<u8>> {
        let device = &self.inner.device;
        unsafe {
            device
                .reset_command_buffer(
                    self.inner.command_buffer,
                    vk::CommandBufferResetFlags::empty(),
                )
                .map_err(|result| vk_err("vkResetCommandBuffer", result))?;
            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            device
                .begin_command_buffer(self.inner.command_buffer, &begin)
                .map_err(|result| vk_err("vkBeginCommandBuffer", result))?;

            let to_transfer = vk::ImageMemoryBarrier::default()
                .old_layout(self.layout)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .image(self.image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });
            device.cmd_pipeline_barrier(
                self.inner.command_buffer,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[to_transfer],
            );

            let clear_value = vk::ClearColorValue {
                float32: [red, green, blue, alpha],
            };
            let range = vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            };
            device.cmd_clear_color_image(
                self.inner.command_buffer,
                self.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &clear_value,
                &[range],
            );

            let to_read = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                .image(self.image)
                .subresource_range(range);
            device.cmd_pipeline_barrier(
                self.inner.command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[to_read],
            );

            let copy = vk::BufferImageCopy::default()
                .buffer_offset(0)
                .buffer_row_length(0)
                .buffer_image_height(0)
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image_extent(vk::Extent3D {
                    width: self.width,
                    height: self.height,
                    depth: 1,
                });
            device.cmd_copy_image_to_buffer(
                self.inner.command_buffer,
                self.image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                self.readback_buffer,
                &[copy],
            );
            device
                .end_command_buffer(self.inner.command_buffer)
                .map_err(|result| vk_err("vkEndCommandBuffer", result))?;

            let command_buffers = [self.inner.command_buffer];
            let submit = vk::SubmitInfo::default().command_buffers(&command_buffers);
            device
                .queue_submit(self.inner.queue, &[submit], self.inner.fence)
                .map_err(|result| vk_err("vkQueueSubmit", result))?;
            device
                .wait_for_fences(&[self.inner.fence], true, u64::MAX)
                .map_err(|result| vk_err("vkWaitForFences", result))?;
            device
                .reset_fences(&[self.inner.fence])
                .map_err(|result| vk_err("vkResetFences", result))?;

            let ptr = device
                .map_memory(
                    self.readback_memory,
                    0,
                    self.readback_size,
                    vk::MemoryMapFlags::empty(),
                )
                .map_err(|result| vk_err("vkMapMemory", result))?;
            let bytes = std::slice::from_raw_parts(ptr as *const u8, self.readback_size as usize)
                .to_vec();
            device.unmap_memory(self.readback_memory);
            self.layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
            Ok(bytes)
        }
    }
}

impl Drop for Target {
    fn drop(&mut self) {
        unsafe {
            let _ = self.inner.device.device_wait_idle();
            self.inner.device.destroy_buffer(self.readback_buffer, None);
            self.inner.device.free_memory(self.readback_memory, None);
            self.inner.device.destroy_image(self.image, None);
            self.inner.device.free_memory(self.image_memory, None);
        }
    }
}

#[pymethods]
impl Target {
    #[getter]
    fn width(&self) -> u32 {
        self.width
    }

    #[getter]
    fn height(&self) -> u32 {
        self.height
    }

    /// Clear the target and return top-down RGBA8 bytes.
    fn clear(&mut self, red: f32, green: f32, blue: f32, alpha: f32) -> PyResult<Vec<u8>> {
        self.clear_and_read(red, green, blue, alpha)
    }

    /// Draw a Skia/Ganesh Vulkan test card and return top-down RGBA8 bytes.
    fn skia_draw_test(&mut self) -> PyResult<Vec<u8>> {
        self.draw_skia_test()
    }
}


fn skia_get_proc(
    inner: &DeviceInner,
    of: gpu::vk::GetProcOf,
) -> *const std::ffi::c_void {
    use ash::vk::Handle;
    unsafe {
        match of {
            gpu::vk::GetProcOf::Instance(instance, name) => {
                match inner.entry.get_instance_proc_addr(
                    vk::Instance::from_raw(instance as _),
                    name,
                ) {
                    Some(function) => function as *const std::ffi::c_void,
                    None => std::ptr::null(),
                }
            }
            gpu::vk::GetProcOf::Device(device, name) => {
                match inner.instance.get_device_proc_addr(
                    vk::Device::from_raw(device as _),
                    name,
                ) {
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
    width: u32,
    height: u32,
    surface: skia_safe::Surface,
    context: gpu::DirectContext,
    backend: Box<gpu::vk::BackendContext<'static>>,
    path: PathBuilder,
    paint: Paint,
    matrix: M44,
    hsb_effect: Option<RuntimeEffect>,
    design3_effect: Option<RuntimeEffect>,
}

impl SkiaTarget {
    fn new(inner: Arc<DeviceInner>, width: u32, height: u32) -> PyResult<Self> {
        use ash::vk::Handle;
        let inner_for_proc = inner.clone();
        let get_proc = Box::leak(Box::new(move |of| {
            skia_get_proc(&inner_for_proc, of)
        }));
        let backend = Box::new(unsafe {
            gpu::vk::BackendContext::new_builder(
                inner.instance.handle().as_raw() as _,
                inner.physical_device.as_raw() as _,
                inner.device.handle().as_raw() as _,
                (inner.queue.as_raw() as _, 0usize),
                &*get_proc,
                Some((1, 3, 0).into()),
            )
            .build()
        });
        let mut context = gpu::direct_contexts::make_vulkan(&backend, None)
            .ok_or_else(|| PyRuntimeError::new_err("Skia make_vulkan returned None"))?;
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
            width,
            height,
            surface,
            context,
            backend,
            path: PathBuilder::new(),
            paint: {
                let mut paint = Paint::default();
                paint.set_anti_alias(true);
                paint
            },
            matrix: M44::new_identity(),
            hsb_effect: None,
            design3_effect: None,
        })
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
            sx, 0.0, 0.0, tx,
            0.0, sy, 0.0, ty,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ]);
        self.matrix = matrix;
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
        self.path.cubic_to(
            Point::new(x1, y1),
            Point::new(x2, y2),
            Point::new(x3, y3),
        );
    }

    fn close_path(&mut self) {
        self.path.close();
    }

    fn set_source_rgba(&mut self, red: f32, green: f32, blue: f32, alpha: f32) {
        self.paint.set_shader(None);
        self.paint.set_color4f(Color4f::new(red, green, blue, alpha), None);
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
        let gradient_colors = GradientColors::new(
            &colors,
            Some(positions.as_slice()),
            TileMode::Clamp,
            None,
        );
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

    fn set_anti_alias(&mut self, enabled: bool) {
        self.paint.set_anti_alias(enabled);
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

    /// Draw a batch of triangles with per-vertex colours.  Skia interpolates
    /// the colours across each triangle, matching the OpenGL vertex-attribute
    /// behaviour for the common ``in_vert`` / ``in_color`` mesh layout.
    fn draw_vertices(
        &mut self,
        positions: Vec<(f32, f32)>,
        colors: Vec<(u8, u8, u8, u8)>,
    ) {
        if positions.len() != colors.len() || positions.len() < 3 {
            return;
        }
        let points: Vec<Point> = positions
            .into_iter()
            .map(|(x, y)| Point::new(x, y))
            .collect();
        let vertex_colors: Vec<Color> = colors
            .into_iter()
            .map(|(r, g, b, a)| Color::from_argb(a, r, g, b))
            .collect();
        let tex_coords = vec![Point::new(0.0, 0.0); points.len()];
        let vertices = Vertices::new_copy(
            VertexMode::Triangles,
            &points,
            &tex_coords,
            &vertex_colors,
            None,
        );
        self.paint.set_shader(None);
        self.paint.set_style(Style::Fill);
        self.paint.set_color(Color::WHITE);
        self.paint.set_alpha(255);
        self.surface
            .canvas()
            .draw_vertices(&vertices, BlendMode::SrcOver, &self.paint);
    }

    /// Evaluate the canonical HSB/RGB polar shader used by Manim's
    /// ``InlineFullScreenQuad`` example as a SkSL runtime effect.  The shader
    /// runs inside Skia's Ganesh Vulkan backend, so the pixel colours are
    /// produced on the GPU rather than by a Python approximation.
    fn draw_hsb_quad(&mut self, time: f32) -> PyResult<()> {
        const SKSL: &str = r#"
uniform float2 u_resolution;
uniform float u_time;

half4 main(float2 fragCoord) {
    float2 st = fragCoord / u_resolution;
    float2 toCenter = float2(0.5) - st;
    float angle = atan(toCenter.y, toCenter.x);
    angle += u_time;
    float radius = length(toCenter) * 2.0;
    float hue = angle / 6.28318530718 + 0.5;
    float3 rgb = clamp(abs(mod(hue * 6.0 + float3(0.0, 4.0, 2.0), 6.0) - 3.0) - 1.0, 0.0, 1.0);
    rgb = rgb * rgb * (3.0 - 2.0 * rgb);
    float3 color = mix(float3(1.0), rgb, radius);
    return half4(color.r, color.g, color.b, 1.0);
}
"#;
        let effect = if let Some(effect) = self.hsb_effect.clone() {
            effect
        } else {
            let effect = RuntimeEffect::make_for_shader(SKSL, None)
                .map_err(|error| PyRuntimeError::new_err(format!("SkSL compile failed: {error}")))?;
            self.hsb_effect = Some(effect.clone());
            effect
        };
        let mut builder = RuntimeShaderBuilder::new(effect);
        builder
            .set_uniform_float("u_resolution", &[self.width as f32, self.height as f32])
            .map_err(|error| PyRuntimeError::new_err(format!("SkSL uniform: {error:?}")))?;
        builder
            .set_uniform_float("u_time", &[time])
            .map_err(|error| PyRuntimeError::new_err(format!("SkSL uniform: {error:?}")))?;
        let shader = builder
            .make_shader(&Matrix::new_identity())
            .ok_or_else(|| PyRuntimeError::new_err("SkSL make_shader returned None"))?;

        self.paint.set_shader(shader);
        self.paint.set_style(Style::Fill);
        self.paint.set_color(Color::WHITE);
        self.paint.set_alpha(255);
        let canvas = self.surface.canvas();
        canvas.save();
        canvas.reset_matrix();
        canvas.draw_rect(
            Rect::new(0.0, 0.0, self.width as f32, self.height as f32),
            &self.paint,
        );
        canvas.restore();
        self.paint.set_shader(None);
        Ok(())
    }

    /// Evaluate the raymarched ``design_3`` full-screen shader.  It is the
    /// named counterpart of ``InlineFullScreenQuad`` and uses the same Skia
    /// Ganesh Vulkan runtime-effect path.
    fn draw_design3_quad(&mut self, time: f32) -> PyResult<()> {
        const SKSL: &str = r#"
uniform float3 u_resolution;
uniform float u_time;

float3 palette(float d) {
    return mix(float3(0.2, 0.7, 0.9), float3(1.0, 0.0, 1.0), d);
}

float2 rotate2(float2 p, float a) {
    float c = cos(a);
    float s = sin(a);
    return p * float2x2(c, s, -s, c);
}

float map3(float3 p) {
    for (int i = 0; i < 8; ++i) {
        float t = u_time * 0.1;
        p.xz = rotate2(p.xz, t);
        p.xy = rotate2(p.xy, t * 1.89);
        p.xz = abs(p.xz);
        p.xz = p.xz - 0.5;
    }
    return dot(sign(p), p) / 5.0;
}

float4 rm3(float3 ro, float3 rd) {
    float t = 0.0;
    float3 col = float3(0.0);
    float d = 0.0;
    for (float i = 0.0; i < 64.0; i++) {
        float3 p = ro + rd * t;
        d = map3(p) * 0.5;
        if (d < 0.02) {
            break;
        }
        if (d > 100.0) {
            break;
        }
        col += palette(length(p) * 0.1) / (400.0 * d);
        t += d;
    }
    return float4(col, 1.0 / (d * 100.0));
}

half4 main(float2 fragCoord) {
    float2 uv = (fragCoord - (u_resolution.xy / 2.0)) / u_resolution.x;
    float3 ro = float3(0.0, 0.0, -50.0);
    ro.xz = rotate2(ro.xz, u_time);
    float3 cf = normalize(-ro);
    float3 cs = normalize(cross(cf, float3(0.0, 1.0, 0.0)));
    float3 cu = normalize(cross(cf, cs));
    float3 uuv = ro + cf * 3.0 + uv.x * cs + uv.y * cu;
    float3 rd = normalize(uuv - ro);
    float4 col = rm3(ro, rd);
    return half4(col.r, col.g, col.b, 1.0);
}
"#;
        let effect = if let Some(effect) = self.design3_effect.clone() {
            effect
        } else {
            let effect = RuntimeEffect::make_for_shader(SKSL, None)
                .map_err(|error| PyRuntimeError::new_err(format!("SkSL compile failed: {error}")))?;
            self.design3_effect = Some(effect.clone());
            effect
        };
        let mut builder = RuntimeShaderBuilder::new(effect);
        builder
            .set_uniform_float(
                "u_resolution",
                &[self.width as f32, self.height as f32, 0.0],
            )
            .map_err(|error| PyRuntimeError::new_err(format!("SkSL uniform: {error:?}")))?;
        builder
            .set_uniform_float("u_time", &[time])
            .map_err(|error| PyRuntimeError::new_err(format!("SkSL uniform: {error:?}")))?;
        let shader = builder
            .make_shader(&Matrix::new_identity())
            .ok_or_else(|| PyRuntimeError::new_err("SkSL make_shader returned None"))?;
        self.paint.set_shader(shader);
        self.paint.set_style(Style::Fill);
        self.paint.set_color(Color::WHITE);
        self.paint.set_alpha(255);
        let canvas = self.surface.canvas();
        canvas.save();
        canvas.reset_matrix();
        canvas.draw_rect(
            Rect::new(0.0, 0.0, self.width as f32, self.height as f32),
            &self.paint,
        );
        canvas.restore();
        self.paint.set_shader(None);
        Ok(())
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
        if !self.surface.read_pixels(
            &info,
            &mut pixels,
            (self.width * 4) as usize,
            (0, 0),
        ) {
            return Err(PyRuntimeError::new_err("Skia surface read_pixels failed"));
        }
        Ok(pixels)
    }

    /// Flush Ganesh and write the RGBA surface directly into a writable
    /// Python buffer (normally a C-contiguous NumPy frame array).  This avoids
    /// the temporary Rust vector and the second Python-side copy done by
    /// ``commit()``.
    fn commit_into(&mut self, target: &Bound<PyAny>) -> PyResult<()> {
        use pyo3::buffer::PyBuffer;

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
        let slice = buffer.as_mut_slice(py).ok_or_else(|| {
            PyValueError::new_err("readback buffer must be C-contiguous")
        })?;
        let data = unsafe {
            std::slice::from_raw_parts_mut(slice.as_ptr() as *mut u8, expected)
        };
        let info = SkImageInfo::new(
            (self.width as i32, self.height as i32),
            ColorType::RGBA8888,
            AlphaType::Unpremul,
            None,
        );
        if !self.surface.read_pixels(&info, data, (self.width * 4) as usize, (0, 0)) {
            return Err(PyRuntimeError::new_err("Skia surface read_pixels failed"));
        }
        Ok(())
    }

    /// Upload host RGBA pixels into the GPU surface. Used after CPU compositing
    /// of point clouds or image mobjects.
    fn load_pixels(&mut self, pixels: Vec<u8>) -> PyResult<()> {
        if pixels.len() != (self.width * self.height * 4) as usize {
            return Err(PyValueError::new_err("pixel buffer has the wrong size"));
        }
        let info = SkImageInfo::new(
            (self.width as i32, self.height as i32),
            ColorType::RGBA8888,
            AlphaType::Unpremul,
            None,
        );
        let mut pixels = pixels;
        let pixmap = skia_safe::Pixmap::new(
            &info,
            pixels.as_mut_slice(),
            (self.width * 4) as usize,
        )
        .ok_or_else(|| PyRuntimeError::new_err("Skia Pixmap::new failed"))?;
        self.surface.write_pixels_from_pixmap(&pixmap, (0, 0));
        self.context
            .flush_and_submit()
            .map_err(|error| PyRuntimeError::new_err(format!("Skia flush failed: {error:?}")))?;
        Ok(())
    }
}

#[pymodule]
fn manim_vk(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<Device>()?;
    module.add_class::<Target>()?;
    module.add_class::<SkiaTarget>()?;
    module.add_class::<window::NativeWindow>()?;
    module.add("__version__", "0.1.0")?;
    Ok(())
}

