//! The exact per-pixel depth of a frame's opaque meshes, from a Vulkan depth
//! buffer.
//!
//! Skia keeps Vulkan's render passes to itself and offers no depth buffer, so
//! the depth comes from a small render pass of our own on the same device and
//! queue: it draws the opaque triangles with a depth test, the way ManimGL
//! draws surfaces, and writes the nearest depth of every pixel into a colour
//! image that Skia then samples (see `display_list::Mesh`).  Depth is
//! normalised to `[0, 1]`, nearer is larger, and `0` means "no opaque mesh";
//! it is split over the red (high 16 bits) and green (low 16 bits) channels.

use std::ffi::CStr;
use std::sync::Arc;

use ash::vk;
use ash::vk::Handle;
use pyo3::exceptions::PyRuntimeError;
use pyo3::PyResult;
use skia_safe::gpu;

use crate::DeviceInner;

const COLOR_FORMAT: vk::Format = vk::Format::R16G16B16A16_UNORM;
const DEPTH_FORMAT: vk::Format = vk::Format::D32_SFLOAT;
/// Floats per vertex: device x and y (pixels), then normalised depth.
pub const VERTEX_FLOATS: usize = 3;

const VERTEX_SHADER: &str = r"#version 450
layout(location = 0) in vec2 position;
layout(location = 1) in float depth;
layout(location = 0) out float v_depth;
layout(push_constant) uniform Viewport { vec2 size; } viewport;
void main() {
    v_depth = depth;
    // Device pixels to Vulkan's clip space (y down, depth 0 nearest).
    gl_Position = vec4(position / viewport.size * 2.0 - 1.0, 1.0 - depth, 1.0);
}
";

const FRAGMENT_SHADER: &str = r"#version 450
layout(location = 0) in float v_depth;
layout(location = 0) out vec4 color;
void main() {
    float scaled = clamp(v_depth, 0.0, 1.0) * 65535.0;
    float high = floor(scaled);
    color = vec4(high / 65535.0, scaled - high, 0.0, 1.0);
}
";

fn vk_error(context: &str, result: vk::Result) -> pyo3::PyErr {
    PyRuntimeError::new_err(format!("depth pass: {context}: {result:?}"))
}

/// GLSL to SPIR-V with naga, the shader translator wgpu is built on.
fn spirv(stage: naga::ShaderStage, source: &str) -> PyResult<Vec<u32>> {
    let module = naga::front::glsl::Frontend::default()
        .parse(&naga::front::glsl::Options::from(stage), source)
        .map_err(|errors| PyRuntimeError::new_err(format!("depth shader: {errors:?}")))?;
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .map_err(|error| PyRuntimeError::new_err(format!("depth shader: {error:?}")))?;
    let options = naga::back::spv::Options {
        // The shaders are written for Vulkan's clip space; do not flip y.
        flags: naga::back::spv::WriterFlags::empty(),
        ..Default::default()
    };
    let pipeline = naga::back::spv::PipelineOptions {
        shader_stage: stage,
        entry_point: "main".into(),
    };
    naga::back::spv::write_vec(&module, &info, &options, Some(&pipeline))
        .map_err(|error| PyRuntimeError::new_err(format!("depth shader: {error:?}")))
}

struct Allocated<T> {
    handle: T,
    memory: vk::DeviceMemory,
}

pub struct DepthPass {
    inner: Arc<DeviceInner>,
    width: u32,
    height: u32,
    color: Allocated<vk::Image>,
    color_view: vk::ImageView,
    depth: Allocated<vk::Image>,
    depth_view: vk::ImageView,
    render_pass: vk::RenderPass,
    framebuffer: vk::Framebuffer,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    pool: vk::CommandPool,
    commands: vk::CommandBuffer,
    fence: vk::Fence,
    vertices: Option<(Allocated<vk::Buffer>, usize)>,
}

impl DepthPass {
    pub fn new(inner: Arc<DeviceInner>, width: u32, height: u32) -> PyResult<Self> {
        for (format, feature) in [
            (
                COLOR_FORMAT,
                vk::FormatFeatureFlags::COLOR_ATTACHMENT | vk::FormatFeatureFlags::SAMPLED_IMAGE,
            ),
            (
                DEPTH_FORMAT,
                vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT,
            ),
        ] {
            let properties = unsafe {
                inner
                    .instance
                    .get_physical_device_format_properties(inner.physical_device, format)
            };
            if !properties.optimal_tiling_features.contains(feature) {
                return Err(PyRuntimeError::new_err(format!(
                    "the Vulkan device cannot use {format:?} for 3D depth"
                )));
            }
        }
        let mut pass = Self {
            inner,
            width,
            height,
            color: Allocated {
                handle: vk::Image::null(),
                memory: vk::DeviceMemory::null(),
            },
            color_view: vk::ImageView::null(),
            depth: Allocated {
                handle: vk::Image::null(),
                memory: vk::DeviceMemory::null(),
            },
            depth_view: vk::ImageView::null(),
            render_pass: vk::RenderPass::null(),
            framebuffer: vk::Framebuffer::null(),
            layout: vk::PipelineLayout::null(),
            pipeline: vk::Pipeline::null(),
            pool: vk::CommandPool::null(),
            commands: vk::CommandBuffer::null(),
            fence: vk::Fence::null(),
            vertices: None,
        };
        // On error, Drop destroys whatever was created.
        pass.create()?;
        Ok(pass)
    }

    fn device(&self) -> &ash::Device {
        &self.inner.device
    }

    fn memory_type(&self, bits: u32, flags: vk::MemoryPropertyFlags) -> PyResult<u32> {
        let properties = unsafe {
            self.inner
                .instance
                .get_physical_device_memory_properties(self.inner.physical_device)
        };
        (0..properties.memory_type_count)
            .find(|&index| {
                bits & (1 << index) != 0
                    && properties.memory_types[index as usize]
                        .property_flags
                        .contains(flags)
            })
            .ok_or_else(|| PyRuntimeError::new_err("depth pass: no suitable Vulkan memory type"))
    }

    fn allocate_image(
        &self,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
        aspect: vk::ImageAspectFlags,
    ) -> PyResult<(Allocated<vk::Image>, vk::ImageView)> {
        let device = self.device();
        let info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D {
                width: self.width,
                height: self.height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let image = unsafe { device.create_image(&info, None) }
            .map_err(|result| vk_error("vkCreateImage", result))?;
        let requirements = unsafe { device.get_image_memory_requirements(image) };
        let memory_type = self.memory_type(
            requirements.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        );
        let memory = memory_type.and_then(|memory_type| {
            let allocate = vk::MemoryAllocateInfo::default()
                .allocation_size(requirements.size)
                .memory_type_index(memory_type);
            unsafe { device.allocate_memory(&allocate, None) }
                .map_err(|result| vk_error("vkAllocateMemory", result))
        });
        let memory = match memory {
            Ok(memory) => memory,
            Err(error) => {
                unsafe { device.destroy_image(image, None) };
                return Err(error);
            }
        };
        let allocated = Allocated {
            handle: image,
            memory,
        };
        let bound = unsafe { device.bind_image_memory(image, memory, 0) }
            .map_err(|result| vk_error("vkBindImageMemory", result));
        let view = bound.and_then(|()| {
            let view_info = vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(format)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: aspect,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });
            unsafe { device.create_image_view(&view_info, None) }
                .map_err(|result| vk_error("vkCreateImageView", result))
        });
        match view {
            Ok(view) => Ok((allocated, view)),
            Err(error) => {
                unsafe {
                    device.destroy_image(image, None);
                    device.free_memory(memory, None);
                }
                Err(error)
            }
        }
    }

    fn create(&mut self) -> PyResult<()> {
        let (color, color_view) = self.allocate_image(
            COLOR_FORMAT,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            vk::ImageAspectFlags::COLOR,
        )?;
        self.color = color;
        self.color_view = color_view;
        let (depth, depth_view) = self.allocate_image(
            DEPTH_FORMAT,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            vk::ImageAspectFlags::DEPTH,
        )?;
        self.depth = depth;
        self.depth_view = depth_view;

        let device = self.inner.device.clone();
        let attachments = [
            vk::AttachmentDescription::default()
                .format(COLOR_FORMAT)
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .final_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
            vk::AttachmentDescription::default()
                .format(DEPTH_FORMAT)
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::DONT_CARE)
                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL),
        ];
        let color_reference = [vk::AttachmentReference {
            attachment: 0,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        }];
        let depth_reference = vk::AttachmentReference {
            attachment: 1,
            layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        };
        let subpasses = [vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&color_reference)
            .depth_stencil_attachment(&depth_reference)];
        // Skia's reads of the previous frame's depth come before our writes,
        // and our writes before Skia's reads of this frame; both happen later
        // or earlier in submission order on the same queue.
        let dependencies = [
            vk::SubpassDependency::default()
                .src_subpass(vk::SUBPASS_EXTERNAL)
                .dst_subpass(0)
                .src_stage_mask(
                    vk::PipelineStageFlags::FRAGMENT_SHADER
                        | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
                )
                .src_access_mask(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE)
                .dst_stage_mask(
                    vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                        | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
                )
                .dst_access_mask(
                    vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                        | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                ),
            vk::SubpassDependency::default()
                .src_subpass(0)
                .dst_subpass(vk::SUBPASS_EXTERNAL)
                .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
                .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags::FRAGMENT_SHADER)
                .dst_access_mask(vk::AccessFlags::SHADER_READ),
        ];
        let render_pass_info = vk::RenderPassCreateInfo::default()
            .attachments(&attachments)
            .subpasses(&subpasses)
            .dependencies(&dependencies);
        self.render_pass = unsafe { device.create_render_pass(&render_pass_info, None) }
            .map_err(|result| vk_error("vkCreateRenderPass", result))?;

        let views = [self.color_view, self.depth_view];
        let framebuffer_info = vk::FramebufferCreateInfo::default()
            .render_pass(self.render_pass)
            .attachments(&views)
            .width(self.width)
            .height(self.height)
            .layers(1);
        self.framebuffer = unsafe { device.create_framebuffer(&framebuffer_info, None) }
            .map_err(|result| vk_error("vkCreateFramebuffer", result))?;

        let push_constants = [vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::VERTEX,
            offset: 0,
            size: 8,
        }];
        let layout_info =
            vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&push_constants);
        self.layout = unsafe { device.create_pipeline_layout(&layout_info, None) }
            .map_err(|result| vk_error("vkCreatePipelineLayout", result))?;
        self.pipeline = self.create_pipeline(&device)?;

        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(self.inner.queue_family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        self.pool = unsafe { device.create_command_pool(&pool_info, None) }
            .map_err(|result| vk_error("vkCreateCommandPool", result))?;
        let allocate = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        self.commands = unsafe { device.allocate_command_buffers(&allocate) }
            .map_err(|result| vk_error("vkAllocateCommandBuffers", result))?[0];
        let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
        self.fence = unsafe { device.create_fence(&fence_info, None) }
            .map_err(|result| vk_error("vkCreateFence", result))?;
        Ok(())
    }

    fn create_pipeline(&self, device: &ash::Device) -> PyResult<vk::Pipeline> {
        let vertex_code = spirv(naga::ShaderStage::Vertex, VERTEX_SHADER)?;
        let fragment_code = spirv(naga::ShaderStage::Fragment, FRAGMENT_SHADER)?;
        let vertex = unsafe {
            device.create_shader_module(
                &vk::ShaderModuleCreateInfo::default().code(&vertex_code),
                None,
            )
        }
        .map_err(|result| vk_error("vkCreateShaderModule", result))?;
        let fragment = unsafe {
            device.create_shader_module(
                &vk::ShaderModuleCreateInfo::default().code(&fragment_code),
                None,
            )
        };
        let fragment = match fragment {
            Ok(module) => module,
            Err(result) => {
                unsafe { device.destroy_shader_module(vertex, None) };
                return Err(vk_error("vkCreateShaderModule", result));
            }
        };
        let entry: &CStr = c"main";
        let stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vertex)
                .name(entry),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(fragment)
                .name(entry),
        ];
        let bindings = [vk::VertexInputBindingDescription {
            binding: 0,
            stride: (VERTEX_FLOATS * 4) as u32,
            input_rate: vk::VertexInputRate::VERTEX,
        }];
        let attributes = [
            vk::VertexInputAttributeDescription {
                location: 0,
                binding: 0,
                format: vk::Format::R32G32_SFLOAT,
                offset: 0,
            },
            vk::VertexInputAttributeDescription {
                location: 1,
                binding: 0,
                format: vk::Format::R32_SFLOAT,
                offset: 8,
            },
        ];
        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(&bindings)
            .vertex_attribute_descriptions(&attributes);
        let assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
        let viewports = [vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: self.width as f32,
            height: self.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        }];
        let scissors = [vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D {
                width: self.width,
                height: self.height,
            },
        }];
        let viewport = vk::PipelineViewportStateCreateInfo::default()
            .viewports(&viewports)
            .scissors(&scissors);
        let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::NONE)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .line_width(1.0);
        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(true)
            .depth_write_enable(true)
            .depth_compare_op(vk::CompareOp::LESS);
        let blend_attachments = [vk::PipelineColorBlendAttachmentState::default()
            .blend_enable(false)
            .color_write_mask(vk::ColorComponentFlags::RGBA)];
        let blend =
            vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend_attachments);
        let info = [vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&assembly)
            .viewport_state(&viewport)
            .rasterization_state(&rasterization)
            .multisample_state(&multisample)
            .depth_stencil_state(&depth_stencil)
            .color_blend_state(&blend)
            .layout(self.layout)
            .render_pass(self.render_pass)
            .subpass(0)];
        let pipeline =
            unsafe { device.create_graphics_pipelines(vk::PipelineCache::null(), &info, None) };
        unsafe {
            device.destroy_shader_module(vertex, None);
            device.destroy_shader_module(fragment, None);
        }
        pipeline
            .map(|pipelines| pipelines[0])
            .map_err(|(_, result)| vk_error("vkCreateGraphicsPipelines", result))
    }

    /// A host-visible vertex buffer of at least `size` bytes, mapped.
    fn vertex_buffer(&mut self, size: usize) -> PyResult<vk::Buffer> {
        if let Some((buffer, capacity)) = &self.vertices {
            if *capacity >= size {
                return Ok(buffer.handle);
            }
        }
        if let Some((old, _)) = self.vertices.take() {
            unsafe {
                self.device().destroy_buffer(old.handle, None);
                self.device().free_memory(old.memory, None);
            }
        }
        let capacity = size.next_power_of_two().max(1 << 16);
        let device = self.inner.device.clone();
        let info = vk::BufferCreateInfo::default()
            .size(capacity as u64)
            .usage(vk::BufferUsageFlags::VERTEX_BUFFER)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = unsafe { device.create_buffer(&info, None) }
            .map_err(|result| vk_error("vkCreateBuffer", result))?;
        let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
        let memory = self
            .memory_type(
                requirements.memory_type_bits,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )
            .and_then(|memory_type| {
                let allocate = vk::MemoryAllocateInfo::default()
                    .allocation_size(requirements.size)
                    .memory_type_index(memory_type);
                unsafe { device.allocate_memory(&allocate, None) }
                    .map_err(|result| vk_error("vkAllocateMemory", result))
            });
        let memory = match memory {
            Ok(memory) => memory,
            Err(error) => {
                unsafe { device.destroy_buffer(buffer, None) };
                return Err(error);
            }
        };
        self.vertices = Some((
            Allocated {
                handle: buffer,
                memory,
            },
            capacity,
        ));
        unsafe { device.bind_buffer_memory(buffer, memory, 0) }
            .map_err(|result| vk_error("vkBindBufferMemory", result))?;
        Ok(buffer)
    }

    /// Draw `vertices` (triangle list, [`VERTEX_FLOATS`] floats each) with a
    /// depth test and submit the pass.  Work Skia recorded before must be
    /// submitted first; work it records after may read the result.
    pub fn render(&mut self, vertices: &[f32]) -> PyResult<()> {
        let device = self.inner.device.clone();
        // The previous pass must be done before its commands and vertices
        // are reused.
        unsafe { device.wait_for_fences(&[self.fence], true, u64::MAX) }
            .map_err(|result| vk_error("vkWaitForFences", result))?;
        let bytes = std::mem::size_of_val(vertices);
        let count = (vertices.len() / VERTEX_FLOATS) as u32;
        let buffer = if count > 0 {
            Some(self.vertex_buffer(bytes)?)
        } else {
            None
        };
        if let (Some(_), Some((allocation, _))) = (buffer, &self.vertices) {
            unsafe {
                let mapped = device
                    .map_memory(
                        allocation.memory,
                        0,
                        bytes as u64,
                        vk::MemoryMapFlags::empty(),
                    )
                    .map_err(|result| vk_error("vkMapMemory", result))?;
                std::ptr::copy_nonoverlapping(
                    vertices.as_ptr() as *const u8,
                    mapped as *mut u8,
                    bytes,
                );
                device.unmap_memory(allocation.memory);
            }
        }
        unsafe {
            device
                .reset_command_buffer(self.commands, vk::CommandBufferResetFlags::empty())
                .map_err(|result| vk_error("vkResetCommandBuffer", result))?;
            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            device
                .begin_command_buffer(self.commands, &begin)
                .map_err(|result| vk_error("vkBeginCommandBuffer", result))?;
            let clear = [
                vk::ClearValue {
                    color: vk::ClearColorValue { float32: [0.0; 4] },
                },
                vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: 1.0,
                        stencil: 0,
                    },
                },
            ];
            let pass = vk::RenderPassBeginInfo::default()
                .render_pass(self.render_pass)
                .framebuffer(self.framebuffer)
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: vk::Extent2D {
                        width: self.width,
                        height: self.height,
                    },
                })
                .clear_values(&clear);
            device.cmd_begin_render_pass(self.commands, &pass, vk::SubpassContents::INLINE);
            if let Some(buffer) = buffer {
                device.cmd_bind_pipeline(
                    self.commands,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.pipeline,
                );
                let size = [self.width as f32, self.height as f32];
                let size_bytes: Vec<u8> =
                    size.iter().flat_map(|value| value.to_ne_bytes()).collect();
                device.cmd_push_constants(
                    self.commands,
                    self.layout,
                    vk::ShaderStageFlags::VERTEX,
                    0,
                    &size_bytes,
                );
                device.cmd_bind_vertex_buffers(self.commands, 0, &[buffer], &[0]);
                device.cmd_draw(self.commands, count, 1, 0, 0);
            }
            device.cmd_end_render_pass(self.commands);
            device
                .end_command_buffer(self.commands)
                .map_err(|result| vk_error("vkEndCommandBuffer", result))?;
            device
                .reset_fences(&[self.fence])
                .map_err(|result| vk_error("vkResetFences", result))?;
            let command_buffers = [self.commands];
            let submit = [vk::SubmitInfo::default().command_buffers(&command_buffers)];
            device
                .queue_submit(self.inner.queue, &submit, self.fence)
                .map_err(|result| vk_error("vkQueueSubmit", result))?;
        }
        Ok(())
    }

    /// The depth image, as Skia sees it after [`Self::render`].
    pub fn image(&self, context: &mut gpu::DirectContext) -> PyResult<skia_safe::Image> {
        let mut info = unsafe {
            gpu::vk::ImageInfo::new(
                self.color.handle.as_raw() as _,
                gpu::vk::Alloc::default(),
                gpu::vk::ImageTiling::OPTIMAL,
                gpu::vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                gpu::vk::Format::R16G16B16A16_UNORM,
                1,
                self.inner.queue_family,
                None,
                None,
                None,
            )
        };
        info.image_usage_flags =
            (vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED).as_raw() as _;
        let texture = unsafe {
            gpu::ganesh::vk::backend_textures::make_vk(
                (self.width as i32, self.height as i32),
                &info,
                "manim depth",
            )
        };
        gpu::ganesh::images::borrow_texture_from(
            context,
            &texture,
            gpu::SurfaceOrigin::TopLeft,
            skia_safe::ColorType::R16G16B16A16UNorm,
            skia_safe::AlphaType::Opaque,
            None,
        )
        .ok_or_else(|| PyRuntimeError::new_err("depth pass: Skia cannot sample the depth image"))
    }
}

impl Drop for DepthPass {
    fn drop(&mut self) {
        let device = &self.inner.device;
        unsafe {
            // Skia may still sample the image in submitted work.
            let _ = device.device_wait_idle();
            if let Some((buffer, _)) = self.vertices.take() {
                device.destroy_buffer(buffer.handle, None);
                device.free_memory(buffer.memory, None);
            }
            device.destroy_fence(self.fence, None);
            device.destroy_command_pool(self.pool, None);
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.layout, None);
            device.destroy_framebuffer(self.framebuffer, None);
            device.destroy_render_pass(self.render_pass, None);
            device.destroy_image_view(self.depth_view, None);
            device.destroy_image(self.depth.handle, None);
            device.free_memory(self.depth.memory, None);
            device.destroy_image_view(self.color_view, None);
            device.destroy_image(self.color.handle, None);
            device.free_memory(self.color.memory, None);
        }
    }
}
