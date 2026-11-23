//! A lightweight Vulkan-backed preview window.
//!
//! Manim's renderer still renders into its offscreen Vulkan target; this
//! window only receives the finished RGBA frame and presents it through
//! wgpu's Vulkan backend.  Keeping presentation separate means preview never
//! changes the pixels that get written to disk.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};

use pixels::{Pixels, PixelsBuilder, SurfaceTexture};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use winit::application::ApplicationHandler;
use winit::dpi::{PhysicalPosition, PhysicalSize};
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop, EventLoopProxy};
use winit::monitor::MonitorHandle;
use winit::window::{Fullscreen, Window, WindowId};

#[derive(Clone, Copy, Debug)]
enum UserEvent {
    Frame,
    Close,
}

/// Map Manim's Y/X position spellings (``UR``, ``LEFT``, ``0,0``) to a
/// physical top-left window position on the selected monitor.
fn initial_position(
    monitor: &MonitorHandle,
    width: u32,
    height: u32,
    spec: &str,
) -> Option<PhysicalPosition<i32>> {
    if spec.is_empty() {
        return None;
    }
    if let Some((x, y)) = spec.split_once(',') {
        let x = x.trim().parse::<i32>().ok()?;
        let y = y.trim().parse::<i32>().ok()?;
        return Some(PhysicalPosition::new(x, y));
    }
    let normalized = match spec {
        "LEFT" => "OL".to_string(),
        "RIGHT" => "OR".to_string(),
        "UP" => "UO".to_string(),
        "DOWN" => "DO".to_string(),
        "ORIGIN" | "CENTER" => "OO".to_string(),
        other if other.len() == 2 => other.to_ascii_uppercase(),
        _ => return None,
    };
    let chars: Vec<char> = normalized.chars().collect();
    if chars.len() != 2 {
        return None;
    }
    let y_frac = match chars[0] {
        'U' => 0.0,
        'O' => 0.5,
        'D' => 1.0,
        _ => return None,
    };
    let x_frac = match chars[1] {
        'L' => 0.0,
        'O' => 0.5,
        'R' => 1.0,
        _ => return None,
    };
    let area = monitor.size();
    let origin = monitor.position();
    let free_x = (area.width as i32 - width as i32).max(0);
    let free_y = (area.height as i32 - height as i32).max(0);
    Some(PhysicalPosition::new(
        origin.x + (x_frac * free_x as f64) as i32,
        origin.y + (y_frac * free_y as f64) as i32,
    ))
}

struct PreviewApp {
    width: u32,
    height: u32,
    title: String,
    fullscreen: bool,
    position: String,
    monitor: usize,
    requested_adapter: String,
    proxy: EventLoopProxy<UserEvent>,
    closing: Arc<AtomicBool>,
    pending: Arc<Mutex<Option<Vec<u8>>>>,
    adapter_name: Arc<Mutex<String>>,
    error: Arc<Mutex<Option<String>>>,
    window: Option<Arc<Window>>,
    pixels: Option<Pixels<'static>>,
}

impl PreviewApp {
    fn redraw(&mut self) {
        let Some(pixels) = self.pixels.as_mut() else { return };
        let frame = self.pending.lock().ok().and_then(|mut slot| slot.take());
        if let Some(frame) = frame {
            if frame.len() == pixels.frame().len() {
                pixels.frame_mut().copy_from_slice(&frame);
                let _ = pixels.render();
            }
        }
    }
}

impl ApplicationHandler<UserEvent> for PreviewApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        if !self.requested_adapter.is_empty() {
            // pixels/wgpu honors this variable when requesting an adapter.
            // It lets preview present on the same GPU as the offscreen
            // renderer on dual-GPU machines.
            std::env::set_var("WGPU_ADAPTER_NAME", &self.requested_adapter);
        }
        let monitors: Vec<MonitorHandle> = event_loop.available_monitors().collect();
        let monitor = monitors
            .get(self.monitor.min(monitors.len().saturating_sub(1)))
            .cloned();
        let mut attributes = Window::default_attributes()
            .with_title(self.title.clone())
            .with_inner_size(PhysicalSize::new(self.width, self.height))
            .with_resizable(true);
        if self.fullscreen {
            attributes = attributes.with_fullscreen(Some(Fullscreen::Borderless(monitor.clone())));
        } else if let Some(monitor) = monitor.as_ref() {
            if let Some(position) = initial_position(monitor, self.width, self.height, &self.position) {
                attributes = attributes.with_position(position);
            }
        }
        let window = match event_loop.create_window(attributes) {
            Ok(window) => Arc::new(window),
            Err(error) => {
                eprintln!("manim_vk preview: failed to create window: {error}");
                if let Ok(mut slot) = self.error.lock() {
                    *slot = Some(format!("failed to create preview window: {error}"));
                }
                self.closing.store(true, Ordering::SeqCst);
                event_loop.exit();
                return;
            }
        };
        let size = window.inner_size();
        let surface = SurfaceTexture::new(size.width.max(1), size.height.max(1), window.clone());
        let pixels = PixelsBuilder::new(self.width, self.height, surface)
            .wgpu_backend(pixels::wgpu::Backends::VULKAN)
            .request_adapter_options(pixels::wgpu::RequestAdapterOptions {
                power_preference: pixels::wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .enable_vsync(true)
            .build();
        match pixels {
            Ok(pixels) => {
                if let Ok(mut name) = self.adapter_name.lock() {
                    *name = pixels.adapter().get_info().name.clone();
                }
                self.pixels = Some(pixels)
            }
            Err(error) => {
                eprintln!("manim_vk preview: failed to create Vulkan surface: {error}");
                if let Ok(mut slot) = self.error.lock() {
                    *slot = Some(format!("failed to create Vulkan preview surface: {error}"));
                }
                self.closing.store(true, Ordering::SeqCst);
                event_loop.exit();
                return;
            }
        }
        self.window = Some(window);
    }

    fn user_event(&mut self, event_loop: &ActiveEventLoop, event: UserEvent) {
        match event {
            UserEvent::Frame => {
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            UserEvent::Close => {
                self.closing.store(true, Ordering::SeqCst);
                event_loop.exit();
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                self.closing.store(true, Ordering::SeqCst);
                event_loop.exit();
            }
            WindowEvent::Resized(size) => {
                if let Some(pixels) = self.pixels.as_mut() {
                    let _ = pixels.resize_surface(size.width.max(1), size.height.max(1));
                }
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            WindowEvent::RedrawRequested => self.redraw(),
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        event_loop.set_control_flow(ControlFlow::Wait);
        let _ = self.proxy;
    }
}

#[pyclass(unsendable, name = "NativeWindow")]
pub struct NativeWindow {
    proxy: EventLoopProxy<UserEvent>,
    closing: Arc<AtomicBool>,
    pending: Arc<Mutex<Option<Vec<u8>>>>,
    adapter_name: Arc<Mutex<String>>,
    error: Arc<Mutex<Option<String>>>,
    width: u32,
    height: u32,
    thread: Option<JoinHandle<()>>,
}

#[pymethods]
impl NativeWindow {
    #[new]
    #[pyo3(signature = (width, height, title = "Manim Community", fullscreen = false, position = "", monitor = 0, adapter_name = ""))]
    fn new(
        width: u32,
        height: u32,
        title: &str,
        fullscreen: bool,
        position: &str,
        monitor: usize,
        adapter_name: &str,
    ) -> PyResult<Self> {
        if width == 0 || height == 0 {
            return Err(PyValueError::new_err("preview window size must be positive"));
        }
        let closing = Arc::new(AtomicBool::new(false));
        let pending = Arc::new(Mutex::new(None));
        let adapter_name_shared = Arc::new(Mutex::new(String::new()));
        let error = Arc::new(Mutex::new(None));
        let (tx, rx) = std::sync::mpsc::channel::<Result<EventLoopProxy<UserEvent>, String>>();

        let thread_closing = closing.clone();
        let thread_pending = pending.clone();
        let thread_adapter_name = adapter_name_shared.clone();
        let thread_error = error.clone();
        let thread_title = title.to_owned();
        let thread_position = position.to_owned();
        let thread_requested_adapter = adapter_name.to_owned();
        let handle = thread::Builder::new()
            .name("manim-vk-preview".into())
            .spawn(move || {
                let mut builder = EventLoop::<UserEvent>::with_user_event();
                #[cfg(target_os = "linux")]
                {
                    use winit::platform::x11::EventLoopBuilderExtX11;
                    builder.with_any_thread(true);
                }
                #[cfg(target_os = "windows")]
                {
                    use winit::platform::windows::EventLoopBuilderExtWindows;
                    builder.with_any_thread(true);
                }
                let event_loop = match builder.build() {
                    Ok(event_loop) => event_loop,
                    Err(error) => {
                        let _ = tx.send(Err(format!("create event loop: {error}")));
                        return;
                    }
                };
                let proxy = event_loop.create_proxy();
                if tx.send(Ok(proxy.clone())).is_err() {
                    return;
                }
                let mut app = PreviewApp {
                    width,
                    height,
                    title: thread_title,
                    fullscreen,
                    position: thread_position,
                    monitor,
                    requested_adapter: thread_requested_adapter,
                    proxy,
                    closing: thread_closing,
                    pending: thread_pending,
                    adapter_name: thread_adapter_name,
                    error: thread_error,
                    window: None,
                    pixels: None,
                };
                if let Err(error) = event_loop.run_app(&mut app) {
                    eprintln!("manim_vk preview: event loop failed: {error}");
                }
            })
            .map_err(|error| PyRuntimeError::new_err(format!("spawn preview thread: {error}")))?;

        let proxy = match rx.recv() {
            Ok(Ok(proxy)) => proxy,
            Ok(Err(message)) => {
                let _ = handle.join();
                return Err(PyRuntimeError::new_err(message));
            }
            Err(error) => {
                let _ = handle.join();
                return Err(PyRuntimeError::new_err(format!("preview thread setup: {error}")));
            }
        };

        Ok(Self {
            proxy,
            closing,
            pending,
            adapter_name: adapter_name_shared,
            error,
            width,
            height,
            thread: Some(handle),
        })
    }

    fn present_rgba(&self, pixels: &[u8]) -> PyResult<()> {
        let expected = self.width as usize * self.height as usize * 4;
        if pixels.len() != expected {
            return Err(PyValueError::new_err(format!(
                "preview frame has {} bytes, expected {expected}",
                pixels.len()
            )));
        }
        let mut slot = self
            .pending
            .lock()
            .map_err(|_| PyRuntimeError::new_err("preview frame lock poisoned"))?;
        *slot = Some(pixels.to_vec());
        drop(slot);
        self.proxy
            .send_event(UserEvent::Frame)
            .map_err(|_| PyRuntimeError::new_err("preview window is closed"))?;
        Ok(())
    }

    fn adapter_name(&self) -> String {
        self.adapter_name
            .lock()
            .map(|name| name.clone())
            .unwrap_or_default()
    }

    fn error(&self) -> Option<String> {
        self.error.lock().ok().and_then(|slot| slot.clone())
    }

    fn is_closing(&self) -> bool {
        self.closing.load(Ordering::SeqCst)
    }

    fn close(&mut self) {
        self.closing.store(true, Ordering::SeqCst);
        let _ = self.proxy.send_event(UserEvent::Close);
        if let Some(handle) = self.thread.take() {
            let _ = handle.join();
        }
    }

    fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }
}

impl Drop for NativeWindow {
    fn drop(&mut self) {
        self.close();
    }
}

