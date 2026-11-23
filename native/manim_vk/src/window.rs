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
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop, EventLoopProxy};
use winit::window::{Window, WindowId};

#[derive(Clone, Copy, Debug)]
enum UserEvent {
    Frame,
    Close,
}

struct PreviewApp {
    width: u32,
    height: u32,
    title: String,
    proxy: EventLoopProxy<UserEvent>,
    closing: Arc<AtomicBool>,
    pending: Arc<Mutex<Option<Vec<u8>>>>,
    adapter_name: Arc<Mutex<String>>,
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
        let attributes = Window::default_attributes()
            .with_title(self.title.clone())
            .with_inner_size(winit::dpi::PhysicalSize::new(self.width, self.height))
            .with_resizable(true);
        let window = match event_loop.create_window(attributes) {
            Ok(window) => Arc::new(window),
            Err(error) => {
                eprintln!("manim_vk preview: failed to create window: {error}");
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
    width: u32,
    height: u32,
    thread: Option<JoinHandle<()>>,
}

#[pymethods]
impl NativeWindow {
    #[new]
    #[pyo3(signature = (width, height, title = "Manim Community"))]
    fn new(width: u32, height: u32, title: &str) -> PyResult<Self> {
        if width == 0 || height == 0 {
            return Err(PyValueError::new_err("preview window size must be positive"));
        }
        let closing = Arc::new(AtomicBool::new(false));
        let pending = Arc::new(Mutex::new(None));
        let adapter_name = Arc::new(Mutex::new(String::new()));
        let (tx, rx) = std::sync::mpsc::channel::<Result<EventLoopProxy<UserEvent>, String>>();

        let thread_closing = closing.clone();
        let thread_pending = pending.clone();
        let thread_adapter_name = adapter_name.clone();
        let thread_title = title.to_owned();
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
                    proxy,
                    closing: thread_closing,
                    pending: thread_pending,
                    adapter_name: thread_adapter_name,
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
            adapter_name,
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

