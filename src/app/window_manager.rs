use std::sync::Arc;

use wgpu::{Surface, SurfaceConfiguration, SurfaceTexture};
use winit::{dpi::PhysicalSize, event_loop::ActiveEventLoop, window::Window};

use crate::app::GPUContext;

pub struct WindowState {
    window: Arc<Window>,
    surface: Surface<'static>,
    config: SurfaceConfiguration,
    is_surface_configured: bool
}

pub struct WindowManager {
    state: Option<WindowState>,
}

impl WindowManager {
    pub fn new() -> Self {
        Self {
            state: None,
        }
    }

    pub fn resume(&mut self, event_loop: &ActiveEventLoop, context: &GPUContext) -> anyhow::Result<()> {
        let window_attributes = Window::default_attributes();
        let window = Arc::new(event_loop.create_window(window_attributes).unwrap());
        let size = window.inner_size();

        let surface = context.instance.create_surface(window.clone())?;
        let surface_caps = surface.get_capabilities(&context.adapter);
        let surface_format = surface_caps.formats.iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
            color_space: wgpu::SurfaceColorSpace::Auto,
        };

        self.state = Some(WindowState {
            window,
            surface,
            config,
            is_surface_configured: false
        });
        Ok(())
    }

    pub fn active(&mut self) -> Option<&mut WindowState> {
        self.state.as_mut()
    }

    pub fn active_ref(&self) -> Option<&WindowState> {
        self.state.as_ref()
    }
}

impl WindowState {

    pub fn get_window(&self) -> Arc<Window> {
        self.window.clone()
    }

    pub fn resize(&mut self, width: u32, height: u32, gpu_context: &GPUContext) {
        if width > 0 && height > 0 {
            self.config.width = width;
            self.config.height = height;
            self.surface.configure(&gpu_context.device, &self.config);
            self.is_surface_configured = true;
        }
    }

    pub fn start_frame(&mut self) {
        self.window.request_redraw()
    }

    pub fn get_window_size(&self) -> PhysicalSize<u32> {
        self.window.inner_size()
    }

    pub fn get_current_texture(&mut self, gpu_context: &GPUContext) -> Option<SurfaceTexture> {

        let out = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(surface_texture) => surface_texture,
            wgpu::CurrentSurfaceTexture::Suboptimal(surface_texture) => {
                surface_texture
            }
            wgpu::CurrentSurfaceTexture::Timeout
            | wgpu::CurrentSurfaceTexture::Occluded
            | wgpu::CurrentSurfaceTexture::Validation => {
                return None;
            }
            wgpu::CurrentSurfaceTexture::Outdated => {
                self.surface.configure(&gpu_context.device, &self.config);
                return None;
            }
            wgpu::CurrentSurfaceTexture::Lost => {
                tracing::error!("Lost gpu device!");
                return None;
            }
        };
        Some(out)
    }

    pub fn get_surface_config(&self) -> &SurfaceConfiguration {
        &self.config
    }
}