use std::sync::Arc;

use egui_winit::EventResponse;
use pollster::FutureExt;
use tracing::Level;
use wgpu::{Color, CommandEncoder};
use winit::{application::ApplicationHandler, event::{KeyEvent, WindowEvent}, event_loop::ActiveEventLoop, keyboard::{KeyCode, PhysicalKey}};

use crate::{
    app::{
        gpu_context::GPUContext, 
        window_manager::WindowManager
    }, gui::GUI, rendering::renderer::{
        Renderer,
        scene::{
            InstanceId,
            mesh::MeshId
        },
    }, simulation::{Simulation, data::ScalarField}
};

mod window_manager;
pub mod gpu_context;

pub struct Application {
    pub gpu_context: Arc<GPUContext>,
    pub gui: GUI,
    pub renderer: Renderer,
    pub window_manager: WindowManager,

    pub simulation: Simulation,

    pub grid_cell_mesh: MeshId,
    pub grid_cell_instance_range: [InstanceId; 2]
}

impl Application {
    pub fn new() -> anyhow::Result<Self> {
        let gpu_context = Arc::new(GPUContext::new().block_on()?);

        Ok(Self {
            renderer: Renderer::new(gpu_context.device.clone()),
            gui: GUI::new(gpu_context.device.clone(), gpu_context.queue.clone()),
            window_manager: WindowManager::new(),

            simulation: Simulation::new(gpu_context.clone()),

            grid_cell_mesh: MeshId(0),
            grid_cell_instance_range: [InstanceId(0), InstanceId(0)],
            gpu_context,
        })
    }

    pub fn rebuild_scene(&mut self) {
        self.renderer.clear_scene();
        self.renderer.clear_meshes();

        let x_res = self.simulation.settings().grid_size[0];
        let y_res = self.simulation.settings().grid_size[1];

        let w = 400.0;
        let h = 400.0;

        let dx = w / x_res as f32;
        let dy = h / y_res as f32;

        let grid_cell_mesh = self.renderer.add_rectangle(dx, dy);

        for y in 0..y_res {
            let y_frac = y as f32 / y_res as f32;
            for x in 0..x_res {
                let x_frac = x as f32 / x_res as f32;
                self.grid_cell_instance_range[1] = self.renderer.add_instance(
                    grid_cell_mesh,
                    w * x_frac + 10.0,
                    h * y_frac + 10.0,
                    Color{r: x_frac as f64, g: 0.0, b: y_frac as f64, a: 1.0});
            }
        }
    }

    pub fn update(&mut self) {
        let mut encoder = self.gpu_context.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("Frame encoder"),
            },
        );

        self.update_gui();
        self.update_simulation(&mut encoder);
        self.update_scene();
        self.present(encoder);
    }

    fn update_gui(&mut self) {
        puffin::profile_function!();
        let Some(gui_state) = self.gui.active() else { return; };
        let solver_names = self.simulation.get_solver_names();
        gui_state.run_ui(
            &mut self.renderer.settings_mut(),
            self.simulation.settings_mut(),
            &solver_names
        );
    }

    fn update_simulation(&mut self, encoder: &mut CommandEncoder) {
        puffin::profile_function!();
        if let Some(scalar_field) = self.renderer.settings().rendered_scalar_field {
            self.simulation.update(encoder, scalar_field);
        }
    }

    fn update_scene(&mut self) {
        puffin::profile_function!();
        let instance_range = self.grid_cell_instance_range[0].0..self.grid_cell_instance_range[1].0;

        for (_cell, instance_idx) in self.simulation.grid.all_cells().zip(instance_range) {
            self.renderer.scene.set_color(
                self.grid_cell_mesh,
                InstanceId(instance_idx),
                Color::RED,
                //self.sampler.sample_cell_color(frame_data, &self.render_settings, &cell)
            );
        }
    }

    pub fn present(&mut self, mut encoder: CommandEncoder) {
        puffin::profile_function!();
        let Some(window_state) = self.window_manager.active() else { return; };
        window_state.start_frame();

        let Some(frame) = window_state.get_current_texture(&self.gpu_context) else { return; };
        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());

        self.renderer.render(&view, &mut encoder);
        let mut command_buffer = self.gui.render(&view, &mut encoder, window_state.get_surface_config());

        command_buffer.push(encoder.finish());
        self.gpu_context.queue.submit(command_buffer);
        self.gpu_context.queue.present(frame);
        
        puffin::GlobalProfiler::lock().new_frame();
    }

    pub fn handle_key(&mut self, event_loop: &ActiveEventLoop, key_code: KeyCode, is_pressed: bool) {
        match (key_code, is_pressed) {
            (KeyCode::Escape, true) => event_loop.exit(),
            _ => {}
        }
    }
}

pub fn read_back_data<T: bytemuck::Pod>(gpu_context: &crate::app::gpu_context::GPUContext, buffer: &wgpu::Buffer, out: &mut [T]) {
    let staging_buffer = gpu_context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging buffer"),
        size: buffer.size(),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = gpu_context.device.create_command_encoder(&wgpu::CommandEncoderDescriptor{
        label: Some("Readback encoder"),
    });

    encoder.copy_buffer_to_buffer(
        buffer,
        0,
        &staging_buffer,
        0,
        buffer.size(),
    );

    gpu_context.queue.submit(Some(encoder.finish()));

    let buffer_slice = staging_buffer.slice(..);

    buffer_slice.map_async(wgpu::MapMode::Read, |_| {});

    let _ = gpu_context.device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None
    });

    let data = buffer_slice.get_mapped_range().unwrap();

    out.copy_from_slice(bytemuck::cast_slice(&data));

    drop(data);
    staging_buffer.unmap();
}

impl ApplicationHandler for Application {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        match self.window_manager.resume(event_loop, &self.gpu_context) {
            Ok(()) => {},
            Err(err) => {
                tracing::event!(Level::ERROR, "Could not resume: {}", err);
                return;
            }
        }

        let Some(window_state) = self.window_manager.active_ref() else { tracing::error!("No window after resume?"); return; };

        self.renderer.create_render_pipeline(
            self.gpu_context.device.clone(),
            self.gpu_context.queue.clone(),
            window_state.get_surface_config(),
            self.simulation.state_ref()
        );
        self.gui.resumed(window_state.get_window(), window_state.get_surface_config());
        self.renderer.update_viewports(&window_state.get_window_size());
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        let Some(window_state) = self.window_manager.active() else { return; };


        let gui_response = 
        if let Some(gui_state) = self.gui.active() {
            gui_state.on_window_event(window_state.get_window(), &event)
        } else {
            EventResponse { consumed: false, repaint: false }
        };

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                window_state.resize(size.width, size.height, &self.gpu_context);
                self.renderer.update_viewports(&window_state.get_window_size());
            }
            WindowEvent::RedrawRequested => {
                self.update();
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(code),
                        state: key_state,
                        ..
                    },
                ..
            } => if !gui_response.consumed { self.handle_key(event_loop, code, key_state.is_pressed()) },
            _ => {}
        }
    }
}