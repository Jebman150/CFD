use std::sync::Arc;

use wgpu::{Color, CommandEncoder, Device, Queue, SurfaceConfiguration, TextureView};
use winit::dpi::PhysicalSize;

use crate::{rendering::renderer::{frame_rendering::frame_dispatcher::{Frame, FrameDispatcher}, globals::RenderGlobals, mesh_registry::MeshRegistry, scene::Scene, settings::RenderSettings}, simulation::data::SimulationData};

mod primitives;

pub mod scene;
mod globals;
mod mesh_registry;
mod frame_rendering;

pub mod settings;


pub struct Renderer {
    settings: RenderSettings,
    render_globals: RenderGlobals,
    pub scene: Scene,
    mesh_registry: MeshRegistry,
    frame_dispatcher: Option<FrameDispatcher>,
}

impl Renderer {
    pub fn new(device: Arc<Device>) -> Self {

        Self {
            settings: RenderSettings::default(&device),
            render_globals: Default::default(),
            scene: Scene::new(),
            mesh_registry: MeshRegistry::new(device),
            frame_dispatcher: None,
        }
    }

    pub fn add_rectangle (
        &mut self,
        w: f32,
        h: f32
    ) -> scene::mesh::MeshId {
        let mesh = primitives::rectangle(w, h, Color::WHITE);
        self.mesh_registry.register_mesh(mesh)
    }

    pub fn add_instance(
        &mut self,
        id: scene::mesh::MeshId,
        x: f32,
        y: f32,
        color: Color,
    ) -> scene::InstanceId {
        self.scene.add_instance(id,
            scene::Instance::new(x, y, color)
        )
    }

    pub fn clear_meshes(&mut self) {
        self.mesh_registry.clear();
    }

    pub fn clear_scene(&mut self) {
        self.scene.instances.clear();
    }

    pub fn render(&mut self, target: &TextureView, encoder: &mut CommandEncoder) {
        let Some(frame_dispatcher) = self.frame_dispatcher.as_mut() else { return; };
        let prepared_frame = self.scene.prepare_frame();
        frame_dispatcher.prepare_instance_buffer(&prepared_frame);
        let mut render_pass = frame_dispatcher.initialize_pass(encoder, target);
        render_pass.set_viewport(
            self.render_globals.simulation_viewport.position[0],
            self.render_globals.simulation_viewport.position[1],
            self.render_globals.simulation_viewport.size[0],
            self.render_globals.simulation_viewport.size[1], 0.0, 1.0);
        let frame = Frame {
            render_pass,
            prepared_data: prepared_frame,
            rendered_scalar_field: self.settings.rendered_scalar_field
        };
        match frame_dispatcher.dispatch_grid_kernel(frame, &self.mesh_registry) {
            Err(msg) => tracing::error!("Grid kernel failed with msg '{}'", msg),
            Ok(()) => {},
        };
    }

    pub fn update_viewports(&mut self, window_size: &PhysicalSize<u32>) {
        self.render_globals.simulation_viewport.set_size([
            window_size.width as f32,
            window_size.height as f32
        ]);
        let Some(frame_dispatcher) = self.frame_dispatcher.as_mut() else { return; };
        frame_dispatcher.update_viewport(&self.render_globals.simulation_viewport);
    }

    pub fn create_render_pipeline(&mut self, device: Arc<Device>, queue: Arc<Queue>, surface_config: &SurfaceConfiguration, simulation_data: &SimulationData) {
        self.frame_dispatcher = Some(FrameDispatcher::new(
            device,
            queue,
            surface_config,
            simulation_data,
            &self.settings.visuals
        ));
    }

    pub fn settings_mut(&mut self) -> &mut RenderSettings {
        &mut self.settings
    }

    pub fn settings(&self) -> &RenderSettings {
        &self.settings
    }
}