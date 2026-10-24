use std::{collections::HashMap, sync::Arc};

use egui::{ClippedPrimitive, FullOutput};
use egui_winit::EventResponse;
use strum::IntoEnumIterator;
use wgpu::{CommandBuffer, CommandEncoder, Device, Queue, SurfaceConfiguration, TextureView};
use winit::{event::WindowEvent, window::Window};

use crate::{rendering::renderer::settings::RenderSettings, simulation::{BackendDevice, data::{ScalarField, SimulationSettings}}};


pub struct GUIState {
    pub egui_ctx: egui::Context,
    pub egui_winit: egui_winit::State,
    pub egui_renderer: egui_wgpu::Renderer,
    pub window: Arc<Window>,

    full_output: Option<FullOutput>,
    paint_jobs: Vec<ClippedPrimitive>,
}

pub struct GUI {
    pub device: Arc<Device>,
    pub queue: Arc<Queue>,

    state: Option<GUIState>,
}

impl GUI {
    pub fn new(device: Arc<Device>, queue: Arc<Queue>) -> Self {

        Self {
            state: None,
            device,
            queue,
        }
    }

    pub fn resumed(&mut self, window: Arc<Window>, surface_config: &SurfaceConfiguration) {
        let egui_ctx = egui::Context::default();

        let egui_winit = egui_winit::State::new(
            egui_ctx.clone(),
            egui::ViewportId::ROOT,
            &window,
            Some(window.scale_factor() as f32),
            None,
            None,
        );

        let egui_renderer = egui_wgpu::Renderer::new(
            &self.device,
            surface_config.format,
            egui_wgpu::RendererOptions::default(),
        );
        
        self.state = Some(GUIState {
            egui_ctx,
            egui_winit,
            egui_renderer,
            window,
            full_output: None,
            paint_jobs: Vec::new()
        });
    }

    pub fn active(&mut self) -> Option<&mut GUIState> {
        self.state.as_mut()
    }

    pub fn render(&mut self, target: &TextureView, encoder: &mut CommandEncoder, surface_config: &SurfaceConfiguration) -> Vec<CommandBuffer> {
        let Some(state) = self.state.as_mut() else { return Vec::new(); };
        let Some(mut full_output) = state.full_output.take() else { return Vec::new(); };
        for (id, deltas) in &full_output.textures_delta.set {
            for delta in deltas {
                state.egui_renderer.update_texture(
                    &self.device,
                    &self.queue,
                    *id,
                    delta,
                );
            }
        }

        let screen_descriptor = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [
                surface_config.width,
                surface_config.height,
            ],
            pixels_per_point: full_output.pixels_per_point,
        };

        let user_command_buffers = state.egui_renderer.update_buffers(
            &self.device,
            &self.queue,
            encoder,
            &state.paint_jobs,
            &screen_descriptor
        );

        let render_pass = encoder.begin_render_pass(
            &wgpu::RenderPassDescriptor {
                label: Some("egui render pass"),
                color_attachments: &[Some(
                    wgpu::RenderPassColorAttachment {
                        view: &target,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    },
                )],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            },
        );

        let mut render_pass = render_pass.forget_lifetime();
        state.egui_renderer.render(
            &mut render_pass,
            &state.paint_jobs,
            &screen_descriptor,
        );

        for id in &full_output.textures_delta.free {
            state.egui_renderer.free_texture(id);
        }

        full_output.textures_delta.clear();

        user_command_buffers
    }
}

impl GUIState {
    pub fn on_window_event(&mut self, window: Arc<Window>, event: &WindowEvent) -> EventResponse {
        self.egui_winit.on_window_event(&window, event)
    }

    pub fn run_ui(
        &mut self,
        render_settings: &mut RenderSettings,
        simulation_settings: &mut SimulationSettings,
        solver_map: &HashMap<usize, &str>,
    ) {
        let raw_input = self.egui_winit.take_egui_input(&self.window);

        let mut full_output = self.egui_ctx.run_ui(raw_input, |ctx| {
            self.get_render_settings(ctx, render_settings);
            self.get_simulation_settings(ctx, simulation_settings, solver_map);
        });
        self.create_paint_jobs(&mut full_output);
        self.full_output = Some(full_output);
    }

    pub fn create_paint_jobs(&mut self, full_output: &mut FullOutput) {
        self.egui_winit.handle_platform_output(
            &self.window,
            std::mem::take(&mut full_output.platform_output),
        );

        let shapes = std::mem::take(&mut full_output.shapes);

        self.paint_jobs = self.egui_ctx.tessellate(
            shapes,
            full_output.pixels_per_point,
        );
    }

    pub fn get_render_settings(
        &self,
        ctx: &egui::Context,
        settings: &mut RenderSettings,
    ) {
        egui::Window::new("Renderer settings")
            .default_width(280.0)
            .show(ctx, |ui| {
                ui.heading("Render settings");

                let current_label = if let Some(scalar_field) = &settings.rendered_scalar_field {
                    <&str>::from(scalar_field)
                } else {
                    "None"
                };

                egui::ComboBox::from_label("Rendered scalar field")
                    .selected_text(current_label)
                    .show_ui(ui, |ui| {
                        for vis_mode in ScalarField::iter() {
                            ui.selectable_value(&mut settings.rendered_scalar_field, Some(vis_mode.clone()), <&str>::from(&vis_mode));
                        }
                        ui.selectable_value(&mut settings.rendered_scalar_field, None, "None");
                    });
            });
    }

    pub fn get_simulation_settings(
        &self,
        ctx: &egui::Context,
        settings: &mut SimulationSettings,
        solver_map: &HashMap<usize, &str>,
    ) {
        egui::Window::new("Simulation settings")
            .show(ctx, |ui| {
                ui.checkbox(&mut settings.is_running, "Run", );
                settings.step = ui.button("Simulate step").clicked();
                

                egui::CollapsingHeader::new("Solver settings")
                    .default_open(false)
                    .show(ui, |ui| {
                        egui::ComboBox::from_label("Solving algorithm")
                            .selected_text(*solver_map.get(&settings.solver_id).unwrap_or(&"No available solver"))
                            .show_ui(ui, |ui| {
                                for (id, name) in solver_map.iter() {
                                    ui.selectable_value(&mut settings.solver_id, *id, *name);
                                }
                            });
                        ui.label("Maximum iterations:");
                        let mut input_text = format!("{}", settings.solver_settings.maxit);
                        let response = ui.add(egui::TextEdit::singleline(&mut input_text));
                        if response.changed() {
                            if let Ok(parsed_val) = input_text.parse::<usize>() {
                                settings.solver_settings.maxit = parsed_val;
                            }
                        }
                        ui.label("Error tolerance:");
                        input_text = format!("{}", settings.solver_settings.tolerance);
                        let response = ui.add(egui::TextEdit::singleline(&mut input_text));
                        if response.changed() {
                            if let Ok(parsed_val) = input_text.parse::<f32>() {
                                settings.solver_settings.tolerance = parsed_val;
                            }
                        }
                    });
                
                    egui::ComboBox::from_label("Device")
                        .selected_text(<&str>::from(settings.device.clone()))
                        .show_ui(ui, |ui| {
                            for device in BackendDevice::iter() {
                                ui.selectable_value(&mut settings.device, device.clone(), <&str>::from(device));
                            }
                        });
            });
    }
}