use macroquad::time::draw_fps;
use macroquad::window::next_frame;

use crate::gui::GUI;
use crate::simulation::{SimulationController, SimulationSettings};
use crate::rendering::{RenderSettings, Renderer};

pub struct Application {
    pub gui: GUI,
    pub controller: SimulationController,
    pub renderer: Renderer,

    pub is_running: bool,

    render_settings: RenderSettings,
    simulation_settings: SimulationSettings,
}

impl Application {
    pub fn new() -> Self {
        Self {
            gui: GUI::new(),
            controller: SimulationController::new(),
            renderer: Renderer::new(),
            is_running: true,

            render_settings: Default::default(),
            simulation_settings: Default::default(),
        }
    }

    pub async fn run(&mut self) -> Result<(), ()> {
        while self.is_running {
            egui_macroquad::ui(|ctx: &egui_macroquad::egui::Context| {
                ctx.set_visuals(egui_macroquad::egui::Visuals::dark());
                self.gui.get_render_settings(ctx, &mut self.render_settings);
                self.gui.get_simulation_settings(ctx, &mut self.simulation_settings, &self.controller.solver_registry);
            });

            self.controller.step(&self.simulation_settings);
            self.simulation_settings.step = false;

            self.renderer.draw(&self.controller.data, &self.render_settings);

            egui_macroquad::draw();
            if self.simulation_settings.is_running {
                puffin::GlobalProfiler::lock().new_frame();
            }
            draw_fps();
            next_frame().await;
        }
        Ok(())
    }
}