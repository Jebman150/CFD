use std::thread;
use std::time::Duration;

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
            self.gui.get_render_settings(&mut self.render_settings);
            self.gui.get_simulation_settings(&mut self.simulation_settings);

            self.controller.step(&self.simulation_settings);
            self.simulation_settings.step = false;

            self.renderer.draw(&self.controller.data, &self.controller.topology, &self.render_settings);

            //thread::sleep(Duration::from_secs(1));
            next_frame().await;
        }
        Ok(())
    }
}