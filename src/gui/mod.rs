use macroquad::{math::{Vec2, vec2}, ui::{hash, root_ui}};

use crate::{rendering::RenderSettings, simulation::SimulationSettings};


pub struct GUI {
}

impl GUI {
    pub fn new() -> Self {
        Self {
        }
    }

    pub fn get_render_settings(&self, settings: &mut RenderSettings) {
        root_ui().window(hash!(), vec2(10.0, 10.0), vec2(150.0, 500.0),
        |ui| {
            ui.checkbox(hash!(), "Draw cells", &mut settings.render_cells);
            ui.checkbox(hash!(), "Draw streamlines", &mut settings.render_streamlines);
            ui.checkbox(hash!(), "Draw velocity debug", &mut settings.render_velocity_debug);
            ui.checkbox(hash!(), "Draw divergence", &mut settings.render_divergence);
            ui.checkbox(hash!(), "Draw pressure", &mut settings.render_pressure);
            ui.checkbox(hash!(), "Draw hr field", &mut settings.render_hr_field);
        });
    }

    pub fn get_simulation_settings(&self, settings: &mut SimulationSettings) {
        root_ui().window(hash!(), vec2(10.0, 500.0), vec2(150.0, 500.0),
        |ui| {
            ui.checkbox(hash!(), "Run", &mut settings.is_running);
            if ui.button(Vec2::new(10.0, 30.0), "Simulate step") {
                settings.step = true;
            }
        });
    }
}