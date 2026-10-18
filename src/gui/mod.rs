use egui_macroquad::egui::{self, Context};
use strum::{self, IntoEnumIterator};

use crate::{rendering::{RenderSettings, VisModeFields}, simulation::{SimulationSettings, pipeline::operator::solver_registry::SolverRegistry}};


pub struct GUI {
}

impl GUI {
    pub fn new() -> Self {
        Self {
        }
    }

    pub fn get_render_settings(&self, ctx: &Context, settings: &mut RenderSettings) {
        egui::Window::new("Render settings")
            .show(ctx, |ui| {
                ui.label("Render settings");
                ui.checkbox(&mut settings.render_cells, "Draw cells");
                ui.checkbox(&mut settings.render_streamlines, "Draw streamlines");
                ui.checkbox(&mut settings.render_velocity_debug, "Draw velocity debug");
                ui.checkbox(&mut settings.render_hr_field, "Draw hr field");

                egui::ComboBox::from_label("Visual mode (fields)")
                    .selected_text(<&str>::from(&settings.vis_mode_fields))
                    .show_ui(ui, |ui| {
                        for vis_mode in VisModeFields::iter() {
                            ui.selectable_value(&mut settings.vis_mode_fields, vis_mode.clone(), <&str>::from(&vis_mode));
                        }
                    })
            });
    }

    pub fn get_simulation_settings(&self, ctx: &Context, settings: &mut SimulationSettings, solver_registry: &SolverRegistry) {
        egui::Window::new("Simulation settings")
            .show(ctx, |ui| {
                ui.checkbox(&mut settings.is_running, "Run", );
                if ui.button("Simulate step").clicked() {
                    settings.step = true;
                }
                

                egui::CollapsingHeader::new("Solver settings")
                    .default_open(false)
                    .show(ui, |ui| {
                        egui::ComboBox::from_label("Solving algorithm")
                            .selected_text(solver_registry.get_solver_name(settings.solver_id).unwrap())
                            .show_ui(ui, |ui| {
                                for (id, name) in solver_registry.list_solver() {
                                    ui.selectable_value(&mut settings.solver_id, id, name);
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
                    })
            });
    }
}