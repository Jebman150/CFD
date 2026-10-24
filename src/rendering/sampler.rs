use wgpu::Color;

use crate::{rendering::sampler::{parameter::{ColorGradient, VisualParameter}, settings::{RenderSettings, VisModeFields}}, simulation::data::{SimulationData, grid::multi_index::CellIndex}};


mod parameter;
pub mod settings;

pub struct Sampler {
    visuals: VisualParameter,
}

fn col_from_rgba(r: u8, g: u8, b: u8, a: u8) -> Color {
    let scale_factor = 1.0/255.0;
    Color {
        r: r as f64 * scale_factor,
        g: g as f64 * scale_factor,
        b: b as f64 * scale_factor,
        a: a as f64 * scale_factor,
    }
}

impl Sampler {
    pub fn new() -> Self {
        let visuals: VisualParameter = VisualParameter {
            fluid_cell: col_from_rgba(30, 30, 30, 255),
            solid_cell: col_from_rgba(10, 10, 10, 255),
            smoke_color: col_from_rgba(200, 200, 200, 200),
            divergence_gradient: ColorGradient {
                low: col_from_rgba(0, 0, 255, 255),
                mid: col_from_rgba(0, 0, 0, 255),
                high: col_from_rgba(255, 0, 0, 255),
                low_val: -5.0,
                high_val: 5.0,
            },
            pressure_gradient: ColorGradient {
                low: col_from_rgba(0, 0, 255, 255),
                mid: col_from_rgba(0, 0, 0, 255),
                high: col_from_rgba(255, 0, 0, 255),
                low_val: -1.0,
                high_val: 1.0,
            },
            velocity_gradient: ColorGradient {
                low: col_from_rgba(0, 0, 200, 255),
                mid: col_from_rgba(0, 200, 0, 255),
                high: col_from_rgba(200, 0, 0, 255),
                low_val: 0.0,
                high_val: 2.0,
            }
        };

        Self {
            visuals,
        }
    }

    pub fn sample_cell_color(&self, sim_data: &SimulationData, settings: &RenderSettings, cell_idx: &CellIndex<2>) -> Color {
        match settings.vis_mode_fields {
            VisModeFields::None => self.cell_color_eval(sim_data, cell_idx),
            VisModeFields::Velocity => self.velocity_eval(sim_data, cell_idx),
            VisModeFields::Divergence => self.divergence_eval(sim_data, cell_idx),
            VisModeFields::Pressure => self.pressure_eval(sim_data, cell_idx),
                    }
        /*
        if settings.render_streamlines {
            self.draw_streamlines(state);
        }
        if settings.render_hr_field {
            self.draw_hr_field(&state.smoke, state.grid.get_delta_x(), [50, 50]);
        }
        if settings.render_velocity_debug {
            self.draw_velocity_debug(state);
        }
        */
        //self.draw_smoke_map(&state, &grid);

    }

    fn cell_color_eval(
        &self,
        state: &SimulationData,
        idx: &CellIndex<2>
    ) -> Color {
        if *state.is_solid.get(idx) {
            self.visuals.solid_cell
        } else {
            self.visuals.fluid_cell
        }
    }

    fn velocity_eval(
        &self,
        state: &SimulationData,
        idx: &CellIndex<2>
    ) -> Color {
        let [vel_x, vel_y] = Self::get_velocity(state, &idx);

        self.visuals.velocity_gradient.get(vel_x.abs() + vel_y.abs())
    }

    fn get_velocity(state: &SimulationData, cell_idx: &CellIndex<2>) -> [f32; 2] {
        let x_faces = state.grid.get_face_indices(&cell_idx, 0);
        let vel_x = (state.velocity[0].get(&x_faces[0]) + state.velocity[0].get(&x_faces[1])) * 0.5;
        let y_faces = state.grid.get_face_indices(&cell_idx, 1);
        let vel_y = (state.velocity[1].get(&y_faces[0]) + state.velocity[1].get(&y_faces[1])) * 0.5;
        [vel_x, vel_y]
    }

    fn divergence_eval(
        &self,
        state: &SimulationData,
        idx: &CellIndex<2>
    ) -> Color {
        self.visuals.divergence_gradient.get(*state.divergence.get(&idx))
    }

    fn pressure_eval(
        &self,
        state: &SimulationData,
        idx: &CellIndex<2>
    ) -> Color {
        self.visuals.pressure_gradient.get(-*state.pressure.get(&idx))
    }

    /*

    fn draw_smoke_map(&self, state: &SimulationData) {
        let cell_width = self.cellsize.0 - self.padding * 2.0 - 1.0;
        let cell_height = self.cellsize.1 - self.padding * 2.0 - 1.0;
        for cell_idx in state.grid.all_cells() {
            let pos_x = self.viewport_pos.0 + (cell_idx.idx[0] as f32) * self.cellsize.0 + self.padding;
            let pos_y = self.viewport_pos.1 + (cell_idx.idx[1] as f32) * self.cellsize.1 + self.padding;

            let color = Self::scale_color(&self.visuals.smoke_color, *state.smoke.get(&cell_idx));

            draw_rectangle(pos_x, pos_y, cell_width, cell_height, color);
        }
    }

    fn draw_hr_field(
        &self,
        data: &CellField2F,
        grid_dx: [f32; 2],
        res: [usize; 2],
    ) {
        let nx = data.layout.dimensions()[0] as f32;
        let ny = data.layout.dimensions()[1] as f32;

        let dx = 1.0 / res[0] as f32;
        let dy = 1.0 / res[1] as f32;

        let cell_width = self.viewport_size.0 * dx - self.padding * 2.0;
        let cell_height = self.viewport_size.1 * dy - self.padding * 2.0;

        for x in 0..res[0] {
            for y in 0..res[1] {

                // normalized coordinates in [0,1]
                let u = x as f32 * dx;
                let v = y as f32 * dy;

                // map to grid physical space
                let pos = [
                    (u * nx) as f32 * grid_dx[0],
                    (v * ny) as f32 * grid_dx[1],
                ];

                let pos_x = self.viewport_pos.0 + u * self.viewport_size.0;
                let pos_y = self.viewport_pos.1 + v * self.viewport_size.1;

                let value = data.sample(pos, grid_dx) as f32;

                let color = Self::scale_color(&self.visuals.smoke_color, value);

                draw_rectangle(pos_x, pos_y, cell_width, cell_height, color);
            }
        }
    }

    fn draw_velocity_debug(&self, state: &SimulationData) {
        let bar_thickness: f32 = 5.0;
        for axis in 0..2 {
            let offset =
            if axis == 1 {
                (self.cellsize.0 / 2.0 - bar_thickness / 2.0, 0.0)
            } else {
                (0.0, self.cellsize.1 / 2.0 - bar_thickness / 2.0)
            };

            for face_idx in state.grid.all_faces(axis) {
                let pos_x = self.viewport_pos.0 + (face_idx.idx[0] as f32) * self.cellsize.0 + offset.0 + self.padding;
                let pos_y = self.viewport_pos.1 + (face_idx.idx[1] as f32) * self.cellsize.1 + offset.1 + self.padding;

                let bar_length = (state.velocity[axis].get(&face_idx) * 10.0) as f32;

                let bar_height = 
                if axis == 0 {
                    bar_thickness
                } else {
                    bar_length
                };

                let bar_width = 
                if axis == 1 {
                    bar_thickness
                } else {
                    bar_length
                };

                draw_rectangle(pos_x, pos_y, bar_width, bar_height, BLUE);
            }

            
        }
    }

    fn draw_streamlines(&self, state: &SimulationData) {
        let line_thickness: f32 = 2.0;

        let nx = state.pressure.layout.dimensions()[0];
        let ny = state.pressure.layout.dimensions()[1];

        let dx = 1.0 / nx as f32;
        let dy = 1.0 / ny as f32;

        let dt = 0.01;

        for x in 0..nx {
            for y in 0..ny {
                let cell_idx = CellIndex::from_index([x, y]);
                let [vel_x, vel_y] = Self::get_velocity(state, &cell_idx);

                let line_start = (
                    self.viewport_pos.0 + (x as f32 * dx + 0.5 * dx) * self.viewport_size.0 + self.padding,
                    self.viewport_pos.1 + (y as f32 * dy + 0.5 * dy) * self.viewport_size.1 + self.padding,
                );

                let line_end = (
                    self.viewport_pos.0 + (x as f32 * dx + vel_x * dt + 0.5 * dx) * self.viewport_size.0 + self.padding,
                    self.viewport_pos.1 + (y as f32 * dy + vel_y * dt + 0.5 * dy) * self.viewport_size.1 + self.padding,
                );

                draw_line(line_start.0, line_start.1, line_end.0, line_end.1, line_thickness, GREEN);
            }  
        }
    }

    fn scale_color(col: &Color, value: f32) -> Color {
        Color::from_rgba(
            (col.r * value * 255.0) as u8,
            (col.g * value * 255.0) as u8,
            (col.b * value * 255.0) as u8,
            (col.a * value * 255.0) as u8)
    }
    */
}