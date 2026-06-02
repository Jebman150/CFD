use macroquad::prelude::*;

mod parameter;

use crate::{engine::{grid::{Grid2D, multi_index::CellIndex}, state::{State2D, field::Field}}, rendering::parameter::VisualParameter};

pub struct Renderer {
    window_size: (f32, f32),
    viewport_pos: (f32, f32),
    viewport_size: (f32, f32),

    cellsize: (f32, f32),
    padding: f32,
    visuals: VisualParameter,
}

impl Renderer {
    pub fn new() -> Self {
        let params: VisualParameter = VisualParameter {
            fluid_cell: Color::from_rgba(30, 30, 30, 255),
            solid_cell: Color::from_rgba(10, 10, 10, 255),
            smoke_color: Color::from_rgba(200, 200, 200, 200),
        };

        Self {
            window_size: (screen_width(), screen_height()),
            viewport_size: (screen_width() - 40.0, screen_height() - 40.0),
            viewport_pos: (20.0, 20.0),
            cellsize: (5.0, 5.0),
            padding: 0.0,
            visuals: params
        }
    }

    pub fn update(&mut self, state: &State2D, grid: &Grid2D) {
        self.clear();

        self.window_size = (screen_width(), screen_height());
        self.viewport_size = (screen_width() - 40.0, screen_height() - 40.0);
        self.cellsize = self.viewport_size;
        self.cellsize.0 /= grid.get_outer_size()[0] as f32;
        self.cellsize.1 /= grid.get_outer_size()[1] as f32;


        self.draw_cells(&state, &grid);
        self.draw_smoke_map(&state, &grid);
        //self.draw_hr_field(&state.smoke, grid, [50, 50]);
        //self.draw_velocity_debug(&state, &grid);
    }

    fn clear(&self) {
        clear_background(BLACK);
    }

    fn draw_cells(&self, state: &State2D, grid: &Grid2D) {
        let cell_width = self.cellsize.0 - self.padding * 2.0;
        let cell_height = self.cellsize.1 - self.padding * 2.0;
        for cell_idx in grid.total_cell_iter() {
            let pos_x = self.viewport_pos.0 + (cell_idx.idx[0] as f32) * self.cellsize.0 + self.padding;
            let pos_y = self.viewport_pos.1 + (cell_idx.idx[1] as f32) * self.cellsize.1 + self.padding;

            let color = 
            if state.cell_types.data[grid.linearize_cell_index(&cell_idx)] == crate::engine::state::field::CellType::Solid {
                self.visuals.solid_cell
            } else { self.visuals.fluid_cell };

            draw_rectangle(pos_x, pos_y, cell_width, cell_height, color);
        }
    }

    fn draw_smoke_map(&self, state: &State2D, grid: &Grid2D) {
        let cell_width = self.cellsize.0 - self.padding * 2.0;
        let cell_height = self.cellsize.1 - self.padding * 2.0;
        for cell_idx in grid.total_cell_iter() {
            let pos_x = self.viewport_pos.0 + (cell_idx.idx[0] as f32) * self.cellsize.0 + self.padding;
            let pos_y = self.viewport_pos.1 + (cell_idx.idx[1] as f32) * self.cellsize.1 + self.padding;

            let color = Self::scale_color(&self.visuals.smoke_color, state.smoke.data[grid.linearize_cell_index(&cell_idx)] as f32);

            draw_rectangle(pos_x, pos_y, cell_width, cell_height, color);
        }
    }

    fn draw_hr_field(
        &self,
        data: &Field<f32, CellIndex<2>>,
        grid: &Grid2D,
        res: [usize; 2],
    ) {
        let nx = grid.get_size(0) as f32;
        let ny = grid.get_size(1) as f32;

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
                    (u * nx) as f32 * grid.get_delta_x(),
                    (v * ny) as f32 * grid.get_delta_x(),
                ];

                let pos_x = self.viewport_pos.0 + u * self.viewport_size.0;
                let pos_y = self.viewport_pos.1 + v * self.viewport_size.1;

                let value = data.sample(grid, pos) as f32;

                let color = Self::scale_color(&self.visuals.smoke_color, value);

                draw_rectangle(pos_x, pos_y, cell_width, cell_height, color);
            }
        }
    }

    fn draw_velocity_debug(&self, state: &State2D, grid: &Grid2D) {
        let bar_thickness: f32 = 5.0;
        for axis in 0..2 {
            let offset =
            if axis == 1 {
                (self.cellsize.0 / 2.0 - bar_thickness / 2.0, 0.0)
            } else {
                (0.0, self.cellsize.1 / 2.0 - bar_thickness / 2.0)
            };

            for mut face_idx in grid.total_face_iter(axis) {
                face_idx.axis = axis;

                let pos_x = self.viewport_pos.0 + (face_idx.idx[0] as f32) * self.cellsize.0 + offset.0 + self.padding;
                let pos_y = self.viewport_pos.1 + (face_idx.idx[1] as f32) * self.cellsize.1 + offset.1 + self.padding;

                let bar_length = (state.velocity[axis].get(&grid, &face_idx) * 10.0) as f32;

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

    fn scale_color(col: &Color, value: f32) -> Color {
        Color::from_rgba(
            (col.r * value * 255.0) as u8,
            (col.g * value * 255.0) as u8,
            (col.b * value * 255.0) as u8,
            (col.a * value * 255.0) as u8)
    }
}

