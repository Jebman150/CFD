use std::{collections::HashMap, sync::Arc};

use strum::IntoEnumIterator;
use wgpu::{BufferUsages, Color, Device, util::BufferInitDescriptor, util::DeviceExt};

use crate::simulation::data::ScalarField;

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ColorGradient {
    pub low: [f32; 4],
    pub mid: [f32; 4],
    pub high: [f32; 4],

    pub low_val: f32,
    pub high_val: f32,
    pub _pad: [f32; 2],
}

impl Default for ColorGradient {
    fn default() -> Self {
        Self {
            low: [0.0, 0.0, 1.0, 1.0],
            mid: [0.0, 0.0, 0.0, 1.0],
            high: [1.0, 0.0, 0.0, 1.0],
            low_val: -1.0,
            high_val: 1.0,
            _pad: [0.0; 2]
        }
    }
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

#[derive(Clone)]
pub struct VisualParameter {
    pub fluid_cell: Color,
    pub solid_cell: Color,
    pub smoke_color: Color,

    pub scalar_field_gradient: HashMap<ScalarField, wgpu::Buffer>,
}

impl VisualParameter {
    fn default(device: &Arc<Device>) -> Self {
        let mut scalar_field_gradient = HashMap::new();

        let velocity_gradient= ColorGradient {
            low: [0.0, 0.0, 0.7, 1.0],
            mid: [0.0, 0.7, 0.0, 1.0],
            high: [0.7, 0.0, 0.0, 1.0],
            low_val: 0.0,
            high_val: 2.0,
            _pad: [0.0; 2]
        };
        let buffer = device.create_buffer_init(
            &BufferInitDescriptor{
                label: Some("Velocity gradient"),
                contents: bytemuck::bytes_of(&velocity_gradient),
                usage: BufferUsages::UNIFORM,
            }
        );
        scalar_field_gradient.insert(ScalarField::Velocity, buffer);

        for f in ScalarField::iter() {
            if scalar_field_gradient.contains_key(&f) {
                continue;
            }
            let buffer = device.create_buffer_init(
                &BufferInitDescriptor{
                    label: Some(&format!("{} gradient", f)),
                    contents: bytemuck::bytes_of(&ColorGradient::default()),
                    usage: BufferUsages::UNIFORM,
                }
            );
            scalar_field_gradient.insert(f, buffer);
        }

        Self {
            fluid_cell: col_from_rgba(30, 30, 30, 255),
            solid_cell: col_from_rgba(10, 10, 10, 255),
            smoke_color: col_from_rgba(200, 200, 200, 200),
            
            scalar_field_gradient
        }
    }
}


#[derive(Clone)]
pub struct RenderSettings {
    pub padding: f32,
    pub bar_width: f32,

    pub render_streamlines: bool,
    pub render_hr_field: bool,
    pub render_velocity_debug: bool,

    pub rendered_scalar_field: Option<ScalarField>,
    pub visuals: VisualParameter,
}

impl RenderSettings {
    pub fn default(device: &Arc<Device>) -> Self {
        Self {
            padding: 10.0,
            bar_width: 100.0,
            
            render_streamlines: true,
            render_hr_field: false,
            render_velocity_debug: false,

            rendered_scalar_field: Some(ScalarField::Velocity),
            visuals: VisualParameter::default(device),
        }
    }
}