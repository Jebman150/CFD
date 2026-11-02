use wgpu::Color;

use crate::rendering::renderer::scene::mesh::{MeshData, Vertex};

fn color_to_array(c: &Color) -> [f32; 4] {
    [c.r as f32, c.g as f32, c.b as f32, c.a as f32]
}


pub fn rectangle(w: f32, h: f32, color: Color) -> MeshData {
    let color = color_to_array(&color);
    MeshData {
        vertices: vec![
            Vertex { position: [0.0, 0.0], color },
            Vertex { position: [0.0, h], color },
            Vertex { position: [w, h], color },
            Vertex { position: [w, 0.0], color },
        ],
        indices: vec![0, 1, 3, 1, 2, 3],
    }
}

pub fn grid(w: f32, h: f32, x_res: usize, y_res: usize, color: Color) -> MeshData {
    let color = color_to_array(&color);
    let mut vertices = Vec::new();
    for y in 0..y_res+1 {
        let y_frac = y as f32 / y_res as f32;
        for x in 0..x_res+1 {
            let x_frac = x as f32 / x_res as f32;
            vertices.push(Vertex {position: [w * x_frac, h * y_frac], color: [x_frac, 0.0, y_frac, 1.0]});
        }
    }

    let mut indices = Vec::new();
    for y in 0..y_res as u32 {
        for x in 0..x_res as u32 {
            let upper_left_idx = y * (x_res + 1) as u32 + x;
            indices.push(upper_left_idx);
            indices.push(upper_left_idx + x_res as u32 + 1);
            indices.push(upper_left_idx + 1);
            
            indices.push(upper_left_idx + x_res as u32 + 1);
            indices.push(upper_left_idx + x_res as u32 + 2);
            indices.push(upper_left_idx + 1);
        }
    }

    MeshData {
        vertices,
        indices
    }
}