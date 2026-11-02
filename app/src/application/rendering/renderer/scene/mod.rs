use std::collections::HashMap;

use wgpu::{Color, VertexBufferLayout};

use crate::rendering::renderer::{frame_rendering::{draw_batch::DrawBatch, prepared_frame::PreparedFrame}, scene::mesh::MeshId};

pub mod mesh;

fn color_to_array(c: &Color) -> [f32; 4] {
    [c.r as f32, c.g as f32, c.b as f32, c.a as f32]
}

pub struct InstanceId (pub usize);

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Instance {
    pub position: [f32; 2],
    pub angle: f32,
    pub scale: [f32; 2],
    pub color: [f32; 4],
    pub instance_id: u32,
}

impl Instance {
    pub fn new(x: f32, y: f32, color: Color) -> Self {
        Self {
            position: [x, y],
            angle: 0.0,
            scale: [1.0, 1.0],
            color: color_to_array(&color),
            instance_id: 0,
        }
    }

    pub fn layout() -> VertexBufferLayout<'static> {
        const ATTRIBUTES: [wgpu::VertexAttribute; 5] =
            wgpu::vertex_attr_array![
                2 => Float32x2,
                3 => Float32,
                4 => Float32x2,
                5 => Float32x4,
                6 => Uint32,
            ];

        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Instance>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &ATTRIBUTES,
        }
    }

    pub fn bytes() -> u64 {
        36
    }

    pub fn set_color(&mut self, color: Color) {
        self.color = color_to_array(&color);
    }
}

pub struct Scene {
    pub instances: HashMap<mesh::MeshId, Vec<Instance>>,
}

impl Scene {
    pub fn new() -> Self {
        Self {
            instances: HashMap::new(),
        }
    }

    pub fn add_instance(&mut self, mesh_id: MeshId, mut instance: Instance) -> InstanceId {
        return if let Some(instances) = self.instances.get_mut(&mesh_id) {
            let index = instances.len();
            instance.instance_id = index as u32;
            instances.push(instance);
            InstanceId(index)
        } else {
            self.instances.insert(mesh_id, vec![instance]);
            InstanceId(0)
        }
    }

    pub fn set_color(&mut self, mesh: MeshId, instance: InstanceId, color: Color) {
        let Some(instances) = self.instances.get_mut(&mesh) else { return; };
        let Some(instance) = instances.get_mut(instance.0) else { return; };
        instance.set_color(color);
    }

    pub fn prepare_frame(&self) -> PreparedFrame {
        let mut frame = PreparedFrame {
            instances: Vec::new(),
            batches: Vec::new(),
        };

        for (&mesh, instances) in &self.instances {
            if instances.is_empty() {
                continue;
            }

            let first_instance = frame.instances.len() as u32;
            let instance_count = instances.len() as u32;

            frame.instances.extend_from_slice(instances);

            frame.batches.push(DrawBatch {
                mesh_id: mesh,
                first_instance,
                instance_count,
            });
        }

        frame
    }
}