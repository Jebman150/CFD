use std::{collections::HashMap, sync::Arc};

use wgpu::{Buffer, BufferUsages, Device, util::{BufferInitDescriptor, DeviceExt}};

use crate::rendering::renderer::scene::mesh::{MeshData, MeshId, Vertex};

pub struct GPUMesh {
    pub vertex_buffer: Buffer,
    pub index_buffer: Buffer,
    pub index_count: u32,
}

impl GPUMesh {
    pub fn from_init_data(vertices: &[Vertex], indices: &[u32], device: &Device) -> Self {
        let vertex_bytes = bytemuck::cast_slice(vertices);
        let index_bytes = bytemuck::cast_slice(indices);

        let vertex_buffer = device.create_buffer_init(
            &BufferInitDescriptor {
                label: Some("Vertex Buffer"),
                contents: vertex_bytes,
                usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
            }
        );

        let index_buffer = device.create_buffer_init(
            &BufferInitDescriptor {
                label: Some("Index Buffer"),
                contents: index_bytes,
                usage: BufferUsages::INDEX | BufferUsages::COPY_DST,
            }
        );

        Self {
            vertex_buffer,
            index_buffer,
            index_count: indices.len() as u32,
        }
    }
}

pub struct MeshRegistry {
    meshes: HashMap<MeshId, GPUMesh>,

    device: Arc<Device>,
}

impl MeshRegistry {
    pub fn new(device: Arc<Device>) -> Self {
        Self {
            meshes: HashMap::new(),
            device,
        }
    }

    pub fn register_mesh(&mut self, mesh: MeshData) -> MeshId {
        let gpu_mesh = GPUMesh::from_init_data(&mesh.vertices, &mesh.indices, &self.device);
        let mesh_id = MeshId(self.meshes.len() as u32);
        self.meshes.insert(mesh_id, gpu_mesh);
        mesh_id
    }

    pub fn get_mesh(&self, mesh_id: &MeshId) -> Option<&GPUMesh> {
        self.meshes.get(mesh_id)
    }

    pub fn clear(&mut self) {
        self.meshes.clear();
    }
}