use crate::rendering::renderer::scene::mesh::MeshId;



pub struct DrawBatch {
    pub mesh_id: MeshId,
    pub first_instance: u32,
    pub instance_count: u32,
}