use crate::{rendering::renderer::{frame_rendering::draw_batch::DrawBatch, scene::Instance}};



pub struct PreparedFrame {
    pub instances: Vec<Instance>,
    pub batches: Vec<DrawBatch>,
}