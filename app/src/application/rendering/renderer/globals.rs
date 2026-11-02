pub struct RenderGlobals {
    pub simulation_viewport: Viewport,
}

impl Default for RenderGlobals {
    fn default() -> Self {
        Self {
            simulation_viewport: Viewport { position: [0.0, 0.0], size: [1000.0, 1000.0] }
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Viewport {
    pub position: [f32; 2],
    pub size: [f32; 2],
}

impl Viewport {
    pub fn new() -> Self {
        Self {
            position: [0.0, 0.0],
            size: [10.0, 10.0],
        }
    }

    pub fn set_size(&mut self, size: [f32; 2]) {
        self.size = size;
    }
}