use std::sync::Arc;

use wgpu::CommandEncoder;

use crate::{GPUContext, ParameterLayout, gpu::init_helper, state::parameter_registry::ParameterRegistry};

#[derive(Debug, Clone)]
pub struct BufferCopy {
    source: wgpu::Buffer,
    buffer: wgpu::Buffer,
}

impl BufferCopy {
    pub fn new(gpu_context: &Arc<GPUContext>, source: &wgpu::Buffer) -> Self {
        Self {
            source: source.clone(),
            buffer: init_helper::create_storage_buffer_with_capacity(
                &gpu_context.device,
                "Buffer copy",
                source.size()
            ),
        }
    }

    pub fn encode_update(&self, command_encoder: &mut CommandEncoder) {
        tracing::info!("Updating buffer copy");
        command_encoder.copy_buffer_to_buffer(
            &self.source,
            0, 
            &self.buffer, 
            0, 
            self.source.size()
        );
    }

    pub fn get_buffer_ref(&self) -> &wgpu::Buffer {
        &self.buffer
    }
}

#[derive(Debug, Clone)]
pub struct ParameterResource {
    u32_params: wgpu::Buffer,
    f32_params: wgpu::Buffer,

    layout: ParameterLayout,
    gpu_context: Arc<GPUContext>
}

impl ParameterResource {
    pub fn new(gpu_context: &Arc<GPUContext>, layout: ParameterLayout) -> Self {
        let u32_count = layout.get_u32_params().len();
        let f32_count = layout.get_f32_params().len();

        Self {
            u32_params: init_helper::create_storage_buffer_with_capacity(
                &gpu_context.device,
                "U32 parameter buffer",
                (std::mem::size_of::<u32>() * u32_count) as u64
            ),
            f32_params: init_helper::create_storage_buffer_with_capacity(
                &gpu_context.device,
                "F32 parameter buffer",
                (std::mem::size_of::<f32>() * f32_count) as u64
            ),
            layout,
            gpu_context: gpu_context.clone()
        }
    }

    pub fn update(&self, parameter_registry: &ParameterRegistry) {
        let param_u32_vec = self.layout.get_u32_params().iter()
            .map(|id| parameter_registry.get_param_u32(id.clone()).unwrap())
            .collect::<Vec<u32>>();

        let param_f32_vec = self.layout.get_f32_params().iter()
            .map(|id| parameter_registry.get_param_f32(id.clone()).unwrap())
            .collect::<Vec<f32>>();

        self.gpu_context.queue.write_buffer(
            &self.u32_params,
            0,
            bytemuck::cast_slice(&param_u32_vec)
        );

        self.gpu_context.queue.write_buffer(
            &self.f32_params,
            0,
            bytemuck::cast_slice(&param_f32_vec)
        );
    }

    pub fn get_u32_buffer_ref(&self) -> &wgpu::Buffer {
        &self.u32_params
    }

    pub fn get_f32_buffer_ref(&self) -> &wgpu::Buffer {
        &self.f32_params
    }
}