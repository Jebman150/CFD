use std::{collections::HashMap, sync::Arc};

use anyhow::Error;
use wgpu::{BindGroup, BindGroupLayoutEntry, ComputePipeline, Device};

use crate::ParamID;

use super::init_helper;

pub struct ParameterLayout {
    f32_params: Vec<ParamID>,
    u32_params: Vec<ParamID>,
}

impl ParameterLayout {
    pub fn new(f32_params: Vec<ParamID>, u32_params: Vec<ParamID>) -> Self {
        Self {
            f32_params,
            u32_params
        }
    }

    pub fn create_f32_buffer(&self, device: &Arc<Device>) -> wgpu::Buffer {
        init_helper::create_storage_buffer::<f32>(
            device,
            "Float parameter buffer", 
            self.f32_params.len()
        )
    }

    pub fn create_u32_buffers(&self, device: &Arc<Device>) -> wgpu::Buffer {
        init_helper::create_storage_buffer::<u32>(
            device,
            "Unsigned integer parameter buffer", 
            self.u32_params.len()
        )
    }

    pub fn get_f32_params(&self) -> &[ParamID] {
        &self.f32_params
    }

    pub fn get_u32_params(&self) -> &[ParamID] {
        &self.u32_params
    }
}

pub trait ComputeKernel {
    fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder
    ) -> anyhow::Result<()>;

    fn get_name(&self) -> &'static str;

    fn get_helper_buffer(&self) -> &[wgpu::Buffer];
}

#[derive(Debug, Clone)]
pub struct LeafKernel {
    name: &'static str,
    pipeline: Option<ComputePipeline>,
    bind_group: Option<BindGroup>,
    helper_buffer: Vec<wgpu::Buffer>,
    work_size: u64,
}

impl LeafKernel {
    pub fn new(name: &'static str) -> Self {
        Self {
            name,
            pipeline: None,
            bind_group: None,
            helper_buffer: Vec::new(),
            work_size: 0,
        }
    }

    pub fn build_pipeline(
        mut self,
        device: &Arc<Device>,
        shader_source: &str,
        entry_point: &str,
        buffer_layout: &[BindGroupLayoutEntry],
        binding_list: &[(u32, &wgpu::Buffer)],
        work_size: u64
    ) -> anyhow::Result<Self> {
        let shader_code = std::fs::read_to_string(shader_source)?;
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(&format!("{} shader module", self.name)),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(&shader_code)),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(&format!("{} bind group layout", self.name)),
            entries: &buffer_layout
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(&format!("{} pipeline layout", self.name)),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        self.pipeline = Some(device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some(&format!("{} compute pipeline", self.name)),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: Some(entry_point),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            }));

        self.bind_group = Some(init_helper::create_bind_group(&device, &binding_list, &bind_group_layout));

        self.work_size = work_size;

        Ok(self)
    }

    pub fn save_helper_buffer(mut self, buffer: wgpu::Buffer) -> Self {
        self.helper_buffer.push(buffer);
        self
    }
}

impl ComputeKernel for LeafKernel {
    
    fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,

    ) -> anyhow::Result<()> {
        let Some(pipeline) = &self.pipeline else { return Err(Error::msg("No pipeline initialized!")); };
        let Some(bind_group) = &self.bind_group else { return Err(Error::msg("No bind group created!")); };

        let mut compute_pass =
            encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("{} compute Pass", self.name)),
                timestamp_writes: None,
            });

        compute_pass.set_pipeline(pipeline);
        compute_pass.set_bind_group(0, bind_group, &[]);


        compute_pass.dispatch_workgroups(self.work_size.div_ceil(64) as u32, 1, 1);
        Ok(())
    }

    fn get_name(&self) -> &'static str {
        &self.name
    }
    
    fn get_helper_buffer(&self) -> &[wgpu::Buffer] {
        &self.helper_buffer
    }
    
}

pub struct MultiStageKernel {
    name: &'static str,
    preprocessor: Vec<LeafKernel>,
    stages: Vec<LeafKernel>,
    postprocessor: Vec<LeafKernel>,
    iteration_count: usize,
    helper_buffer: HashMap<usize, wgpu::Buffer>,
}

impl MultiStageKernel {
    pub fn new(
        name: &'static str,
        preprocessor: Vec<LeafKernel>, 
        stages: Vec<LeafKernel>,
        postprocessor: Vec<LeafKernel>,
        iterations: usize,
        helper_buffers: HashMap<usize, wgpu::Buffer>
    ) -> Self {
        Self {
            name,
            preprocessor,
            stages,
            postprocessor,
            iteration_count: iterations,
            helper_buffer: helper_buffers,
        }
    }
}

impl ComputeKernel for MultiStageKernel {
    fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder
    ) -> anyhow::Result<()> {
        for prep in &self.preprocessor {
            prep.encode(encoder)?;
        }
        for _iter in 0..self.iteration_count {
            for stage in &self.stages {
                stage.encode(encoder)?;
            }
        }
        for postp in &self.postprocessor {
            postp.encode(encoder)?;
        }
        Ok(())
    }

    fn get_name(&self) -> &'static str {
        &self.name
    }
    
    fn get_helper_buffer(&self) -> &[wgpu::Buffer] {
        todo!()
    }

    
}