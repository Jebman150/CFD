use std::{collections::HashMap, sync::Arc};

use anyhow::Error;
use wgpu::{BindGroup, BindGroupLayout, BindGroupLayoutEntry, ComputePipeline, Device};

use crate::simulation::gpu_backend::pipeline::operator::SimulationOperator;

pub struct ComputeKernel {
    name: &'static str,
    pipeline: Option<ComputePipeline>,
    bind_group: Option<BindGroup>,
    work_size: usize,
}

impl ComputeKernel {
    pub fn new(name: &'static str) -> Self {
        Self {
            name,
            pipeline: None,
            bind_group: None,
            work_size: 0,
        }
    }

    pub fn build_pipeline(
        mut self,
        device: &Arc<Device>,
        shader_source: &str,
        entry_point: &str,
        buffer_layout: &[BindGroupLayoutEntry],
        buffer_list: &[(u32, &wgpu::Buffer)],
        work_size: usize,
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

        self.bind_group = Some(create_bind_group(&device, &buffer_list, &bind_group_layout));

        self.work_size = work_size;
        Ok(self)
    }
}

impl SimulationOperator for ComputeKernel {
    fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder
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
}

pub struct MultiStageKernel {
    name: &'static str,
    stages: Vec<ComputeKernel>,
    iteration_count: usize,
    helper_buffer: HashMap<usize, wgpu::Buffer>,
}

impl SimulationOperator for MultiStageKernel {
    fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder
    ) -> anyhow::Result<()> {
        for _iter in 0..self.iteration_count {
            for stage in &self.stages {
                stage.encode(encoder);
            }
        }
        Ok(())
    }

    fn get_name(&self) -> &'static str {
        &self.name
    }
}

pub fn create_bind_group(device: &Arc<Device>, buffer_list: &[(u32, &wgpu::Buffer)], layout: &BindGroupLayout) -> wgpu::BindGroup {
    let mut group_entries = Vec::new();

    for buffer in buffer_list.iter() {
        group_entries.push(wgpu::BindGroupEntry {
            binding: buffer.0,
            resource: buffer.1.as_entire_binding(),
        });
    }

    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Bind group"),
        layout: layout,
        entries: &group_entries
    })
}

pub fn add_layout_buffer(entry_list: &mut Vec<wgpu::BindGroupLayoutEntry>, binding: u32, read_only: bool) {
    entry_list.push(wgpu::BindGroupLayoutEntry {
        binding: binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    });
}

pub fn add_layout_uniform(entry_list: &mut Vec<wgpu::BindGroupLayoutEntry>, binding: u32) {
    entry_list.push(wgpu::BindGroupLayoutEntry {
        binding: binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    });
}