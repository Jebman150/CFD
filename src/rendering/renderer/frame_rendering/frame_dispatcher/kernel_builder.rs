use std::{collections::HashMap, fs, hash::Hash, sync::Arc};

use anyhow::Error;
use wgpu::{BindGroup, BindGroupLayout, BindGroupLayoutEntry, Device, PolygonMode, PrimitiveTopology, RenderPipeline, ShaderModule, TextureFormat};

use crate::rendering::renderer::scene::{Instance, mesh::Vertex};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KernelID(usize);

impl std::fmt::Display for KernelID {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BindGroupID(usize);

impl std::fmt::Display for BindGroupID {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

pub struct RenderKernel {
    pipeline: RenderPipeline,
    bind_group_layout: BindGroupLayout,
}

pub struct BindGroupManager {
    device: Arc<Device>,

    bind_groups: HashMap<BindGroupID, BindGroup>
}

impl BindGroupManager {
    pub fn new(device: Arc<Device>) -> Self {
        Self {
            device,

            bind_groups: HashMap::new(),
        }
    }

    pub fn create_bind(&mut self, buffer_list: &Vec<(u32, &wgpu::Buffer)>, layout: &BindGroupLayout) -> BindGroupID {
        let mut group_entries = Vec::new();

        for buffer in buffer_list.iter() {
            group_entries.push(wgpu::BindGroupEntry {
                binding: buffer.0,
                resource: buffer.1.as_entire_binding(),
            });
        }

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bind group"),
            layout: &layout,
            entries: &group_entries
        });

        let id = BindGroupID(self.bind_groups.len());
        self.bind_groups.insert(id, bind_group);
        id
    }

    pub fn get_bind(&self, bind_group_id: BindGroupID) -> anyhow::Result<&BindGroup> {
        self.bind_groups.get(&bind_group_id)
            .ok_or(
                Error::msg(format!("No bind group with ID {}", bind_group_id))
            )
    }
}


pub struct RenderKernelManager {
    device: Arc<Device>,
    kernels: HashMap<KernelID, RenderKernel>,
}

impl RenderKernelManager {
    pub fn new(device: Arc<Device>) -> Self {
        Self {
            device,

            kernels: HashMap::new(),
        }
    }

    pub fn new_kernel(&mut self, shader_source: &str) -> RenderKernelBuilder<'_> {
        RenderKernelBuilder::new(self, self.device.clone(), shader_source)
    }

    pub fn add_kernel(&mut self, kernel: RenderKernel) -> KernelID {
        let id = KernelID(self.kernels.len());
        self.kernels.insert(id, kernel);
        id
    }

    pub fn get_layout(&self, kernel: KernelID) -> anyhow::Result<&BindGroupLayout> {
        self.kernels
            .get(&kernel)
            .map(|k| &k.bind_group_layout)
            .ok_or(
                Error::msg(format!("No kernel with ID {}", kernel))
            )
    }

    pub fn get_kernel_pipeline(&self, kernel: KernelID) -> anyhow::Result<&RenderPipeline> {
        self.kernels
            .get(&kernel)
            .map(|k| &k.pipeline)
            .ok_or(
                Error::msg(format!("No kernel with ID {}", kernel))
            )
    }
}

pub struct RenderKernelBuilder<'a> {
    device: Arc<Device>,

    shader_module: ShaderModule,
    buffer_layout_entries: Vec<BindGroupLayoutEntry>,
    
    manager: &'a mut RenderKernelManager,
}

impl<'a> RenderKernelBuilder<'a> {
    pub fn new(manager: &'a mut RenderKernelManager, device: Arc<Device>, shader_source: &str) -> Self {
        let shader_code = fs::read_to_string(shader_source).unwrap();
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Rendering Shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(&shader_code)),
        });

        Self {
            device,
            shader_module,
            buffer_layout_entries: Vec::new(),

            manager,
        }
    }

    pub fn add_buffer(&mut self, binding: u32) -> &mut Self {
        self.buffer_layout_entries.push(wgpu::BindGroupLayoutEntry {
            binding: binding,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });

        self
    }

    pub fn add_uniform(&mut self, binding: u32) -> &mut Self {
        self.buffer_layout_entries.push(wgpu::BindGroupLayoutEntry {
            binding: binding,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });

        self
    }

    pub fn create(&mut self, surface_format: TextureFormat, polygon_mode: PolygonMode, topology: PrimitiveTopology) -> KernelID {
        let bind_group_layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Render bind group layout"),
            entries: &self.buffer_layout_entries
        });

        let pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render pipeline Layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = self.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &self.shader_module,
                entry_point: Some("vs_main"),
                buffers: &[Some(Vertex::layout()), Some(Instance::layout())],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &self.shader_module,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: topology,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: polygon_mode,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview_mask: None,
            cache: None,
        });

        self.manager.add_kernel(
            RenderKernel {
                bind_group_layout,
                pipeline,
            }
        )
    }
}