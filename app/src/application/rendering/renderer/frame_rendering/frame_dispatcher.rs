use std::{collections::HashMap, sync::Arc};

use anyhow::Error;
use strum::IntoEnumIterator;
use wgpu::{Buffer, BufferDescriptor, BufferUsages, CommandEncoder, Device, Queue, RenderPass, SurfaceConfiguration, TextureView};

use crate::{rendering::renderer::{frame_rendering::{frame_dispatcher::kernel_builder::{BindGroupID, BindGroupManager, KernelID, RenderKernelManager}, prepared_frame::PreparedFrame}, globals::Viewport, mesh_registry::MeshRegistry, scene::Instance, settings::VisualParameter}, simulation::data::{ScalarField, SimulationData}};

mod kernel_builder;

pub struct Frame <'frame> {
    pub render_pass: RenderPass<'frame>,
    pub prepared_data: PreparedFrame,
    pub rendered_scalar_field: Option<ScalarField>
}

struct DynamicBuffer {
    pub buffer: Buffer,
    pub capacity: u64
}

impl DynamicBuffer {
    pub fn new(device: &Arc<Device>, capacity: u64) -> Self {
        let buffer = device.create_buffer(
            &BufferDescriptor {
                label: Some("Instance Buffer"),
                size: capacity,
                usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }
        );

        Self {
            buffer,
            capacity
        }
    }

    pub fn grow_to_capacity(&mut self, device: &Arc<Device>, required: u64) {
        while self.capacity < required {
            self.capacity *= 2;
        }
        self.buffer = device.create_buffer(
            &BufferDescriptor {
                label: Some("Instance Buffer"),
                size: self.capacity,
                usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }
        );
    }
}

pub struct FrameDispatcher {
    kernel_manager: RenderKernelManager,
    bind_group_manager: BindGroupManager,

    instance_buffer: DynamicBuffer,

    viewport_buffer: Buffer,

    bind_group_map: HashMap<ScalarField, BindGroupID>,
    grid_kernel: KernelID,

    queue: Arc<Queue>,
    device: Arc<Device>,
}

impl FrameDispatcher {
    pub fn new(
        device: Arc<Device>,
        queue: Arc<Queue>,
        surface_config: &SurfaceConfiguration,
        simulation_buffer: &SimulationData,
        visuals: &VisualParameter,
    ) -> Self {
        let mut kernel_manager = RenderKernelManager::new(device.clone());
        let mut bind_group_manager = BindGroupManager::new(device.clone());

        let grid_kernel = kernel_manager.new_kernel("shaders/graphics/render_grid.wgsl")
            .add_uniform(0)
            .add_uniform(1)
            .add_buffer(2)
            .create(surface_config.format,
                wgpu::PolygonMode::Fill,
                wgpu::PrimitiveTopology::TriangleList
            );

        let mut default_viewport = Viewport::new();
        default_viewport.set_size([1000.0, 1000.0]);

        let viewport_buffer = create_uniform_buffer::<Viewport>(&device);

        let bind_group_map = ScalarField::iter()
            .map(|f| (
                f,
                bind_group_manager.create_bind(&vec![
                    (0, &viewport_buffer),
                    (1, &visuals.scalar_field_gradient.get(&f).unwrap()),
                    (2, &simulation_buffer.scalar_fields.get(&f).unwrap())
                ], kernel_manager.get_layout(grid_kernel).unwrap())
            )).collect();

        Self {
            instance_buffer: DynamicBuffer::new(&device, 128),
            grid_kernel,

            kernel_manager,
            bind_group_manager,

            viewport_buffer,
            bind_group_map,

            queue,
            device,
        }
    }

    pub fn update_viewport(&self, viewport: &Viewport) {
        self.queue.write_buffer(
            &self.viewport_buffer,
            0,
            bytemuck::bytes_of(viewport)
        );
    }

    pub fn initialize_pass<'frame>(&self, encoder: &'frame mut CommandEncoder, target: &TextureView) -> RenderPass<'frame> {
        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &target,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.001,
                        g: 0.001,
                        b: 0.002,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
            multiview_mask: None,
        })
    }

    pub fn prepare_instance_buffer(&mut self, prepared_frame: &PreparedFrame) {
        let required_space = prepared_frame.instances.len() as u64 * Instance::bytes();
        self.instance_buffer.grow_to_capacity(&self.device, required_space);
        if self.instance_buffer.capacity < prepared_frame.instances.len() as u64 * Instance::bytes() {
            tracing::error!("Instance buffer not large enough ({} < {})", self.instance_buffer.capacity, prepared_frame.instances.len() as u64 * Instance::bytes());
            return;
        }
        self.queue.write_buffer(&self.instance_buffer.buffer, 0, bytemuck::cast_slice(&prepared_frame.instances));
    }

    pub fn dispatch_grid_kernel(&self, mut frame: Frame, mesh_registry: &MeshRegistry) -> anyhow::Result<()> {
        let Some(rendered_field) = frame.rendered_scalar_field else { return Ok(()); };
        let Some(bind_group_id) = self.bind_group_map.get(&rendered_field) else {
            return Err(Error::msg(format!("No bind group generated for field {}", rendered_field)));
        }; 
        let render_pass = &mut frame.render_pass;

        render_pass.set_pipeline(self.kernel_manager.get_kernel_pipeline(self.grid_kernel)?);
        render_pass.set_bind_group(0, self.bind_group_manager.get_bind(*bind_group_id)?, &[]);

        render_pass.set_vertex_buffer(1, self.instance_buffer.buffer.slice(..));

        for batch in &frame.prepared_data.batches {
            let Some(mesh) = mesh_registry.get_mesh(&batch.mesh_id) else {
                continue;
            };

            render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
            render_pass.set_index_buffer(
                mesh.index_buffer.slice(..),
                wgpu::IndexFormat::Uint32,
            );

            let end = batch.first_instance + batch.instance_count;

            render_pass.draw_indexed(
                0..mesh.index_count,
                0,
                batch.first_instance..end,
            );
        }
        Ok(())
    }
}

fn create_uniform_buffer<T>(device: &Arc<Device>) -> Buffer {
    device.create_buffer(
            &wgpu::BufferDescriptor {
                label: Some("Uniform Buffer"),
                size: std::mem::size_of::<T>() as u64,
                usage: wgpu::BufferUsages::UNIFORM
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            },
        )
}