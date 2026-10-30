use std::sync::Arc;

use wgpu::{BindGroupLayout, BufferUsages, Device};

pub fn create_storage_buffer<T: bytemuck::Pod>(device: &Arc<Device>, label: &str, element_num: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: (std::mem::size_of::<T>() * element_num) as u64,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

pub fn create_storage_buffer_with_capacity(device: &Arc<Device>, label: &str, capacity: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: capacity,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
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