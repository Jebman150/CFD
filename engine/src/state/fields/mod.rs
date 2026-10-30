use std::sync::Arc;

use wgpu::{Device, Queue};

use crate::{gpu, state::fields::layout::GridLayout};

/// Defining an SI Unit dimension with the following convention:
/// 
/// m     kg       s       A      K      mol cd
/// 
/// meter kilogram seconds ampere kelvin mol candela
/// 
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SIUnitDimensions(pub [i32; 7]);
const DIMENSION_LABELS: [&'static str; 7] = ["m", "kg", "s", "A", "K", "mol", "cd"];

pub mod layout;

impl std::fmt::Display for SIUnitDimensions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for d in 0..7 {
            let exponent = self.0[d];
            if exponent != 0 {
                write!(f, "{}^{} ", DIMENSION_LABELS[d], exponent)?;
            }
        }
        Ok(())
    }
}

impl SIUnitDimensions {
    pub const PRESSURE: Self = Self(
        [-1, 1, -2, 0, 0, 0, 0]
    );

    pub const VELOCITY: Self = Self(
        [1, 0, -1, 0, 0, 0, 0]
    );

    pub const NO_UNIT: Self = Self(
        [0, 0, 0, 0, 0, 0, 0]
    );

    pub const LENGTH: Self = Self(
        [1, 0, 0, 0, 0, 0, 0]
    );

    pub const TIME: Self = Self(
        [0, 0, 1, 0, 0, 0, 0]
    );

    /// Divergence is computed as the
    /// spacial derivative, therefore
    /// the spacial dimension is decreased
    /// by 1:
    /// m/s => 1/s
    pub fn div(other: &Self) -> Self {
        let mut dims = other.0;
        dims[0] -= 1;
        Self (dims)
    }
}

pub struct FieldMetadata {
    pub si_unit: SIUnitDimensions,
    pub layout: Box<dyn GridLayout<2>>,
}

pub enum VectorFieldStorage {
    Collocated(Field),
    Scattered([Field; 2]),
}

pub struct Field {
    metadata: FieldMetadata,
    data: wgpu::Buffer,
}

impl Field {
    pub fn new<T>(device: &Arc<Device>, si_unit: SIUnitDimensions, layout: Box<dyn GridLayout<2>>) -> Self 
    where
        T: bytemuck::Pod
    {
        let total_element_num = layout.total_count();
        Self {
            metadata: FieldMetadata { si_unit, layout },
            data: gpu::init_helper::create_storage_buffer::<T>(device, "Field buffer", total_element_num)
        }
    }

    pub fn metadata(&self) -> &FieldMetadata {
        &self.metadata
    }

    pub fn get_buffer_ref(&self) -> &wgpu::Buffer {
        &self.data
    }

    pub async fn read_buffer(&self, device: &Arc<Device>, queue: &Arc<Queue>) -> anyhow::Result<Vec<f32>> {
        let staging_buffer = device.create_buffer(
            &wgpu::BufferDescriptor {
                label: Some("Readback staging buffer"),
                size: self.data.size(),
                usage: wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            },
        );

        let mut encoder = device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("Readback encoder"),
            },
        );

        encoder.copy_buffer_to_buffer(
            &self.data,
            0,
            &staging_buffer,
            0,
            staging_buffer.size(),
        );

        queue.submit(Some(encoder.finish()));

        let slice = staging_buffer.slice(..);

        let (tx, rx) = std::sync::mpsc::channel();

        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });

        device.poll(wgpu::wgt::PollType::Wait { submission_index: None, timeout: None })?;

        rx.recv()??;

        let mapped = slice.get_mapped_range()?;

        let result = bytemuck::cast_slice::<u8, f32>(&mapped).to_vec();

        drop(mapped);
        staging_buffer.unmap();

        Ok(result)
    }

    pub fn write_buffer(&self, queue: &Arc<Queue>, data: &[f32]) {
        queue.write_buffer(
            &self.data,
            0,
            bytemuck::cast_slice(data)
        );
    }
}