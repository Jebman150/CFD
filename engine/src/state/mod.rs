use std::{collections::HashMap, hash::Hash, sync::Arc};

use anyhow::Error;
use tracing::instrument;
use wgpu::{Device, Queue};

use crate::{GPUContext, ParamID, state::{fields::{Field, FieldMetadata, SIUnitDimensions, VectorFieldStorage}, parameter_registry::ParameterRegistry}};

pub mod fields;
pub mod parameter_registry;


#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ScalarFieldID(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VectorFieldID(pub usize);

pub struct Parameter {
    value: f32,
}

pub struct SimulationState {
    gpu_device: Arc<Device>,
    parameters: ParameterRegistry,
    scalar_fields: HashMap<ScalarFieldID, Field>,
    vector_fields: HashMap<VectorFieldID, VectorFieldStorage>,

    scalar_id_counter: usize,
    vector_id_counter: usize,
}

impl SimulationState {
    #[instrument]
    pub fn new(gpu_context: &Arc<GPUContext>) -> anyhow::Result<Self> {
        tracing::info!("Initializing simulation state");
        Ok(Self {
            gpu_device: gpu_context.device.clone(),
            parameters: ParameterRegistry::new(),
            scalar_fields: HashMap::new(),
            vector_fields: HashMap::new(),

            scalar_id_counter: 0,
            vector_id_counter: 0,
        })
    }

    pub fn add_scalar_field(&mut self, si_units: SIUnitDimensions, size: [usize; 2]) -> ScalarFieldID {
        let new_id = ScalarFieldID(self.scalar_id_counter);
        let layout = fields::layout::ScalarLayout::new(size);
        self.scalar_fields.insert(new_id, Field::new::<f32>(&self.gpu_device, si_units, Box::new(layout)));
        tracing::info!("Inserted new scalar field with id {}", new_id.0);
        tracing::info!(" - SI Units:    {}", si_units);
        tracing::info!(" - Size:        {:?}", size);
        self.scalar_id_counter += 1;
        new_id
    }

    pub fn add_vector_field(&mut self, si_units: SIUnitDimensions, size: [usize; 2]) -> VectorFieldID {
        let new_id = VectorFieldID(self.vector_id_counter);
        let layout = fields::layout::VectorLayout::new(size, 0);
        self.vector_fields.insert(new_id, VectorFieldStorage::Collocated(
            Field::new::<[f32; 2]>(&self.gpu_device, si_units, Box::new(layout))
        ));

        tracing::info!("Inserted new collocated vector field with id {}", new_id.0);
        tracing::info!(" - SI Units:    {}", si_units);
        tracing::info!(" - Size:        {:?}", size);
        self.vector_id_counter += 1;
        new_id
    }

    pub fn add_scattered_vector_field(&mut self, si_units: SIUnitDimensions, size: [usize; 2]) -> VectorFieldID {
        let new_id = VectorFieldID(self.vector_id_counter);

        let mut u_dim = size;
        u_dim[0] += 1;
        let u_layout = fields::layout::VectorLayout::new(u_dim, 0);
        let mut v_dim = size;
        v_dim[1] += 1;
        let v_layout = fields::layout::VectorLayout::new(v_dim, 1);

        self.vector_fields.insert(
            new_id,
            VectorFieldStorage::Scattered([
                Field::new::<f32>(&self.gpu_device, si_units, Box::new(u_layout)),
                Field::new::<f32>(&self.gpu_device, si_units, Box::new(v_layout)),
            ])
        );

        tracing::info!("Inserted new scattered vector field with id {}", new_id.0);
        tracing::info!(" - SI Units:    {}", si_units);
        tracing::info!(" - U Size:        {:?}", u_dim);
        tracing::info!(" - V Size:        {:?}", v_dim);
        self.vector_id_counter += 1;
        new_id
    }

    pub fn get_scalar_field_metadata(&self, id: ScalarFieldID) -> anyhow::Result<&FieldMetadata> {
        self.scalar_fields.get(&id).map(|f| f.metadata()).ok_or(Error::msg(format!("No field with id {:?}", id)))
    }

    pub fn get_vector_field_metadata(&self, id: VectorFieldID) -> anyhow::Result<&FieldMetadata> {
        self.vector_fields.get(&id).map(|f| 
            match f {
                VectorFieldStorage::Collocated(field) => field.metadata(),
                VectorFieldStorage::Scattered(fields) => {
                    fields[0].metadata()
                }  
            }).ok_or(Error::msg(format!("No field with id {:?}", id)))
    }

    pub fn get_vector_field(&self, id: VectorFieldID) -> anyhow::Result<&VectorFieldStorage> {
        self.vector_fields.get(&id).ok_or(Error::msg(format!("No vector field with id {:?}", id)))
    }

    pub fn get_scalar_field(&self, id: ScalarFieldID) -> anyhow::Result<&Field> {
        self.scalar_fields.get(&id).ok_or(Error::msg(format!("No scalar field with id {:?}", id)))
    }

    pub fn get_f32_parameter_unit(&self, id: ParamID) -> anyhow::Result<SIUnitDimensions> {
        self.parameters.get_param_f32_units(id)
    }

    pub fn get_u32_parameter_unit(&self, id: ParamID) -> anyhow::Result<SIUnitDimensions> {
        self.parameters.get_param_u32_units(id)
    }

    pub fn parameter_mut(&mut self) -> &mut ParameterRegistry {
        &mut self.parameters
    }

    pub fn parameter_ref(&self) -> &ParameterRegistry {
        &self.parameters
    }

    pub async fn read_scalar_field(&self, queue: &Arc<Queue>, id: ScalarFieldID) -> anyhow::Result<Vec<f32>> {
        let Some(field) = self.scalar_fields.get(&id) else {
            return Err(Error::msg(format!("Could not find scalar field with id {:?}", id)));
        };

        let staging_buffer = self.gpu_device.create_buffer(
            &wgpu::BufferDescriptor {
                label: Some("Readback staging buffer"),
                size: field.get_buffer_ref().size(),
                usage: wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            },
        );

        let mut encoder = self.gpu_device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("Readback encoder"),
            },
        );

        encoder.copy_buffer_to_buffer(
            field.get_buffer_ref(),
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

        self.gpu_device.poll(wgpu::wgt::PollType::Wait { submission_index: None, timeout: None })?;

        rx.recv()??;

        let mapped = slice.get_mapped_range()?;

        let result = bytemuck::cast_slice::<u8, f32>(&mapped).to_vec();

        drop(mapped);
        staging_buffer.unmap();

        Ok(result)
    }
}