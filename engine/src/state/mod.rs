use std::{collections::HashMap, hash::Hash, sync::Arc};

use anyhow::Error;
use pollster::FutureExt;
use tracing::instrument;
use wgpu::{Device, Queue};

use crate::{GPUContext, ParamID, state::{fields::{Field, FieldMetadata, SIUnitDimensions, VectorFieldStorage, layout::GridLayout}, parameter_registry::ParameterRegistry}};

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
        tracing::debug!("Initializing simulation state");
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
        tracing::debug!("Inserted new scalar field with id {}", new_id.0);
        tracing::debug!(" - SI Units:    {}", si_units);
        tracing::debug!(" - Size:        {:?}", size);
        self.scalar_id_counter += 1;
        new_id
    }

    pub fn add_vector_field(&mut self, si_units: SIUnitDimensions, size: [usize; 2]) -> VectorFieldID {
        let new_id = VectorFieldID(self.vector_id_counter);
        let layout = fields::layout::VectorLayout::new(size, 0);
        self.vector_fields.insert(new_id, VectorFieldStorage::Collocated(
            Field::new::<[f32; 2]>(&self.gpu_device, si_units, Box::new(layout))
        ));

        tracing::debug!("Inserted new collocated vector field with id {}", new_id.0);
        tracing::debug!(" - SI Units:    {}", si_units);
        tracing::debug!(" - Size:        {:?}", size);
        self.vector_id_counter += 1;
        new_id
    }

    pub fn add_scattered_vector_field(&mut self, si_units: SIUnitDimensions, size: [usize; 2]) -> VectorFieldID {
        let new_id = VectorFieldID(self.vector_id_counter);

        let u_layout = fields::layout::VectorLayout::new(size, 0);
        let v_layout = fields::layout::VectorLayout::new(size, 1);

        self.vector_fields.insert(
            new_id,
            VectorFieldStorage::Scattered([
                Field::new::<f32>(&self.gpu_device, si_units, Box::new(u_layout.clone())),
                Field::new::<f32>(&self.gpu_device, si_units, Box::new(v_layout.clone())),
            ])
        );

        tracing::debug!("Inserted new scattered vector field with id {}", new_id.0);
        tracing::debug!(" - SI Units:    {}", si_units);
        tracing::debug!(" - U Size:        {:?}", u_layout.dimensions());
        tracing::debug!(" - V Size:        {:?}", v_layout.dimensions());
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
        field.read_buffer(&self.gpu_device, queue).await
    }

    pub fn write_scalar_field(&self, queue: &Arc<Queue>, id: ScalarFieldID, data: &[f32]) -> anyhow::Result<()> {
        let Some(field) = self.scalar_fields.get(&id) else {
            return Err(Error::msg(format!("Could not find scalar field with id {:?}", id)));
        };
        field.write_buffer(queue, data);
        Ok(())
    }
}