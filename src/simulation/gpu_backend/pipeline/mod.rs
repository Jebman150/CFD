use std::time::Instant;

use anyhow::Error;

use crate::simulation::{data::{
        Frame, SolverSettings
    }, gpu_backend::pipeline::operator::SimulationOperator};

pub mod operator;
pub mod gpu_helper;

pub struct SimulationPipeline {
    operations: Vec<Box<dyn SimulationOperator>>,
}

impl SimulationPipeline {
    pub fn new() -> Self {
        Self {
            operations: Vec::new(),
        }
    }

    pub fn add_step(&mut self, op: Box<dyn SimulationOperator>) -> &mut Self {
        self.operations.push(op);
        self
    }

    pub fn run(&mut self, frame: &mut Frame) -> anyhow::Result<u128> {
        let start = Instant::now();
        for op in &mut self.operations {
            if let Err(msg) = op.encode(frame.command_encoder) {
                return Err(Error::msg(format!("{} failed with msg: {}", op.get_name(), msg)));
            }
        }
        Ok(start.elapsed().as_micros())
    }

    pub fn set_solver(&mut self) -> Result<(), String> {

        Ok(())
    }

    pub fn modify_solver_settings(&mut self, _settings: &SolverSettings) -> Result<(), String> {

        Ok(())
    }
}